"""Cross-check our CPCB AQI engine against published city AQI values (AQ-02).

WHAT THIS CAN AND CANNOT PROVE
------------------------------
IndiaPopularCity.csv holds daily pollutant readings for 22 cities (2015-2020) plus a
published AQI column. It looks like a perfect test set, but it cannot be used as one,
for a reason worth writing down:

    The pollutant columns are city-level MEANS ACROSS STATIONS.
    The published AQI is a city-level aggregate of PER-STATION AQIs.

AQI of an average is not the average of AQIs -- the formula is piecewise linear with
different slopes per band, so the two disagree in both directions. Chasing an exact
match here would mean tuning a correct engine to reproduce a different calculation.

So the exactness proof lives in `tests/test_aqi.py`, where every expected value is
hand-computed from the CPCB table. THIS script is a consistency check: our AQI should
track the published one closely (high correlation, same category most of the time)
without matching it digit for digit.

It also reports defects it finds in the reference data itself, which is useful to know
before anyone else tries to use that file.

Writes:
    <reports>/aqi_validation.md

Usage:
    python pipeline/validate_aqi.py --raw-root "D:/.../India datasets"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.aqi import CATEGORIES, compute_aqi_frame  # noqa: E402

DEFAULT_REPORTS = REPO_ROOT / "reports"
CITY_FILE = "IndiaPopularCity.csv"

CITY_COLUMN_RENAMES = {
    "PM2.5": "pm2_5", "PM10": "pm10", "NO2": "no2",
    "SO2": "so2", "NH3": "nh3", "CO": "co", "O3": "o3",
}

# What we require of a consistency check. These are deliberately not "within 5",
# because the two calculations are not the same calculation. See the module docstring.
MIN_CORRELATION = 0.90
MIN_WITHIN_50 = 85.0        # percent
MIN_CATEGORY_MATCH = 65.0   # percent

# The CPCB scale stops at 500. Published rows above it were computed without a cap.
AQI_SCALE_MAX = 500

# CO in this file is meant to be mg/m3, where a typical city sits below ~5.
# A city whose median is far above that is reporting something else.
SUSPICIOUS_CO_MEDIAN = 5.0

log = logging.getLogger("validate_aqi")


# ---------------------------------------------------------------------------
# Loading and finding defects in the reference data
# ---------------------------------------------------------------------------

def load_city_data(raw_root: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_root / CITY_FILE, parse_dates=["Date"])
    return df.rename(columns=CITY_COLUMN_RENAMES)


def find_reference_defects(df: pd.DataFrame) -> dict:
    """Problems in the published file, found before we compare anything to it."""
    published = df[df["AQI"].notna()]
    above_scale = published[published["AQI"] > AQI_SCALE_MAX]

    co_medians = df.groupby("City")["co"].median()
    bad_co_cities = sorted(co_medians[co_medians > SUSPICIOUS_CO_MEDIAN].index)

    return {
        "rows_above_scale": len(above_scale),
        "max_published_aqi": float(published["AQI"].max()) if len(published) else 0.0,
        "cities_above_scale": above_scale["City"].value_counts().to_dict(),
        "bad_co_cities": bad_co_cities,
        "co_medians": co_medians,
    }


def compare(df: pd.DataFrame, exclude_cities: list[str]) -> pd.DataFrame:
    """Recompute AQI and line it up against the published value, on comparable rows only."""
    computed = compute_aqi_frame(df)
    result = pd.concat([df[["City", "Date", "AQI", "AQI_Bucket"]], computed], axis=1)

    result = result[result["AQI"].notna()].copy()
    result["comparable"] = (
        result["aqi"].notna()
        & (result["AQI"] <= AQI_SCALE_MAX)
        & ~result["City"].isin(exclude_cities)
    )
    result["difference"] = result["aqi"] - result["AQI"]
    result["category_matches"] = result["aqi_category"] == result["AQI_Bucket"]
    return result


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def score(comparable: pd.DataFrame) -> dict:
    if comparable.empty:
        return {"correlation": 0.0, "within_50": 0.0, "within_25": 0.0, "category_match": 0.0}
    difference = comparable["difference"].abs()
    return {
        "correlation": float(comparable["aqi"].corr(comparable["AQI"])),
        "within_25": float(100 * difference.le(25).mean()),
        "within_50": float(100 * difference.le(50).mean()),
        "category_match": float(100 * comparable["category_matches"].mean()),
    }


def build_summary(result: pd.DataFrame, defects: dict, scores: dict, total_rows: int) -> str:
    comparable = result[result["comparable"]]
    lines: list[str] = []
    lines.append("# AQI engine cross-check — AQ-02")
    lines.append("")
    lines.append(f"Generated {date.today().isoformat()} by `pipeline/validate_aqi.py`.")
    lines.append("")
    lines.append("> **What this proves.** Exact correctness of the CPCB formula is proved by")
    lines.append("> `tests/test_aqi.py`, where every expected value is hand-computed from the")
    lines.append("> official breakpoint table. This report is a *consistency* check against")
    lines.append("> published city AQI values, which are computed from per-station AQIs rather")
    lines.append("> than from the city-average pollutants in the same file. The two are not the")
    lines.append("> same calculation, so they track each other closely but do not match exactly.")
    lines.append("")

    lines.append("## Result")
    lines.append("")
    lines.append("| Check | Target | Actual | Pass |")
    lines.append("|---|---:|---:|:--:|")
    lines.append(f"| Correlation with published AQI | ≥{MIN_CORRELATION:.2f} | {scores['correlation']:.3f} "
                 f"| {'✅' if scores['correlation'] >= MIN_CORRELATION else '❌'} |")
    lines.append(f"| Within ±50 AQI points | ≥{MIN_WITHIN_50:.0f}% | {scores['within_50']:.1f}% "
                 f"| {'✅' if scores['within_50'] >= MIN_WITHIN_50 else '❌'} |")
    lines.append(f"| Same CPCB category | ≥{MIN_CATEGORY_MATCH:.0f}% | {scores['category_match']:.1f}% "
                 f"| {'✅' if scores['category_match'] >= MIN_CATEGORY_MATCH else '❌'} |")
    lines.append(f"| Within ±25 AQI points | — | {scores['within_25']:.1f}% | — |")
    lines.append("")
    lines.append(f"- Rows in file: **{total_rows:,}**")
    lines.append(f"- Rows with a published AQI: **{len(result):,}**")
    lines.append(f"- Rows comparable after exclusions: **{len(comparable):,}**")
    lines.append("")

    lines.append("## Defects found in the reference file")
    lines.append("")
    lines.append("Worth knowing before anyone trusts this dataset's AQI column.")
    lines.append("")
    lines.append(f"**1. {defects['rows_above_scale']:,} rows have a published AQI above "
                 f"{AQI_SCALE_MAX}** (highest: {defects['max_published_aqi']:.0f}). The CPCB index")
    lines.append("stops at 500, so whoever produced that column extrapolated past the Severe band")
    lines.append("instead of capping. Those rows are excluded here.")
    lines.append("")
    if defects["cities_above_scale"]:
        lines.append("| City | Rows above 500 |")
        lines.append("|---|---:|")
        for city, count in list(defects["cities_above_scale"].items())[:8]:
            lines.append(f"| {city} | {count:,} |")
        lines.append("")
    if defects["bad_co_cities"]:
        lines.append(f"**2. CO looks like the wrong unit for: "
                     f"{', '.join(defects['bad_co_cities'])}.** CPCB reports CO in mg/m³, where a")
        lines.append("city median sits near 1. These cities are far above that, which inflates their")
        lines.append("CO sub-index and therefore their AQI. Excluded from the comparison.")
        lines.append("")
        lines.append("| City | Median CO |")
        lines.append("|---|---:|")
        for city in defects["bad_co_cities"]:
            lines.append(f"| {city} | {defects['co_medians'][city]:.2f} |")
        lines.append("")

    refused = result[result["aqi"].isna()]
    lines.append("## Rows where we refused to compute an AQI")
    lines.append("")
    lines.append("The file publishes an AQI for these, but they fail CPCB's own validity rules.")
    lines.append("Refusing is correct behaviour.")
    lines.append("")
    if refused.empty:
        lines.append("- None")
    else:
        lines.append("| Reason | Rows |")
        lines.append("|---|---:|")
        for reason, count in refused["aqi_reason"].value_counts().items():
            lines.append(f"| {reason} | {count:,} |")
    lines.append("")

    lines.append("## Agreement by AQI band")
    lines.append("")
    lines.append("| Published band | Rows | Within ±50 | Mean difference |")
    lines.append("|---|---:|---:|---:|")
    edges = [0] + [upper for upper, _, _ in CATEGORIES]
    labels = [label for _, label, _ in CATEGORIES]
    banded = comparable.copy()
    banded["band"] = pd.cut(banded["AQI"], bins=edges, labels=labels, include_lowest=True)
    for band, group in banded.groupby("band", observed=True):
        lines.append(f"| {band} | {len(group):,} | {100 * group['difference'].abs().le(50).mean():.1f}% "
                     f"| {group['difference'].mean():+.1f} |")
    lines.append("")

    lines.append("## Agreement by dominant pollutant")
    lines.append("")
    lines.append("| Dominant pollutant | Rows | Within ±50 |")
    lines.append("|---|---:|---:|")
    for pollutant, group in comparable.groupby("dominant_pollutant", observed=True):
        lines.append(f"| {pollutant} | {len(group):,} | {100 * group['difference'].abs().le(50).mean():.1f}% |")
    lines.append("")

    lines.append("## Category distribution")
    lines.append("")
    lines.append("| Category | Published | Ours |")
    lines.append("|---|---:|---:|")
    published_counts = comparable["AQI_Bucket"].value_counts()
    our_counts = comparable["aqi_category"].value_counts()
    for _, label, _ in CATEGORIES:
        lines.append(f"| {label} | {published_counts.get(label, 0):,} | {our_counts.get(label, 0):,} |")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=os.environ.get("VAYU_RAW_ROOT"),
                   help=f"Folder holding {CITY_FILE} (or set VAYU_RAW_ROOT)")
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    args = p.parse_args(argv)
    if not args.raw_root:
        p.error("--raw-root is required (or set VAYU_RAW_ROOT)")
    if not (args.raw_root / CITY_FILE).exists():
        p.error(f"{CITY_FILE} not found in {args.raw_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)
    args.reports.mkdir(parents=True, exist_ok=True)

    df = load_city_data(args.raw_root)
    log.info("Loaded %d rows for %d cities", len(df), df["City"].nunique())

    defects = find_reference_defects(df)
    if defects["bad_co_cities"]:
        log.warning("CO unit looks wrong for: %s (excluded)", ", ".join(defects["bad_co_cities"]))
    if defects["rows_above_scale"]:
        log.warning("%d published rows exceed the 0-500 CPCB scale (excluded)",
                    defects["rows_above_scale"])

    result = compare(df, exclude_cities=defects["bad_co_cities"])
    scores = score(result[result["comparable"]])

    summary = build_summary(result, defects, scores, total_rows=len(df))
    (args.reports / "aqi_validation.md").write_text(summary, encoding="utf-8")

    log.info("correlation %.3f | within +/-50 %.1f%% | category match %.1f%%",
             scores["correlation"], scores["within_50"], scores["category_match"])
    log.info("Report: %s", args.reports / "aqi_validation.md")

    passed = (scores["correlation"] >= MIN_CORRELATION
              and scores["within_50"] >= MIN_WITHIN_50
              and scores["category_match"] >= MIN_CATEGORY_MATCH)
    if not passed:
        log.error("Cross-check FAILED -- investigate before trusting the engine")
        return 1
    log.info("Cross-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
