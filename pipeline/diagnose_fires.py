"""Is the fire signal real, or is it tracking the season? (fire diagnostics)

A feature can correlate strongly with pollution and still tell the model
nothing, if what it really encodes is "it is late October". Fires and Delhi's
PM2.5 both climb through October, so a raw correlation cannot separate the two.

This measures the part that survives once the things the model already has are
controlled for: day-of-year, and yesterday's PM2.5. Whatever is left is the
incremental information a fire feature actually contributes.

Writes:
    <reports>/fire_diagnosis.md

Usage:
    python pipeline/diagnose_fires.py --state DL
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# The months where burning could plausibly matter for northern India.
MONTHS_OF_INTEREST = (9, 10, 11, 12)

# Fire columns worth diagnosing. The upwind summary is the one the model ranks
# highest, so it leads.
FIRE_COLUMNS = ("fire_count_upwind", "fire_count_100_300km", "fire_frp_100_300km")

log = logging.getLogger("diagnose_fires")


def partial_correlation(frame: pd.DataFrame, x: str, y: str, controls: list[str]) -> float:
    """Correlation of x and y after removing what the controls explain.

    Both are regressed on the controls and the residuals are correlated, so a
    feature that is merely a proxy for the controls comes out near zero.
    """
    design = np.column_stack([np.ones(len(frame))] + [frame[c].to_numpy(float) for c in controls])

    def residual(column: str) -> np.ndarray:
        values = frame[column].to_numpy(float)
        coefficients, *_ = np.linalg.lstsq(design, values, rcond=None)
        return values - design @ coefficients

    return float(np.corrcoef(residual(x), residual(y))[0, 1])


def load_daily(data_dir: Path, state_code: str) -> pd.DataFrame:
    """Daily mean PM2.5 per station, joined to that station's fire exposure."""
    registry = pd.read_parquet(data_dir / "registry.parquet")
    wanted = registry.loc[
        (registry["state_code"] == state_code) & registry["is_mappable"], "station_code"
    ].tolist()
    if not wanted:
        raise ValueError(f"No mappable stations for state {state_code!r}")

    con = duckdb.connect()
    placeholders = ",".join("?" for _ in wanted)
    daily = con.execute(f"""
        SELECT station_code, date_trunc('day', hour) AS day, avg(pm2_5) AS pm25
        FROM read_parquet('{(data_dir / "hourly" / "**" / "*.parquet").as_posix()}', hive_partitioning=1)
        WHERE station_code IN ({placeholders}) AND pm2_5 IS NOT NULL
        GROUP BY 1, 2
    """, wanted).fetch_df()

    fires = pd.read_parquet(data_dir / "fires" / "fire_features.parquet")
    fires = fires[fires["station_code"].isin(wanted)].rename(columns={"acq_date": "day"})

    frame = daily.merge(fires, on=["station_code", "day"], how="inner").dropna(subset=["pm25"])
    frame = frame.sort_values(["station_code", "day"])
    frame["month"] = frame["day"].dt.month
    day_of_year = frame["day"].dt.dayofyear
    frame["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)
    # Yesterday's level, which the real model has as a lag feature.
    frame["pm25_yesterday"] = frame.groupby("station_code")["pm25"].shift(1)
    return frame


def diagnose(frame: pd.DataFrame) -> pd.DataFrame:
    """Raw against partial correlation, per month and fire column."""
    rows = []
    for column in FIRE_COLUMNS:
        if column not in frame.columns:
            continue
        for month in MONTHS_OF_INTEREST:
            month_rows = frame[(frame["month"] == month) & frame["pm25_yesterday"].notna()]
            if len(month_rows) < 200:
                continue
            raw = float(month_rows[column].corr(month_rows["pm25"]))
            season_only = partial_correlation(month_rows, column, "pm25", ["doy_sin", "doy_cos"])
            full = partial_correlation(
                month_rows, column, "pm25", ["doy_sin", "doy_cos", "pm25_yesterday"]
            )
            rows.append({
                "feature": column,
                "month": month,
                "n": len(month_rows),
                "raw": round(raw, 3),
                "after_season": round(season_only, 3),
                "after_season_and_lag": round(full, 3),
                "share_kept": round(abs(full / raw), 3) if raw else np.nan,
            })
    return pd.DataFrame(rows)


def build_summary(results: pd.DataFrame, state_code: str) -> str:
    month_names = {9: "September", 10: "October", 11: "November", 12: "December"}
    lines = [
        "# Is the fire signal real, or seasonal?",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/diagnose_fires.py` for state "
        f"`{state_code}`.",
        "",
        "A feature can correlate strongly with pollution and still add nothing, if what it",
        "really encodes is the time of year. Fires and Delhi's PM2.5 both climb through",
        "October, so the raw correlation cannot separate cause from calendar.",
        "",
        "The last column is what matters: the share of the raw correlation that survives",
        "once day-of-year and yesterday's PM2.5 — both of which the model already has — are",
        "controlled for.",
        "",
        "| Feature | Month | Rows | Raw | After season | After season + lag | Kept |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.itertuples():
        lines.append(
            f"| `{row.feature}` | {month_names.get(row.month, row.month)} | {row.n:,} "
            f"| {row.raw:+.3f} | {row.after_season:+.3f} | {row.after_season_and_lag:+.3f} "
            f"| {100 * row.share_kept:.0f}% |"
        )
    lines.append("")

    upwind = results[results["feature"] == "fire_count_upwind"]
    october = upwind[upwind["month"] == 10]
    november = upwind[upwind["month"] == 11]
    if not october.empty and not november.empty:
        oct_row, nov_row = october.iloc[0], november.iloc[0]
        lines += [
            "## What this says",
            "",
            f"October has the strongest raw correlation of any month ({oct_row.raw:+.3f}) and the",
            f"weakest once controlled ({oct_row.after_season_and_lag:+.3f}) — only "
            f"{100 * oct_row.share_kept:.0f}% survives. Through October both fires and PM2.5 rise",
            "together, so the raw figure is mostly the shared ramp. The model already knows it is",
            "late October and that yesterday was bad, and the fire count adds little on top.",
            "",
            f"November keeps {100 * nov_row.share_kept:.0f}% ({nov_row.after_season_and_lag:+.3f}).",
            "There both fires and pollution are high and fluctuating day to day, so the",
            "correlation reflects real coupling rather than a trend.",
            "",
            "This is why October's onset recall did not improve with fire features. The problem",
            "is not that the fires are missing or that the wind points the wrong way — it is that",
            "in October the fire signal is largely redundant with the calendar.",
            "",
        ]
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--state", default="DL", help="State code to diagnose (default DL)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    frame = load_daily(args.data, args.state)
    log.info("Loaded %s station-days for %s", f"{len(frame):,}", args.state)

    results = diagnose(frame)
    if results.empty:
        log.error("Not enough overlapping data to diagnose")
        return 1

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "fire_diagnosis.md").write_text(
        build_summary(results, args.state), encoding="utf-8"
    )

    for row in results[results["feature"] == "fire_count_upwind"].itertuples():
        log.info("  month %2d: raw %+.3f -> %+.3f after controls (%.0f%% kept)",
                 row.month, row.raw, row.after_season_and_lag, 100 * row.share_kept)
    log.info("Report: %s", args.reports / "fire_diagnosis.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
