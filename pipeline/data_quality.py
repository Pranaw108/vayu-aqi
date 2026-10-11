"""One data-quality report across every stage (observability).

Each pipeline step already writes its own report, which means checking the
project's health meant opening a dozen files. This gathers the numbers that
actually decide whether the data is fit to train on, and adds two checks nothing
else does:

  constant features  — a column with one value teaches nothing and costs a slot
                       in every tree's feature sample
  outlier stations   — a station far from its regional peers, which is how KA06
                       was found by hand after it had already ranked first
                       nationally

Writes:
    <reports>/data_quality.md

Usage:
    python pipeline/data_quality.py
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
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# A station this many robust deviations from its state's median is suspect.
OUTLIER_THRESHOLD = 3.5

# A feature is near-constant when its single commonest value covers this much of
# the column. Comparing distinct-count to row-count instead flags any binary
# flag as near-constant, which wrongly caught is_weekend and is_burning_season.
NEAR_CONSTANT_DOMINANCE = 0.999

log = logging.getLogger("data_quality")


def stage_presence(data_dir: Path) -> list[dict]:
    """Which pipeline outputs exist, so a missing stage is obvious."""
    stages = [
        ("registry", data_dir / "registry.parquet", "pipeline/registry.py"),
        ("curated 15-min", data_dir / "curated", "pipeline/normalise.py"),
        ("hourly", data_dir / "hourly", "pipeline/build_hourly.py"),
        ("weather", data_dir / "weather" / "weather_hourly.parquet", "pipeline/fetch_weather.py"),
        ("sensor quarantine", data_dir / "sensor_quarantine.json", "pipeline/fetch_weather.py"),
        ("fires", data_dir / "fires" / "fires.parquet", "pipeline/fetch_fires.py"),
        ("fire features", data_dir / "fires" / "fire_features.parquet", "pipeline/build_fire_features.py"),
    ]
    return [
        {"stage": name, "present": path.exists(), "build_with": command}
        for name, path, command in stages
    ]


def station_outliers(data_dir: Path) -> pd.DataFrame:
    """Stations whose long-run PM2.5 is far from their state's median.

    Uses the median and the median absolute deviation rather than the mean and
    standard deviation, because one extreme station would otherwise drag the
    very baseline it is being compared against.
    """
    hourly = (data_dir / "hourly" / "**" / "*.parquet").as_posix()
    con = duckdb.connect()
    per_station = con.execute(f"""
        SELECT h.station_code, r.state_code, r.city,
               count(h.pm2_5) AS hours,
               round(avg(h.pm2_5), 1) AS mean_pm25,
               round(max(h.pm2_5), 1) AS max_pm25
        FROM read_parquet('{hourly}', hive_partitioning=1) h
        JOIN read_parquet('{(data_dir / "registry.parquet").as_posix()}') r USING (station_code)
        WHERE h.pm2_5 IS NOT NULL
        GROUP BY 1, 2, 3
        HAVING count(h.pm2_5) > 5000
    """).fetch_df()

    rows = []
    for state, group in per_station.groupby("state_code"):
        if len(group) < 4:
            continue
        median = group["mean_pm25"].median()
        deviation = (group["mean_pm25"] - median).abs().median()
        if deviation == 0:
            continue
        # 0.6745 scales the median absolute deviation to be comparable with a
        # standard deviation for normal data.
        score = 0.6745 * (group["mean_pm25"] - median) / deviation
        flagged = group.assign(state_median=round(median, 1), score=score.round(2))
        rows.append(flagged[flagged["score"].abs() >= OUTLIER_THRESHOLD])

    if not rows:
        return pd.DataFrame(columns=["station_code", "state_code", "city", "mean_pm25", "state_median", "score"])
    return pd.concat(rows).sort_values("score", ascending=False)


def feature_health(data_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Constant, near-constant and mostly-missing columns in the feature table."""
    tables = sorted((data_dir / "features").glob("target=*/horizon=*/features.parquet"))
    if not tables:
        return pd.DataFrame(), {}

    newest = max(tables, key=lambda p: p.stat().st_mtime)
    frame = pd.read_parquet(newest)
    skip = {"station_code", "hour", "target", "target_hour", "baseline_persistence"}

    rows = []
    for column in frame.columns:
        if column in skip:
            continue
        series = frame[column]
        distinct = int(series.nunique(dropna=True))
        present = float(series.notna().mean())
        dominance = float(series.value_counts(normalize=True, dropna=True).iloc[0]) if distinct else 1.0
        issue = None
        if distinct <= 1:
            issue = "constant"
        elif dominance >= NEAR_CONSTANT_DOMINANCE:
            issue = f"near-constant ({100 * dominance:.2f}% one value)"
        elif present < 0.5:
            issue = "mostly missing"
        if issue:
            rows.append({"feature": column, "issue": issue,
                         "distinct_values": distinct, "percent_present": round(100 * present, 1)})

    meta = {
        "table": str(newest.relative_to(data_dir)),
        "rows": len(frame),
        "columns": len(frame.columns),
        "features": len(frame.columns) - len(skip & set(frame.columns)),
    }
    return pd.DataFrame(rows), meta


def build_summary(stages: list[dict], outliers: pd.DataFrame,
                  features: pd.DataFrame, meta: dict) -> str:
    lines = [
        "# Data quality",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/data_quality.py`.",
        "",
        "## Pipeline stages",
        "",
        "| Stage | Present | Build with |",
        "|---|:--:|---|",
    ]
    for stage in stages:
        mark = "yes" if stage["present"] else "**NO**"
        lines.append(f"| {stage['stage']} | {mark} | `{stage['build_with']}` |")
    lines.append("")

    lines += [
        "## Outlier stations",
        "",
        f"A station whose long-run mean PM2.5 sits {OUTLIER_THRESHOLD} or more robust",
        "deviations from its own state's median. Computed with the median and median",
        "absolute deviation, so one extreme station cannot drag the baseline it is",
        "measured against.",
        "",
        "This is the check that was missing when KA06 was found by hand, after it had",
        "already ranked worst in India in a city table.",
        "",
    ]
    if outliers.empty:
        lines.append("- None flagged.")
    else:
        lines += [
            "| Station | State | City | Mean PM2.5 | State median | Score |",
            "|---|---|---|---:|---:|---:|",
        ]
        for row in outliers.itertuples():
            lines.append(f"| {row.station_code} | {row.state_code} | {row.city} "
                         f"| {row.mean_pm25} | {row.state_median} | {row.score:+.2f} |")
    lines.append("")

    lines += ["## Feature table health", ""]
    if not meta:
        lines.append("- No feature table built yet.")
    else:
        lines += [
            f"- Table: `{meta['table']}`",
            f"- Rows: **{meta['rows']:,}** · columns: **{meta['columns']}** "
            f"· features: **{meta['features']}**",
            "",
            "A constant column teaches nothing and still takes a slot in every tree's",
            "feature sample, which is how 56 fire columns cost 13 points of onset recall.",
            "",
        ]
        if features.empty:
            lines.append("- No constant, near-constant or mostly-missing features.")
        else:
            lines += ["| Feature | Issue | Distinct values | % present |", "|---|---|---:|---:|"]
            for row in features.itertuples():
                lines.append(f"| `{row.feature}` | {row.issue} | {row.distinct_values:,} "
                             f"| {row.percent_present} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--fail-on-issues", action="store_true",
                   help="Exit non-zero if any outlier station or constant feature is found")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    stages = stage_presence(args.data)
    missing = [s["stage"] for s in stages if not s["present"]]
    if missing:
        log.warning("Missing stages: %s", ", ".join(missing))

    outliers = pd.DataFrame()
    if (args.data / "hourly").is_dir() and (args.data / "registry.parquet").exists():
        outliers = station_outliers(args.data)
    features, meta = feature_health(args.data)

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "data_quality.md").write_text(
        build_summary(stages, outliers, features, meta), encoding="utf-8"
    )

    log.info("Stages present: %d/%d", len(stages) - len(missing), len(stages))
    log.info("Outlier stations flagged: %d", len(outliers))
    log.info("Problem features: %d", len(features))
    log.info("Report: %s", args.reports / "data_quality.md")

    if args.fail_on_issues and (not outliers.empty or not features.empty):
        log.error("Issues found and --fail-on-issues was given")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
