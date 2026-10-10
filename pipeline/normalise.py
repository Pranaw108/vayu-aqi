"""Turn the raw CPCB station CSVs into one clean, partitioned Parquet dataset (DF-01).

The raw folder holds 546 station CSVs spread over 20 state folders. They do not share a
layout: there are 113 different header rows, with column counts from 4 to 22. This script
lines them all up to one canonical schema, cleans obvious sensor faults, and writes:

    <out>/curated/state=<STATE>/year=<YYYY>/*.parquet   one row per 15-minute reading
    <out>/reports/normalise_summary.md                  human-readable report
    <out>/reports/station_completeness.csv              % present per pollutant, per station

Cleaning is deliberately conservative. We only remove readings that are physically
impossible (negative concentrations, humidity above 100%). We do NOT fill gaps here --
gap filling belongs in the feature step, where the model's needs are known.

Usage:
    python pipeline/normalise.py --raw-root "D:/.../India datasets" --sample 5   # quick check
    python pipeline/normalise.py --raw-root "D:/.../India datasets"              # full run
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Raw header name -> the name we use everywhere else. Anything not listed keeps its own name.
# We rename only where the raw name is awkward in SQL ("pm2.5") or ambiguous ("ozone").
COLUMN_RENAMES = {
    "pm2.5": "pm2_5",
    "ozone": "o3",
}

# Every column that can appear in a station file, in the order we store them.
# Built from the union of all 113 raw header rows.
POLLUTANTS = ["pm2_5", "pm10", "no", "no2", "nox", "nh3", "so2", "co", "o3"]
WEATHER = ["at", "rh", "ws", "wd", "vws", "bp"]
VOCS = ["benzene", "toluene", "xylene", "mp_xylene", "o_xylene", "eth_benzene"]
MEASURES = POLLUTANTS + WEATHER + VOCS
CANONICAL = ["timestamp", "station_code", "state"] + MEASURES

# Plausible range for each measure. Readings outside these are sensor faults, not air.
# CO is reported in mg/m3 by CPCB; every other concentration is ug/m3.
VALID_RANGE = {
    "pm2_5": (0, 1000),
    "pm10": (0, 2000),
    "no": (0, 2000),
    "no2": (0, 2000),
    "nox": (0, 2000),
    "nh3": (0, 2000),
    "so2": (0, 2000),
    "co": (0, 50),
    "o3": (0, 1000),
    "at": (-20, 60),      # degrees C
    "rh": (0, 100),       # percent
    "ws": (0, 50),        # m/s
    "wd": (0, 360),       # degrees
    "vws": (-50, 50),     # vertical wind, can be negative
    "bp": (300, 1100),    # station pressure, units vary between stations
    **{voc: (0, 1000) for voc in VOCS},
}

# Files that are metadata, not station readings.
NON_STATION_FILES = {"stations.csv", "IndiaPopularCity.csv"}

log = logging.getLogger("normalise")


# ---------------------------------------------------------------------------
# Counting what we changed
# ---------------------------------------------------------------------------

@dataclass
class CleanStats:
    """How many values each cleaning rule removed, across all files."""
    out_of_range: Counter = field(default_factory=Counter)
    pm_inverted: int = 0           # pm2_5 > pm10, which is impossible
    duplicate_rows: int = 0
    unparsable_timestamps: int = 0
    missing_columns: Counter = field(default_factory=Counter)  # column -> files lacking it


@dataclass
class StationResult:
    """One station's outcome, used for the completeness report."""
    station_code: str
    state: str
    rows: int
    first: pd.Timestamp | None
    last: pd.Timestamp | None
    present: dict[str, float]      # measure -> % of rows that have a value


# ---------------------------------------------------------------------------
# Reading one station file
# ---------------------------------------------------------------------------

def find_station_files(raw_root: Path) -> list[Path]:
    """Every station CSV, sorted, ignoring the two metadata files at the top level."""
    files = [p for p in raw_root.glob("*/*.csv") if p.name not in NON_STATION_FILES]
    return sorted(files)


def read_station_csv(path: Path, stats: CleanStats) -> pd.DataFrame:
    """Read one station file and reshape it to the canonical schema.

    Missing columns become all-null, so files with 8 columns and files with 22
    can be stacked without any values sliding into the wrong column.
    """
    station_code = path.stem                 # "DL01.csv" -> "DL01"
    state = path.parent.name                 # "Delhi_NCT"

    df = pd.read_csv(path, low_memory=False)
    df = df.rename(columns=COLUMN_RENAMES)

    # Record which canonical columns this file simply does not have.
    for column in MEASURES:
        if column not in df.columns:
            stats.missing_columns[column] += 1

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    stats.unparsable_timestamps += int(df["timestamp"].isna().sum())
    df = df.dropna(subset=["timestamp"])

    df["station_code"] = station_code
    df["state"] = state

    # reindex adds the missing columns as NaN and drops anything unexpected
    df = df.reindex(columns=CANONICAL)

    for column in MEASURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def clean_readings(df: pd.DataFrame, stats: CleanStats) -> pd.DataFrame:
    """Drop impossible values and duplicate rows. Gaps are left as gaps on purpose."""
    # A station should report each timestamp once. Keep the last row when it does not.
    before = len(df)
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    stats.duplicate_rows += before - len(df)

    for column, (low, high) in VALID_RANGE.items():
        values = df[column]
        bad = values.notna() & ((values < low) | (values > high))
        count = int(bad.sum())
        if count:
            stats.out_of_range[column] += count
            df.loc[bad, column] = pd.NA

    # PM2.5 particles are a subset of PM10, so PM2.5 can never be larger.
    # When it is, one of the two sensors is wrong and we cannot tell which. Drop both.
    inverted = df["pm2_5"].notna() & df["pm10"].notna() & (df["pm2_5"] > df["pm10"])
    count = int(inverted.sum())
    if count:
        stats.pm_inverted += count
        df.loc[inverted, ["pm2_5", "pm10"]] = pd.NA

    return df.sort_values("timestamp").reset_index(drop=True)


def measure_completeness(df: pd.DataFrame) -> dict[str, float]:
    """Percent of rows that have a value, per measure. This is our data quality signal."""
    if df.empty:
        return {column: 0.0 for column in MEASURES}
    return {column: round(100 * df[column].notna().mean(), 1) for column in MEASURES}


# ---------------------------------------------------------------------------
# Writing Parquet
# ---------------------------------------------------------------------------

def build_arrow_schema() -> pa.Schema:
    """One fixed schema for every file, so partitions always stack cleanly."""
    fields = [
        pa.field("timestamp", pa.timestamp("s")),
        pa.field("station_code", pa.string()),
        pa.field("state", pa.string()),
        pa.field("year", pa.int16()),
    ]
    fields += [pa.field(column, pa.float32()) for column in MEASURES]
    return pa.schema(fields)


def write_station(df: pd.DataFrame, out_dir: Path, schema: pa.Schema) -> None:
    """Append one station to the dataset, split into state/year folders.

    Partitioning by state and year lets DuckDB skip whole folders when a query
    filters on either, which is most queries this project will run.
    """
    if df.empty:
        return
    df = df.copy()
    df["year"] = df["timestamp"].dt.year.astype("int16")
    for column in MEASURES:
        df[column] = df[column].astype("float32")

    table = pa.Table.from_pandas(df[schema.names], schema=schema, preserve_index=False)
    station_code = df["station_code"].iloc[0]
    pq.write_to_dataset(
        table,
        root_path=str(out_dir),
        partition_cols=["state", "year"],
        # station code in the filename keeps two stations from overwriting each other
        basename_template=f"{station_code}-{{i}}.parquet",
        compression="zstd",
        existing_data_behavior="overwrite_or_ignore",
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_completeness_csv(results: list[StationResult], path: Path) -> None:
    rows = []
    for r in results:
        row = {
            "station_code": r.station_code,
            "state": r.state,
            "rows": r.rows,
            "first_reading": r.first,
            "last_reading": r.last,
        }
        row.update({f"pct_{k}": v for k, v in r.present.items()})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def build_summary(results: list[StationResult], stats: CleanStats, out_dir: Path,
                  elapsed_min: float, sampled: int | None) -> str:
    """Markdown report. Reviewers read this instead of re-running the script."""
    total_rows = sum(r.rows for r in results)
    lines: list[str] = []
    lines.append("# Station data normalisation — DF-01")
    lines.append("")
    note = f" (**sample run**: first {sampled} files only)" if sampled else ""
    lines.append(f"Generated {date.today().isoformat()} by `pipeline/normalise.py`{note}.")
    lines.append("")
    lines.append(f"- Stations processed: **{len(results):,}**")
    lines.append(f"- Readings written: **{total_rows:,}**")
    lines.append(f"- Output: `{out_dir}`")
    lines.append(f"- Runtime: **{elapsed_min:.1f} min**")
    lines.append("")

    if results:
        first = min(r.first for r in results if r.first is not None)
        last = max(r.last for r in results if r.last is not None)
        lines.append(f"- Time span: **{first:%Y-%m-%d}** to **{last:%Y-%m-%d}**")
        lines.append("")

    lines.append("## Rows per state")
    lines.append("")
    lines.append("| State | Stations | Readings |")
    lines.append("|---|---:|---:|")
    by_state: dict[str, list[StationResult]] = {}
    for r in results:
        by_state.setdefault(r.state, []).append(r)
    for state in sorted(by_state):
        group = by_state[state]
        lines.append(f"| {state} | {len(group)} | {sum(g.rows for g in group):,} |")
    lines.append("")

    lines.append("## Average completeness by measure")
    lines.append("")
    lines.append("Percent of stored rows that carry a value. Low numbers here are the")
    lines.append("real limit on what the model can learn.")
    lines.append("")
    lines.append("| Measure | Mean % present | Stations with >80% |")
    lines.append("|---|---:|---:|")
    for column in MEASURES:
        values = [r.present[column] for r in results]
        mean_pct = sum(values) / len(values) if values else 0
        good = sum(1 for v in values if v > 80)
        lines.append(f"| {column} | {mean_pct:.1f} | {good} |")
    lines.append("")

    lines.append("## Values removed by cleaning")
    lines.append("")
    lines.append("| Rule | Values removed |")
    lines.append("|---|---:|")
    for column, count in stats.out_of_range.most_common():
        lines.append(f"| {column} outside plausible range | {count:,} |")
    lines.append(f"| pm2_5 > pm10 (both dropped) | {stats.pm_inverted:,} |")
    lines.append(f"| duplicate timestamps | {stats.duplicate_rows:,} |")
    lines.append(f"| unparsable timestamps | {stats.unparsable_timestamps:,} |")
    lines.append("")

    lines.append("## Columns absent from source files")
    lines.append("")
    lines.append("These became all-null columns. Not an error -- CPCB stations monitor")
    lines.append("different sets of pollutants.")
    lines.append("")
    lines.append("| Column | Files without it |")
    lines.append("|---|---:|")
    for column, count in stats.missing_columns.most_common():
        lines.append(f"| {column} | {count} |")
    lines.append("")

    lines.append("## Stations with weak PM coverage")
    lines.append("")
    lines.append("AQI needs PM2.5 or PM10. Stations below 50% on both are near-useless")
    lines.append("for AQI and should be checked before they reach the model.")
    lines.append("")
    weak = [r for r in results if r.present["pm2_5"] < 50 and r.present["pm10"] < 50]
    if weak:
        lines.append("| Station | State | % pm2_5 | % pm10 | Rows |")
        lines.append("|---|---|---:|---:|---:|")
        for r in sorted(weak, key=lambda x: x.present["pm2_5"]):
            lines.append(f"| {r.station_code} | {r.state} | {r.present['pm2_5']} "
                         f"| {r.present['pm10']} | {r.rows:,} |")
    else:
        lines.append("- None")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--raw-root", type=Path, default=os.environ.get("VAYU_RAW_ROOT"),
                   help="Folder holding the state subfolders of station CSVs "
                        "(or set VAYU_RAW_ROOT)")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output folder; Parquet goes to <out>/curated (default: data/)")
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS,
                   help="Where the summary report is written (default: reports/)")
    p.add_argument("--only", nargs="+", metavar="STATE",
                   help="Process only these state folders, e.g. Delhi_NCT Bihar")
    p.add_argument("--sample", type=int, metavar="N",
                   help="Process only the first N station files (quick check)")
    p.add_argument("--since-year", type=int, default=None,
                   help="Keep only readings from this year onward (use 2018 for model training)")
    args = p.parse_args(argv)
    if not args.raw_root:
        p.error("--raw-root is required (or set VAYU_RAW_ROOT)")
    if not args.raw_root.is_dir():
        p.error(f"--raw-root is not a folder: {args.raw_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    files = find_station_files(args.raw_root)
    if args.only:
        wanted = set(args.only)
        files = [f for f in files if f.parent.name in wanted]
        if not files:
            log.error("No station files found for state(s): %s", ", ".join(sorted(wanted)))
            return 2
    if args.sample:
        files = files[: args.sample]

    curated_dir = args.out / "curated"
    curated_dir.mkdir(parents=True, exist_ok=True)
    args.reports.mkdir(parents=True, exist_ok=True)

    log.info("Normalising %d station files -> %s", len(files), curated_dir)
    schema = build_arrow_schema()
    stats = CleanStats()
    results: list[StationResult] = []
    start = time.time()

    for n, path in enumerate(files, 1):
        try:
            df = read_station_csv(path, stats)
            df = clean_readings(df, stats)
            if args.since_year:
                df = df[df["timestamp"].dt.year >= args.since_year]
            write_station(df, curated_dir, schema)
        except Exception as exc:
            # One bad file must not stop a run that takes many minutes.
            log.error("  %s failed: %s: %s", path.name, type(exc).__name__, exc)
            continue

        results.append(StationResult(
            station_code=path.stem,
            state=path.parent.name,
            rows=len(df),
            first=df["timestamp"].min() if not df.empty else None,
            last=df["timestamp"].max() if not df.empty else None,
            present=measure_completeness(df),
        ))
        if n % 25 == 0 or n == len(files):
            log.info("  %d/%d files, %s readings so far",
                     n, len(files), f"{sum(r.rows for r in results):,}")

    elapsed_min = (time.time() - start) / 60
    write_completeness_csv(results, args.reports / "station_completeness.csv")
    summary = build_summary(results, stats, curated_dir, elapsed_min, args.sample)
    (args.reports / "normalise_summary.md").write_text(summary, encoding="utf-8")

    log.info("Done in %.1f min: %d stations, %s readings",
             elapsed_min, len(results), f"{sum(r.rows for r in results):,}")
    log.info("Report: %s", args.reports / "normalise_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
