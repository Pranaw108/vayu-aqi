"""Aggregate the 15-minute readings into an hourly table (DF-03).

The curated archive is every 15 minutes, which is right for training but too
fine for the API to scan on each request. This builds one hourly row per
station, which is what the service reads and what the model's lag features are
built from.

It also records how many of the four possible readings each hour actually had,
so a thin hour can be told apart from a full one later.

Writes:
    <out>/hourly/year=<YYYY>/*.parquet
    <reports>/hourly_summary.md

Usage:
    python pipeline/build_hourly.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date
from pathlib import Path

import duckdb

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Columns that identify a row rather than measure something.
KEY_COLUMNS = {"timestamp", "station_code", "state", "year"}

log = logging.getLogger("build_hourly")


def measure_columns(con: duckdb.DuckDBPyConnection, source_glob: str) -> list[str]:
    """Measure columns, read from the data itself so there is no second list to keep in step."""
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{source_glob}', hive_partitioning=1) LIMIT 1")
    return [name for name, *_ in schema.fetchall() if name not in KEY_COLUMNS]


def build(con: duckdb.DuckDBPyConnection, source_glob: str, out_dir: Path) -> None:
    """Write the hourly table, partitioned by year."""
    measures = measure_columns(con, source_glob)
    averages = ",\n        ".join(f'avg("{m}") AS "{m}"' for m in measures)

    con.execute(f"""
        COPY (
            SELECT
                station_code,
                any_value(state) AS state,
                date_trunc('hour', timestamp) AS hour,
                year(timestamp) AS year,
                count(*) AS readings,
                {averages}
            FROM read_parquet('{source_glob}', hive_partitioning=1)
            GROUP BY station_code, date_trunc('hour', timestamp), year(timestamp)
        )
        TO '{out_dir.as_posix()}'
        (FORMAT PARQUET, PARTITION_BY (year), COMPRESSION ZSTD, OVERWRITE_OR_IGNORE)
    """)


def summarise(con: duckdb.DuckDBPyConnection, hourly_glob: str, source_glob: str, elapsed: float) -> str:
    hourly_rows, stations, first, last = con.execute(f"""
        SELECT count(*), count(DISTINCT station_code), min(hour), max(hour)
        FROM read_parquet('{hourly_glob}', hive_partitioning=1)
    """).fetchone()
    source_rows = con.execute(f"SELECT count(*) FROM read_parquet('{source_glob}', hive_partitioning=1)").fetchone()[0]

    # An hour with 4 readings is complete; fewer means the station reported less.
    full, thin = con.execute(f"""
        SELECT sum(CASE WHEN readings >= 4 THEN 1 ELSE 0 END), sum(CASE WHEN readings < 4 THEN 1 ELSE 0 END)
        FROM read_parquet('{hourly_glob}', hive_partitioning=1)
    """).fetchone()

    per_year = con.execute(f"""
        SELECT year, count(*) AS hours, count(DISTINCT station_code) AS stations
        FROM read_parquet('{hourly_glob}', hive_partitioning=1)
        GROUP BY year ORDER BY year
    """).fetchall()

    lines = [
        "# Hourly table — DF-03",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/build_hourly.py`.",
        "",
        f"- Source rows (15-minute): **{source_rows:,}**",
        f"- Hourly rows written: **{hourly_rows:,}**",
        f"- Stations: **{stations:,}**",
        f"- Span: **{first:%Y-%m-%d}** to **{last:%Y-%m-%d}**",
        f"- Complete hours (4 readings): **{full:,}** · thin hours: **{thin:,}**",
        f"- Build time: **{elapsed:.1f}s**",
        "",
        "## Hours per year",
        "",
        "| Year | Hours | Stations |",
        "|---|---:|---:|",
    ]
    lines += [f"| {y} | {h:,} | {s} |" for y, h, s in per_year]
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Folder holding curated/ (default: data/)")
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    curated = args.data / "curated"
    if not curated.is_dir():
        log.error("No curated data at %s — run pipeline/normalise.py first", curated)
        return 2

    out_dir = args.data / "hourly"
    args.reports.mkdir(parents=True, exist_ok=True)

    source_glob = (curated / "**" / "*.parquet").as_posix()
    hourly_glob = (out_dir / "**" / "*.parquet").as_posix()

    con = duckdb.connect()
    log.info("Aggregating 15-minute readings to hourly -> %s", out_dir)
    start = time.time()
    build(con, source_glob, out_dir)
    elapsed = time.time() - start

    summary = summarise(con, hourly_glob, source_glob, elapsed)
    (args.reports / "hourly_summary.md").write_text(summary, encoding="utf-8")

    rows = con.execute(f"SELECT count(*) FROM read_parquet('{hourly_glob}', hive_partitioning=1)").fetchone()[0]
    log.info("Done in %.1fs: %s hourly rows", elapsed, f"{rows:,}")
    log.info("Report: %s", args.reports / "hourly_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
