"""Fetch historical weather for the training window (MD-01 input).

Pulls ERA5 hourly weather from Open-Meteo for every grid cell the stations sit
on, so the model can learn how dispersion affects pollution. Free, no API key.

The 462 mappable stations share only 245 grid cells, so that is how many calls
this makes. Each cell is cached, so an interrupted run resumes.

It also validates the spatial join: CPCB stations report their own temperature
and humidity, so ERA5's values should correlate strongly with them. A weak
correlation would mean stations are matched to the wrong grid cell, which is a
mistake that is otherwise invisible.

Writes:
    <data>/weather/weather_hourly.parquet   grid_key + hour + weather
    <data>/weather_cache/*.parquet          one file per grid cell
    <reports>/weather_summary.md

Usage:
    python pipeline/fetch_weather.py                      # 2018-2024
    python pipeline/fetch_weather.py --start 2023-01-01   # shorter window
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.openmeteo import GridCell, WeatherClient, station_grid_map  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Model training starts in 2018; earlier station data is too sparse to trust.
DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2024, 12, 31)

# Below this, the station is probably matched to the wrong grid cell.
MIN_TEMPERATURE_CORRELATION = 0.7

log = logging.getLogger("fetch_weather")


def fetch_all_cells(client: WeatherClient, mapping: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    """One archive call per grid cell, skipping any already cached."""
    cells = mapping[["grid_lat", "grid_lon", "grid_key"]].drop_duplicates().reset_index(drop=True)
    log.info("Fetching %d grid cells for %s to %s", len(cells), start, end)

    frames = []
    for n, row in enumerate(cells.itertuples(), 1):
        cell = GridCell(latitude=row.grid_lat, longitude=row.grid_lon)
        try:
            frames.append(client.archive(cell, start, end))
        except Exception as error:
            # One unreachable cell must not lose the other 244.
            log.error("  %s failed: %s: %s", cell.key, type(error).__name__, error)
            continue
        if n % 25 == 0 or n == len(cells):
            log.info("  %d/%d cells · %d calls made", n, len(cells), client.calls_made)

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def check_spatial_join(data_dir: Path, weather: pd.DataFrame, mapping: pd.DataFrame) -> pd.DataFrame:
    """Compare ERA5 temperature and humidity against what the stations measured.

    This is the test that the station-to-grid-cell match is actually correct.
    """
    con = duckdb.connect()
    hourly_glob = (data_dir / "hourly" / "**" / "*.parquet").as_posix()

    # A sample of stations that actually report their own weather.
    candidates = con.execute(f"""
        SELECT station_code, count(*) AS n
        FROM read_parquet('{hourly_glob}', hive_partitioning=1)
        WHERE "at" IS NOT NULL AND rh IS NOT NULL AND hour >= '2023-01-01'
        GROUP BY station_code HAVING count(*) > 2000
        ORDER BY n DESC LIMIT 40
    """).fetch_df()

    rows = []
    for code in candidates["station_code"]:
        grid_key = mapping.loc[mapping["station_code"] == code, "grid_key"]
        if grid_key.empty:
            continue
        station = con.execute(f"""
            SELECT hour AS timestamp, "at" AS station_temp_c, rh AS station_humidity_pct
            FROM read_parquet('{hourly_glob}', hive_partitioning=1)
            WHERE station_code = ? AND hour >= '2023-01-01'
        """, [code]).fetch_df()

        cell_weather = weather[weather["grid_key"] == grid_key.iloc[0]]
        merged = station.merge(cell_weather, on="timestamp", how="inner").dropna(
            subset=["station_temp_c", "temp_c", "station_humidity_pct", "humidity_pct"]
        )
        if len(merged) < 500:
            continue
        rows.append({
            "station_code": code,
            "grid_key": grid_key.iloc[0],
            "hours_compared": len(merged),
            "temp_correlation": round(merged["station_temp_c"].corr(merged["temp_c"]), 3),
            "humidity_correlation": round(merged["station_humidity_pct"].corr(merged["humidity_pct"]), 3),
            "mean_temp_difference": round((merged["station_temp_c"] - merged["temp_c"]).mean(), 2),
        })
    return pd.DataFrame(rows)


def build_summary(weather: pd.DataFrame, mapping: pd.DataFrame, joined: pd.DataFrame,
                  calls: int, elapsed_min: float, start: date, end: date) -> str:
    lines = [
        "# Historical weather — MD-01 input",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/fetch_weather.py`.",
        "",
        f"- Window: **{start}** to **{end}**",
        f"- Stations covered: **{mapping['station_code'].nunique()}**",
        f"- Grid cells fetched: **{weather['grid_key'].nunique()}** "
        f"(of {mapping['grid_key'].nunique()} needed)",
        f"- Hourly rows: **{len(weather):,}**",
        f"- API calls: **{calls}** (free tier allows 10,000/day)",
        f"- Runtime: **{elapsed_min:.1f} min**",
        "",
        "Source: [Open-Meteo](https://open-meteo.com/) ERA5 reanalysis, no API key, "
        "non-commercial use.",
        "",
        "## Coverage per variable",
        "",
        "| Variable | % present |",
        "|---|---:|",
    ]
    for column in weather.columns:
        if column in ("grid_key", "timestamp"):
            continue
        lines.append(f"| {column} | {100 * weather[column].notna().mean():.1f} |")
    lines.append("")

    lines += [
        "## Is the station-to-grid match correct?",
        "",
        "CPCB stations measure their own temperature and humidity. ERA5 is an independent",
        "source, so a strong correlation means each station is matched to the right grid",
        "cell. A weak one would mean the coordinates are wrong — a mistake that is",
        "otherwise invisible.",
        "",
    ]
    if joined.empty:
        lines.append("- Not enough overlapping station weather to check.")
    else:
        good = int((joined["temp_correlation"] >= MIN_TEMPERATURE_CORRELATION).sum())
        lines += [
            f"- Stations checked: **{len(joined)}**",
            f"- Median temperature correlation: **{joined['temp_correlation'].median():.3f}**",
            f"- Median humidity correlation: **{joined['humidity_correlation'].median():.3f}**",
            f"- Above the {MIN_TEMPERATURE_CORRELATION} threshold: **{good} / {len(joined)}**",
            "",
            "| Station | Grid cell | Hours | Temp r | Humidity r | Mean temp diff (°C) |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for row in joined.sort_values("temp_correlation").itertuples():
            lines.append(
                f"| {row.station_code} | {row.grid_key} | {row.hours_compared:,} "
                f"| {row.temp_correlation} | {row.humidity_correlation} | {row.mean_temp_difference} |"
            )
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--start", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_START)
    p.add_argument("--end", type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(), default=DEFAULT_END)
    p.add_argument("--limit-cells", type=int, help="Fetch only the first N grid cells (quick check)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    registry_path = args.data / "registry.parquet"
    if not registry_path.exists():
        log.error("No registry at %s — run pipeline/registry.py first", registry_path)
        return 2

    mapping = station_grid_map(pd.read_parquet(registry_path))
    if args.limit_cells:
        keep = mapping["grid_key"].drop_duplicates().head(args.limit_cells)
        mapping = mapping[mapping["grid_key"].isin(keep)]

    client = WeatherClient(args.data / "weather_cache")
    start_time = time.time()
    weather = fetch_all_cells(client, mapping, args.start, args.end)
    elapsed_min = (time.time() - start_time) / 60

    if weather.empty:
        log.error("No weather fetched")
        return 1

    out_dir = args.data / "weather"
    out_dir.mkdir(parents=True, exist_ok=True)
    weather.to_parquet(out_dir / "weather_hourly.parquet", index=False, compression="zstd")

    log.info("Checking the station-to-grid match against station-measured weather")
    joined = check_spatial_join(args.data, weather, mapping)

    args.reports.mkdir(parents=True, exist_ok=True)
    summary = build_summary(weather, mapping, joined, client.calls_made, elapsed_min, args.start, args.end)
    (args.reports / "weather_summary.md").write_text(summary, encoding="utf-8")

    log.info("Done in %.1f min: %s rows, %d calls", elapsed_min, f"{len(weather):,}", client.calls_made)
    if not joined.empty:
        log.info("Median temperature correlation with station sensors: %.3f", joined["temp_correlation"].median())
        weak = joined[joined["temp_correlation"] < MIN_TEMPERATURE_CORRELATION]
        if not weak.empty:
            log.warning("%d station(s) correlate weakly — check their coordinates: %s",
                        len(weak), ", ".join(weak["station_code"]))
    log.info("Report: %s", args.reports / "weather_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
