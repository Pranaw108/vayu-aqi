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
import json
import logging
import sys
import time
from datetime import date, datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.openmeteo import GridCell, WeatherClient, station_grid_map  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Model training starts in 2018; earlier station data is too sparse to trust.
DEFAULT_START = date(2018, 1, 1)
DEFAULT_END = date(2024, 12, 31)

# Correlation alone is not enough to judge a sensor.
MIN_TEMPERATURE_CORRELATION = 0.7

# In a place with little seasonal swing — Kerala barely moves all year —
# correlation is unstable even for a perfect sensor, so there is nothing to
# judge. Measured: KL07 scores r=0.115 with a station spread of 2.5 C against
# reanalysis's 2.9 C, which is agreement, not a fault.
MIN_TEMPERATURE_SPREAD_C = 3.0

# A working sensor should vary about as much as reality does. Far outside this
# means the sensor is damped or exaggerating, whatever its correlation says.
SPREAD_RATIO_BOUNDS = (0.7, 1.4)

# A sensor this far off in absolute terms is miscalibrated.
MAX_MEAN_TEMPERATURE_DIFFERENCE_C = 3.0

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
        station_spread = merged["station_temp_c"].std()
        reanalysis_spread = merged["temp_c"].std()
        rows.append({
            "station_code": code,
            "grid_key": grid_key.iloc[0],
            "hours_compared": len(merged),
            "temp_correlation": round(merged["station_temp_c"].corr(merged["temp_c"]), 3),
            "humidity_correlation": round(merged["station_humidity_pct"].corr(merged["humidity_pct"]), 3),
            "mean_temp_difference": round((merged["station_temp_c"] - merged["temp_c"]).mean(), 2),
            "station_spread_c": round(station_spread, 2),
            "reanalysis_spread_c": round(reanalysis_spread, 2),
            "spread_ratio": round(station_spread / reanalysis_spread, 2) if reanalysis_spread else np.nan,
        })
    frame = pd.DataFrame(rows)
    if not frame.empty:
        frame["verdict"] = [classify_sensor(row) for row in frame.itertuples()]
    return frame


def classify_sensor(row) -> str:
    """Judge one station's weather sensor: ok, faulty, or not assessable.

    Correlation on its own gets this wrong both ways, which is why there are
    three tests rather than one:
      - too little spread to judge  -> not_assessable, and left alone
      - spread far from reanalysis  -> faulty, whatever the correlation says
      - large absolute offset       -> faulty
    """
    if not np.isfinite(row.station_spread_c) or row.station_spread_c < MIN_TEMPERATURE_SPREAD_C:
        return "not_assessable"

    low, high = SPREAD_RATIO_BOUNDS
    if np.isfinite(row.spread_ratio) and not (low <= row.spread_ratio <= high):
        return "faulty"
    if abs(row.mean_temp_difference) > MAX_MEAN_TEMPERATURE_DIFFERENCE_C:
        return "faulty"
    if row.temp_correlation < MIN_TEMPERATURE_CORRELATION:
        return "faulty"
    return "ok"


def write_quarantine(data_dir: Path, joined: pd.DataFrame) -> dict:
    """List stations whose own weather sensor disagrees with reanalysis.

    `features.py` reads this and drops those stations' local weather columns, so a
    sensor reading 60 C cannot teach the model anything. Derived from the
    measurement rather than hardcoded, so it updates whenever this script reruns.
    """
    quarantine = {
        "generated": date.today().isoformat(),
        "threshold": MIN_TEMPERATURE_CORRELATION,
        "stations_checked": int(len(joined)),
        "note": "Only stations with enough overlapping weather could be checked, so this is the "
                "known-bad list, not a complete one. Stations in 'not_assessable' have too little "
                "temperature spread for correlation to mean anything (Kerala barely varies all "
                "year) and are deliberately left alone rather than quarantined on a weak score.",
        "stations": {},
    }
    if not joined.empty:
        weak = joined[joined["verdict"] == "faulty"]
        quarantine["not_assessable"] = sorted(joined.loc[joined["verdict"] == "not_assessable", "station_code"])
        quarantine["stations"] = {
            row.station_code: {
                "temp_correlation": float(row.temp_correlation),
                "humidity_correlation": float(row.humidity_correlation),
                "mean_temp_difference": float(row.mean_temp_difference),
                "spread_ratio": float(row.spread_ratio),
                "hours_compared": int(row.hours_compared),
            }
            for row in weak.itertuples()
        }

    path = data_dir / "sensor_quarantine.json"
    path.write_text(json.dumps(quarantine, indent=2), encoding="utf-8")
    log.info("Quarantined %d station weather sensors -> %s", len(quarantine["stations"]), path)
    return quarantine


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
    write_quarantine(args.data, joined)

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
