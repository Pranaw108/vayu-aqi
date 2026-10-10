"""Build the model's feature table (MD-01).

Forecasting rule that governs this whole file: **every feature must be knowable
at the moment the prediction is made.** The target is a pollutant concentration
`horizon` hours in the future; the features may only use what was already
observed, plus weather forecasts, which are genuinely available ahead of time.

This is where the project's biggest trap lives. AQI is a deterministic function
of PM2.5 and PM10, so a model given the same hour's PM would score near-perfectly
and have learned arithmetic, not air quality. Hence: the target is a pollutant,
never AQI, and no feature reads the target hour's own pollutants.

Writes:
    <data>/features/target=<pollutant>/horizon=<H>/*.parquet
    <reports>/features_summary.md

Usage:
    python pipeline/features.py --horizon 24 --sample-stations 20   # quick check
    python pipeline/features.py --horizon 24                        # all stations
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

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

# Pre-2018 station data is too sparse to learn from.
TRAIN_START = "2018-01-01"

# What we forecast. PM drives the AQI in almost every Indian city.
TARGETS = ("pm2_5", "pm10")

# Hours back that the model may look. 168 is one week, which captures the
# weekly traffic cycle.
LAG_HOURS = (1, 2, 3, 6, 12, 24, 48, 168)
ROLLING_WINDOWS = (6, 24, 168)

# Pollutants usable as inputs. The target's own past is allowed; the target
# hour itself is not.
INPUT_POLLUTANTS = ("pm2_5", "pm10", "no2", "so2", "nh3", "co", "o3")
OBSERVED_WEATHER = ("at", "rh", "ws", "bp")
FORECAST_WEATHER = ("temp_c", "humidity_pct", "wind_speed_ms", "pressure_hpa", "precip_mm", "boundary_layer_m")

# Below this share of rows, forecast weather is too thin to learn dispersion from.
MIN_WEATHER_COVERAGE = 0.5

# Stubble burning in Punjab and Haryana drives the worst north-Indian episodes.
BURNING_SEASON_MONTHS = (10, 11)

log = logging.getLogger("features")


def load_station_hours(con: duckdb.DuckDBPyConnection, data_dir: Path,
                       station_codes: list[str] | None) -> pd.DataFrame:
    """Hourly readings joined to that station's grid-cell weather."""
    hourly_glob = (data_dir / "hourly" / "**" / "*.parquet").as_posix()
    weather_path = (data_dir / "weather" / "weather_hourly.parquet").as_posix()
    registry_path = (data_dir / "registry.parquet").as_posix()

    station_filter = ""
    params: list = [TRAIN_START]
    if station_codes:
        placeholders = ",".join("?" for _ in station_codes)
        station_filter = f"AND h.station_code IN ({placeholders})"
        params += station_codes

    has_weather = Path(weather_path).exists()
    weather_select = ", ".join(f'w."{c}"' for c in FORECAST_WEATHER) if has_weather else ""
    weather_join = ""
    if has_weather:
        weather_select = ", " + weather_select
        weather_join = f"""
            LEFT JOIN read_parquet('{weather_path}') w
              ON w.grid_key = r.grid_key AND w.timestamp = h.hour
        """

    return con.execute(f"""
        WITH grid AS (
            SELECT station_code,
                   printf('%.2f_%.2f', round(latitude / 0.25) * 0.25, round(longitude / 0.25) * 0.25) AS grid_key,
                   latitude, longitude, elevation_m
            FROM read_parquet('{registry_path}')
            WHERE is_mappable
        )
        SELECT h.station_code, h.hour, h.readings,
               h.pm2_5, h.pm10, h.no2, h.so2, h.nh3, h.co, h.o3,
               h."at", h.rh, h.ws, h.wd, h.bp,
               r.latitude, r.longitude, r.elevation_m
               {weather_select}
        FROM read_parquet('{hourly_glob}', hive_partitioning=1) h
        JOIN grid r USING (station_code)
        {weather_join}
        WHERE h.hour >= ? {station_filter}
        ORDER BY h.station_code, h.hour
    """, params).fetch_df()


def add_calendar_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Time-of-day and season as cyclical pairs, so 23:00 sits next to 00:00."""
    hours = frame["hour"]
    frame["hour_sin"] = np.sin(2 * np.pi * hours.dt.hour / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * hours.dt.hour / 24)
    frame["doy_sin"] = np.sin(2 * np.pi * hours.dt.dayofyear / 365.25)
    frame["doy_cos"] = np.cos(2 * np.pi * hours.dt.dayofyear / 365.25)
    frame["weekday"] = hours.dt.weekday
    frame["is_weekend"] = (hours.dt.weekday >= 5).astype("int8")
    frame["is_burning_season"] = hours.dt.month.isin(BURNING_SEASON_MONTHS).astype("int8")
    return frame


def load_quarantine(data_dir: Path) -> dict[str, dict]:
    """Stations whose own weather sensor was shown to disagree with reanalysis."""
    path = data_dir / "sensor_quarantine.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("stations", {})


def drop_quarantined_sensors(frame: pd.DataFrame, quarantined: dict[str, dict]) -> pd.DataFrame:
    """Null the local weather columns for stations with a proven-bad sensor.

    A station reporting 60 C does not just add noise — the model would learn a
    relationship between a wrong temperature and real pollution. The rows stay,
    because their pollutant readings are still good; only the weather goes.
    """
    if not quarantined:
        return frame
    affected = frame["station_code"].isin(quarantined)
    if not affected.any():
        return frame

    columns = [c for c in (*OBSERVED_WEATHER, "wd") if c in frame.columns]
    frame.loc[affected, columns] = np.nan
    log.info("Dropped local weather for %d quarantined station(s) across %s rows: %s",
             frame.loc[affected, "station_code"].nunique(), f"{int(affected.sum()):,}",
             ", ".join(sorted(set(frame.loc[affected, "station_code"]))))
    return frame


def weather_coverage(frame: pd.DataFrame) -> float:
    """Share of rows that actually have forecast weather. 0 if the columns are absent."""
    present = [c for c in FORECAST_WEATHER if c in frame.columns]
    if not present:
        return 0.0
    return float(frame[present].notna().any(axis=1).mean())


def warn_on_thin_weather(frame: pd.DataFrame) -> float:
    """Say loudly when weather is mostly missing.

    A model built this way is autoregressive only: it has no way to know that
    the wind is about to pick up, which is most of why pollution clears. It
    still trains and still scores, so without this warning the gap is invisible.
    """
    coverage = weather_coverage(frame)
    if coverage < MIN_WEATHER_COVERAGE:
        log.warning("=" * 72)
        log.warning("Forecast weather present on only %.1f%% of rows (want >=%.0f%%).",
                    100 * coverage, 100 * MIN_WEATHER_COVERAGE)
        log.warning("The model will be autoregressive only and will not learn dispersion.")
        log.warning("Run pipeline/fetch_weather.py to completion first.")
        log.warning("=" * 72)
    else:
        log.info("Forecast weather present on %.1f%% of rows", 100 * coverage)
    return coverage


def add_history_features(frame: pd.DataFrame, station_column: str = "station_code") -> pd.DataFrame:
    """Lags and rolling statistics, all strictly from before the current hour.

    `shift(1)` before every rolling window is what keeps the current hour out of
    its own feature. Without it the model would read the answer.

    Columns are collected and concatenated once — adding ~120 columns one at a
    time leaves the frame badly fragmented.
    """
    grouped = frame.groupby(station_column, sort=False)
    new_columns: dict[str, pd.Series] = {}

    for pollutant in INPUT_POLLUTANTS:
        if pollutant not in frame.columns:
            continue
        for lag in LAG_HOURS:
            new_columns[f"{pollutant}_lag{lag}"] = grouped[pollutant].shift(lag)
        past = grouped[pollutant].shift(1)
        for window in ROLLING_WINDOWS:
            rolled = past.groupby(frame[station_column], sort=False).rolling(window, min_periods=max(2, window // 4))
            new_columns[f"{pollutant}_mean{window}"] = rolled.mean().reset_index(level=0, drop=True)
            new_columns[f"{pollutant}_std{window}"] = rolled.std().reset_index(level=0, drop=True)

    for measure in OBSERVED_WEATHER:
        if measure not in frame.columns:
            continue
        for lag in (1, 24):
            new_columns[f"{measure}_lag{lag}"] = grouped[measure].shift(lag)

    # Wind direction is circular, so 350 and 10 degrees are 20 apart, not 340.
    # Built from lagged values: the current hour's wind is not known in advance.
    if "wd" in frame.columns:
        for lag in (1, 24):
            lagged = grouped["wd"].shift(lag)
            radians = np.deg2rad(lagged)
            new_columns[f"wd_sin_lag{lag}"] = np.sin(radians)
            new_columns[f"wd_cos_lag{lag}"] = np.cos(radians)

    return pd.concat([frame, pd.DataFrame(new_columns, index=frame.index)], axis=1)


def add_neighbour_features(frame: pd.DataFrame, registry: pd.DataFrame, neighbours: int = 3) -> pd.DataFrame:
    """Mean of the nearest stations an hour ago. Pollution advects, so they saw it first."""
    coords = registry.set_index("station_code")[["latitude", "longitude"]]
    present = [c for c in frame["station_code"].unique() if c in coords.index]
    if len(present) <= neighbours:
        frame["neighbour_pm2_5_lag1"] = np.nan
        return frame

    points = coords.loc[present]
    # Equirectangular distance is plenty for ranking neighbours within one country.
    lat = np.deg2rad(points["latitude"].to_numpy())
    lon = np.deg2rad(points["longitude"].to_numpy())
    mean_lat = lat.mean()
    x = (lon[:, None] - lon[None, :]) * np.cos(mean_lat)
    y = lat[:, None] - lat[None, :]
    distance = np.sqrt(x**2 + y**2)
    np.fill_diagonal(distance, np.inf)
    nearest = {present[i]: [present[j] for j in np.argsort(distance[i])[:neighbours]] for i in range(len(present))}

    lagged = frame[["station_code", "hour", "pm2_5"]].copy()
    lagged["pm2_5_prev"] = lagged.groupby("station_code", sort=False)["pm2_5"].shift(1)
    wide = lagged.pivot_table(index="hour", columns="station_code", values="pm2_5_prev")

    neighbour_mean = pd.DataFrame(index=wide.index)
    for code, others in nearest.items():
        usable = [o for o in others if o in wide.columns]
        neighbour_mean[code] = wide[usable].mean(axis=1) if usable else np.nan

    tidy = neighbour_mean.stack(future_stack=True).rename("neighbour_pm2_5_lag1").reset_index()
    tidy.columns = ["hour", "station_code", "neighbour_pm2_5_lag1"]
    return frame.merge(tidy, on=["station_code", "hour"], how="left")


def add_target(frame: pd.DataFrame, target: str, horizon: int) -> pd.DataFrame:
    """The value we are predicting: this pollutant, `horizon` hours ahead."""
    frame["target"] = frame.groupby("station_code", sort=False)[target].shift(-horizon)
    frame["target_hour"] = frame["hour"] + pd.Timedelta(hours=horizon)
    # The persistence baseline every model must beat: "the same as now".
    frame["baseline_persistence"] = frame[target]
    return frame


def leakage_check(frame: pd.DataFrame, target: str, horizon: int) -> list[str]:
    """Name any feature that could not be known at prediction time.

    A feature is suspect if it is a bare current-hour measurement rather than a
    lag, a rolling statistic, a calendar term, a station constant or a forecast.
    """
    allowed_suffixes = ("_lag1", "_lag2", "_lag3", "_lag6", "_lag12", "_lag24", "_lag48", "_lag168")
    # wd_sin_lag1, wd_cos_lag24 and friends end in a lag suffix already.
    allowed_exact = {
        "station_code", "hour", "target", "target_hour", "baseline_persistence",
        "latitude", "longitude", "elevation_m", "readings",
        "hour_sin", "hour_cos", "doy_sin", "doy_cos", "weekday", "is_weekend", "is_burning_season",
        "neighbour_pm2_5_lag1",
        *FORECAST_WEATHER,
    }

    suspect = []
    for column in frame.columns:
        if column in allowed_exact:
            continue
        if column.endswith(allowed_suffixes):
            continue
        if "_mean" in column or "_std" in column:
            continue
        suspect.append(column)
    return suspect


def build(con: duckdb.DuckDBPyConnection, data_dir: Path, target: str, horizon: int,
          station_codes: list[str] | None) -> tuple[pd.DataFrame, list[str]]:
    """Full feature table for one target and horizon."""
    frame = load_station_hours(con, data_dir, station_codes)
    if frame.empty:
        return frame, []

    registry = pd.read_parquet(data_dir / "registry.parquet")
    frame = drop_quarantined_sensors(frame, load_quarantine(data_dir))
    warn_on_thin_weather(frame)
    frame = add_calendar_features(frame)
    frame = add_history_features(frame)
    frame = add_neighbour_features(frame, registry)
    frame = add_target(frame, target, horizon)

    # Rows with no known answer teach nothing.
    frame = frame[frame["target"].notna()]

    # Drop the raw current-hour columns the model must never see.
    suspect = leakage_check(frame, target, horizon)
    frame = frame.drop(columns=suspect)
    return frame, suspect


def build_summary(frame: pd.DataFrame, dropped: list[str], target: str, horizon: int,
                  elapsed_min: float, stations: int) -> str:
    feature_columns = [c for c in frame.columns
                       if c not in ("station_code", "hour", "target", "target_hour", "baseline_persistence")]
    error = (frame["baseline_persistence"] - frame["target"]).abs()

    coverage = weather_coverage(frame)
    weather_note = (
        f"**{100 * coverage:.1f}%**"
        if coverage >= MIN_WEATHER_COVERAGE
        else f"**{100 * coverage:.1f}% — too thin; this model cannot learn dispersion**"
    )
    lines = [
        f"# Features — {target}, +{horizon}h (MD-01)",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/features.py`.",
        "",
        f"- Target: **{target}**, **{horizon}** hours ahead",
        f"- Training window starts: **{TRAIN_START}**",
        f"- Stations: **{stations}**",
        f"- Rows with a known answer: **{len(frame):,}**",
        f"- Features: **{len(feature_columns)}**",
        f"- Rows with forecast weather: {weather_note}",
        f"- Build time: **{elapsed_min:.1f} min**",
        "",
        "## Persistence baseline",
        "",
        "The number any model has to beat. Predicting *the value right now* is a",
        "surprisingly strong forecast for air quality, and a model that cannot beat it",
        "has learned nothing.",
        "",
        f"- Mean absolute error: **{error.mean():.1f} µg/m³**",
        f"- Median absolute error: **{error.median():.1f} µg/m³**",
        f"- RMSE: **{np.sqrt((error**2).mean()):.1f} µg/m³**",
        "",
        "## Leakage guard",
        "",
        "Columns dropped because they could not be known at prediction time:",
        "",
    ]
    lines += [f"- `{c}`" for c in sorted(dropped)] or ["- None"]
    lines += [
        "",
        "Every surviving feature is a lag, a rolling statistic over past hours only, a",
        "calendar term, a station constant, or a weather forecast — which is genuinely",
        "available ahead of time.",
        "",
        "## Feature coverage",
        "",
        "| Feature | % present |",
        "|---|---:|",
    ]
    coverage = [(c, 100 * frame[c].notna().mean()) for c in feature_columns]
    for name, pct in sorted(coverage, key=lambda x: x[1]):
        lines.append(f"| {name} | {pct:.1f} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--target", choices=TARGETS, default="pm2_5")
    p.add_argument("--horizon", type=int, default=24, help="Hours ahead to predict (default 24)")
    p.add_argument("--sample-stations", type=int, help="Use only the first N stations (quick check)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    if not (args.data / "hourly").is_dir():
        log.error("No hourly data — run pipeline/build_hourly.py first")
        return 2

    registry = pd.read_parquet(args.data / "registry.parquet")
    codes = registry.loc[registry["is_mappable"], "station_code"].tolist()
    if args.sample_stations:
        codes = codes[: args.sample_stations]

    con = duckdb.connect()
    log.info("Building features for %s at +%dh over %d stations", args.target, args.horizon, len(codes))
    start = time.time()
    frame, dropped = build(con, args.data, args.target, args.horizon, codes)
    elapsed_min = (time.time() - start) / 60

    if frame.empty:
        log.error("No rows produced — check the training window and station list")
        return 1

    out_dir = args.data / "features" / f"target={args.target}" / f"horizon={args.horizon}"
    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / "features.parquet", index=False, compression="zstd")

    args.reports.mkdir(parents=True, exist_ok=True)
    name = f"features_{args.target}_h{args.horizon}.md"
    (args.reports / name).write_text(
        build_summary(frame, dropped, args.target, args.horizon, elapsed_min, len(codes)), encoding="utf-8"
    )

    log.info("Done in %.1f min: %s rows, %d features", elapsed_min, f"{len(frame):,}", frame.shape[1])
    log.info("Dropped %d current-hour columns as unusable: %s", len(dropped), ", ".join(sorted(dropped)[:8]))
    log.info("Report: %s", args.reports / name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
