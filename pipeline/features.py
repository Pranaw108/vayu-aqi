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

# The handful of fire columns that carry the signal, chosen from the full set's
# own feature importance and its correlation with observed PM2.5. Offering all 56
# at frozen hyperparameters cost 13 points of onset recall, so "all" is no longer
# the default.
CORE_FIRE_FEATURES = (
    "fire_count_upwind",
    "fire_count_100_300km",
    "fire_frp_100_300km",
    "fire_count_300_500km",
    "fire_count_upwind_d1",
    "fire_count_100_300km_d1",
)

# Below this share of rows, forecast weather is too thin to learn dispersion from.
MIN_WEATHER_COVERAGE = 0.5

# Fires are seasonal, so most rows legitimately have none. This threshold is
# about whether the HISTORY has been fetched, not about how often fires burn:
# a quarter of the archive downloaded looks identical to a quiet fire year
# unless it is checked. The weather join taught this the hard way at 1.7%.
MIN_FIRE_DAY_COVERAGE = 0.9

# Stubble burning in Punjab and Haryana drives the worst north-Indian episodes.
BURNING_SEASON_MONTHS = (10, 11)

log = logging.getLogger("features")


def load_station_hours(con: duckdb.DuckDBPyConnection, data_dir: Path,
                       station_codes: list[str] | None, horizon: int = 24) -> pd.DataFrame:
    """Hourly readings joined to weather, twice, at two different times.

    `fc_*`  weather at the hour being PREDICTED. This is the point of forecast
            weather: at prediction time a forecast for that hour exists, so
            using it is legitimate and it is the only genuinely new signal —
            "wind picks up tomorrow" cannot be read off today's readings.
    `wx_*`  reanalysis at the current hour. Useful mainly because it fills in
            for stations whose own sensors are missing or quarantined.

    Joining weather only at the current hour, as this did at first, hands the
    model something it already has from its own lagged sensors, which is why
    adding complete weather changed the score by 0.0%.
    """
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
    weather_select = ""
    weather_join = ""
    if has_weather:
        forecast_cols = ", ".join(f'fc."{c}" AS fc_{c}' for c in FORECAST_WEATHER)
        now_cols = ", ".join(f'wx."{c}" AS wx_{c}' for c in FORECAST_WEATHER)
        weather_select = f", {forecast_cols}, {now_cols}"
        weather_join = f"""
            LEFT JOIN read_parquet('{weather_path}') fc
              ON fc.grid_key = r.grid_key
             AND fc.timestamp = h.hour + INTERVAL {horizon} HOUR
            LEFT JOIN read_parquet('{weather_path}') wx
              ON wx.grid_key = r.grid_key AND wx.timestamp = h.hour
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


def join_fire_features(frame: pd.DataFrame, data_dir: Path, horizon: int,
                       selection: str = "core") -> pd.DataFrame:
    """Attach per-station daily fire exposure.

    Joined on the CURRENT day, never the predicted day. A forecast made now can
    only know about fires already detected; fires on the target day have not
    happened yet. The d1/d2/d3 columns reach further back, so every fire column
    is history.
    """
    fire_path = data_dir / "fires" / "fire_features.parquet"
    if not fire_path.exists():
        log.warning("No fire features at %s — run pipeline/build_fire_features.py", fire_path)
        return frame

    fires = pd.read_parquet(fire_path)
    fires = fires.rename(columns={"acq_date": "join_day"})

    if selection == "none":
        log.info("Fire features skipped (--fire-features none)")
        return frame
    if selection == "core":
        keep = ["station_code", "join_day"] + [c for c in CORE_FIRE_FEATURES if c in fires.columns]
        fires = fires[keep]
    frame["join_day"] = pd.to_datetime(frame["hour"]).dt.normalize()

    merged = frame.merge(fires, on=["station_code", "join_day"], how="left")
    fire_columns = [c for c in merged.columns if c.startswith("fire_")]
    merged[fire_columns] = merged[fire_columns].fillna(0.0)

    # The question that matters is how much of the archive exists, not how often
    # fires burn: a station-day with no fires is a real zero, but a station-day
    # the fetch never reached is a fake one and looks the same.
    fetched_days = set(fires["join_day"].unique())
    needed_days = merged["join_day"].unique()
    day_coverage = float(np.isin(needed_days, list(fetched_days)).mean()) if len(needed_days) else 0.0

    non_zero = (merged[[c for c in fire_columns if not c.endswith(("_d1", "_d2", "_d3"))]].sum(axis=1) > 0).mean()
    log.info("Fire features joined: %d columns · %.1f%% of days fetched · non-zero on %.1f%% of rows",
             len(fire_columns), 100 * day_coverage, 100 * non_zero)
    if day_coverage < MIN_FIRE_DAY_COVERAGE:
        log.warning("=" * 72)
        log.warning("Fire history covers only %.1f%% of the days in this feature table "
                    "(want >=%.0f%%).", 100 * day_coverage, 100 * MIN_FIRE_DAY_COVERAGE)
        log.warning("Unfetched days look exactly like days with no fires, so the model")
        log.warning("would learn that fires rarely matter. Let fetch_fires.py finish.")
        log.warning("=" * 72)
    return merged.drop(columns=["join_day"])


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

    # Drop only the columns proven bad for each station. A sound thermometer
    # with a failed hygrometer should keep its temperature.
    for station, details in quarantined.items():
        columns = [c for c in details.get("quarantine_columns", ["at", "rh"]) if c in frame.columns]
        if not columns:
            continue
        frame.loc[frame["station_code"] == station, columns] = np.nan
    log.info("Dropped local weather for %d quarantined station(s) across %s rows: %s",
             frame.loc[affected, "station_code"].nunique(), f"{int(affected.sum()):,}",
             ", ".join(sorted(set(frame.loc[affected, "station_code"]))))
    return frame


def weather_coverage(frame: pd.DataFrame) -> float:
    """Share of rows that actually have forecast weather. 0 if the columns are absent."""
    present = [f"fc_{c}" for c in FORECAST_WEATHER if f"fc_{c}" in frame.columns]
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
    }

    suspect = []
    for column in frame.columns:
        if column in allowed_exact:
            continue
        # fc_* is weather at the predicted hour (a forecast exists for it);
        # wx_* is reanalysis at the current hour; fire_* is detections up to
        # today. All three are knowable when the forecast is made.
        if column.startswith(("fc_", "wx_", "fire_")):
            continue
        if column.endswith(allowed_suffixes):
            continue
        if "_mean" in column or "_std" in column:
            continue
        suspect.append(column)
    return suspect


def build(con: duckdb.DuckDBPyConnection, data_dir: Path, target: str, horizon: int,
          station_codes: list[str] | None, fire_selection: str = "core") -> tuple[pd.DataFrame, list[str]]:
    """Full feature table for one target and horizon."""
    frame = load_station_hours(con, data_dir, station_codes, horizon)
    if frame.empty:
        return frame, []

    registry = pd.read_parquet(data_dir / "registry.parquet")
    frame = drop_quarantined_sensors(frame, load_quarantine(data_dir))
    warn_on_thin_weather(frame)
    frame = add_calendar_features(frame)
    frame = add_history_features(frame)
    frame = add_neighbour_features(frame, registry)
    frame = join_fire_features(frame, data_dir, horizon, fire_selection)
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
    p.add_argument("--stations", nargs="+", metavar="CODE",
                   help="Use exactly these stations, e.g. --stations DL01 DL26")
    p.add_argument("--state", help="Use only this state folder, e.g. --state Delhi_NCT")
    p.add_argument("--sample-stations", type=int,
                   help="Use a RANDOM sample of N stations. Random, not the first N: station "
                        "codes are grouped by state, so taking the first N quietly tests one "
                        "region. The first such run used AP01-AP08 and concluded weather does "
                        "not help, having tested it where inversions matter least.")
    p.add_argument("--seed", type=int, default=0, help="Sample seed, so a run can be repeated")
    p.add_argument("--fire-features", choices=("core", "all", "none"), default="core",
                   help="How many fire columns to include. 'all' (56) cost 13 points of onset "
                        "recall against no fires at all; 'core' is the handful that carry signal")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    if not (args.data / "hourly").is_dir():
        log.error("No hourly data — run pipeline/build_hourly.py first")
        return 2

    registry = pd.read_parquet(args.data / "registry.parquet")
    mappable = registry[registry["is_mappable"]]

    if args.stations:
        codes = [c for c in args.stations if c in set(mappable["station_code"])]
        missing = sorted(set(args.stations) - set(codes))
        if missing:
            log.warning("Not mappable or not found, skipped: %s", ", ".join(missing))
    elif args.state:
        codes = mappable.loc[mappable["state_code"] == args.state, "station_code"].tolist()
        if not codes:
            states = ", ".join(sorted(mappable["state_code"].dropna().unique()))
            log.error("No mappable stations for state %r. Available: %s", args.state, states)
            return 2
    elif args.sample_stations:
        codes = (
            mappable["station_code"]
            .sample(min(args.sample_stations, len(mappable)), random_state=args.seed)
            .tolist()
        )
    else:
        codes = mappable["station_code"].tolist()

    if not codes:
        log.error("No stations selected")
        return 2
    covered = mappable.loc[mappable["station_code"].isin(codes), "state_code"].value_counts()
    log.info("States covered: %s", ", ".join(f"{s}={n}" for s, n in covered.items()))

    con = duckdb.connect()
    log.info("Building features for %s at +%dh over %d stations", args.target, args.horizon, len(codes))
    start = time.time()
    frame, dropped = build(con, args.data, args.target, args.horizon, codes, args.fire_features)
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
