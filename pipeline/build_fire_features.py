"""Build per-station daily fire features from FIRMS detections (LI-04).

Turns raw fire detections into something a station-hour row can join to:
fire counts and radiative power in distance bands, plus an upwind-weighted
figure that only counts fires the wind was actually blowing from.

Wind direction comes from ERA5 rather than the stations' own sensors, because
it is complete and 15 station hygrometers and thermometers already failed
validation. It is averaged over the day as a CIRCULAR mean — averaging 350 and
10 degrees arithmetically gives 180, which points the opposite way.

Writes:
    <data>/fires/fire_features.parquet    station_code, acq_date, fire_*
    <reports>/fire_features_summary.md

Usage:
    python pipeline/build_fire_features.py
"""

from __future__ import annotations

import argparse
import glob
import logging
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.firms import drop_low_confidence  # noqa: E402
from pipeline.fire_features import GRID_DEGREES, build_fire_features  # noqa: E402

DEFAULT_DATA = REPO_ROOT / "data"
DEFAULT_REPORTS = REPO_ROOT / "reports"

log = logging.getLogger("build_fire_features")


def load_detections(data_dir: Path) -> pd.DataFrame:
    """Read the finished fires file, or fall back to whatever chunks are cached.

    The fallback matters while the historical fetch is still running: features
    can be developed and checked against real detections rather than guesses.
    """
    final = data_dir / "fires" / "fires.parquet"
    if final.exists():
        log.info("Reading %s", final)
        return pd.read_parquet(final)

    chunks = sorted(glob.glob(str(data_dir / "fire_cache" / "*.parquet")))
    if not chunks:
        raise FileNotFoundError(
            f"No fire data. Run pipeline/fetch_fires.py (nothing in {data_dir / 'fire_cache'})"
        )
    log.warning("fires.parquet not written yet — using %d cached chunks (fetch still running)", len(chunks))
    return pd.concat([pd.read_parquet(f) for f in chunks], ignore_index=True).drop_duplicates()


def circular_mean_degrees(degrees: pd.Series) -> float:
    """Mean of compass bearings. Arithmetic means are wrong for angles."""
    radians = np.deg2rad(degrees.dropna())
    if radians.empty:
        return np.nan
    return float(np.rad2deg(np.arctan2(np.sin(radians).mean(), np.cos(radians).mean())) % 360)


def daily_station_wind(data_dir: Path, registry: pd.DataFrame) -> pd.DataFrame:
    """Daily mean wind direction per station, from ERA5, as a circular mean."""
    weather_path = data_dir / "weather" / "weather_hourly.parquet"
    if not weather_path.exists():
        log.warning("No weather at %s — upwind features will be skipped", weather_path)
        return pd.DataFrame()

    weather = pd.read_parquet(weather_path, columns=["grid_key", "timestamp", "wind_direction_deg"])
    weather["acq_date"] = pd.to_datetime(weather["timestamp"]).dt.normalize()

    radians = np.deg2rad(weather["wind_direction_deg"])
    weather["sin"] = np.sin(radians)
    weather["cos"] = np.cos(radians)
    daily = weather.groupby(["grid_key", "acq_date"], as_index=False)[["sin", "cos"]].mean()
    daily["wind_from_deg"] = np.rad2deg(np.arctan2(daily["sin"], daily["cos"])) % 360

    mappable = registry[registry["is_mappable"]].copy()
    mappable["grid_key"] = (
        (mappable["latitude"] / GRID_DEGREES).round().mul(GRID_DEGREES).map("{:.2f}".format)
        + "_"
        + (mappable["longitude"] / GRID_DEGREES).round().mul(GRID_DEGREES).map("{:.2f}".format)
    )
    return (
        mappable[["station_code", "grid_key"]]
        .merge(daily[["grid_key", "acq_date", "wind_from_deg"]], on="grid_key", how="inner")
        .drop(columns="grid_key")
    )


def build_summary(features: pd.DataFrame, detections: int, stations: int, elapsed_min: float) -> str:
    fire_columns = [c for c in features.columns if c.startswith("fire_") and not c.endswith(("_d1", "_d2", "_d3"))]
    by_month = (
        features.assign(month=pd.to_datetime(features["acq_date"]).dt.month)
        .groupby("month", as_index=False)[fire_columns]
        .mean()
    )
    month_names = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    lines = [
        "# Fire features per station",
        "",
        f"Generated {date.today().isoformat()} by `pipeline/build_fire_features.py`.",
        "",
        f"- Detections used: **{detections:,}**",
        f"- Stations: **{stations}**",
        f"- Station-days: **{len(features):,}**",
        f"- Features: **{len(fire_columns)}** plus 1-, 2- and 3-day lags",
        f"- Runtime: **{elapsed_min:.1f} min**",
        "",
        "Upwind figures use ERA5 wind direction, averaged over the day as a circular",
        "mean, and weight each fire by FRP divided by distance within a 60 degree cone.",
        "",
        "## Monthly means",
        "",
        "October and November should stand out at northern stations — that is paddy",
        "stubble season, and it coincides with the winter inversion that traps the smoke.",
        "",
        "| Month | " + " | ".join(fire_columns) + " |",
        "|---" * (len(fire_columns) + 1) + "|",
    ]
    for row in by_month.itertuples():
        values = " | ".join(f"{getattr(row, c):,.1f}" for c in fire_columns)
        lines.append(f"| {month_names[int(row.month)]} | {values} |")
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    p.add_argument("--state", help="Only build for this state code, e.g. DL")
    p.add_argument("--keep-low-confidence", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    args = parse_args(argv)

    registry = pd.read_parquet(args.data / "registry.parquet")
    if args.state:
        registry = registry[registry["state_code"] == args.state]
    stations = registry[registry["is_mappable"]][["station_code", "latitude", "longitude"]]
    if stations.empty:
        log.error("No mappable stations selected")
        return 2

    detections = load_detections(args.data)
    before = len(detections)
    if not args.keep_low_confidence:
        detections = drop_low_confidence(detections)
    log.info("Detections: %s kept of %s (low confidence dropped)", f"{len(detections):,}", f"{before:,}")

    wind = daily_station_wind(args.data, registry)
    if not wind.empty:
        log.info("Daily wind direction for %d stations", wind["station_code"].nunique())

    start = time.time()
    features = build_fire_features(detections, stations, wind)
    elapsed_min = (time.time() - start) / 60

    if features.empty:
        log.error("No fire features produced")
        return 1

    out_dir = args.data / "fires"
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_parquet(out_dir / "fire_features.parquet", index=False, compression="zstd")

    args.reports.mkdir(parents=True, exist_ok=True)
    (args.reports / "fire_features_summary.md").write_text(
        build_summary(features, len(detections), stations["station_code"].nunique(), elapsed_min),
        encoding="utf-8",
    )

    log.info("Done in %.1f min: %s station-days, %d columns",
             elapsed_min, f"{len(features):,}", features.shape[1])
    log.info("Report: %s", args.reports / "fire_features_summary.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
