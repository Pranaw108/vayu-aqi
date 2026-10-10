"""Turn fire detections into per-station features (MD-01 extension).

Why this exists: the first backtest lost to persistence on high-pollution
episodes, because stubble burning drives those episodes and nothing in the
station or weather data sees it.

Two ideas matter here:

1. **Distance bands.** A fire 50 km away matters more than one 400 km away, but
   not in a way worth fitting a curve to, so fires are counted in bands and the
   model decides what each band is worth.

2. **Upwind weighting.** Smoke travels with the wind. A fire 200 km upwind
   affects a station; the same fire downwind does not. Wind direction is already
   in the archive, so this is cheap and is where most of the real signal is.

Fires are aggregated to a daily 0.25 degree grid first. Satellites pass roughly
twice a day, so finer time resolution would be invented precision, and the grid
makes the per-station join a lookup instead of 18 million distance calculations.

Known limitation: snapping a fire to a cell centre moves it by up to about 20 km,
which can put it in the neighbouring distance band. Delhi to Punjab is 286 km but
grids to 301 km. The bands are modelling choices rather than physical thresholds,
so this is acceptable — see `test_gridding_can_shift_a_fire_across_a_band_boundary`.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

GRID_DEGREES = 0.25

# Bands in km. Punjab's burning belt sits roughly 200-400 km from Delhi, so the
# outer band has to reach that far to be useful.
DISTANCE_BANDS = ((0, 100), (100, 300), (300, 500))

# How many days of fire history to carry. Smoke from a big burn lingers.
FIRE_LAG_DAYS = (0, 1, 2, 3)

EARTH_RADIUS_KM = 6371.0

# A fire counts as upwind if it sits within this angle of where the wind is
# coming from. Smoke plumes spread, so this is generous rather than a ray.
UPWIND_HALF_ANGLE_DEGREES = 60.0

log = logging.getLogger("fire_features")


def to_daily_grid(detections: pd.DataFrame) -> pd.DataFrame:
    """Fires summed onto a daily 0.25 degree grid."""
    if detections.empty:
        return pd.DataFrame(columns=["acq_date", "cell_lat", "cell_lon", "fire_count", "total_frp"])

    frame = detections.copy()
    frame["cell_lat"] = (frame["latitude"] / GRID_DEGREES).round() * GRID_DEGREES
    frame["cell_lon"] = (frame["longitude"] / GRID_DEGREES).round() * GRID_DEGREES
    frame["acq_date"] = pd.to_datetime(frame["acq_date"]).dt.normalize()

    return (
        frame.groupby(["acq_date", "cell_lat", "cell_lon"], as_index=False)
        .agg(fire_count=("frp", "size"), total_frp=("frp", "sum"))
    )


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Great-circle distance. Equirectangular error is too large at 500 km."""
    lat1, lon1, lat2, lon2 = map(np.deg2rad, (lat1, lon1, lat2, lon2))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


def bearing_degrees(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Compass bearing from point 1 to point 2, 0 = north, clockwise."""
    lat1r, lat2r = np.deg2rad(lat1), np.deg2rad(lat2)
    dlon = np.deg2rad(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2r)
    y = np.cos(lat1r) * np.sin(lat2r) - np.sin(lat1r) * np.cos(lat2r) * np.cos(dlon)
    return np.rad2deg(np.arctan2(x, y)) % 360


def station_cell_links(stations: pd.DataFrame, cells: pd.DataFrame,
                       max_km: float = DISTANCE_BANDS[-1][1]) -> pd.DataFrame:
    """Every station-to-cell pair within range, with its distance and bearing.

    Computed once. After this, a station-day's fire exposure is a grouped sum.
    """
    unique_cells = cells[["cell_lat", "cell_lon"]].drop_duplicates().reset_index(drop=True)
    if unique_cells.empty or stations.empty:
        return pd.DataFrame(columns=["station_code", "cell_lat", "cell_lon", "distance_km", "bearing_deg"])

    station_lat = stations["latitude"].to_numpy()[:, None]
    station_lon = stations["longitude"].to_numpy()[:, None]
    cell_lat = unique_cells["cell_lat"].to_numpy()[None, :]
    cell_lon = unique_cells["cell_lon"].to_numpy()[None, :]

    distance = haversine_km(station_lat, station_lon, cell_lat, cell_lon)
    within = distance <= max_km
    station_index, cell_index = np.nonzero(within)

    return pd.DataFrame({
        "station_code": stations["station_code"].to_numpy()[station_index],
        "cell_lat": unique_cells["cell_lat"].to_numpy()[cell_index],
        "cell_lon": unique_cells["cell_lon"].to_numpy()[cell_index],
        "distance_km": distance[station_index, cell_index],
        # Bearing from the station to the fire, so "upwind" can be judged later.
        "bearing_deg": bearing_degrees(
            stations["latitude"].to_numpy()[station_index],
            stations["longitude"].to_numpy()[station_index],
            unique_cells["cell_lat"].to_numpy()[cell_index],
            unique_cells["cell_lon"].to_numpy()[cell_index],
        ),
    })


def band_label(low: int, high: int) -> str:
    return f"{low}_{high}km"


def daily_station_fires(cells: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Fire count and FRP per station per day, in distance bands."""
    if cells.empty or links.empty:
        return pd.DataFrame(columns=["station_code", "acq_date"])

    joined = cells.merge(links, on=["cell_lat", "cell_lon"], how="inner")
    frames = []
    for low, high in DISTANCE_BANDS:
        band = joined[(joined["distance_km"] >= low) & (joined["distance_km"] < high)]
        if band.empty:
            continue
        label = band_label(low, high)
        summed = (
            band.groupby(["station_code", "acq_date"], as_index=False)
            .agg(**{f"fire_count_{label}": ("fire_count", "sum"), f"fire_frp_{label}": ("total_frp", "sum")})
        )
        frames.append(summed.set_index(["station_code", "acq_date"]))

    if not frames:
        return pd.DataFrame(columns=["station_code", "acq_date"])
    return pd.concat(frames, axis=1).fillna(0.0).reset_index()


def daily_upwind_fires(cells: pd.DataFrame, links: pd.DataFrame, wind: pd.DataFrame) -> pd.DataFrame:
    """Fire exposure counting only fires the wind was blowing from.

    `wind` needs station_code, acq_date and wind_from_deg — the direction the
    wind came from, which is the convention both CPCB and Open-Meteo use.
    """
    if cells.empty or links.empty or wind.empty:
        return pd.DataFrame(columns=["station_code", "acq_date", "fire_frp_upwind"])

    joined = cells.merge(links, on=["cell_lat", "cell_lon"], how="inner")
    joined = joined.merge(wind, on=["station_code", "acq_date"], how="inner")
    if joined.empty:
        return pd.DataFrame(columns=["station_code", "acq_date", "fire_frp_upwind"])

    # Smallest angle between the fire's bearing and where the wind came from.
    difference = (joined["bearing_deg"] - joined["wind_from_deg"]).abs() % 360
    angle = np.minimum(difference, 360 - difference)
    upwind = joined[angle <= UPWIND_HALF_ANGLE_DEGREES]
    if upwind.empty:
        return pd.DataFrame(columns=["station_code", "acq_date"])

    # Upwind totals per distance band, with no inverse-distance weighting.
    # An earlier version divided FRP by distance, which gave a 10km fire thirty
    # times the weight of an equivalent fire 300km away — and measured against
    # Delhi PM2.5, fires within 100km correlate at r=+0.04 while those at
    # 100-300km reach +0.50. The weighting was emphasising the fires that do not
    # matter. Bands let the model learn the distance response instead.
    frames = []
    for low, high in DISTANCE_BANDS:
        band = upwind[(upwind["distance_km"] >= low) & (upwind["distance_km"] < high)]
        if band.empty:
            continue
        label = band_label(low, high)
        frames.append(
            band.groupby(["station_code", "acq_date"], as_index=False)
            .agg(**{f"fire_count_upwind_{label}": ("fire_count", "sum"),
                    f"fire_frp_upwind_{label}": ("total_frp", "sum")})
            .set_index(["station_code", "acq_date"])
        )

    # Kept as well, so the model can use a single summary if it prefers one.
    totals = (
        upwind.groupby(["station_code", "acq_date"], as_index=False)
        .agg(fire_count_upwind=("fire_count", "sum"), fire_frp_upwind=("total_frp", "sum"))
        .set_index(["station_code", "acq_date"])
    )
    frames.append(totals)
    return pd.concat(frames, axis=1).fillna(0.0).reset_index()


def add_fire_lags(daily: pd.DataFrame) -> pd.DataFrame:
    """Yesterday's and the day before's fires, since smoke lingers."""
    if daily.empty:
        return daily
    daily = daily.sort_values(["station_code", "acq_date"]).reset_index(drop=True)
    value_columns = [c for c in daily.columns if c.startswith("fire_")]
    grouped = daily.groupby("station_code", sort=False)

    lagged = {}
    for column in value_columns:
        for days in FIRE_LAG_DAYS:
            if days == 0:
                continue
            lagged[f"{column}_d{days}"] = grouped[column].shift(days)
    return pd.concat([daily, pd.DataFrame(lagged, index=daily.index)], axis=1)


def build_fire_features(detections: pd.DataFrame, stations: pd.DataFrame,
                        wind: pd.DataFrame | None = None) -> pd.DataFrame:
    """Full per-station daily fire feature table, ready to join on date."""
    cells = to_daily_grid(detections)
    if cells.empty:
        log.warning("No fire detections — fire features will be empty")
        return pd.DataFrame(columns=["station_code", "acq_date"])

    links = station_cell_links(stations, cells)
    log.info("%d station-cell pairs within %dkm", len(links), DISTANCE_BANDS[-1][1])

    daily = daily_station_fires(cells, links)
    if wind is not None and not wind.empty:
        upwind = daily_upwind_fires(cells, links, wind)
        if not upwind.empty:
            daily = daily.merge(upwind, on=["station_code", "acq_date"], how="left")

    fire_columns = [c for c in daily.columns if c.startswith("fire_")]
    daily[fire_columns] = daily[fire_columns].fillna(0.0)
    return add_fire_lags(daily)
