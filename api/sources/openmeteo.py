"""Open-Meteo weather client — free, no API key (LI-02).

Weather is the strongest signal the forecast model can have. Pollution builds up
when air is still and the boundary layer is low, and clears when it is windy or
raining. Without it the model can only guess at dispersion.

Two endpoints, deliberately both:
  archive  — ERA5 reanalysis, used to build the TRAINING features
  forecast — used at inference time

Training on reanalysis and predicting from a forecast is a real mismatch, but a
much smaller one than having no future weather at all. It is noted in the model
report rather than hidden.

Units are in every column name. Open-Meteo reports wind in km/h while the CPCB
archive reports it in m/s, and silently mixing those would be a bad, quiet bug.

Free tier: 10,000 calls/day, non-commercial. Requests are deduplicated onto a
0.25 degree grid, which is Open-Meteo's own resolution, so nothing is lost:
462 stations collapse to 245 grid cells.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Matches Open-Meteo's own grid, so rounding to it loses no real detail.
GRID_DEGREES = 0.25

# Open-Meteo's name -> ours, with the unit in the name.
WEATHER_VARIABLES = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "humidity_pct",
    "wind_speed_10m": "wind_speed_kmh",
    "wind_direction_10m": "wind_direction_deg",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
    # How deep the mixing layer is. A low boundary layer traps pollution.
    "boundary_layer_height": "boundary_layer_m",
}

REQUEST_TIMEOUT_SECONDS = 90
MAX_RETRIES = 5

# Open-Meteo prices a request by how much data it returns, so a 7-year pull is
# far more expensive than the plain call count suggests. A 0.4s pause still
# earned HTTP 429 part-way through a 245-cell fetch.
PAUSE_BETWEEN_CALLS_SECONDS = 2.0

# A rate limit needs real backoff, not the few seconds a network blip needs.
RATE_LIMIT_PAUSE_SECONDS = 90

log = logging.getLogger("openmeteo")


@dataclass(frozen=True)
class GridCell:
    """One Open-Meteo grid point that any number of stations can share."""
    latitude: float
    longitude: float

    @property
    def key(self) -> str:
        return f"{self.latitude:.2f}_{self.longitude:.2f}"


def grid_cell(latitude: float, longitude: float) -> GridCell:
    """Snap a station's coordinates to the weather grid."""
    return GridCell(
        latitude=round(round(latitude / GRID_DEGREES) * GRID_DEGREES, 2),
        longitude=round(round(longitude / GRID_DEGREES) * GRID_DEGREES, 2),
    )


def station_grid_map(registry: pd.DataFrame) -> pd.DataFrame:
    """Map each mappable station to its grid cell, so weather is fetched once per cell."""
    mappable = registry[registry["is_mappable"]].copy()
    cells = [grid_cell(row.latitude, row.longitude) for row in mappable.itertuples()]
    mappable["grid_lat"] = [c.latitude for c in cells]
    mappable["grid_lon"] = [c.longitude for c in cells]
    mappable["grid_key"] = [c.key for c in cells]
    return mappable[["station_code", "latitude", "longitude", "grid_lat", "grid_lon", "grid_key"]]


class WeatherClient:
    """Fetches hourly weather, caching each grid cell on disk.

    The cache is what makes a re-run cheap: a cell already on disk is never
    requested again, so an interrupted historical pull resumes instead of
    starting over.
    """

    def __init__(self, cache_dir: Path | str):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0

    # ---------------------------------------------------------------- fetching

    def _get(self, url: str, params: dict) -> dict:
        """One GET with retries. Open-Meteo has no uptime guarantee on the free tier."""
        query = urllib.parse.urlencode({**params, "timezone": "UTC"}, doseq=True)
        full_url = f"{url}?{query}"

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(full_url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    self.calls_made += 1
                    return json.load(response)
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
                if attempt == MAX_RETRIES:
                    raise
                rate_limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
                wait = RATE_LIMIT_PAUSE_SECONDS if rate_limited else 2**attempt
                log.warning("  %s — retry %d/%d in %ds", error, attempt, MAX_RETRIES, wait)
                time.sleep(wait)
        raise RuntimeError("unreachable")

    @staticmethod
    def _to_frame(payload: dict) -> pd.DataFrame:
        """Turn Open-Meteo's column-per-variable JSON into a tidy hourly frame."""
        hourly = payload.get("hourly")
        if not hourly or not hourly.get("time"):
            return pd.DataFrame(columns=["timestamp", *WEATHER_VARIABLES.values()])

        frame = pd.DataFrame({"timestamp": pd.to_datetime(hourly["time"])})
        for source_name, our_name in WEATHER_VARIABLES.items():
            frame[our_name] = pd.to_numeric(hourly.get(source_name), errors="coerce")

        # m/s so it is directly comparable with the CPCB archive's own wind column.
        frame["wind_speed_ms"] = (frame["wind_speed_kmh"] / 3.6).round(3)
        return frame

    def archive(self, cell: GridCell, start: date, end: date, refresh: bool = False) -> pd.DataFrame:
        """Historical hourly weather for one grid cell, cached on disk."""
        cache_file = self.cache_dir / f"{cell.key}__{start:%Y%m%d}_{end:%Y%m%d}.parquet"
        if cache_file.exists() and not refresh:
            return pd.read_parquet(cache_file)

        payload = self._get(ARCHIVE_URL, {
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "hourly": list(WEATHER_VARIABLES),
        })
        frame = self._to_frame(payload)
        frame.insert(0, "grid_key", cell.key)
        frame.to_parquet(cache_file, index=False, compression="zstd")
        time.sleep(PAUSE_BETWEEN_CALLS_SECONDS)
        return frame

    def forecast(self, cell: GridCell, days: int = 3) -> pd.DataFrame:
        """Forecast hourly weather for one grid cell. Not cached — it changes."""
        payload = self._get(FORECAST_URL, {
            "latitude": cell.latitude,
            "longitude": cell.longitude,
            "hourly": list(WEATHER_VARIABLES),
            "forecast_days": days,
        })
        frame = self._to_frame(payload)
        frame.insert(0, "grid_key", cell.key)
        return frame
