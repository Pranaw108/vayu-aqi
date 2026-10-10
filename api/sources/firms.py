"""NASA FIRMS active-fire client (LI-04).

Stubble burning in Punjab and Haryana is the main driver of north India's worst
winter pollution. Nothing in the station or weather data sees it, which is why
the first model lost to persistence on exactly those episodes.

FIRMS publishes satellite fire detections: a location, a date and a fire
radiative power (FRP), which is roughly how much the fire is emitting.

Needs a free MAP_KEY: https://firms.modaps.eosdis.nasa.gov/api/map_key/
Set it as the FIRMS_MAP_KEY environment variable. Every endpoint rejects
requests without one, so this cannot run until that is set.

Limits: 5,000 transactions per 10 minutes, and an area query covers at most 5
days, so a 7-year pull is about 512 requests.
"""

from __future__ import annotations

import io
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

AREA_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

# VIIRS at 375m, science-quality. Covers 2012-01-20 onward, so the whole
# training window. The NRT feeds only cover recent months.
DEFAULT_SOURCE = "VIIRS_SNPP_SP"

# The API's own ceiling for one area request.
MAX_DAYS_PER_REQUEST = 5

# west, south, east, north — mainland India plus a margin, so fires just over
# the border still count. Smoke does not stop at a border.
INDIA_BBOX = (67.0, 5.0, 99.0, 38.5)

REQUEST_TIMEOUT_SECONDS = 120
MAX_RETRIES = 4
PAUSE_BETWEEN_CALLS_SECONDS = 1.0
RATE_LIMIT_PAUSE_SECONDS = 90

# Low-confidence detections include a lot of false positives.
MIN_CONFIDENCE = "n"  # VIIRS uses l / n / h for low, nominal, high

log = logging.getLogger("firms")


class MissingMapKey(RuntimeError):
    """Raised early, with instructions, rather than failing deep inside a loop."""


@dataclass(frozen=True)
class BoundingBox:
    west: float
    south: float
    east: float
    north: float

    def as_param(self) -> str:
        return f"{self.west},{self.south},{self.east},{self.north}"


def date_chunks(start: date, end: date, days: int = MAX_DAYS_PER_REQUEST) -> list[tuple[date, int]]:
    """Split a range into (start_date, day_count) pieces the API will accept."""
    chunks = []
    cursor = start
    while cursor <= end:
        span = min(days, (end - cursor).days + 1)
        chunks.append((cursor, span))
        cursor += timedelta(days=span)
    return chunks


class FirmsClient:
    """Fetches fire detections, caching each chunk on disk so reruns are cheap."""

    def __init__(self, cache_dir: Path | str, map_key: str | None = None, source: str = DEFAULT_SOURCE):
        self.map_key = map_key or os.environ.get("FIRMS_MAP_KEY", "")
        self.source = source
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.calls_made = 0

    def require_key(self) -> None:
        if not self.map_key:
            raise MissingMapKey(
                "FIRMS_MAP_KEY is not set. Get a free key by email at "
                "https://firms.modaps.eosdis.nasa.gov/api/map_key/ then set "
                "FIRMS_MAP_KEY before running."
            )

    def _get_csv(self, url: str) -> pd.DataFrame:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(url, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    self.calls_made += 1
                    text = response.read().decode("utf-8", "replace")
                break
            except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
                if attempt == MAX_RETRIES:
                    raise
                rate_limited = isinstance(error, urllib.error.HTTPError) and error.code == 429
                wait = RATE_LIMIT_PAUSE_SECONDS if rate_limited else 2**attempt
                log.warning("  %s — retry %d/%d in %ds", error, attempt, MAX_RETRIES, wait)
                time.sleep(wait)

        if "Invalid MAP_KEY" in text:
            raise MissingMapKey("FIRMS rejected the MAP_KEY. Check FIRMS_MAP_KEY.")
        # A chunk with no fires returns just a header, which is a valid answer.
        return pd.read_csv(io.StringIO(text)) if text.strip() else pd.DataFrame()

    def cache_path(self, start: date, days: int) -> Path:
        """Where one chunk is cached. Named so callers can find it without refetching."""
        return self.cache_dir / f"{self.source}__{start:%Y%m%d}_{days}d.parquet"

    def fetch_chunk(self, bbox: BoundingBox, start: date, days: int, refresh: bool = False) -> pd.DataFrame:
        """Fire detections in one box over at most 5 days."""
        self.require_key()
        if days > MAX_DAYS_PER_REQUEST:
            raise ValueError(f"FIRMS allows at most {MAX_DAYS_PER_REQUEST} days per request, got {days}")

        cache_file = self.cache_path(start, days)
        if cache_file.exists() and not refresh:
            return pd.read_parquet(cache_file)

        url = f"{AREA_URL}/{self.map_key}/{self.source}/{bbox.as_param()}/{days}/{start:%Y-%m-%d}"
        frame = normalise_detections(self._get_csv(url))
        frame.to_parquet(cache_file, index=False, compression="zstd")
        time.sleep(PAUSE_BETWEEN_CALLS_SECONDS)
        return frame


def normalise_detections(raw: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns the features need, with consistent names and types."""
    wanted = ["latitude", "longitude", "acq_date", "frp", "confidence", "daynight"]
    if raw.empty:
        return pd.DataFrame(columns=wanted)

    frame = raw.rename(columns={"bright_ti4": "brightness"})
    for column in wanted:
        if column not in frame.columns:
            frame[column] = pd.NA

    frame = frame[wanted].copy()
    frame["acq_date"] = pd.to_datetime(frame["acq_date"], errors="coerce")
    frame["latitude"] = pd.to_numeric(frame["latitude"], errors="coerce")
    frame["longitude"] = pd.to_numeric(frame["longitude"], errors="coerce")
    # FRP is how much the fire is radiating, so it stands in for how much smoke.
    frame["frp"] = pd.to_numeric(frame["frp"], errors="coerce").fillna(0.0)
    return frame.dropna(subset=["latitude", "longitude", "acq_date"])


def drop_low_confidence(frame: pd.DataFrame) -> pd.DataFrame:
    """Low-confidence VIIRS detections carry many false positives."""
    if frame.empty or "confidence" not in frame.columns:
        return frame
    return frame[frame["confidence"].astype(str).str.lower() != "l"]
