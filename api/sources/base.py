"""Shared shape for every live air-quality source (LI-01).

All sources are put behind one interface so the service can fail over between
them without caring which is answering. Providers go down, licences change and
rate limits bite; swapping an implementation should not mean rewriting
ingestion.

Two rules every source must honour, because both have already caused real bugs
in this project:

1. Report CONCENTRATIONS, not AQI sub-indices. The AQI engine turns ug/m3 into
   an index; handing it an index would convert twice. WAQI returns sub-indices
   and is therefore not usable as a source here.
2. Carry the reading's own timestamp. A service that cannot say how old a number
   is will eventually present a stale one as current.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Canonical pollutant names, matching the curated archive and the AQI engine.
POLLUTANTS = ("pm2_5", "pm10", "no2", "so2", "nh3", "co", "o3")

# How close a live station must be to a registry station to be the same one.
# Two CPCB stations in one city sit kilometres apart, so this is generous enough
# to absorb coordinate rounding without merging neighbours.
MATCH_RADIUS_KM = 2.0

EARTH_RADIUS_KM = 6371.0


@dataclass(frozen=True)
class Reading:
    """One pollutant measurement at one station at one time."""
    station_code: str
    timestamp: datetime
    pollutant: str
    value: float
    unit: str
    source: str

    def age_hours(self, now: datetime | None = None) -> float:
        reference = now or datetime.now(timezone.utc)
        observed = self.timestamp
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=timezone.utc)
        return (reference - observed).total_seconds() / 3600


class SourceUnavailable(RuntimeError):
    """The source could not be reached or refused the request."""


class MissingCredential(RuntimeError):
    """Raised early, with instructions, rather than failing mid-poll."""


class AirQualitySource(ABC):
    """One live provider."""

    name: str

    @abstractmethod
    def fetch_current(self) -> list[Reading]:
        """Latest reading for every station this source knows about."""

    @abstractmethod
    def healthy(self) -> bool:
        """Cheap liveness probe, used by the failover chain."""


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Distance from one point to many, in kilometres."""
    lat1r, lon1r = np.deg2rad(lat1), np.deg2rad(lon1)
    lat2r, lon2r = np.deg2rad(lat2), np.deg2rad(lon2)
    dlat, dlon = lat2r - lat1r, lon2r - lon1r
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1r) * np.cos(lat2r) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))


class StationMatcher:
    """Matches a live station to a registry station_code by position.

    Coordinates rather than names, deliberately. CPCB station names are
    inconsistent across feeds, and 106 registry rows carry a pollution-control
    board acronym where the city should be — matching on that would be guesswork.
    Positions are unambiguous.
    """

    def __init__(self, registry: pd.DataFrame, radius_km: float = MATCH_RADIUS_KM):
        mappable = registry[registry["is_mappable"]]
        self.codes = mappable["station_code"].to_numpy()
        self.latitudes = mappable["latitude"].to_numpy(dtype=float)
        self.longitudes = mappable["longitude"].to_numpy(dtype=float)
        self.radius_km = radius_km
        self.unmatched: list[tuple[str, float, float]] = []

    def match(self, name: str, latitude: float, longitude: float) -> str | None:
        """Nearest registry station within the radius, or None."""
        if latitude is None or longitude is None or not np.isfinite([latitude, longitude]).all():
            self.unmatched.append((name, latitude, longitude))
            return None

        distances = haversine_km(latitude, longitude, self.latitudes, self.longitudes)
        nearest = int(np.argmin(distances))
        if distances[nearest] > self.radius_km:
            self.unmatched.append((name, latitude, longitude))
            return None
        return str(self.codes[nearest])


def readings_to_frame(readings: list[Reading]) -> pd.DataFrame:
    """One row per station-hour, one column per pollutant — the store's shape."""
    if not readings:
        return pd.DataFrame(columns=["station_code", "timestamp", *POLLUTANTS, "source"])

    tidy = pd.DataFrame([
        {"station_code": r.station_code, "timestamp": r.timestamp,
         "pollutant": r.pollutant, "value": r.value, "source": r.source}
        for r in readings
    ])
    wide = tidy.pivot_table(
        index=["station_code", "timestamp", "source"],
        columns="pollutant", values="value", aggfunc="last",
    ).reset_index()
    wide.columns.name = None
    for pollutant in POLLUTANTS:
        if pollutant not in wide.columns:
            wide[pollutant] = np.nan
    return wide[["station_code", "timestamp", *POLLUTANTS, "source"]]


def first_healthy(sources: list[AirQualitySource]) -> AirQualitySource | None:
    """First source that answers, in the order given."""
    for source in sources:
        try:
            if source.healthy():
                return source
        except Exception:
            continue
    return None
