"""OpenAQ live client — the failover source (LI-01).

OpenAQ mirrors India's CPCB network and, unlike WAQI, reports concentrations in
ug/m3 rather than AQI sub-indices, which is what the AQI engine needs.

Needs a free API key: https://explore.openaq.org/register
Set it as OPENAQ_API_KEY. Free tier allows 60 requests a minute and 2,000 an
hour, which is ample for an hourly poll.

This is the failover rather than the primary because CPCB via data.gov.in is the
same network the archive came from, so the model sees the sensors it trained on.
"""

from __future__ import annotations

import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime
from pathlib import Path

import pandas as pd

from .base import (
    AirQualitySource,
    MissingCredential,
    Reading,
    SourceUnavailable,
    StationMatcher,
)

BASE_URL = "https://api.openaq.org/v3"

# OpenAQ's parameter names on the left, ours on the right.
PARAMETER_NAMES = {
    "pm25": "pm2_5",
    "pm10": "pm10",
    "no2": "no2",
    "so2": "so2",
    "nh3": "nh3",
    "co": "co",
    "o3": "o3",
}

# India, so a bounding-box query returns only relevant stations.
INDIA_BBOX = "67.0,5.0,99.0,38.5"

REQUEST_TIMEOUT_SECONDS = 60
MAX_RETRIES = 3
PAGE_LIMIT = 1000

log = logging.getLogger("openaq")


class OpenAqSource(AirQualitySource):
    name = "openaq"

    def __init__(self, registry: pd.DataFrame, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("OPENAQ_API_KEY", "")
        self.matcher = StationMatcher(registry)
        self.calls_made = 0

    def require_key(self) -> None:
        if not self.api_key:
            raise MissingCredential(
                "OPENAQ_API_KEY is not set. Register free at "
                "https://explore.openaq.org/register and set OPENAQ_API_KEY."
            )

    def _get(self, path: str, params: dict) -> dict:
        self.require_key()
        url = f"{BASE_URL}/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers={"X-API-Key": self.api_key})

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
                    self.calls_made += 1
                    return json.load(response)
            except urllib.error.HTTPError as error:
                if error.code in (401, 403):
                    raise MissingCredential("OpenAQ rejected the key. Check OPENAQ_API_KEY.") from error
                if attempt == MAX_RETRIES:
                    raise SourceUnavailable(f"OpenAQ returned HTTP {error.code}") from error
                time.sleep(2**attempt)
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt == MAX_RETRIES:
                    raise SourceUnavailable(f"OpenAQ unreachable: {error}") from error
                time.sleep(2**attempt)
        raise SourceUnavailable("OpenAQ: retries exhausted")

    def healthy(self) -> bool:
        try:
            self._get("parameters", {"limit": 1})
            return True
        except (MissingCredential, SourceUnavailable):
            return False

    def fetch_current(self) -> list[Reading]:
        """Latest measurement per Indian station, mapped to registry codes."""
        payload = self._get("locations", {"bbox": INDIA_BBOX, "limit": PAGE_LIMIT})
        locations = payload.get("results", [])
        log.info("OpenAQ returned %d Indian locations", len(locations))

        readings: list[Reading] = []
        for location in locations:
            coordinates = location.get("coordinates") or {}
            station_code = self.matcher.match(
                location.get("name", ""),
                coordinates.get("latitude"),
                coordinates.get("longitude"),
            )
            if station_code is None:
                continue
            readings.extend(self._readings_for(location, station_code))

        if self.matcher.unmatched:
            log.info("%d OpenAQ locations matched no registry station within "
                     "the radius and were skipped", len(self.matcher.unmatched))
        return readings

    def _readings_for(self, location: dict, station_code: str) -> list[Reading]:
        """Turn one location's latest values into Readings, skipping bad units."""
        out: list[Reading] = []
        for sensor in location.get("sensors", []) or []:
            parameter = (sensor.get("parameter") or {})
            our_name = PARAMETER_NAMES.get(parameter.get("name", ""))
            if our_name is None:
                continue

            latest = sensor.get("latest") or {}
            value = latest.get("value")
            if value is None:
                continue

            unit = parameter.get("units", "")
            # Anything not in a mass concentration cannot go to the AQI engine.
            if unit not in ("µg/m³", "ug/m3", "mg/m³", "mg/m3", "ppm"):
                log.debug("skipping %s at %s: unexpected unit %r", our_name, station_code, unit)
                continue

            timestamp = latest.get("datetime", {}).get("utc") if isinstance(latest.get("datetime"), dict) else None
            out.append(Reading(
                station_code=station_code,
                timestamp=_parse_time(timestamp),
                pollutant=our_name,
                value=float(value),
                unit=unit,
                source=self.name,
            ))
        return out


def _parse_time(value: str | None) -> datetime:
    """Parse OpenAQ's ISO timestamp, falling back to now if absent."""
    if not value:
        return datetime.now()
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return datetime.now()


def load_key_from_env_file(repo_root: Path) -> None:
    """Read OPENAQ_API_KEY from .env if it is not already in the environment."""
    if os.environ.get("OPENAQ_API_KEY"):
        return
    env_file = repo_root / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        if line.startswith("OPENAQ_API_KEY="):
            os.environ["OPENAQ_API_KEY"] = line.split("=", 1)[1].strip()
            return
