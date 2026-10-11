"""Tests for the shared live-source plumbing (LI-01).

The coordinate matcher and the unit rules are what stop a live feed quietly
reporting the wrong city or the wrong kind of number — both of which already
happened while probing providers.

Run:  pytest tests/test_sources_base.py -v
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.sources.base import (  # noqa: E402
    MATCH_RADIUS_KM,
    POLLUTANTS,
    AirQualitySource,
    Reading,
    StationMatcher,
    first_healthy,
    haversine_km,
    readings_to_frame,
)


def registry() -> pd.DataFrame:
    return pd.DataFrame({
        "station_code": ["DL01", "DL02", "MH01"],
        "latitude": [28.610948, 28.628191, 19.0436],
        "longitude": [77.038456, 77.241044, 73.0185],
        "is_mappable": [True, True, True],
    })


# ------------------------------------------------------------------ matching

def test_a_station_at_the_same_place_matches():
    m = StationMatcher(registry())
    assert m.match("Dwarka", 28.610948, 77.038456) == "DL01"


def test_a_station_a_few_hundred_metres_away_still_matches():
    """Coordinate rounding between feeds must not break the match."""
    m = StationMatcher(registry())
    assert m.match("Dwarka Sector 8", 28.6120, 77.0390) == "DL01"


def test_the_nearer_of_two_city_stations_wins():
    m = StationMatcher(registry())
    assert m.match("somewhere in east Delhi", 28.6280, 77.2405) == "DL02"


def test_a_station_in_another_city_does_not_match():
    """The trap that prompted this: a provider returned Shanghai for 'delhi'."""
    m = StationMatcher(registry())
    assert m.match("Shanghai", 31.2047, 121.4489) is None
    assert m.unmatched, "an unmatched station must be recorded, not silently dropped"


def test_missing_coordinates_do_not_match():
    m = StationMatcher(registry())
    assert m.match("no position", None, None) is None
    assert m.match("nan position", float("nan"), float("nan")) is None


def test_a_station_just_outside_the_radius_is_rejected():
    m = StationMatcher(registry(), radius_km=1.0)
    # Roughly 2 km north of DL01.
    assert m.match("too far", 28.6290, 77.038456) is None


def test_haversine_is_symmetric_and_zero_at_a_point():
    d = haversine_km(28.61, 77.21, np.array([28.61]), np.array([77.21]))
    assert d[0] == pytest.approx(0.0, abs=1e-6)
    a = haversine_km(28.61, 77.21, np.array([30.90]), np.array([75.85]))
    b = haversine_km(30.90, 75.85, np.array([28.61]), np.array([77.21]))
    assert a[0] == pytest.approx(b[0], rel=1e-9)


def test_default_radius_is_tight_enough_to_separate_neighbours():
    """DL01 and DL02 are about 20 km apart, so the radius must not merge them."""
    gap = float(haversine_km(28.610948, 77.038456, np.array([28.628191]), np.array([77.241044]))[0])
    assert gap > MATCH_RADIUS_KM


# ------------------------------------------------------------------ reshaping

def test_readings_become_one_row_per_station_hour():
    when = datetime(2024, 11, 5, 10, tzinfo=timezone.utc)
    readings = [
        Reading("DL01", when, "pm2_5", 180.0, "ug/m3", "openaq"),
        Reading("DL01", when, "pm10", 320.0, "ug/m3", "openaq"),
        Reading("DL02", when, "pm2_5", 150.0, "ug/m3", "openaq"),
    ]
    frame = readings_to_frame(readings)
    assert len(frame) == 2
    assert set(POLLUTANTS) <= set(frame.columns)
    row = frame[frame.station_code == "DL01"].iloc[0]
    assert row.pm2_5 == 180.0 and row.pm10 == 320.0
    # A pollutant nobody reported is absent, not zero.
    assert pd.isna(row.no2)


def test_no_readings_gives_an_empty_frame_with_the_right_columns():
    frame = readings_to_frame([])
    assert frame.empty
    assert set(POLLUTANTS) <= set(frame.columns)


def test_a_reading_knows_how_old_it_is():
    now = datetime(2024, 11, 5, 12, tzinfo=timezone.utc)
    r = Reading("DL01", now - timedelta(hours=3), "pm2_5", 100.0, "ug/m3", "cpcb")
    assert r.age_hours(now) == pytest.approx(3.0)


# ------------------------------------------------------------------ failover

class FakeSource(AirQualitySource):
    def __init__(self, name, ok, raises=False):
        self.name, self._ok, self._raises = name, ok, raises

    def healthy(self):
        if self._raises:
            raise RuntimeError("boom")
        return self._ok

    def fetch_current(self):
        return []


def test_failover_picks_the_first_source_that_answers():
    chosen = first_healthy([FakeSource("down", False), FakeSource("up", True)])
    assert chosen.name == "up"


def test_a_source_that_raises_is_skipped_not_fatal():
    chosen = first_healthy([FakeSource("angry", False, raises=True), FakeSource("up", True)])
    assert chosen.name == "up"


def test_all_sources_down_returns_none_rather_than_raising():
    assert first_healthy([FakeSource("a", False), FakeSource("b", False)]) is None
