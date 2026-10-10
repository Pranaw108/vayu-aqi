"""Tests for the DuckDB query layer (DF-03).

These run against the real curated archive, so they check actual behaviour
rather than mocks. Skipped with a clear message if the data has not been built.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from datetime import timedelta
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.store import COMPARABLE_WINDOW_HOURS, Store  # noqa: E402

needs_data = pytest.mark.skipif(
    not (REPO_ROOT / "data" / "hourly").is_dir(),
    reason="No hourly data — run pipeline/normalise.py then pipeline/build_hourly.py",
)


@pytest.fixture(scope="module")
def store():
    s = Store()
    yield s
    s.close()


@needs_data
def test_missing_data_fails_loudly(tmp_path):
    """An empty data folder must raise, not return an empty result that looks fine."""
    with pytest.raises(FileNotFoundError):
        Store(tmp_path)


@needs_data
def test_health_reports_honest_age(store):
    health = store.health()
    assert health["stations"] > 500
    assert health["mappable_stations"] < health["stations"]  # placeholder coords excluded
    assert health["data_age_hours"] > 0
    # The archive ends in 2024, so this must never claim to be live.
    assert health["is_live"] is False


@needs_data
def test_stations_default_to_mappable_only(store):
    mappable = store.stations()
    everything = store.stations(mappable_only=False)
    assert len(mappable) < len(everything)
    assert mappable["is_mappable"].all()


@needs_data
def test_mappable_stations_have_usable_coordinates(store):
    stations = store.stations()
    assert stations["latitude"].notna().all()
    assert stations["longitude"].notna().all()
    # Inside India's bounding box.
    assert stations["latitude"].between(6, 37.5).all()
    assert stations["longitude"].between(68, 97.5).all()


@needs_data
def test_repeat_call_returns_the_same_values(store):
    first = store.stations()
    second = store.stations()
    assert first["aqi"].fillna(-1).tolist() == second["aqi"].fillna(-1).tolist()


@needs_data
def test_some_stations_are_refused_an_aqi(store):
    """The CPCB validity rules must actually fire on real data, not just in unit tests."""
    stations = store.stations()
    refused = stations[stations["aqi"].isna()]
    assert len(refused) > 0, "no station was refused — the validity rules may not be reached"
    assert refused["aqi_reason"].notna().all()
    assert set(refused["aqi_reason"]) <= {"fewer_than_3_pollutants", "no_pm2_5_or_pm10", "insufficient_data"}


@needs_data
def test_every_aqi_is_in_range_and_categorised(store):
    scored = store.stations().dropna(subset=["aqi"])
    assert scored["aqi"].between(0, 500).all()
    assert scored["aqi_category"].notna().all()
    assert scored["dominant_pollutant"].notna().all()


@needs_data
def test_freshest_station_is_zero_hours_behind(store):
    stations = store.stations(mappable_only=False)
    assert stations["hours_behind_latest"].min() == 0


@needs_data
def test_cities_exclude_stations_from_a_different_period(store):
    cities, note = store.cities()
    assert note["stations_excluded_as_not_comparable"] > 0, (
        "nothing was excluded — the comparability window may not be applied"
    )
    assert note["comparable_within_hours"] == COMPARABLE_WINDOW_HOURS

    # A city can own both comparable and non-comparable stations (Mumbai does),
    # so check the count each city was built from, not every station sharing its name.
    stations = store.stations()
    comparable = stations[
        stations["has_real_city"]
        & stations["aqi"].notna()
        & (stations["hours_behind_latest"] <= COMPARABLE_WINDOW_HOURS)
    ]
    expected = comparable.groupby(["city", "state_code"]).size()
    for row in cities.itertuples():
        assert row.stations == expected[(row.city, row.state_code)]
    assert int(cities["stations"].sum()) == note["stations_used"]


@needs_data
def test_cities_skip_agency_named_stations(store):
    """A pollution-control-board acronym is not a city, so it must not be ranked."""
    cities, _ = store.cities()
    acronyms = {"HSPCB", "KSPCB", "UPPCB", "RSPCB", "BSPCB", "MPCB", "TNPCB", "OSPCB", "MPPCB", "CECB"}
    assert not (set(cities["city"]) & acronyms)


@needs_data
def test_city_aqi_matches_its_category(store):
    from api.aqi import category_for

    cities, _ = store.cities()
    for row in cities.itertuples():
        assert row.aqi_category == category_for(row.aqi)[0]


@needs_data
def test_station_detail_includes_pollutants(store):
    code = store.stations().iloc[0]["station_code"]
    detail = store.station(code)
    assert detail["station_code"] == code
    assert "pollutants" in detail
    assert set(detail["pollutants"]) <= {"pm2_5", "pm10", "no2", "so2", "nh3", "co", "o3"}


@needs_data
def test_unknown_station_returns_none(store):
    assert store.station("NOPE99") is None


@needs_data
def test_history_respects_the_requested_range(store):
    code = store.stations().iloc[0]["station_code"]
    end = store.as_of()
    start = end - timedelta(days=3)
    history = store.history(code, start, end)
    assert not history.empty
    assert history["timestamp"].min() >= pd.Timestamp(start)
    assert history["timestamp"].max() <= pd.Timestamp(end)
    # Oldest first, so a chart can plot it directly.
    assert history["timestamp"].is_monotonic_increasing


@needs_data
def test_history_of_an_empty_range_is_empty_not_an_error(store):
    code = store.stations().iloc[0]["station_code"]
    start = pd.Timestamp("1990-01-01")
    history = store.history(code, start, start + timedelta(days=1))
    assert history.empty
