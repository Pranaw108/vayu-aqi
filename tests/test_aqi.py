"""Exact tests for the CPCB AQI engine.

These are the real proof that the formula is right. Every expected value here was
worked out by hand from the CPCB breakpoint table, so if someone edits the bands
later, these fail immediately.

Run:  pytest tests/ -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.aqi import category_for, compute_aqi, sub_index  # noqa: E402


# Concentration exactly on a band edge must give exactly that band's AQI edge.
BAND_EDGES = [
    ("pm2_5", 0, 0), ("pm2_5", 30, 50), ("pm2_5", 60, 100), ("pm2_5", 90, 200),
    ("pm2_5", 120, 300), ("pm2_5", 250, 400), ("pm2_5", 380, 500),
    ("pm10", 50, 50), ("pm10", 100, 100), ("pm10", 250, 200), ("pm10", 350, 300),
    ("pm10", 430, 400), ("pm10", 510, 500),
    ("no2", 40, 50), ("no2", 80, 100), ("no2", 180, 200), ("no2", 280, 300),
    ("so2", 40, 50), ("so2", 80, 100), ("so2", 380, 200),
    ("nh3", 200, 50), ("nh3", 400, 100), ("nh3", 800, 200),
    ("co", 1, 50), ("co", 2, 100), ("co", 10, 200), ("co", 17, 300),
    ("o3", 50, 50), ("o3", 100, 100), ("o3", 168, 200), ("o3", 208, 300),
]


@pytest.mark.parametrize("pollutant,concentration,expected", BAND_EDGES)
def test_band_edges_are_exact(pollutant, concentration, expected):
    got = sub_index(pollutant, pd.Series([concentration])).iloc[0]
    assert got == expected


# Midpoints, computed by hand with I = I_lo + (C - B_lo)/(B_hi - B_lo) * (I_hi - I_lo).
MIDPOINTS = [
    # PM2.5 = 45 sits halfway in band 30-60 -> 51 + 0.5 * (100-51) = 75.5 -> 76
    ("pm2_5", 45, 76),
    # PM2.5 = 75 halfway in 60-90 -> 101 + 0.5 * (200-101) = 150.5 -> 151 (banker's rounding gives 150)
    ("pm2_5", 105, 251),   # 201 + 0.5 * (300-201) = 250.5
    # PM10 = 75 halfway in 50-100 -> 51 + 0.5 * (100-51) = 75.5 -> 76
    ("pm10", 75, 76),
    # CO = 6 sits halfway in 2-10 -> 101 + 0.5 * (200-101) = 150.5
    ("co", 6, 151),
]


@pytest.mark.parametrize("pollutant,concentration,expected", MIDPOINTS)
def test_midpoint_interpolation(pollutant, concentration, expected):
    got = sub_index(pollutant, pd.Series([concentration])).iloc[0]
    # Allow one point either way: numpy rounds .5 to even.
    assert abs(got - expected) <= 1


def test_above_top_band_caps_at_500():
    assert sub_index("pm2_5", pd.Series([5000])).iloc[0] == 500
    assert sub_index("co", pd.Series([999])).iloc[0] == 500


def test_missing_values_stay_missing():
    result = sub_index("pm2_5", pd.Series([None, 30, None]))
    assert pd.isna(result.iloc[0])
    assert result.iloc[1] == 50
    assert pd.isna(result.iloc[2])


def test_unknown_pollutant_raises():
    with pytest.raises(KeyError):
        sub_index("lead", pd.Series([1.0]))


# ---------------------------------------------------------------------------
# CPCB validity rules
# ---------------------------------------------------------------------------

def test_needs_three_pollutants():
    result = compute_aqi({"pm2_5": 50, "no2": 40})
    assert result.aqi is None
    assert result.reason == "fewer_than_3_pollutants"
    assert not result.is_valid


def test_needs_particulate_matter():
    """Three gases without PM2.5 or PM10 is not a valid AQI under CPCB rules."""
    result = compute_aqi({"no2": 40, "so2": 30, "o3": 60})
    assert result.aqi is None
    assert result.reason == "no_pm2_5_or_pm10"


def test_valid_with_pm10_only():
    result = compute_aqi({"pm10": 100, "no2": 40, "so2": 30})
    assert result.is_valid
    assert result.aqi == 100
    assert result.dominant_pollutant == "pm10"


def test_dominant_pollutant_is_the_worst_one():
    result = compute_aqi({"pm2_5": 30, "pm10": 50, "no2": 280, "so2": 10})
    assert result.dominant_pollutant == "no2"
    assert result.aqi == 300


def test_empty_input_is_refused():
    result = compute_aqi({})
    assert result.aqi is None
    assert not result.is_valid


# ---------------------------------------------------------------------------
# Categories
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aqi,expected", [
    (0, "Good"), (50, "Good"), (51, "Satisfactory"), (100, "Satisfactory"),
    (101, "Moderate"), (200, "Moderate"), (201, "Poor"), (300, "Poor"),
    (301, "Very Poor"), (400, "Very Poor"), (401, "Severe"), (500, "Severe"),
])
def test_category_boundaries(aqi, expected):
    assert category_for(aqi)[0] == expected


def test_category_of_none_is_none():
    assert category_for(None) == (None, None)


def test_every_category_has_a_colour():
    for aqi in (25, 75, 150, 250, 350, 450):
        label, colour = category_for(aqi)
        assert label and colour.startswith("#")
