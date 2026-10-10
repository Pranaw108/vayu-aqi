"""Tests for the weather-sensor verdict (part of MD-01's data hygiene).

Correlation alone gets this wrong in both directions, which real data proved:
a Kerala station scored r=0.115 while agreeing with reanalysis (there is almost
no seasonal swing to correlate), and three stations miscalibrated by over 3 C
scored r=0.78-0.80 and slipped through. Hence three tests, not one.

Run:  pytest tests/test_sensor_verdict.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.fetch_weather import classify_sensor  # noqa: E402


def sensor(temp_correlation=0.9, station_spread_c=8.0, spread_ratio=1.0, mean_temp_difference=0.5):
    return SimpleNamespace(
        temp_correlation=temp_correlation,
        station_spread_c=station_spread_c,
        spread_ratio=spread_ratio,
        mean_temp_difference=mean_temp_difference,
    )


def test_a_healthy_sensor_passes():
    """Modelled on DL26: r=0.807, spread ratio 1.04."""
    assert classify_sensor(sensor(temp_correlation=0.807, spread_ratio=1.04)) == "ok"


def test_low_spread_is_not_judged():
    """KL07: r=0.115 but spread 2.53 C against reanalysis 2.93 C — that is agreement.

    Quarantining this would throw away a working sensor.
    """
    verdict = classify_sensor(sensor(temp_correlation=0.115, station_spread_c=2.53, spread_ratio=0.86))
    assert verdict == "not_assessable"


def test_a_damped_sensor_is_faulty():
    """BR08: varies only 61% as much as reality."""
    assert classify_sensor(sensor(temp_correlation=0.095, spread_ratio=0.61)) == "faulty"


def test_an_exaggerating_sensor_is_faulty():
    """KL20: varies 2.5x as much as reality."""
    assert classify_sensor(sensor(temp_correlation=0.16, station_spread_c=6.19, spread_ratio=2.50)) == "faulty"


def test_a_miscalibrated_sensor_is_caught_despite_good_correlation():
    """UP25, UP16 and DL13: r around 0.78-0.80 but over 3 C out.

    This is the case plain correlation missed entirely.
    """
    assert classify_sensor(sensor(temp_correlation=0.79, spread_ratio=1.0, mean_temp_difference=3.2)) == "faulty"


def test_a_negative_offset_is_caught_too():
    assert classify_sensor(sensor(temp_correlation=0.85, mean_temp_difference=-4.0)) == "faulty"


def test_low_correlation_with_adequate_spread_is_faulty():
    """TG05: spread ratio fine, offset fine, but it does not track reality."""
    assert classify_sensor(sensor(temp_correlation=0.201, station_spread_c=5.06, spread_ratio=1.10)) == "faulty"


@pytest.mark.parametrize("ratio", [0.69, 1.41])
def test_spread_ratio_bounds_are_enforced(ratio):
    assert classify_sensor(sensor(spread_ratio=ratio)) == "faulty"


@pytest.mark.parametrize("ratio", [0.71, 1.0, 1.39])
def test_spread_ratios_inside_the_bounds_pass(ratio):
    assert classify_sensor(sensor(spread_ratio=ratio)) == "ok"


def test_a_missing_spread_is_not_judged():
    assert classify_sensor(sensor(station_spread_c=float("nan"))) == "not_assessable"
