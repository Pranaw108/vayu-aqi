"""Tests for the humidity-sensor verdict.

Relative humidity swings 20-90% in a normal day, so the absolute-offset test
that works for temperature is meaningless here. These check the physics tests
that replace it, using the real measurements that motivated them.

Run:  pytest tests/test_humidity_verdict.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.fetch_weather import classify_humidity_sensor  # noqa: E402


def sensor(humidity_spread_pct=18.0, humidity_spread_ratio=1.0,
           stuck_saturated_share=0.0, pinned_zero_share=0.0):
    return SimpleNamespace(
        humidity_spread_pct=humidity_spread_pct,
        humidity_spread_ratio=humidity_spread_ratio,
        stuck_saturated_share=stuck_saturated_share,
        pinned_zero_share=pinned_zero_share,
    )


def test_a_healthy_hygrometer_passes():
    assert classify_humidity_sensor(sensor()) == "ok"


def test_saturated_on_dry_hours_is_faulty():
    """WB09 reads 100% humidity on 32.7% of hours when no rain fell.

    Correlation cannot see this: the sensor still follows the daily cycle.
    """
    assert classify_humidity_sensor(sensor(stuck_saturated_share=0.3265)) == "faulty"


def test_a_small_amount_of_saturation_is_normal():
    """Genuine fog and dew exist, so the test must not fire on every station."""
    assert classify_humidity_sensor(sensor(stuck_saturated_share=0.01)) == "ok"


def test_a_true_zero_reading_is_faulty():
    """Outdoor air is never at 0% relative humidity."""
    assert classify_humidity_sensor(sensor(pinned_zero_share=0.02)) == "faulty"


def test_an_exaggerating_hygrometer_is_faulty():
    """KL20 swings 1.95x as much as reanalysis."""
    assert classify_humidity_sensor(sensor(humidity_spread_ratio=1.95)) == "faulty"


def test_a_damped_hygrometer_is_faulty():
    assert classify_humidity_sensor(sensor(humidity_spread_ratio=0.4)) == "faulty"


def test_low_spread_is_not_judged():
    """Nothing to assess if the sensor barely moves at all."""
    assert classify_humidity_sensor(sensor(humidity_spread_pct=3.0)) == "not_assessable"


def test_no_absolute_offset_test_is_applied():
    """The point of this verdict: a 10-point mean offset must not alone condemn
    a hygrometer, because RH's natural range makes that threshold arbitrary."""
    # Nothing in the signature even accepts a mean offset; a sensor that is
    # otherwise healthy passes regardless of how far its mean sits from ERA5.
    assert classify_humidity_sensor(sensor(humidity_spread_ratio=1.05)) == "ok"


@pytest.mark.parametrize("ratio", [0.61, 1.0, 1.59])
def test_ratios_inside_the_wider_humidity_bounds_pass(ratio):
    """Humidity bounds are wider than temperature's, because RH is noisier."""
    assert classify_humidity_sensor(sensor(humidity_spread_ratio=ratio)) == "ok"
