"""Tests for the serving-time deadband.

The point of these is the onset case. A deadband that quietly suppressed onset
predictions would look like an improvement on aggregate error while destroying
the only thing the model does better than persistence.

Run:  pytest tests/test_forecast.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from api.forecast import DEFAULT_DEADBAND, apply_deadband, deadband_share  # noqa: E402


def test_a_small_predicted_change_falls_back_to_persistence():
    """On calm hours the model is 5.2% worse, so it should not be used."""
    out = apply_deadband(np.array([52.0]), np.array([50.0]), threshold=5.0)
    assert out[0] == 50.0


def test_a_large_predicted_change_keeps_the_model():
    out = apply_deadband(np.array([160.0]), np.array([50.0]), threshold=5.0)
    assert out[0] == 160.0


def test_an_onset_prediction_is_never_suppressed():
    """The failure mode this design exists to avoid.

    Routing on *current* conditions would send every calm hour to persistence —
    and an onset starts from a calm hour, so the alarm could never fire. Routing
    on the predicted change keeps it.
    """
    calm_now = np.array([40.0, 45.0, 38.0])
    model_sees_onset = np.array([150.0, 160.0, 175.0])
    out = apply_deadband(model_sees_onset, calm_now, threshold=5.0)
    assert np.array_equal(out, model_sees_onset), "an onset call was replaced by persistence"


def test_a_predicted_drop_also_keeps_the_model():
    """Direction must not matter — clearing air is as useful to forecast."""
    out = apply_deadband(np.array([60.0]), np.array([200.0]), threshold=5.0)
    assert out[0] == 60.0


def test_exactly_at_the_threshold_keeps_the_model():
    """Defined boundary: strictly below the threshold falls back."""
    out = apply_deadband(np.array([55.0]), np.array([50.0]), threshold=5.0)
    assert out[0] == 55.0


def test_a_missing_current_value_keeps_the_model():
    """With no reading now there is no persistence forecast to fall back to."""
    out = apply_deadband(np.array([80.0]), np.array([np.nan]), threshold=5.0)
    assert out[0] == 80.0


def test_a_zero_threshold_always_keeps_the_model():
    model = np.array([51.0, 90.0, 20.0])
    current = np.array([50.0, 50.0, 50.0])
    assert np.array_equal(apply_deadband(model, current, threshold=0.0), model)


def test_a_huge_threshold_always_falls_back():
    model = np.array([51.0, 90.0, 20.0])
    current = np.array([50.0, 50.0, 50.0])
    assert np.array_equal(apply_deadband(model, current, threshold=1e6), current)


def test_mixed_rows_are_routed_independently():
    model = np.array([52.0, 160.0, 48.0, 10.0])
    current = np.array([50.0, 50.0, 50.0, 50.0])
    out = apply_deadband(model, current, threshold=5.0)
    assert np.array_equal(out, np.array([50.0, 160.0, 50.0, 10.0]))


def test_share_reports_how_often_the_deadband_engages():
    model = np.array([52.0, 160.0, 48.0, 10.0])
    current = np.array([50.0, 50.0, 50.0, 50.0])
    assert deadband_share(model, current, threshold=5.0) == pytest.approx(0.5)


def test_share_is_zero_when_nothing_is_comparable():
    assert deadband_share(np.array([np.nan]), np.array([np.nan])) == 0.0


def test_the_default_threshold_is_the_one_that_was_specified():
    assert DEFAULT_DEADBAND == 5.0
