"""Tests for the partial-correlation diagnostic.

The whole point of this tool is to tell a real signal from a seasonal proxy, so
the tests build both cases from scratch and check it separates them.

Run:  pytest tests/test_diagnose_fires.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.diagnose_fires import partial_correlation  # noqa: E402


def test_a_pure_seasonal_proxy_is_exposed():
    """A feature that is only a function of the season must come out near zero.

    This is the October case: fires and PM2.5 both rise through the month, so a
    raw correlation looks strong while the feature adds nothing the calendar
    does not already say.
    """
    rng = np.random.default_rng(0)
    day = np.arange(1, 1200) % 365
    season = np.sin(2 * np.pi * day / 365.25)
    frame = pd.DataFrame({
        "doy_sin": season,
        "doy_cos": np.cos(2 * np.pi * day / 365.25),
        # Both are the season plus independent noise, with no link to each other.
        "fire": season * 100 + rng.normal(0, 5, day.size),
        "pm25": season * 80 + rng.normal(0, 5, day.size),
    })
    raw = frame["fire"].corr(frame["pm25"])
    partial = partial_correlation(frame, "fire", "pm25", ["doy_sin", "doy_cos"])

    assert raw > 0.9, "the proxy should look strong before controlling"
    assert abs(partial) < 0.2, f"the proxy should collapse once controlled, got {partial:+.3f}"


def test_a_genuine_signal_survives_the_controls():
    """A feature that really drives the outcome must keep its correlation."""
    rng = np.random.default_rng(1)
    day = np.arange(1, 1200) % 365
    season = np.sin(2 * np.pi * day / 365.25)
    fire = rng.normal(0, 20, day.size)          # independent of the season
    frame = pd.DataFrame({
        "doy_sin": season,
        "doy_cos": np.cos(2 * np.pi * day / 365.25),
        "fire": fire,
        "pm25": season * 80 + fire * 2.0 + rng.normal(0, 5, day.size),
    })
    partial = partial_correlation(frame, "fire", "pm25", ["doy_sin", "doy_cos"])
    assert partial > 0.8, f"a real driver should survive, got {partial:+.3f}"


def test_controlling_for_the_outcome_lag_reduces_an_autocorrelated_signal():
    """Yesterday's value explains a lot of today's, so it belongs in the controls."""
    rng = np.random.default_rng(2)
    n = 2000
    pm = np.zeros(n)
    for i in range(1, n):
        pm[i] = 0.8 * pm[i - 1] + rng.normal(0, 10)
    frame = pd.DataFrame({
        "doy_sin": np.zeros(n), "doy_cos": np.zeros(n),
        "pm25": pm,
        "pm25_yesterday": np.r_[0, pm[:-1]],
        # Tracks yesterday's level rather than adding anything new.
        "fire": np.r_[0, pm[:-1]] + rng.normal(0, 3, n),
    })
    raw = abs(frame["fire"].corr(frame["pm25"]))
    partial = abs(partial_correlation(frame, "fire", "pm25", ["pm25_yesterday"]))
    assert raw > 0.6
    assert partial < raw / 2, "a feature echoing the lag should shrink once the lag is controlled"


def test_partial_correlation_of_a_constant_control_changes_nothing():
    rng = np.random.default_rng(3)
    frame = pd.DataFrame({
        "ones": np.ones(500),
        "a": rng.normal(0, 1, 500),
    })
    frame["b"] = frame["a"] * 2 + rng.normal(0, 0.1, 500)
    raw = frame["a"].corr(frame["b"])
    partial = partial_correlation(frame, "a", "b", ["ones"])
    assert partial == pytest.approx(raw, abs=1e-6)
