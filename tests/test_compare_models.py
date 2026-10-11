"""Tests for the shared comparison path.

This script exists to stop mis-comparisons, so the tests are mostly about it
refusing to compare things that are not comparable.

Run:  pytest tests/test_compare_models.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.compare_models import EPISODE_THRESHOLD, Scored, accuracy, detection  # noqa: E402


def make_scored(tag="t", prediction=None, actual=None, current=None) -> Scored:
    return Scored(tag=tag, features=10, trees=50,
                  prediction=np.asarray(prediction, dtype=float),
                  actual=np.asarray(actual, dtype=float),
                  current=np.asarray(current, dtype=float))


def test_a_perfect_model_scores_zero_error_and_full_skill():
    truth = np.array([50.0, 150.0, 200.0])
    s = make_scored(prediction=truth, actual=truth, current=np.array([60.0, 140.0, 180.0]))
    result = accuracy(s)
    assert result["mae"] == 0.0
    assert result["skill"] == 100.0


def test_a_model_matching_persistence_scores_zero_skill():
    current = np.array([50.0, 150.0, 200.0])
    actual = np.array([60.0, 140.0, 190.0])
    s = make_scored(prediction=current, actual=actual, current=current)
    assert accuracy(s)["skill"] == pytest.approx(0.0)


def test_rows_with_missing_values_are_excluded_from_both_sides():
    s = make_scored(prediction=np.array([10.0, np.nan, 30.0]),
                    actual=np.array([12.0, 20.0, 33.0]),
                    current=np.array([11.0, 20.0, 31.0]))
    assert accuracy(s)["rows"] == 2


def test_a_perfect_persistence_window_reports_no_skill_rather_than_raising():
    """Dividing by persistence's error raised on a window where it was exact."""
    truth = np.array([10.0, 30.0])
    s = make_scored(prediction=truth, actual=truth, current=truth)
    assert np.isnan(accuracy(s)["skill"])


def test_detection_only_counts_hours_that_could_contain_an_onset():
    """An hour already above the threshold cannot be an onset."""
    s = make_scored(
        prediction=np.array([130.0, 200.0, 90.0, 140.0]),
        actual=np.array([130.0, 210.0, 95.0, 150.0]),
        # Two hours are already elevated, so only two can hold an onset.
        current=np.array([50.0, 180.0, 60.0, 70.0]),
    )
    result = detection(s)
    assert result["calm_hours"] == 3
    assert result["onsets"] == 2


def test_a_model_that_ranks_onsets_first_gets_high_average_precision():
    rng = np.random.default_rng(0)
    current = np.full(400, 60.0)
    actual = np.where(rng.random(400) < 0.25, 200.0, 70.0)
    # Prediction tracks the truth, so ranking is near-perfect.
    prediction = actual + rng.normal(0, 3, 400)
    result = detection(make_scored(prediction=prediction, actual=actual, current=current))
    assert result["average_precision"] > 0.95
    assert result["roc_auc"] > 0.95


def test_a_model_that_ranks_randomly_gets_baseline_average_precision():
    rng = np.random.default_rng(1)
    current = np.full(4000, 60.0)
    actual = np.where(rng.random(4000) < 0.25, 200.0, 70.0)
    prediction = rng.normal(100, 30, 4000)   # unrelated to the outcome
    result = detection(make_scored(prediction=prediction, actual=actual, current=current))
    # With a 25% positive rate, random ranking lands near 0.25.
    assert 0.18 < result["average_precision"] < 0.33
    assert 0.45 < result["roc_auc"] < 0.55


def test_the_curve_spans_the_whole_sweep_and_recall_falls_as_the_threshold_rises():
    rng = np.random.default_rng(2)
    current = np.full(2000, 60.0)
    actual = np.where(rng.random(2000) < 0.3, 200.0, 70.0)
    prediction = actual + rng.normal(0, 20, 2000)
    curve = detection(make_scored(prediction=prediction, actual=actual, current=current))["curve"]
    recalls = [point["recall"] for point in curve]
    assert recalls == sorted(recalls, reverse=True), "recall must not rise with the threshold"
    assert curve[0]["threshold"] < EPISODE_THRESHOLD < curve[-1]["threshold"]
