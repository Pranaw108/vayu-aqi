"""Tests for the backtest harness (MD-04).

The fold boundaries are the whole evaluation. If a fold trains on data that
comes after its test period, every score in the report is inflated and nothing
raises an error — so the boundaries are tested directly.

Run:  pytest tests/test_train.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.train import (  # noqa: E402
    MIN_TRAIN_DAYS,
    TEST_DAYS,
    make_folds,
    score,
    season_of,
    skill,
)


def hourly_frame(days: int, start: str = "2018-01-01") -> pd.DataFrame:
    index = pd.date_range(start, periods=days * 24, freq="h")
    return pd.DataFrame({"hour": index, "target": np.arange(len(index), dtype=float)})


# --------------------------------------------------------------- fold boundaries

def test_folds_never_train_on_data_after_their_test_period():
    """The defining property of rolling-origin validation."""
    frame = hourly_frame(days=1200)
    for train_end, test_start, test_end in make_folds(frame, folds=8):
        assert train_end <= test_start, "training window reaches past the start of its test period"
        assert test_start < test_end


def test_folds_do_not_overlap_each_other():
    frame = hourly_frame(days=1200)
    folds = make_folds(frame, folds=8)
    for earlier, later in zip(folds, folds[1:]):
        assert earlier[2] <= later[1], "two folds test the same hours"


def test_folds_run_forwards_in_time():
    frame = hourly_frame(days=1200)
    starts = [f[1] for f in make_folds(frame, folds=6)]
    assert starts == sorted(starts)


def test_each_fold_tests_the_expected_span():
    frame = hourly_frame(days=1200)
    for _, test_start, test_end in make_folds(frame, folds=5):
        assert (test_end - test_start) == pd.Timedelta(days=TEST_DAYS)


def test_every_fold_keeps_the_minimum_training_history():
    frame = hourly_frame(days=1200)
    first_hour = frame["hour"].min()
    for train_end, _, _ in make_folds(frame, folds=10):
        assert (train_end - first_hour) >= pd.Timedelta(days=MIN_TRAIN_DAYS)


def test_too_little_history_produces_no_folds():
    """Better to produce nothing than to evaluate on a few weeks and call it validated."""
    assert make_folds(hourly_frame(days=30), folds=5) == []


def test_fold_count_is_capped_by_available_history():
    frame = hourly_frame(days=500)
    folds = make_folds(frame, folds=50)
    assert 0 < len(folds) < 50


# --------------------------------------------------------------- scoring

def test_score_ignores_missing_values():
    actual = np.array([10.0, np.nan, 30.0])
    predicted = np.array([12.0, 5.0, 33.0])
    result = score(actual, predicted)
    assert result["n"] == 2
    assert result["mae"] == 2.5


def test_score_of_a_perfect_forecast_is_zero():
    actual = np.array([10.0, 20.0, 30.0])
    assert score(actual, actual.copy())["mae"] == 0.0


def test_score_with_nothing_comparable_is_not_a_crash():
    result = score(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert result["n"] == 0
    assert np.isnan(result["mae"])


def test_skill_is_positive_only_when_the_model_is_better():
    assert skill(model_mae=20, baseline_mae=40) == 50.0
    assert skill(model_mae=40, baseline_mae=40) == 0.0
    # Worse than doing nothing must read as negative, not be hidden.
    assert skill(model_mae=60, baseline_mae=40) < 0


def test_skill_of_a_zero_baseline_is_not_infinite():
    assert np.isnan(skill(model_mae=5, baseline_mae=0))


# --------------------------------------------------------------- seasons

def test_indian_seasons_are_mapped_not_calendar_quarters():
    assert season_of(1) == "winter"
    assert season_of(12) == "winter"
    assert season_of(4) == "summer"
    assert season_of(7) == "monsoon"
    assert season_of(10) == "post-monsoon"
    assert season_of(11) == "post-monsoon"


def test_every_month_has_a_season():
    assert {season_of(m) for m in range(1, 13)} == {"winter", "summer", "monsoon", "post-monsoon"}
