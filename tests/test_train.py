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


# ------------------------------------------------- selection bias (a real mistake)

def test_an_unbiased_forecast_looks_biased_when_rows_are_chosen_by_outcome():
    """Encodes the error this project actually made.

    The mechanism matters. A forecast is made first and the outcome then varies
    around it, so `outcome = forecast + error`. Selecting rows where the OUTCOME
    came out high therefore selects rows where the error happened to be positive,
    and the forecast looks too low. Selecting low outcomes does the reverse.

    An earlier report read a 44 ug/m3 "severity gap" on onsets and a "+21.5 calm
    bias" off exactly these splits, and concluded the model compressed toward the
    mean. Conditioned on the present instead, its bias is under 2 ug/m3 in every
    band.
    """
    rng = np.random.default_rng(0)
    # The forecast is the primary quantity; the outcome deviates from it.
    forecast = rng.lognormal(mean=4.2, sigma=0.7, size=80_000)
    outcome = forecast + rng.normal(0, 40, size=forecast.size)

    overall_bias = float(np.mean(forecast - outcome))
    assert abs(overall_bias) < 1.0, "this forecast is unbiased by construction"

    high_outcome = outcome >= 120
    low_outcome = outcome < 120
    bias_on_high = float(np.mean(forecast[high_outcome] - outcome[high_outcome]))
    bias_on_low = float(np.mean(forecast[low_outcome] - outcome[low_outcome]))

    assert bias_on_high < -5.0, "selecting high outcomes must look like under-prediction"
    assert bias_on_low > 5.0, "selecting low outcomes must look like over-prediction"


def test_conditioning_on_the_present_leaves_an_unbiased_predictor_unbiased():
    """The valid split: condition on what is known when the forecast is made."""
    rng = np.random.default_rng(1)
    current = rng.lognormal(mean=4.2, sigma=0.7, size=60_000)
    truth = current * rng.lognormal(mean=0.0, sigma=0.4, size=current.size)
    prediction = truth + rng.normal(0, 20, size=truth.size)

    for lo, hi in [(0, 50), (50, 120), (120, 400)]:
        band = (current >= lo) & (current < hi)
        if band.sum() < 100:
            continue
        bias = float(np.mean(prediction[band] - truth[band]))
        assert abs(bias) < 2.0, f"band {lo}-{hi} should stay unbiased, got {bias:+.2f}"
