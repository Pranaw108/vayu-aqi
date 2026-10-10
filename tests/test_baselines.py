"""Tests for the baseline forecasters (MD-02).

The baselines decide whether the real model counts as working, so a broken or
leaking baseline is as damaging as a broken model — it moves the bar the model
is measured against.

Run:  pytest tests/test_baselines.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.baselines import (  # noqa: E402
    Persistence,
    PersistenceClimatologyBlend,
    SeasonalClimatology,
    all_baselines,
)


def make_frame(hours: int = 24 * 60, station: str = "X", start: str = "2024-01-01") -> pd.DataFrame:
    """Hourly rows with a daily cycle plus a trend, so averages are meaningful."""
    index = pd.date_range(start, periods=hours, freq="h")
    rng = np.random.default_rng(0)
    hour_of_day = index.hour.to_numpy()
    values = 80 + 25 * np.sin(2 * np.pi * hour_of_day / 24) + rng.normal(0, 5, hours)
    return pd.DataFrame({
        "station_code": [station] * hours,
        "hour": index,
        "baseline_persistence": values,
        "target": np.roll(values, -24),
    })


def test_persistence_returns_the_current_value():
    frame = make_frame(100)
    predicted = Persistence().fit(frame).predict(frame)
    assert np.allclose(predicted, frame["baseline_persistence"].to_numpy())


def test_climatology_uses_only_the_rows_it_was_fitted_on():
    """The leak that matters: an average over all data would contain the test period."""
    frame = make_frame(24 * 120)
    split = frame["hour"].quantile(0.6)
    train = frame[frame["hour"] < split]
    test = frame[frame["hour"] >= split]

    fitted_on_train = SeasonalClimatology().fit(train).predict(test)
    fitted_on_everything = SeasonalClimatology().fit(frame).predict(test)

    # If fitting on train alone gave the same answer as fitting on everything,
    # the "train-only" fit was not actually restricted.
    assert not np.allclose(fitted_on_train, fitted_on_everything), (
        "climatology fitted on train matches one fitted on all data — it is seeing the test period"
    )


def test_climatology_learns_the_daily_shape():
    """Predictions should vary across the day, not collapse to one number."""
    frame = make_frame(24 * 90)
    predicted = SeasonalClimatology().fit(frame).predict(frame)
    assert np.nanstd(predicted) > 1.0


def test_climatology_falls_back_for_an_unseen_station():
    """A station absent from training must still get a number, not a crash or a NaN."""
    train = make_frame(24 * 60, station="KNOWN")
    test = make_frame(48, station="BRAND_NEW")
    predicted = SeasonalClimatology().fit(train).predict(test)
    assert np.isfinite(predicted).all()


def test_climatology_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        SeasonalClimatology().predict(make_frame(10))


def test_blend_sits_between_its_two_parts():
    frame = make_frame(24 * 90)
    split = frame["hour"].quantile(0.7)
    train, test = frame[frame["hour"] < split], frame[frame["hour"] >= split]

    recent = Persistence().fit(train).predict(test)
    average = SeasonalClimatology().fit(train).predict(test)
    blended = PersistenceClimatologyBlend().fit(train).predict(test)

    low, high = np.minimum(recent, average), np.maximum(recent, average)
    usable = np.isfinite(recent) & np.isfinite(average)
    assert (blended[usable] >= low[usable] - 1e-6).all()
    assert (blended[usable] <= high[usable] + 1e-6).all()


def test_blend_handles_a_missing_recent_value():
    """With no current reading, the blend must fall back to the average alone."""
    train = make_frame(24 * 60)
    test = make_frame(24).copy()
    test.loc[test.index[:5], "baseline_persistence"] = np.nan

    model = PersistenceClimatologyBlend().fit(train)
    predicted = model.predict(test)
    assert np.isfinite(predicted[:5]).all()


def test_every_baseline_returns_one_prediction_per_row():
    frame = make_frame(24 * 60)
    split = frame["hour"].quantile(0.7)
    train, test = frame[frame["hour"] < split], frame[frame["hour"] >= split]
    for baseline in all_baselines():
        predicted = baseline.fit(train).predict(test)
        assert len(predicted) == len(test), f"{baseline.name} returned the wrong number of rows"
        assert predicted.dtype.kind == "f"


def test_baselines_are_distinguishable():
    """Three baselines that agree exactly would make the comparison pointless."""
    frame = make_frame(24 * 90)
    split = frame["hour"].quantile(0.7)
    train, test = frame[frame["hour"] < split], frame[frame["hour"] >= split]
    predictions = [b.fit(train).predict(test) for b in all_baselines()]
    assert not np.allclose(predictions[0], predictions[1])
    assert not np.allclose(predictions[0], predictions[2])
