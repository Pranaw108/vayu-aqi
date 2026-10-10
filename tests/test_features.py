"""Tests for the feature pipeline (MD-01), mostly about leakage.

Leakage is the failure that matters here. A leaking feature produces a model
that scores beautifully and forecasts nothing, and it does not raise an error —
so it has to be tested for directly rather than reasoned about.

Run:  pytest tests/test_features.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.features import (  # noqa: E402
    INPUT_POLLUTANTS,
    LAG_HOURS,
    add_history_features,
    add_target,
    build,
    leakage_check,
    drop_quarantined_sensors,
    load_quarantine,
    warn_on_thin_weather,
    weather_coverage,
)

DATA = REPO_ROOT / "data"
HORIZON = 24

needs_data = pytest.mark.skipif(
    not (DATA / "hourly").is_dir(),
    reason="No hourly data — run pipeline/build_hourly.py first",
)


@pytest.fixture(scope="module")
def features():
    """A small real feature table: enough stations to be meaningful, few enough to be quick."""
    registry = pd.read_parquet(DATA / "registry.parquet")
    codes = registry.loc[registry["is_mappable"], "station_code"].tolist()[:4]
    con = duckdb.connect()
    frame, dropped = build(con, DATA, "pm2_5", HORIZON, codes)
    return frame, dropped


# ----------------------------------------------------------------- leakage

@needs_data
def test_the_target_column_itself_is_dropped(features):
    """The model must never see the pollutant it is predicting at the current hour."""
    frame, dropped = features
    assert "pm2_5" in dropped
    assert "pm2_5" not in frame.columns


@needs_data
def test_no_bare_current_hour_measurement_survives(features):
    frame, _ = features
    for pollutant in INPUT_POLLUTANTS:
        assert pollutant not in frame.columns, f"{pollutant} is a current-hour value and must be dropped"
    for weather in ("at", "rh", "ws", "wd", "bp"):
        assert weather not in frame.columns


@needs_data
def test_no_feature_almost_perfectly_predicts_the_target(features):
    """A feature correlating ~1.0 with a value 24 hours away is leaking.

    This is the check that would have caught predicting AQI from the same hour's
    PM, which is the classic way these projects fool themselves.
    """
    frame, _ = features
    ignore = {"target", "target_hour", "hour", "station_code", "baseline_persistence"}
    numeric = frame.drop(columns=[c for c in ignore if c in frame.columns]).select_dtypes("number")

    correlations = numeric.corrwith(frame["target"]).abs().dropna()
    worst = correlations.sort_values(ascending=False).head(5)
    assert (correlations < 0.99).all(), f"suspiciously predictive features:\n{worst}"


@needs_data
def test_persistence_is_strong_but_not_perfect(features):
    """Sanity check on the baseline: correlated, clearly not identical."""
    frame, _ = features
    r = frame["baseline_persistence"].corr(frame["target"])
    assert 0.3 < r < 0.99, f"persistence correlation {r:.3f} looks wrong for a {HORIZON}h horizon"


@needs_data
def test_longer_lags_are_less_informative(features):
    """Information should fade with distance. If it does not, something is wrong."""
    frame, _ = features
    r_short = abs(frame["pm2_5_lag1"].corr(frame["target"]))
    r_long = abs(frame["pm2_5_lag168"].corr(frame["target"]))
    assert r_short > r_long, f"lag1 ({r_short:.3f}) should beat lag168 ({r_long:.3f})"


# ----------------------------------------------------------------- correctness

def test_lags_are_exactly_the_shifted_values():
    """Built on a tiny hand-made frame, so the expected values are obvious."""
    hours = pd.date_range("2024-01-01", periods=12, freq="h")
    frame = pd.DataFrame({
        "station_code": ["X"] * 12,
        "hour": hours,
        "pm2_5": np.arange(12, dtype=float),
    })
    out = add_history_features(frame.copy())
    # At position 5, lag1 is position 4's value.
    assert out["pm2_5_lag1"].iloc[5] == 4.0
    assert out["pm2_5_lag3"].iloc[5] == 2.0
    # The first rows have no history.
    assert pd.isna(out["pm2_5_lag1"].iloc[0])


def test_rolling_mean_excludes_the_current_hour():
    """The decisive check: a rolling window must not contain the hour it describes."""
    hours = pd.date_range("2024-01-01", periods=20, freq="h")
    frame = pd.DataFrame({
        "station_code": ["X"] * 20,
        "hour": hours,
        "pm2_5": np.arange(20, dtype=float),
    })
    out = add_history_features(frame.copy())
    # mean6 at position 10 covers positions 4..9 (values 4..9), never position 10.
    expected = np.mean([4, 5, 6, 7, 8, 9])
    assert out["pm2_5_mean6"].iloc[10] == pytest.approx(expected)
    assert out["pm2_5_mean6"].iloc[10] != pytest.approx(np.mean(range(5, 11)))


def test_lags_do_not_bleed_between_stations():
    """Two stations in one frame must not borrow each other's history."""
    hours = pd.date_range("2024-01-01", periods=6, freq="h")
    frame = pd.DataFrame({
        "station_code": ["A"] * 6 + ["B"] * 6,
        "hour": list(hours) * 2,
        "pm2_5": [1.0] * 6 + [100.0] * 6,
    })
    out = add_history_features(frame.copy())
    # B's first row has no history of its own, so it must be null, not A's value.
    assert pd.isna(out[out["station_code"] == "B"]["pm2_5_lag1"].iloc[0])
    assert out[out["station_code"] == "B"]["pm2_5_lag1"].iloc[1] == 100.0


def test_target_is_the_future_value():
    hours = pd.date_range("2024-01-01", periods=30, freq="h")
    frame = pd.DataFrame({
        "station_code": ["X"] * 30,
        "hour": hours,
        "pm2_5": np.arange(30, dtype=float),
    })
    out = add_target(frame.copy(), "pm2_5", horizon=24)
    assert out["target"].iloc[0] == 24.0
    assert out["target_hour"].iloc[0] == hours[24]
    # Nothing is known 24 hours past the end.
    assert pd.isna(out["target"].iloc[-1])


def test_leakage_check_flags_a_raw_pollutant_column():
    """Prove the guard can fire — a guard that never fires protects nothing."""
    frame = pd.DataFrame({
        "station_code": ["X"], "hour": [pd.Timestamp("2024-01-01")],
        "pm2_5": [10.0],          # current hour, must be flagged
        "pm2_5_lag1": [9.0],      # legitimate
        "pm2_5_mean24": [8.0],    # legitimate
        "hour_sin": [0.0],        # legitimate
        "fc_temp_c": [20.0],      # forecast for the predicted hour, legitimate
        "wx_temp_c": [19.0],      # reanalysis at the current hour, legitimate
        "target": [11.0],
    })
    suspect = leakage_check(frame, "pm2_5", 24)
    assert "pm2_5" in suspect
    assert "pm2_5_lag1" not in suspect
    assert "pm2_5_mean24" not in suspect
    assert "hour_sin" not in suspect
    assert "fc_temp_c" not in suspect
    assert "wx_temp_c" not in suspect


def test_every_lag_horizon_is_built():
    hours = pd.date_range("2024-01-01", periods=200, freq="h")
    frame = pd.DataFrame({
        "station_code": ["X"] * 200,
        "hour": hours,
        "pm2_5": np.random.default_rng(0).normal(80, 20, 200),
    })
    out = add_history_features(frame.copy())
    for lag in LAG_HOURS:
        assert f"pm2_5_lag{lag}" in out.columns


# ----------------------------------------------------------------- weather guard

def test_weather_coverage_is_zero_when_the_columns_are_absent():
    frame = pd.DataFrame({"station_code": ["X"], "hour": [pd.Timestamp("2024-01-01")]})
    assert weather_coverage(frame) == 0.0


def test_weather_coverage_counts_rows_with_any_weather():
    frame = pd.DataFrame({
        "fc_temp_c": [20.0, np.nan, np.nan, 22.0],
        "fc_wind_speed_ms": [np.nan, np.nan, np.nan, 3.0],
        "fc_boundary_layer_m": [np.nan] * 4,
    })
    assert weather_coverage(frame) == 0.5


def test_thin_weather_is_warned_about(caplog):
    """A near-empty weather join must be loud, not silent.

    This is the bug it catches: a 1.7%-weather run trained and scored fine while
    being autoregressive only, and nothing said so.
    """
    frame = pd.DataFrame({"fc_temp_c": [20.0] + [np.nan] * 99})
    with caplog.at_level("WARNING"):
        coverage = warn_on_thin_weather(frame)
    assert coverage == pytest.approx(0.01)
    assert any("autoregressive" in record.message for record in caplog.records)


def test_good_weather_coverage_does_not_warn(caplog):
    frame = pd.DataFrame({"fc_temp_c": [20.0] * 90 + [np.nan] * 10})
    with caplog.at_level("WARNING"):
        warn_on_thin_weather(frame)
    assert not [r for r in caplog.records if r.levelname == "WARNING"]


# ----------------------------------------------------------------- sensor quarantine

def test_quarantined_station_loses_its_local_weather():
    """A proven-bad sensor must not reach the model at all."""
    frame = pd.DataFrame({
        "station_code": ["UP09", "UP09", "DL26"],
        "hour": pd.date_range("2024-01-01", periods=3, freq="h"),
        "at": [60.0, 59.0, 21.0],
        "rh": [10.0, 11.0, 55.0],
        "ws": [1.0, 1.1, 2.0],
        "wd": [90.0, 95.0, 180.0],
        "pm2_5": [100.0, 101.0, 102.0],
    })
    out = drop_quarantined_sensors(
        frame.copy(), {"UP09": {"quarantine_columns": ["at", "rh"], "temp_correlation": 0.41}}
    )

    bad = out[out["station_code"] == "UP09"]
    assert bad[["at", "rh"]].isna().all().all()
    # Wind is not dropped: there is no independent check on it, so dropping it
    # would discard real data on suspicion rather than on evidence.
    assert bad["ws"].notna().all()
    # The pollutant readings are still good, so the rows stay.
    assert bad["pm2_5"].notna().all()
    # A healthy station is untouched.
    good = out[out["station_code"] == "DL26"]
    assert good["at"].iloc[0] == 21.0


def test_no_quarantine_file_changes_nothing(tmp_path):
    assert load_quarantine(tmp_path) == {}
    frame = pd.DataFrame({"station_code": ["X"], "at": [25.0]})
    assert drop_quarantined_sensors(frame.copy(), {})["at"].iloc[0] == 25.0


def test_quarantine_file_is_read(tmp_path):
    (tmp_path / "sensor_quarantine.json").write_text(
        '{"stations": {"BR08": {"temp_correlation": 0.095}}}', encoding="utf-8"
    )
    assert "BR08" in load_quarantine(tmp_path)


def test_forecast_weather_is_distinguished_from_current_weather():
    """fc_* and wx_* must both survive the leakage guard, and mean different things.

    Joining weather only at the current hour was a real bug: the model already
    has today's conditions from its own lagged sensors, so complete weather
    changed the score by 0.0%. The value is in the hour being predicted.
    """
    frame = pd.DataFrame({
        "station_code": ["X"], "hour": [pd.Timestamp("2024-01-01")],
        "fc_wind_speed_ms": [5.0],      # forecast for the target hour
        "wx_wind_speed_ms": [1.0],      # reanalysis now
        "target": [50.0],
    })
    suspect = leakage_check(frame, "pm2_5", 24)
    assert suspect == []


def test_coverage_is_measured_on_the_forecast_columns():
    """Current-hour weather being present must not mask a missing forecast."""
    frame = pd.DataFrame({
        "wx_temp_c": [20.0] * 10,       # present
        "fc_temp_c": [np.nan] * 10,     # absent — this is what matters
    })
    assert weather_coverage(frame) == 0.0


def test_only_the_proven_bad_column_is_dropped():
    """A sound thermometer with a failed hygrometer must keep its temperature."""
    frame = pd.DataFrame({
        "station_code": ["WB09", "WB09"],
        "hour": pd.date_range("2024-01-01", periods=2, freq="h"),
        "at": [21.0, 21.5],
        "rh": [100.0, 100.0],
        "ws": [2.0, 2.1],
    })
    out = drop_quarantined_sensors(frame.copy(), {"WB09": {"quarantine_columns": ["rh"]}})
    assert out["rh"].isna().all()
    assert out["at"].notna().all()
    assert out["ws"].notna().all()
