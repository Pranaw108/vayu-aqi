"""Tests for the data-quality checks.

The near-constant rule is the one worth pinning down: measured by cardinality it
flagged every binary flag in the project as a problem, which is wrong.

Run:  pytest tests/test_data_quality.py -v
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from pipeline.data_quality import (  # noqa: E402
    NEAR_CONSTANT_DOMINANCE,
    OUTLIER_THRESHOLD,
    stage_presence,
)


def classify(series: pd.Series) -> str | None:
    """Mirror of the rule inside feature_health, for a single column."""
    distinct = int(series.nunique(dropna=True))
    present = float(series.notna().mean())
    dominance = float(series.value_counts(normalize=True, dropna=True).iloc[0]) if distinct else 1.0
    if distinct <= 1:
        return "constant"
    if dominance >= NEAR_CONSTANT_DOMINANCE:
        return "near-constant"
    if present < 0.5:
        return "mostly missing"
    return None


def test_a_single_valued_column_is_constant():
    assert classify(pd.Series([5.0] * 1000)) == "constant"


def test_a_balanced_binary_flag_is_fine():
    """is_weekend is 2/7 of rows and is_burning_season 2/12. Both are real features."""
    weekend = pd.Series([1] * 2000 + [0] * 5000)
    burning = pd.Series([1] * 2000 + [0] * 10000)
    assert classify(weekend) is None
    assert classify(burning) is None


def test_an_almost_always_zero_flag_is_near_constant():
    """A flag that fires once in 10,000 rows carries nothing useful."""
    rare = pd.Series([1] + [0] * 19999)
    assert classify(rare) == "near-constant"


def test_a_mostly_missing_column_is_flagged():
    # Needs more than one distinct value, or it is constant rather than missing.
    rng = np.random.default_rng(5)
    present = pd.Series(rng.normal(50, 10, 100))
    half_missing = pd.concat([present, pd.Series([np.nan] * 400)], ignore_index=True)
    assert classify(half_missing) == "mostly missing"


def test_a_normal_continuous_feature_passes():
    rng = np.random.default_rng(0)
    assert classify(pd.Series(rng.normal(80, 20, 5000))) is None


def test_stage_presence_reports_what_is_missing(tmp_path):
    stages = stage_presence(tmp_path)
    assert all(not s["present"] for s in stages)
    # Every row must say how to build the thing it reports missing.
    assert all(s["build_with"].startswith("pipeline/") for s in stages)


def test_outlier_threshold_is_strict_enough_to_be_rare():
    """3.5 robust deviations should not fire on ordinary spread."""
    rng = np.random.default_rng(1)
    values = pd.Series(rng.normal(50, 8, 2000))
    median = values.median()
    deviation = (values - median).abs().median()
    scores = 0.6745 * (values - median).abs() / deviation
    assert (scores >= OUTLIER_THRESHOLD).mean() < 0.01
