"""Baseline forecasters (MD-02).

These exist so that "the model works" is a measurable claim. Air quality is
strongly autocorrelated, so predicting *whatever it is right now* is already a
good forecast — in Delhi winter it gets within roughly 25-30 ug/m3 at 24 hours.
A gradient-boosted model that scores 24 has learned nothing.

Every baseline is fitted on training rows only and then asked about unseen test
rows, exactly like the real model. Climatology is where this matters: an average
computed over the whole dataset would contain the test period, which is a quiet
way to leak the future into a "simple" baseline and make the real model look bad
by comparison.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd

# Weight on persistence in the blend. Air quality is autocorrelated enough that
# the recent value deserves more weight than the long-run average.
PERSISTENCE_WEIGHT = 0.7


class Baseline(ABC):
    """Same fit/predict shape as the real model, so scoring code is shared."""

    name: str

    @abstractmethod
    def fit(self, train: pd.DataFrame, target_column: str = "target") -> "Baseline":
        ...

    @abstractmethod
    def predict(self, test: pd.DataFrame) -> np.ndarray:
        ...


class Persistence(Baseline):
    """Tomorrow is the same as now.

    `baseline_persistence` is the pollutant's current value, carried through the
    feature pipeline for exactly this purpose.
    """

    name = "persistence"

    def fit(self, train: pd.DataFrame, target_column: str = "target") -> "Persistence":
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        return test["baseline_persistence"].to_numpy(dtype=float)


class SeasonalClimatology(Baseline):
    """The long-run average for this station, this month and this hour.

    Captures the two cycles that dominate Indian air quality: the winter peak
    and the daily traffic and inversion cycle.
    """

    name = "climatology"

    def __init__(self):
        self.by_station_month_hour: pd.Series | None = None
        self.by_month_hour: pd.Series | None = None
        self.global_mean: float = np.nan

    @staticmethod
    def _keys(frame: pd.DataFrame) -> pd.DataFrame:
        hours = pd.to_datetime(frame["hour"])
        return pd.DataFrame({
            "station_code": frame["station_code"].to_numpy(),
            "month": hours.dt.month.to_numpy(),
            "hour_of_day": hours.dt.hour.to_numpy(),
        })

    def fit(self, train: pd.DataFrame, target_column: str = "target") -> "SeasonalClimatology":
        keys = self._keys(train)
        keys[target_column] = train[target_column].to_numpy()

        self.by_station_month_hour = keys.groupby(["station_code", "month", "hour_of_day"])[target_column].mean()
        # Fallbacks, so a station or hour unseen in training still gets an answer.
        self.by_month_hour = keys.groupby(["month", "hour_of_day"])[target_column].mean()
        self.global_mean = float(keys[target_column].mean())
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        if self.by_station_month_hour is None:
            raise RuntimeError("fit() must be called before predict()")

        keys = self._keys(test)
        index = pd.MultiIndex.from_arrays([keys["station_code"], keys["month"], keys["hour_of_day"]])
        predicted = self.by_station_month_hour.reindex(index).to_numpy(dtype=float)

        coarse = self.by_month_hour.reindex(
            pd.MultiIndex.from_arrays([keys["month"], keys["hour_of_day"]])
        ).to_numpy(dtype=float)
        predicted = np.where(np.isnan(predicted), coarse, predicted)
        return np.where(np.isnan(predicted), self.global_mean, predicted)


class PersistenceClimatologyBlend(Baseline):
    """Weighted average of the two. Often beats naive machine learning."""

    name = "persistence+climatology"

    def __init__(self, persistence_weight: float = PERSISTENCE_WEIGHT):
        self.persistence_weight = persistence_weight
        self.persistence = Persistence()
        self.climatology = SeasonalClimatology()

    def fit(self, train: pd.DataFrame, target_column: str = "target") -> "PersistenceClimatologyBlend":
        self.persistence.fit(train, target_column)
        self.climatology.fit(train, target_column)
        return self

    def predict(self, test: pd.DataFrame) -> np.ndarray:
        recent = self.persistence.predict(test)
        average = self.climatology.predict(test)
        blended = self.persistence_weight * recent + (1 - self.persistence_weight) * average
        # Where there is no recent value, fall back to the average alone.
        return np.where(np.isnan(recent), average, blended)


def all_baselines() -> list[Baseline]:
    """The three every model is scored against."""
    return [Persistence(), SeasonalClimatology(), PersistenceClimatologyBlend()]
