"""CPCB National Air Quality Index — the official Indian AQI formula (AQ-01).

AQI is a formula, not a model. Each pollutant is converted to a 0-500 "sub-index"
by looking up which band its concentration falls in and interpolating inside that
band. The station's AQI is simply the worst sub-index, and the pollutant that
produced it is the "dominant pollutant".

Two rules from CPCB that are easy to miss and change the answer:

  1. Each pollutant uses its own averaging window. PM2.5, PM10, NO2, SO2 and NH3
     use a 24-hour mean. CO and O3 use the highest 8-hour rolling mean of the day,
     because those two are short-exposure hazards.
  2. An AQI needs at least THREE pollutants, and one of them must be PM2.5 or PM10.
     Below that, CPCB does not define an AQI. We return None, never a guess -- a
     made-up number on a public health service is worse than no number.

Units: CO is mg/m3. Every other concentration is ug/m3.

The top ("Severe") band is open-ended in the CPCB tables ("250+" for PM2.5). We use
the widely published upper bounds below so the interpolation stays defined, and cap
the result at 500. `pipeline/validate_aqi.py` checks this against real published AQI
values -- if the convention is wrong, mismatches show up in the Severe band.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# (concentration_low, concentration_high, aqi_low, aqi_high) per pollutant.
# Bands are continuous: each band starts where the previous one ended, so a reading
# like 30.5 ug/m3 PM2.5 cannot fall through a gap.
BREAKPOINTS: dict[str, list[tuple[float, float, float, float]]] = {
    "pm2_5": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
              (90, 120, 201, 300), (120, 250, 301, 400), (250, 380, 401, 500)],
    "pm10":  [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
              (250, 350, 201, 300), (350, 430, 301, 400), (430, 510, 401, 500)],
    "no2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 180, 101, 200),
              (180, 280, 201, 300), (280, 400, 301, 400), (400, 520, 401, 500)],
    "so2":   [(0, 40, 0, 50), (40, 80, 51, 100), (80, 380, 101, 200),
              (380, 800, 201, 300), (800, 1600, 301, 400), (1600, 2620, 401, 500)],
    "nh3":   [(0, 200, 0, 50), (200, 400, 51, 100), (400, 800, 101, 200),
              (800, 1200, 201, 300), (1200, 1800, 301, 400), (1800, 2400, 401, 500)],
    "co":    [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
              (10, 17, 201, 300), (17, 34, 301, 400), (34, 50, 401, 500)],
    "o3":    [(0, 50, 0, 50), (50, 100, 51, 100), (100, 168, 101, 200),
              (168, 208, 201, 300), (208, 748, 301, 400), (748, 1000, 401, 500)],
}

# How each pollutant is averaged before its sub-index is computed.
AVERAGING_HOURS = {
    "pm2_5": 24, "pm10": 24, "no2": 24, "so2": 24, "nh3": 24,
    "co": 8, "o3": 8,
}
# CO and O3 take the WORST 8-hour window of the day, not the mean of all of them.
ROLLING_MAX_POLLUTANTS = {"co", "o3"}

# CPCB needs three pollutants, and particulate matter must be one of them.
MIN_POLLUTANTS = 3
PARTICULATE = ("pm2_5", "pm10")

# (upper AQI bound, label, colour). Colours are the official CPCB scale.
CATEGORIES = [
    (50,  "Good",         "#55a84f"),
    (100, "Satisfactory", "#a3c853"),
    (200, "Moderate",     "#e0cf2a"),
    (300, "Poor",         "#f29c33"),
    (400, "Very Poor",    "#e93f33"),
    (500, "Severe",       "#96261d"),
]

# Health guidance shown next to each category. Wording follows CPCB's public advisory.
CATEGORY_ADVICE = {
    "Good": "Air quality is satisfactory. Minimal health risk.",
    "Satisfactory": "Minor breathing discomfort possible for sensitive people.",
    "Moderate": "Breathing discomfort for people with asthma, lung or heart disease.",
    "Poor": "Breathing discomfort for most people on prolonged exposure.",
    "Very Poor": "Respiratory illness on prolonged exposure. Avoid outdoor exertion.",
    "Severe": "Serious health effects for everyone. Avoid all outdoor activity.",
}

POLLUTANT_LABELS = {
    "pm2_5": "PM2.5", "pm10": "PM10", "no2": "NO₂",
    "so2": "SO₂", "nh3": "NH₃", "co": "CO", "o3": "O₃",
}


@dataclass(frozen=True)
class AQIResult:
    """One station's AQI at one point in time."""
    aqi: int | None
    category: str | None
    dominant_pollutant: str | None
    sub_indices: dict[str, float]
    reason: str | None = None       # why aqi is None, when it is

    @property
    def is_valid(self) -> bool:
        return self.aqi is not None


# ---------------------------------------------------------------------------
# Sub-index for one pollutant
# ---------------------------------------------------------------------------

def sub_index(pollutant: str, concentration: pd.Series | float) -> pd.Series:
    """Convert concentrations to 0-500 sub-index values for one pollutant.

    Works on a whole column at once so it stays fast over millions of readings.
    Values below zero or missing come back as NaN.
    """
    if pollutant not in BREAKPOINTS:
        raise KeyError(f"No CPCB breakpoints defined for {pollutant!r}")

    values = pd.Series(concentration, dtype="float64")
    result = pd.Series(np.nan, index=values.index, dtype="float64")

    for low_c, high_c, low_i, high_i in BREAKPOINTS[pollutant]:
        # The lowest band includes its floor; the rest start above their floor,
        # so a value on a boundary lands in exactly one band.
        in_band = (values > low_c) & (values <= high_c) if low_c > 0 else (values >= 0) & (values <= high_c)
        if not in_band.any():
            continue
        span = high_c - low_c
        fraction = (values[in_band] - low_c) / span
        result[in_band] = low_i + fraction * (high_i - low_i)

    # Anything above the top band is still Severe. Cap rather than extrapolate.
    top_concentration = BREAKPOINTS[pollutant][-1][1]
    result[values > top_concentration] = 500.0

    return result.round(0)


def category_for(aqi: float | None) -> tuple[str | None, str | None]:
    """Turn an AQI number into its CPCB category name and colour."""
    if aqi is None or pd.isna(aqi):
        return None, None
    for upper, label, colour in CATEGORIES:
        if aqi <= upper:
            return label, colour
    return CATEGORIES[-1][1], CATEGORIES[-1][2]


# ---------------------------------------------------------------------------
# Averaging windows
# ---------------------------------------------------------------------------

def rolling_concentrations(readings: pd.DataFrame, timestamp_column: str = "timestamp") -> pd.DataFrame:
    """Apply each pollutant's CPCB averaging window to a single station's readings.

    Expects readings at a regular interval (the archive is every 15 minutes), sorted
    or not, with a timestamp column. Returns one row per original timestamp holding
    the averaged concentration that the sub-index should be computed from.
    """
    df = readings.sort_values(timestamp_column).set_index(timestamp_column)
    averaged = pd.DataFrame(index=df.index)

    for pollutant, hours in AVERAGING_HOURS.items():
        if pollutant not in df.columns:
            continue
        window = f"{hours}h"
        # min_periods keeps a window from being reported when it is mostly empty:
        # a "24-hour mean" built from two readings is not a 24-hour mean.
        rolling = df[pollutant].rolling(window, min_periods=max(2, hours // 2))
        if pollutant in ROLLING_MAX_POLLUTANTS:
            # Worst 8-hour exposure in the last 24 hours.
            eight_hour_mean = rolling.mean()
            averaged[pollutant] = eight_hour_mean.rolling("24h", min_periods=1).max()
        else:
            averaged[pollutant] = rolling.mean()

    return averaged.reset_index()


def rolling_concentrations_by_station(
    readings: pd.DataFrame,
    station_column: str = "station_code",
    timestamp_column: str = "timestamp",
) -> pd.DataFrame:
    """Same windows as `rolling_concentrations`, for many stations in one pass.

    Calling the single-station version in a loop costs far more in pandas
    overhead than in actual work, so the service uses this instead.
    """
    frames = []
    for station, group in readings.groupby(station_column, sort=False):
        averaged = rolling_concentrations(group, timestamp_column)
        averaged.insert(0, station_column, station)
        frames.append(averaged)

    if not frames:
        return pd.DataFrame(columns=[station_column, timestamp_column])
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Station AQI
# ---------------------------------------------------------------------------

def compute_aqi_frame(averaged: pd.DataFrame) -> pd.DataFrame:
    """Compute AQI for every row of already-averaged concentrations.

    Adds: aqi, aqi_category, dominant_pollutant, n_pollutants, aqi_reason.
    Rows that fail the CPCB validity rules get aqi = NaN and a reason.
    """
    available = [p for p in BREAKPOINTS if p in averaged.columns]
    if not available:
        raise ValueError("No known pollutant columns found to compute AQI from")

    sub = pd.DataFrame(
        {p: sub_index(p, averaged[p]) for p in available},
        index=averaged.index,
    )

    out = pd.DataFrame(index=averaged.index)
    out["n_pollutants"] = sub.notna().sum(axis=1)

    has_particulate = pd.Series(False, index=averaged.index)
    for pm in PARTICULATE:
        if pm in sub.columns:
            has_particulate |= sub[pm].notna()

    enough_pollutants = out["n_pollutants"] >= MIN_POLLUTANTS
    is_valid = enough_pollutants & has_particulate

    out["aqi"] = sub.max(axis=1).where(is_valid)
    # idxmax needs at least one non-null value in the row, so guard it.
    any_value = sub.notna().any(axis=1)
    dominant = pd.Series(pd.NA, index=averaged.index, dtype="object")
    dominant[any_value] = sub[any_value].idxmax(axis=1)
    out["dominant_pollutant"] = dominant.where(is_valid)

    out["aqi_reason"] = np.select(
        [is_valid, ~has_particulate & enough_pollutants, ~enough_pollutants],
        [None, "no_pm2_5_or_pm10", "fewer_than_3_pollutants"],
        default="insufficient_data",
    )
    out["aqi_category"] = [category_for(v)[0] for v in out["aqi"]]

    return pd.concat([out, sub.add_prefix("subindex_")], axis=1)


def compute_aqi(concentrations: dict[str, float]) -> AQIResult:
    """Single-reading AQI, for the API to serve one station right now.

    `concentrations` holds already-averaged values keyed by canonical pollutant name.
    """
    sub_indices: dict[str, float] = {}
    for pollutant, value in concentrations.items():
        if pollutant not in BREAKPOINTS or value is None or pd.isna(value):
            continue
        index_value = sub_index(pollutant, pd.Series([value])).iloc[0]
        if not pd.isna(index_value):
            sub_indices[pollutant] = float(index_value)

    if len(sub_indices) < MIN_POLLUTANTS:
        return AQIResult(None, None, None, sub_indices, "fewer_than_3_pollutants")
    if not any(pm in sub_indices for pm in PARTICULATE):
        return AQIResult(None, None, None, sub_indices, "no_pm2_5_or_pm10")

    dominant = max(sub_indices, key=sub_indices.get)
    aqi_value = int(round(sub_indices[dominant]))
    category, _colour = category_for(aqi_value)
    return AQIResult(aqi_value, category, dominant, sub_indices)
