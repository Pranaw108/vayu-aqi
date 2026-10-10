"""Serving-time forecast rules (AP-01 support).

The Delhi backtest showed the model is 5.2% WORSE than simply repeating the
current value on calm hours, which are 40% of all hours, while being +33.7% on
episode onsets. So it should not be used everywhere.

The obvious fix is wrong. Routing on the *current* conditions — "use
persistence when it is calm and stable" — would silence the model exactly where
its value is: an onset BEGINS from calm conditions, so every hour routed to
persistence would be an hour the onset could not be called. That would throw
away the 49.9% onset detection rate to win back accuracy on quiet days.

The deadband routes on the model's own predicted CHANGE instead. If the model
says little will change, trust persistence, because on those hours persistence
is measurably better. If the model says a lot will change, trust the model,
because that is the case it is good at. Nothing is suppressed on the basis of
how things look now.
"""

from __future__ import annotations

import numpy as np

# µg/m³. Below this predicted change, persistence wins on the backtest.
DEFAULT_DEADBAND = 5.0


def apply_deadband(
    model_prediction: np.ndarray,
    current_value: np.ndarray,
    threshold: float = DEFAULT_DEADBAND,
) -> np.ndarray:
    """Return persistence where the model predicts little change, else the model.

    `current_value` is the pollutant reading now, which is also the persistence
    forecast. Where it is missing there is nothing to fall back to, so the
    model's prediction is used regardless of the predicted change.
    """
    model_prediction = np.asarray(model_prediction, dtype=float)
    current_value = np.asarray(current_value, dtype=float)

    predicted_change = np.abs(model_prediction - current_value)
    use_persistence = (predicted_change < threshold) & ~np.isnan(current_value)
    return np.where(use_persistence, current_value, model_prediction)


def deadband_share(
    model_prediction: np.ndarray,
    current_value: np.ndarray,
    threshold: float = DEFAULT_DEADBAND,
) -> float:
    """Share of hours the deadband would hand to persistence.

    Worth watching: if this approaches 1.0 the model is doing nothing, and if it
    is near 0.0 the deadband is not engaging.
    """
    model_prediction = np.asarray(model_prediction, dtype=float)
    current_value = np.asarray(current_value, dtype=float)
    usable = ~(np.isnan(model_prediction) | np.isnan(current_value))
    if not usable.any():
        return 0.0
    change = np.abs(model_prediction[usable] - current_value[usable])
    return float((change < threshold).mean())
