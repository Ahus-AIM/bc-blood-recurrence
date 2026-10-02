from __future__ import annotations

import numpy as np

ZERO_DURATION_DAYS = 1e-6
CENSORING_OFFSET_DAYS = 1e-6


def model_duration_days(durations: np.ndarray) -> np.ndarray:
    """Encode recorded zero days for the EBM survival objective without changing clinical durations."""
    values = np.asarray(durations, dtype=np.float64)
    if np.any(~np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Survival durations must be finite and nonnegative.")
    return np.where(values == 0, ZERO_DURATION_DAYS, values)


def signed_survival_durations(durations: np.ndarray, events: np.ndarray) -> np.ndarray:
    """Encode events positively and censorings negatively, just after same-day events.

    The censoring offset affects only model ordering; recorded durations stay unchanged.
    """
    values = model_duration_days(durations)
    return np.where(events, values, -(values + CENSORING_OFFSET_DAYS))
