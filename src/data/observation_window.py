from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.data.types import PatientEndpointStatus

OBSERVATION_WINDOW_VERSION = 2


def observation_end_day(recurrence_day: int | None, censoring_day: int | None) -> int | None:
    days = tuple(day for day in (recurrence_day, censoring_day) if day is not None)
    return min(days) if days else None


def measurement_end_day(
    recurrence_day: int | None,
    censoring_day: int | None,
    *,
    include_end_day: bool,
) -> int | None:
    """Exclusive boundary; optionally count the first event or censoring day once.

    Inclusion is independent of outcome type and does not shift clinical dates
    or change the event/censoring tie rules used to construct targets.
    """
    end_day = observation_end_day(recurrence_day, censoring_day)
    if end_day is not None and include_end_day:
        return end_day + 1
    return end_day


def measurement_window_mask(
    measurements: pd.DataFrame,
    statuses: Mapping[str, PatientEndpointStatus],
    *,
    include_end_day: bool,
) -> pd.Series:
    end_days = {
        patient_id: measurement_end_day(
            None if status.input_recurrence_event is None else status.input_recurrence_event.days_since_diagnosis,
            status.right_censoring_day,
            include_end_day=include_end_day,
        )
        for patient_id, status in statuses.items()
    }
    boundaries = pd.to_numeric(measurements["patient_id"].map(end_days), errors="coerce")
    return boundaries.isna() | (measurements["days_since_diagnosis"] < boundaries)
