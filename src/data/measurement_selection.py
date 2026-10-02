from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import pandas as pd

from src.data.recurrence_measurement_history import MEASUREMENT_COLUMNS
from src.data.types import PreparedPatientData


def selected_cohort_measurements(prepared_data: PreparedPatientData) -> pd.DataFrame:
    """Select final model measurements, including history only for change features."""
    patient_ids = frozenset(
        item.patient_id for split in ("train", "val", "test") for item in prepared_data.items_by_split.get(split, ())
    )
    measurements = prepared_data.measurement_df
    return measurements.loc[measurements["patient_id"].isin(patient_ids)].copy()


def selected_recurrence_measurements(prepared_data: PreparedPatientData) -> pd.DataFrame:
    """Return all study measurements before model-only early-row and sparse-day exclusions."""
    measurements = selected_cohort_measurements(prepared_data).loc[:, list(MEASUREMENT_COLUMNS)]
    history = prepared_data.recurrence_history
    if history is None:
        return measurements
    report_only = history.read()
    report_only = report_only.loc[report_only["patient_id"].isin(measurements["patient_id"].unique())]
    return pd.concat([measurements, report_only], ignore_index=True).sort_values("source_index").reset_index(drop=True)


@contextmanager
def recurrence_report_measurements(prepared_data: PreparedPatientData) -> Iterator[None]:
    """Share report-only values while writing one report, then release the reporting data."""
    history = prepared_data.recurrence_history
    if history is None:
        yield
    else:
        with history.materialized():
            yield
