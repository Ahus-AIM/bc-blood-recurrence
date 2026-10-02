from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.dataset_config import PatientDatasetConfig
from src.data.types import BloodTestChangeInputs

BLOOD_TEST_CHANGE_VERSION = 2


def blood_test_change_inputs(
    measurements: pd.DataFrame,
    patient_ids: np.ndarray,
    landmark_days: np.ndarray,
    test_keys: tuple[str, ...],
    config: PatientDatasetConfig,
) -> BloodTestChangeInputs:
    """Find each test's latest observed past value, including when its current value is missing."""
    past = np.full((len(patient_ids), len(test_keys)), np.nan, dtype=np.float32)
    if not len(patient_ids) or measurements.empty:
        return BloodTestChangeInputs(test_keys, past)
    queries = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "latest_day": landmark_days - config.blood_test_change_min_days,
            "earliest_day": landmark_days - config.blood_test_change_max_days,
            "row_pos": np.arange(len(patient_ids)),
        }
    ).sort_values("latest_day", kind="stable")
    selected = measurements.loc[measurements["patient_id"].isin(patient_ids)]
    column_by_key = {key: index for index, key in enumerate(test_keys)}
    for key, rows in selected.groupby("test_key", sort=False, observed=True):
        if key not in column_by_key:
            continue
        history = rows[["patient_id", "days_since_diagnosis", "result"]].rename(
            columns={"days_since_diagnosis": "past_day"},
        )
        history["past_day"] = history["past_day"].astype(queries["latest_day"].dtype)
        matched = pd.merge_asof(
            queries,
            history.sort_values("past_day", kind="stable"),
            by="patient_id",
            left_on="latest_day",
            right_on="past_day",
            direction="backward",
        )
        valid = matched["past_day"].notna() & matched["past_day"].ge(matched["earliest_day"])
        positions = matched.loc[valid, "row_pos"].to_numpy(dtype=np.int64)
        past[positions, column_by_key[key]] = matched.loc[valid, "result"].to_numpy(dtype=np.float32)
    return BloodTestChangeInputs(test_keys, past)


def require_matching_change_inputs(
    current: BloodTestChangeInputs | None,
    shared: BloodTestChangeInputs | None,
) -> None:
    """Protect cached imputations when raw differences hide changed past observations."""
    if current is None and shared is None:
        return
    if current is None or shared is None or current.test_keys != shared.test_keys:
        raise ValueError("Shared blood-test past-value schema differs between compatible tasks.")
    if not np.array_equal(current.past_values, shared.past_values, equal_nan=True):
        raise ValueError("Shared blood-test past values differ between compatible tasks.")
