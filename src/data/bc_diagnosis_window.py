from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from src.data.patient_parsing import canonical_csv, clean_text

BC_DIAGNOSIS_REQUIRED = {
    "patient_id": ("patient_id", "id", "ID"),
    "bc_diagnosis_day": ("bc_diagnosis_day", "days_since_diagnosis", "Dager_siden_BC_diagnose"),
}
BC_DIAGNOSIS_OPTIONAL = {
    "code": ("code", "Kode"),
    "description": ("description", "text", "Tekst"),
}
BASE_SPLITS = ("train", "val", "test", "unknown")


def load_bc_diagnosis_days(csv_path: Path) -> pd.DataFrame:
    """Load canonical patient/day rows used for BC-window filtering and timeline plots."""
    diagnosis_df = canonical_csv(csv_path, BC_DIAGNOSIS_REQUIRED, BC_DIAGNOSIS_OPTIONAL)
    parsed_df = diagnosis_df[["patient_id", "bc_diagnosis_day", "code", "description"]].copy()
    parsed_df["patient_id"] = parsed_df["patient_id"].map(clean_text)
    parsed_df["bc_diagnosis_day"] = pd.to_numeric(parsed_df["bc_diagnosis_day"], errors="coerce")
    parsed_df = parsed_df.dropna(subset=["patient_id", "bc_diagnosis_day"])
    parsed_df["bc_diagnosis_day"] = parsed_df["bc_diagnosis_day"].astype(int)
    parsed_df["code"] = parsed_df["code"].map(clean_text)
    parsed_df["description"] = parsed_df["description"].map(clean_text)
    return parsed_df.sort_values(["patient_id", "bc_diagnosis_day"], kind="mergesort").reset_index(drop=True)


def restrict_blood_tests_to_bc_diagnosis_window(
    blood_df: pd.DataFrame,
    diagnosis_days_csv_path: Path,
    split_by_patient: Mapping[str, str],
    *,
    days_before: int,
    days_after: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Keep numeric blood-test rows inside an inclusive window around extracted BC diagnosis days."""
    diagnosis_df = load_bc_diagnosis_days(diagnosis_days_csv_path)
    diagnosis_days_by_patient = diagnosis_day_arrays(diagnosis_df)
    keep_mask = blood_test_window_mask(blood_df, diagnosis_days_by_patient, days_before, days_after)
    counts = filter_counts(blood_df, keep_mask, diagnosis_days_by_patient.keys(), split_by_patient)
    filtered_df = blood_df.loc[keep_mask].reset_index(drop=True)
    return filtered_df, counts


def disabled_filter_counts(days_before: int, days_after: int) -> dict[str, object]:
    return {
        "enabled": False,
        "bc_diagnosis_window_days_before": days_before,
        "bc_diagnosis_window_days_after": days_after,
    }


def diagnosis_day_arrays(diagnosis_df: pd.DataFrame) -> dict[str, np.ndarray]:
    return {
        str(patient_id): np.sort(group["bc_diagnosis_day"].to_numpy(dtype=np.int32))
        for patient_id, group in diagnosis_df.groupby("patient_id", sort=False)
    }


def blood_test_window_mask(
    blood_df: pd.DataFrame,
    diagnosis_days_by_patient: Mapping[str, np.ndarray],
    days_before: int,
    days_after: int,
) -> np.ndarray:
    keep_mask = np.zeros(len(blood_df), dtype=bool)
    if blood_df.empty:
        return keep_mask
    blood_days = blood_df["days_since_diagnosis"].to_numpy(dtype=np.int32)
    for patient_id, positions in blood_df.groupby("patient_id", sort=False).indices.items():
        diagnosis_days = diagnosis_days_by_patient.get(str(patient_id))
        if diagnosis_days is None or len(diagnosis_days) == 0:
            continue
        position_array = np.asarray(positions, dtype=np.int64)
        keep_mask[position_array] = patient_window_mask(
            blood_days[position_array],
            diagnosis_days,
            days_before,
            days_after,
        )
    return keep_mask


def patient_window_mask(
    blood_days: np.ndarray,
    diagnosis_days: np.ndarray,
    days_before: int,
    days_after: int,
) -> np.ndarray:
    earliest_allowed_diagnosis_day = blood_days - days_after
    latest_allowed_diagnosis_day = blood_days + days_before
    diagnosis_indices = np.searchsorted(diagnosis_days, earliest_allowed_diagnosis_day, side="left")
    has_candidate = diagnosis_indices < len(diagnosis_days)
    keep_mask = np.zeros(len(blood_days), dtype=bool)
    keep_mask[has_candidate] = diagnosis_days[diagnosis_indices[has_candidate]] <= latest_allowed_diagnosis_day[
        has_candidate
    ]
    return keep_mask


def filter_counts(
    blood_df: pd.DataFrame,
    keep_mask: np.ndarray,
    c50_patient_ids: Iterable[str],
    split_by_patient: Mapping[str, str],
) -> dict[str, object]:
    c50_patient_id_set = frozenset(str(patient_id) for patient_id in c50_patient_ids)
    return {
        "enabled": True,
        "global": count_summary(blood_df, keep_mask, c50_patient_id_set),
        "by_split": split_count_summaries(blood_df, keep_mask, c50_patient_id_set, split_by_patient),
    }


def split_count_summaries(
    blood_df: pd.DataFrame,
    keep_mask: np.ndarray,
    c50_patient_ids: frozenset[str],
    split_by_patient: Mapping[str, str],
) -> dict[str, dict[str, int]]:
    if blood_df.empty:
        return {split: count_summary(blood_df, keep_mask, c50_patient_ids) for split in BASE_SPLITS}
    split_labels = blood_df["patient_id"].map(split_by_patient).fillna("unknown").astype(str)
    observed_splits = tuple(sorted(set(split_labels) - set(BASE_SPLITS)))
    summaries = {}
    for split in (*BASE_SPLITS, *observed_splits):
        row_mask = (split_labels == split).to_numpy()
        summaries[split] = count_summary(blood_df.loc[row_mask], keep_mask[row_mask], c50_patient_ids)
    return summaries


def count_summary(
    blood_df: pd.DataFrame,
    keep_mask: np.ndarray,
    c50_patient_ids: frozenset[str],
) -> dict[str, int]:
    input_patients = frozenset(str(patient_id) for patient_id in blood_df["patient_id"].unique())
    kept_df = blood_df.loc[keep_mask]
    kept_patients = frozenset(str(patient_id) for patient_id in kept_df["patient_id"].unique())
    input_patient_days = unique_patient_day_count(blood_df)
    kept_patient_days = unique_patient_day_count(kept_df)
    return {
        "input_bloodtest_rows": int(len(blood_df)),
        "kept_bloodtest_rows": int(keep_mask.sum()),
        "dropped_bloodtest_rows": int(len(blood_df) - keep_mask.sum()),
        "input_patient_days": input_patient_days,
        "kept_patient_days": kept_patient_days,
        "dropped_patient_days": input_patient_days - kept_patient_days,
        "input_patients": len(input_patients),
        "kept_patients": len(kept_patients),
        "patients_without_kept_bloodtests": len(input_patients - kept_patients),
        "patients_without_c50_days": len(input_patients - c50_patient_ids),
    }


def unique_patient_day_count(blood_df: pd.DataFrame) -> int:
    if blood_df.empty:
        return 0
    return int(blood_df[["patient_id", "days_since_diagnosis"]].drop_duplicates().shape[0])
