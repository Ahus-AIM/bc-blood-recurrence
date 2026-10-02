from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.patient_parsing import first_existing_column, optional_existing_column

PATIENT_ID_COLUMNS = ("patient_id", "id")
DAY_COLUMNS = ("days_since_diagnosis", "diagnosis_to_bloodtest", "day")
TEST_NAME_COLUMNS = ("test_name", "analysis", "blood_test")
RESULT_COLUMNS = ("result", "value", "numeric_result")
UNIT_COLUMNS = ("unit", "units")
TEST_KEY_SEPARATOR = "|||"


@dataclass(frozen=True)
class BloodTestSourceData:
    """Clean deduplicated numeric measurements and counts of discarded source rows."""

    measurements: pd.DataFrame
    dropped_counts: dict[str, int]


def load_blood_tests(csv_path: str | Path) -> tuple[pd.DataFrame, dict[str, int], int]:
    """Load clean, same-day-deduplicated blood tests in stable source order."""
    source = load_blood_test_source(csv_path)
    return source.measurements, source.dropped_counts, sum(source.dropped_counts.values())


def load_blood_test_source(csv_path: str | Path) -> BloodTestSourceData:
    """Read and clean the numeric measurements shared by model and descriptive reports."""
    source_df = pd.read_csv(csv_path)
    blood_df = canonical_blood_columns(source_df)
    blood_df["patient_id"] = blood_df["patient_id"].astype(str).str.strip()
    blood_df["test_name"] = blood_df["test_name"].astype(str).str.strip()
    blood_df["unit"] = blood_df["unit"].fillna("").astype(str).str.strip()
    blood_df["test_key"] = make_test_keys(blood_df["test_name"], blood_df["unit"])
    blood_df["source_index"] = range(len(blood_df))
    blood_df["result"] = pd.to_numeric(blood_df["result"], errors="coerce")
    dropped_mask = blood_df["result"].isna()
    dropped_counts = {"non_parseable_result": int(dropped_mask.sum())}
    blood_df = blood_df.loc[~dropped_mask].copy()
    blood_df["days_since_diagnosis"] = blood_df["days_since_diagnosis"].astype(int)
    blood_df["result"] = blood_df["result"].astype(float)
    duplicate_mask = blood_df.duplicated(
        ["patient_id", "days_since_diagnosis", "test_key"],
        keep="last",
    )
    dropped_counts["same_day_duplicate"] = int(duplicate_mask.sum())
    blood_df = blood_df.loc[~duplicate_mask]
    return BloodTestSourceData(blood_df.reset_index(drop=True), dropped_counts)


def canonical_blood_columns(source_df: pd.DataFrame) -> pd.DataFrame:
    patient_column = first_existing_column(source_df, PATIENT_ID_COLUMNS)
    day_column = first_existing_column(source_df, DAY_COLUMNS)
    test_name_column = first_existing_column(source_df, TEST_NAME_COLUMNS)
    result_column = first_existing_column(source_df, RESULT_COLUMNS)
    unit_column = optional_existing_column(source_df, UNIT_COLUMNS)
    blood_df = pd.DataFrame(
        {
            "patient_id": source_df[patient_column],
            "days_since_diagnosis": source_df[day_column],
            "test_name": source_df[test_name_column],
            "result": source_df[result_column],
            "unit": "" if unit_column is None else source_df[unit_column],
        }
    )
    return blood_df


def make_test_keys(test_names: pd.Series, units: pd.Series) -> pd.Series:
    has_unit = units.astype(str).str.len() > 0
    keys = test_names.astype(str).copy()
    keys.loc[has_unit] = keys.loc[has_unit] + TEST_KEY_SEPARATOR + units.loc[has_unit].astype(str)
    return keys
