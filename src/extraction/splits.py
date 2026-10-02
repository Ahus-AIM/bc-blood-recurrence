"""Temporal split export from notebook 03."""

from __future__ import annotations

import re

import pandas as pd

from src.data.temporal_splits import split_for_diagnosis_year, validate_temporal_cutoffs


def parse_recurrence_days(value: str) -> list[int] | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    return [int(item) for item in re.findall(r"(-?\d+)", str(value))]


def extract_splits(
    endpoint_df: pd.DataFrame, blood_test_df: pd.DataFrame, dataset_config: dict[str, object]
) -> pd.DataFrame:
    """Exclude early recurrence/death and split remaining patients by diagnosis year."""
    min_landmark_day = int(dataset_config["min_landmark_day"])
    death_before_diagnosis = endpoint_df[endpoint_df["days_from_diagnosis_to_death"] < 0]["id"]

    endpoint_alive_during_diagnosis_df = endpoint_df[~endpoint_df["id"].isin(death_before_diagnosis)]

    first_recurrence_days = endpoint_alive_during_diagnosis_df["diagnosis_to_ddfs_in_days"].apply(
        lambda days: days[0] if isinstance(days, list) else pd.NA
    )

    mbc_ids = endpoint_alive_during_diagnosis_df.loc[first_recurrence_days < min_landmark_day, "id"]

    ebc_ids = endpoint_alive_during_diagnosis_df[~endpoint_alive_during_diagnosis_df["id"].isin(mbc_ids)]["id"]

    ebc_patients_with_blood_test_ids = blood_test_df[
        (blood_test_df["days_since_diagnosis"] >= 0) & (blood_test_df["id"].isin(ebc_ids))
    ]["id"]

    train_end_year = int(dataset_config["train_end_year"])

    test_start_year = int(dataset_config["test_start_year"])

    validate_temporal_cutoffs(train_end_year, test_start_year)

    split_by_row = endpoint_df["bc_diagnosis_year"].map(
        lambda year: split_for_diagnosis_year(int(year), train_end_year, test_start_year)
    )

    train_ids = endpoint_df[(endpoint_df["id"].isin(ebc_patients_with_blood_test_ids)) & (split_by_row == "train")][
        "id"
    ]

    test_ids = endpoint_df[(endpoint_df["id"].isin(ebc_patients_with_blood_test_ids)) & (split_by_row == "test")]["id"]

    val_ids = endpoint_df[(endpoint_df["id"].isin(ebc_patients_with_blood_test_ids)) & (split_by_row == "val")]["id"]

    return pd.concat(
        [
            pd.DataFrame({"id": train_ids.values, "split": "train"}),
            pd.DataFrame({"id": val_ids.values, "split": "val"}),
            pd.DataFrame({"id": test_ids.values, "split": "test"}),
        ]
    ).sort_values(["id"])
