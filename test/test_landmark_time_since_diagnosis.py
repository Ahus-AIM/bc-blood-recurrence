from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from src.data.dataset_config import PatientDatasetConfig
from src.data.landmarks import build_patient_landmark_table
from src.data.types import PatientBloodTestItem


def test_time_since_diagnosis_feature_uses_landmark_day_years() -> None:
    config = dataset_config(include_time_since_diagnosis=True)
    table = build_patient_landmark_table(dataset(config), bootstrap=False)

    assert table.feature_names == ("hemoglobin", "time_since_diagnosis_years")
    np.testing.assert_allclose(table.features[:, 1], np.array([0.0, 364 / 365, 1.0, 729 / 365, 2.0], dtype=np.float32))
    assert np.isnan(table.features[:, 0]).all()


def test_time_since_diagnosis_feature_does_not_replace_age_feature() -> None:
    config = dataset_config(include_time_since_diagnosis=True, include_diagnosis_age=True)
    table = build_patient_landmark_table(dataset(config), bootstrap=False)

    assert table.feature_names == ("hemoglobin", "time_since_diagnosis_years", "age_at_diagnosis")
    np.testing.assert_allclose(table.features[:, 1], np.array([0.0, 364 / 365, 1.0, 729 / 365, 2.0], dtype=np.float32))
    np.testing.assert_allclose(table.features[:, 2], np.full(5, 55.5, dtype=np.float32))


def dataset_config(**overrides: object) -> PatientDatasetConfig:
    config = {
        "split": "train",
        "bloodtest_csv_path": "blood.csv",
        "endpoint_csv_path": "endpoint.csv",
        "split_csv_path": "split.csv",
    }
    config.update(overrides)
    return PatientDatasetConfig.from_mapping(config, config_dir=Path.cwd())


def dataset(config: PatientDatasetConfig) -> SimpleNamespace:
    return SimpleNamespace(
        config=config,
        items=(patient_item(config),),
        selected_test_keys=("hemoglobin",),
        selected_recent_diagnosis_codes=(),
        selected_recent_diagnosis_feature_names=(),
        prepared_data=prepared_data(),
    )


def patient_item(config: PatientDatasetConfig) -> PatientBloodTestItem:
    return PatientBloodTestItem(
        patient_id="patient-1",
        split=config.split,
        diagnosis_year=2010,
        age_at_diagnosis=55.5,
        blood_days_since_diagnosis=torch.empty(0, dtype=torch.long),
        blood_test_type_ids=torch.empty(0, dtype=torch.long),
        blood_values=torch.empty(0, dtype=torch.float32),
        input_recurrence_day=None,
        endpoint_day=None,
        death_day=None,
        right_censoring_day=1000,
        last_observation_day=1000,
        source_last_observation_day=1000,
        distant_recurrence_events=(),
        valid_landmark_days=(0, 364, 365, 729, 730),
    )


def prepared_data() -> SimpleNamespace:
    return SimpleNamespace(
        measurement_df=empty_measurement_frame(),
        bc_diagnosis_window_filter_counts={},
        selected_recent_diagnosis_codes=(),
        selected_recent_diagnosis_feature_names=(),
        recent_diagnosis_descriptions={},
        recent_diagnosis_train_patient_counts={},
        input_fingerprint="test",
        dropped_bloodtest_counts={},
        dropped_clinical_event_counts={},
    )


def empty_measurement_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=["patient_id", "days_since_diagnosis", "test_key", "result", "source_index"])
