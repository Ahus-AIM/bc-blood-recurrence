from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.data.blood_test_display import feature_display_label
from src.data.dataset import input_fingerprint
from src.data.dataset_config import PatientDatasetConfig
from src.script.ebm_survival_config import dataset_config_mapping
from src.script.single_bloodtest_benchmark_reporting import bloodtest_rows, model_context_features
from src.script.survival_training import feature_subset_prepared_training, prepare_survival_training
from src.script.train_single_bloodtest_benchmark import benchmark_dataset_config
from test.test_patient_bloodtest import blood_row, config, endpoint_row


@pytest.mark.parametrize(
    ("prior_day", "expected"),
    [(9, np.nan), (10, -6.0), (23, -6.0), (24, np.nan), (30, np.nan), (31, np.nan)],
)
def test_change_window_is_inclusive_and_preserves_decreases(
    tmp_path: Path, prior_day: int, expected: float,
) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P0", prior_day, result=10.0),
            blood_row("P0", 30, result=4.0),
        ],
        [endpoint_row("P0", last_entry_days=100)],
    )
    cfg.update(include_blood_test_change=True, blood_test_change_min_days=7, blood_test_change_max_days=20)
    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)
    row = np.flatnonzero(table.landmark_days == 30)[0]

    assert table.feature_names == ("B-Hb|||g/dL", "B-Hb|||g/dL change")
    assert table.features[row, 0] == 4.0
    np.testing.assert_allclose(table.features[row, 1], expected, equal_nan=True)


def test_change_uses_latest_eligible_result_from_same_patient_and_test(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P0", 10, result=10.0),
            blood_row("P0", 20, result=16.0),
            blood_row("P0", 20, result=18.0),
            blood_row("P0", 25, result=100.0),
            blood_row("P0", 30, result=24.0),
            blood_row("P0", 23, test_name="Other", result=500.0),
            blood_row("P1", 23, result=900.0),
        ],
        [endpoint_row("P0"), endpoint_row("P1")],
    )
    cfg.update(include_blood_test_change=True, blood_test_change_min_days=7, blood_test_change_max_days=20)
    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)
    column = table.feature_names.index("B-Hb|||g/dL change")
    row = np.flatnonzero((table.patient_ids == "P0") & (table.landmark_days == 30))[0]

    assert table.features[row, column] == 6.0
    bootstrap = dataset.to_landmark_table(bootstrap=True)
    expected = {
        (patient_id, day): features
        for patient_id, day, features in zip(table.patient_ids, table.landmark_days, table.features, strict=True)
    }
    np.testing.assert_allclose(
        bootstrap.features,
        [expected[(patient_id, day)] for patient_id, day in zip(bootstrap.patient_ids, bootstrap.landmark_days, strict=True)],
        equal_nan=True,
    )


def test_changes_subtract_completed_values_without_changing_landmarks_or_carrying_changes(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P0", day, test_name=test, unit="u", result=value)
            for test, values in (("A", [(0, 10), (10, 6), (20, 4)]), ("B", [(0, 10), (10, 12), (20, 13), (30, 14)]))
            for day, value in values
        ],
        [endpoint_row("P0", last_entry_days=100)],
    )
    cfg.update(selected_blood_tests=["A|||u", "B|||u"], blood_test_change_max_days=20)
    baseline = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)
    dataset = PatientBloodTestDataset({**cfg, "include_blood_test_change": True})
    table = dataset.to_landmark_table(bootstrap=False)
    prepared = prepare_survival_training(
        dataset,
        table,
        {"enabled": True, "method": "locf", "locf_max_age_days": 20, "locf_fallback_method": "median"},
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )

    np.testing.assert_array_equal(table.landmark_days, baseline.landmark_days)
    np.testing.assert_array_equal(table.patient_ids, baseline.patient_ids)
    np.testing.assert_array_equal(table.durations, baseline.durations)
    np.testing.assert_array_equal(table.events, baseline.events)
    np.testing.assert_array_equal(table.sample_weights, baseline.sample_weights)
    assert np.isnan(table.features[0, 2:]).all()
    assert np.isnan(table.features[3, [0, 2]]).all()
    assert prepared.train_imputation.features[3, 0] == 4.0
    assert prepared.train_imputation.features[3, 2] == 0.0
    assert prepared.train_imputation.derived_mask[3, 2]
    assert prepared.train_imputation.carried_forward_mask[3, 0]
    assert not prepared.train_imputation.carried_forward_mask[:, 2:].any()
    assert not prepared.train_imputation.clipped_mask[:, 2:].any()

    rows = bloodtest_rows(("A|||u", "B|||u"), {}, tmp_path, include_blood_test_change=True)
    single_test = feature_subset_prepared_training(prepared, rows[1]["features"])
    assert single_test.imputer is prepared.imputer
    np.testing.assert_array_equal(single_test.train_table.landmark_days, table.landmark_days)
    np.testing.assert_array_equal(single_test.train_imputation.features[3], [4.0, 0.0])


def test_benchmark_honors_change_option_and_keeps_baseline_clinical(tmp_path: Path) -> None:
    cfg = dataset_config_mapping(minimal_config())
    cfg.update(include_blood_test_change=True, include_diagnosis_age=True, include_time_since_diagnosis=True)
    dataset_config = benchmark_dataset_config({"dataset": cfg}, tmp_path)
    rows = bloodtest_rows(
        ("A|||u", "B|||u"), {}, tmp_path,
        context_features=model_context_features(dataset_config),
        include_blood_test_change=dataset_config.include_blood_test_change,
    )

    assert dataset_config.include_blood_test_change
    context = ("time_since_diagnosis_years", "age_at_diagnosis")
    assert rows[0]["features"] == context
    assert rows[1]["features"] == ("A|||u", "A|||u change", *context)
    assert rows[2]["features"] == ("B|||u", "B|||u change", *context)
    assert rows[3]["features"] == ("A|||u", "B|||u", "A|||u change", "B|||u change", *context)




def test_change_settings_affect_fingerprint_and_metadata(tmp_path: Path) -> None:
    cfg = config(tmp_path, [blood_row("P0", 10)], [endpoint_row("P0")])
    dataset = PatientBloodTestDataset({**cfg, "include_blood_test_change": True})
    table = dataset.to_landmark_table(bootstrap=False)
    fingerprint = dataset.prepared_data.input_fingerprint

    assert table.metadata["include_blood_test_change"] is True
    assert table.metadata["blood_test_change_min_days"] == 7
    assert table.metadata["blood_test_change_max_days"] == 365
    assert fingerprint != input_fingerprint(replace(dataset.config, include_blood_test_change=False))
    assert fingerprint != input_fingerprint(replace(dataset.config, blood_test_change_min_days=8))
    assert fingerprint != input_fingerprint(replace(dataset.config, blood_test_change_max_days=180))


@pytest.mark.parametrize(("minimum", "maximum"), [(0, 20), (-1, 20), (7, 6)])
def test_change_window_rejects_nonhistorical_or_reversed_bounds(minimum: int, maximum: int) -> None:
    cfg = dataset_config_mapping(minimal_config())
    cfg.update(blood_test_change_min_days=minimum, blood_test_change_max_days=maximum)
    with pytest.raises(ValueError, match="blood_test_change"):
        PatientDatasetConfig.from_mapping(cfg)


def test_legacy_rate_models_are_not_silently_reinterpreted() -> None:
    cfg = dataset_config_mapping(minimal_config())
    with pytest.raises(ValueError, match="retrain"):
        PatientDatasetConfig.from_mapping({**cfg, "include_rate_of_change": True})
    with pytest.raises(ValueError, match="blood_test_change_min_days"):
        PatientDatasetConfig.from_mapping({**cfg, "include_blood_test_change": True, "rate_of_change_min_days": 90})
    legacy = PatientDatasetConfig.from_mapping({**cfg, "include_rate_of_change": False, "rate_of_change_min_days": 90})
    assert not legacy.include_blood_test_change


def test_change_label_retains_test_unit() -> None:
    label = feature_display_label("CRP|||mg/L change", {"CRP": "C-reactive protein"})
    assert label.full_label == "C-reactive protein change (mg/L)"
    assert label.x_axis_label == "mg/L"


def minimal_config() -> PatientDatasetConfig:
    return PatientDatasetConfig.from_mapping(
        {"bloodtest_csv_path": "blood.csv", "endpoint_csv_path": "endpoint.csv", "split_csv_path": "split.csv"}
    )
