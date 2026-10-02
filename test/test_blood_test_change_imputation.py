from __future__ import annotations

import pickle
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.model.imputation import fit_feature_imputer, transform_landmark_table
from src.model.imputation_config import ImputationConfig
from src.model.survival_evaluation import require_matching_landmark_feature_rows
from src.script.survival_training import prepare_survival_training
from test.test_patient_bloodtest import blood_row, config, endpoint_row


def change_dataset(tmp_path: Path, *, train_current: float = 10, train_past: float = 3) -> PatientBloodTestDataset:
    records = {
        "P": ((20, "A", train_past), (20, "B", 2), (40, "A", train_current), (40, "B", 8)),
        "Q": ((20, "A", train_past), (20, "B", 2), (40, "A", train_current), (40, "B", 8)),
        "V": ((20, "B", 2), (40, "A", 21), (40, "B", 8)),
        "W": ((20, "A", 9), (20, "B", 2), (40, "B", 8)),
    }
    cfg = config(
        tmp_path,
        [
            blood_row(patient, day, test_name=test, result=value, unit="u")
            for patient, measurements in records.items()
            for day, test, value in measurements
        ],
        [endpoint_row(patient, last_entry_days=100) for patient in records],
        split_by_patient={"P": "train", "Q": "train", "V": "val", "W": "test"},
    )
    ages = tmp_path / "ages.csv"
    pd.DataFrame({"patient_id": list(records), "age_at_diagnosis": [50, 60, 70, 80]}).to_csv(ages, index=False)
    cfg.update(
        min_landmark_day=31,
        selected_blood_tests=["A|||u", "B|||u"],
        include_blood_test_change=True,
        include_time_since_diagnosis=True,
        include_diagnosis_age=True,
        bc_diagnosis_patient_age_csv_path=str(ages),
    )
    return PatientBloodTestDataset(cfg)




@pytest.mark.parametrize(
    ("current", "past", "split", "expected_current", "expected_change"),
    [(-2, 3, "test", 0, -9), (10, -3, "val", 21, 21)],
)
def test_imputed_inputs_are_clipped_before_subtraction_but_signed_changes_are_not(
    tmp_path: Path,
    current: float,
    past: float,
    split: str,
    expected_current: float,
    expected_change: float,
) -> None:
    dataset = change_dataset(tmp_path, train_current=current, train_past=past)
    prepared = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {"enabled": True, "method": "mice"},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    # Negative observed values are retained when fitting and deriving training changes.
    assert prepared.train_imputation.features[0, 2] == current - past
    table = PatientBloodTestDataset(
        replace(dataset.config, split=split), prepared_data=dataset.prepared_data
    ).to_landmark_table(bootstrap=False)
    completed = transform_landmark_table(prepared.imputer, table)
    assert completed.features[0, 0] == pytest.approx(expected_current)
    assert completed.features[0, 2] == pytest.approx(expected_change)
    assert not completed.clipped_mask[0, 2]
    assert completed.past_mice_mask[0, 2] == (split == "val")


def test_past_value_never_uses_locf_even_when_previous_landmark_had_an_observed_past(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P", day, test_name="A", result=value, unit="u") for day, value in ((0, 10), (10, 20), (40, 30))]
        + [
            blood_row("P", day, test_name="B", result=index + 1, unit="u")
            for index, day in enumerate((0, 10, 20, 40, 50, 65))
        ],
        [endpoint_row("P", last_entry_days=100)],
    )
    cfg.update(
        selected_blood_tests=["A|||u", "B|||u"],
        include_blood_test_change=True,
        blood_test_change_min_days=7,
        blood_test_change_max_days=20,
    )
    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)
    prepared = prepare_survival_training(
        dataset,
        table,
        {"enabled": True, "method": "locf", "locf_max_age_days": 30, "locf_fallback_method": "median"},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    np.testing.assert_allclose(
        table.blood_test_change_inputs.past_values[:, 0], [np.nan, 10, 20, np.nan, 30, np.nan], equal_nan=True
    )
    assert prepared.train_imputation.features[-1, 0] == 30  # Current A carries its day-40 observation.
    assert prepared.train_imputation.features[-1, 2] == 10  # Past A uses its training median 20, never LOCF 30.
    assert prepared.imputer.value_imputer.locf_feature_names == ("A|||u", "B|||u")
    assert prepared.train_imputation.carried_forward_mask[-1, 0]
    assert not prepared.train_imputation.carried_forward_mask[:, 2:4].any()
    assert not prepared.train_imputation.past_mice_mask.any()  # Median fallback never contributes to MICE.


def test_bootstrap_and_saved_imputers_preserve_value_pairs_and_legacy_behavior(tmp_path: Path) -> None:
    dataset = change_dataset(tmp_path)
    landmarks = BenchmarkLandmarks(dataset)
    table = landmarks.table(dataset.config)
    prepared = prepare_survival_training(
        dataset,
        table,
        {"enabled": True, "method": "locf", "locf_fallback_method": "mice"},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    restored = pickle.loads(pickle.dumps(prepared.imputer))
    draw = landmarks.table(dataset.config, bootstrap_run=0)
    assert len(draw.blood_test_change_inputs.past_values) == len(draw.patient_ids)
    transformed = transform_landmark_table(restored, draw, reference_table=table)
    rows = {patient: index for index, patient in enumerate(table.patient_ids)}
    np.testing.assert_array_equal(
        transformed.features, prepared.train_imputation.features[[rows[p] for p in draw.patient_ids]]
    )

    validation = landmarks.table(replace(dataset.config, split="val"))
    new_result = transform_landmark_table(restored, validation)
    legacy = fit_feature_imputer(
        table.features,
        table.feature_names,
        ImputationConfig.from_mapping({"enabled": True, "method": "mice"}),
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    old_result = transform_landmark_table(pickle.loads(pickle.dumps(legacy)), validation)
    assert new_result.features[0, 2] == pytest.approx(18)
    assert old_result.features[0, 2] == pytest.approx(7)  # Existing saved models retain direct-change imputation.
    assert old_result.past_mice_mask is None  # Legacy models did not impute a separate past value.


def test_shared_feature_cache_checks_past_values_hidden_by_missing_current_values(tmp_path: Path) -> None:
    dataset = change_dataset(tmp_path)
    table = PatientBloodTestDataset(
        replace(dataset.config, split="test"), prepared_data=dataset.prepared_data
    ).to_landmark_table(bootstrap=False)
    assert np.isnan(table.features[0, 2])
    changed_values = table.blood_test_change_inputs.past_values.copy()
    changed_values[0, 0] = 99
    changed = replace(
        table, blood_test_change_inputs=replace(table.blood_test_change_inputs, past_values=changed_values)
    )
    with pytest.raises(ValueError, match="past values differ"):
        require_matching_landmark_feature_rows(changed, table)


def test_disabled_imputation_preserves_missing_changes_and_all_missing_past_fails_clearly(tmp_path: Path) -> None:
    dataset = change_dataset(tmp_path)
    table = dataset.to_landmark_table(bootstrap=False)
    disabled = prepare_survival_training(
        dataset, table, {"enabled": False}, nonnegative_feature_names=dataset.selected_test_keys
    )
    validation = PatientBloodTestDataset(
        replace(dataset.config, split="val"), prepared_data=dataset.prepared_data
    ).to_landmark_table(bootstrap=False)
    assert np.isnan(transform_landmark_table(disabled.imputer, validation).features[0, 2])
    empty_past = replace(table.blood_test_change_inputs, past_values=np.full((2, 2), np.nan))
    with pytest.raises(ValueError, match="all training values missing.*past"):
        prepare_survival_training(
            dataset,
            replace(table, blood_test_change_inputs=empty_past),
            {"enabled": True, "method": "mice"},
            nonnegative_feature_names=dataset.selected_test_keys,
        )
