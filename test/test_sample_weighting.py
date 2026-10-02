from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from test.test_patient_bloodtest import endpoints_with_split_years


@pytest.mark.parametrize(
    "mode",
    ("person_uniform", "person_time_weighted", "landmark_uniform"),
)
def test_sample_weighting_accepts_canonical_modes(mode: str) -> None:
    config = PatientDatasetConfig.from_mapping(
        {
            "bloodtest_csv_path": "blood.csv",
            "endpoint_csv_path": "endpoint.csv",
            "split_csv_path": "split.csv",
            "training_sample_weighting": mode,
            "bootstrap_sample_weighting": mode,
        }
    )

    assert config.training_sample_weighting == mode
    assert config.bootstrap_sample_weighting == mode


def test_landmark_uniform_training_gives_every_landmark_unit_weight(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P0", 20),
            blood_row("P0", 50),
            blood_row("P1", 10),
        ],
        [endpoint_row("P0"), endpoint_row("P1")],
        training_sample_weighting="landmark_uniform",
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=False)

    np.testing.assert_array_equal(table.sample_weights, np.ones(4))
    assert table.sample_weight_interval_days is None


def test_landmark_uniform_bootstrap_samples_globally_across_valid_landmarks(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P1", 10),
            blood_row("P1", 20),
            blood_row("P1", 50),
        ],
        [endpoint_row("P0"), endpoint_row("P1")],
        bootstrap_sample_weighting="landmark_uniform",
        train_bootstrap_samples=4_000,
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=True)

    assert np.mean(table.patient_ids == "P0") == pytest.approx(0.25, abs=0.03)
    np.testing.assert_array_equal(
        table.bootstrap_selection_probabilities,
        np.full(4_000, 0.25),
    )


def test_person_time_weighted_training_weights_follow_landmark_coverage(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P0", 20),
            blood_row("P0", 50),
            blood_row("P1", 10),
            blood_row("P1", 60),
        ],
        [endpoint_row("P0"), endpoint_row("P1")],
        training_sample_weighting="person_time_weighted",
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=False)

    np.testing.assert_allclose(
        table.sample_weights,
        np.asarray([5 / 18, 5 / 6, 25 / 18, 25 / 18, 10 / 9]),
    )
    assert table.sample_weight_interval_days.tolist() == [10, 30, 50, 50, 40]
    assert table.metadata["training_sample_weighting"] == "person_time_weighted"


def test_training_sample_weighting_defaults_to_person_uniform(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P0", 10), blood_row("P0", 20), blood_row("P1", 10)],
        [endpoint_row("P0"), endpoint_row("P1")],
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=False)

    np.testing.assert_allclose(table.sample_weights, np.asarray([0.75, 0.75, 1.5]))
    assert table.sample_weight_interval_days is None
    assert table.metadata["training_sample_weighting"] == "person_uniform"


def test_person_time_weights_end_at_first_clinical_boundary_not_horizon(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("recurrence", 10),
            blood_row("recurrence", 20),
            blood_row("death", 10),
            blood_row("death", 20),
            blood_row("censor", 10),
            blood_row("censor", 20),
        ],
        [
            endpoint_row("recurrence", recurrence_day=70, death_day=90),
            endpoint_row("death", death_day=70),
            endpoint_row("censor", right_censoring_day=50),
        ],
        censoring_horizon_days=5,
        endpoint_ddfs_code_filter_regex=r"C787\d*",
        training_sample_weighting="person_time_weighted",
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=False)

    assert table.sample_weight_interval_days.tolist() == [10, 30, 10, 50, 10, 50]


def test_training_sample_weighting_rejects_unknown_mode(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P0", 10)],
        [endpoint_row("P0")],
        training_sample_weighting="unknown",
    )

    with pytest.raises(ValueError, match="training_sample_weighting"):
        PatientBloodTestDataset(config)


def test_bootstrap_sample_weighting_rejects_unknown_mode(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P0", 10)],
        [endpoint_row("P0")],
        bootstrap_sample_weighting="unknown",
    )

    with pytest.raises(ValueError, match="bootstrap_sample_weighting"):
        PatientBloodTestDataset(config)


@pytest.mark.parametrize("field", ("training_sample_weighting", "bootstrap_sample_weighting"))
@pytest.mark.parametrize("legacy_mode", ("uniform", "time_interval"))
def test_current_config_rejects_legacy_weighting_modes(field: str, legacy_mode: str) -> None:
    with pytest.raises(ValueError, match=field):
        PatientDatasetConfig.from_mapping(
            {
                "bloodtest_csv_path": "blood.csv",
                "endpoint_csv_path": "endpoint.csv",
                "split_csv_path": "split.csv",
                field: legacy_mode,
            }
        )


def test_person_time_weighted_mode_does_not_change_validation_weights(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P0", 20),
            blood_row("train-reference", 10),
        ],
        [endpoint_row("P0"), endpoint_row("train-reference")],
        split="val",
        split_by_patient={"P0": "val", "train-reference": "train"},
        training_sample_weighting="person_time_weighted",
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=False)

    assert table.sample_weights.tolist() == [1.0, 1.0]
    assert table.sample_weight_interval_days is None


def test_person_time_weighted_bootstrap_reports_full_selection_probabilities(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P0", 20),
            blood_row("P0", 50),
            blood_row("train-reference", 10),
        ],
        [endpoint_row("P0"), endpoint_row("train-reference")],
        split="val",
        split_by_patient={"P0": "val", "train-reference": "train"},
        selected_blood_tests=["B-Hb|||g/dL"],
        bootstrap_sample_weighting="person_time_weighted",
        censoring_horizon_days=5,
        validation_bootstrap_samples=20,
    )

    table = PatientBloodTestDataset(config).to_landmark_table(bootstrap=True)

    expected_by_day = {10: 1 / 9, 20: 1 / 3, 50: 5 / 9}
    expected = np.asarray([expected_by_day[int(day)] for day in table.landmark_days])
    np.testing.assert_allclose(
        table.bootstrap_selection_probabilities,
        expected,
    )
    assert table.metadata["bootstrap_sample_weighting"] == "person_time_weighted"


def test_person_uniform_bootstrap_defaults_to_deterministic_equal_probabilities(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P0", 10), blood_row("P0", 20), blood_row("P1", 10)],
        [endpoint_row("P0"), endpoint_row("P1")],
        selected_blood_tests=["B-Hb|||g/dL"],
        train_bootstrap_samples=20,
    )
    dataset = PatientBloodTestDataset(config)

    first = dataset.to_landmark_table(bootstrap=True, bootstrap_run=2)
    repeated = dataset.to_landmark_table(bootstrap=True, bootstrap_run=2)

    assert dataset.config.bootstrap_sample_weighting == "person_uniform"
    assert first.landmark_days.tolist() == repeated.landmark_days.tolist()
    expected_probabilities = np.asarray(
        [0.25 if patient_id == "P0" else 0.5 for patient_id in first.patient_ids]
    )
    np.testing.assert_array_equal(first.bootstrap_selection_probabilities, expected_probabilities)
    np.testing.assert_array_equal(
        first.bootstrap_selection_probabilities,
        repeated.bootstrap_selection_probabilities,
    )


def test_bootstrap_modes_keep_patient_draws_and_are_independently_deterministic(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("P0", 10),
            blood_row("P0", 20),
            blood_row("P1", 10),
            blood_row("P1", 50),
        ],
        [endpoint_row("P0"), endpoint_row("P1")],
        selected_blood_tests=["B-Hb|||g/dL"],
        train_bootstrap_samples=100,
    )
    uniform = PatientBloodTestDataset(config).to_landmark_table(bootstrap=True, bootstrap_run=4)
    time_dataset = PatientBloodTestDataset(
        {**config, "bootstrap_sample_weighting": "person_time_weighted"}
    )
    time_based = time_dataset.to_landmark_table(bootstrap=True, bootstrap_run=4)
    repeated_time_based = time_dataset.to_landmark_table(bootstrap=True, bootstrap_run=4)

    np.testing.assert_array_equal(uniform.patient_ids, time_based.patient_ids)
    np.testing.assert_array_equal(time_based.landmark_days, repeated_time_based.landmark_days)
    assert uniform.landmark_days.tolist() != time_based.landmark_days.tolist()


def test_person_time_weighting_rejects_non_positive_interval(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P0", 10), blood_row("P0", 20)],
        [endpoint_row("P0")],
        training_sample_weighting="person_time_weighted",
    )
    dataset = PatientBloodTestDataset(config)
    dataset.items = (replace(dataset.items[0], right_censoring_day=20),)

    with pytest.raises(
        ValueError,
        match=r"patient P0 at landmark day 20 with boundary day 20",
    ):
        dataset.to_landmark_table(bootstrap=False)


def blood_row(patient_id: str, day: int) -> dict[str, object]:
    return {
        "id": patient_id,
        "days_since_diagnosis": day,
        "test_name": "B-Hb",
        "result": 1.0,
        "unit": "g/dL",
    }


def endpoint_row(
    patient_id: str,
    *,
    recurrence_day: int | None = None,
    death_day: int | None = None,
    right_censoring_day: int = 100,
) -> dict[str, object]:
    return {
        "id": patient_id,
        "bc_diagnosis_year": None,
        "diagnosis_to_ddfs_in_days": "" if recurrence_day is None else recurrence_day,
        "ddfs_code": "" if recurrence_day is None else "C780",
        "ddfs_description": "",
        "days_from_diagnosis_to_death": "" if death_day is None else death_day,
        "days_from_diagnosis_to_last_entry": right_censoring_day,
    }


def dataset_config(
    tmp_path: Path,
    blood_rows: list[dict[str, object]],
    endpoint_rows: list[dict[str, object]],
    *,
    split: str = "train",
    split_by_patient: dict[str, str] | None = None,
    **overrides: object,
) -> dict[str, object]:
    blood_path = tmp_path / "blood.csv"
    endpoint_path = tmp_path / "endpoint.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(blood_rows).to_csv(blood_path, index=False)
    patient_ids = [str(row["id"]) for row in endpoint_rows]
    assignments = split_by_patient if split_by_patient is not None else dict.fromkeys(patient_ids, "train")
    pd.DataFrame(endpoints_with_split_years(endpoint_rows, assignments)).to_csv(endpoint_path, index=False)
    pd.DataFrame({"id": patient_ids, "split": [assignments[patient_id] for patient_id in patient_ids]}).to_csv(
        split_path, index=False
    )
    return {
        "split": split,
        "bloodtest_csv_path": str(blood_path),
        "endpoint_csv_path": str(endpoint_path),
        "split_csv_path": str(split_path),
        "num_most_common_blood_tests": 1,
        "input_recurrence_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "endpoint_ddfs_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "recurrence_endpoint_name": "distant_recurrence",
        "censoring_horizon_days": 100,
        "random_seed": 42,
        "set_all_bloodtest_values_to_one": False,
        **overrides,
    }
