from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.cohort_selection import (
    MEASUREMENT_EXCLUSION_STAGE_COLUMN,
    STAGE_BC_DIAGNOSIS_WINDOW,
    STAGE_EARLY_OTHER_CANCER,
    STAGE_MIN_BLOOD_TEST_TYPES,
    STAGE_SPARSE_MODEL_MEASUREMENTS,
    STAGE_EARLY_MODEL_MEASUREMENTS,
    STAGE_OBSERVATION_WINDOW,
)
from src.data.landmarks import patient_targets
from src.data.sample_weighting import bootstrap_sample_count, bootstrap_seed
from src.data.dataset import load_patient_data_sources, validate_final_measurement_boundaries
from src.data.dataset_config import PatientDatasetConfig
from src.data.types import PatientEndpointStatus

OTHER_CANCER_CONFIG = {
    "exclude_and_censor_other_cancer_patients": True,
    "other_cancer_code_filter_regex": r"C(?:0[0-9]|[1-4][0-9]|5[1-9]|6[0-9]|7[0-6]|8[0-9]|9[0-7])\d*",
}


def blood_row(
    patient_id: str,
    day: int,
    *,
    test_name: str = "B-Hb",
    result: float = 1.0,
    unit: str = "g/dL",
) -> dict[str, object]:
    return {
        "id": patient_id,
        "days_since_diagnosis": day,
        "test_name": test_name,
        "result": result,
        "unit": unit,
    }


def endpoint_row(
    patient_id: str,
    *,
    diagnosis_year: int | None = None,
    diagnosis_to_ddfs: object = "",
    ddfs_code: object = "",
    ddfs_description: object = "",
    diagnosis_to_death: object = "",
    last_entry_days: object = 3650,
) -> dict[str, object]:
    return {
        "id": patient_id,
        "bc_diagnosis_year": diagnosis_year,
        "diagnosis_to_ddfs_in_days": diagnosis_to_ddfs,
        "ddfs_code": ddfs_code,
        "ddfs_description": ddfs_description,
        "days_from_diagnosis_to_death": diagnosis_to_death,
        "days_from_diagnosis_to_last_entry": last_entry_days,
    }


def clinical_row(
    patient_id: str,
    day: int,
    *,
    event_type: str = "diagnosis",
    code: str = "D1",
    description: str = "diagnosis",
) -> dict[str, object]:
    return {
        "id": patient_id,
        "days_since_diagnosis": day,
        "event_type": event_type,
        "code": code,
        "description": description,
    }


def bc_diagnosis_row(
    patient_id: str,
    day: int,
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "bc_diagnosis_day": day,
        "code": "C509",
        "description": "Breast cancer",
    }


def dips_row(
    patient_id: str,
    day: object,
    *,
    datatype: str,
    code: str,
    description: str,
) -> dict[str, object]:
    return {
        "ID": patient_id,
        "Dager_siden_BC_diagnose": day,
        "Datatype": datatype,
        "Kode": code,
        "Tekst": description,
    }


def config(
    tmp_path: Path,
    blood_rows: list[dict[str, object]],
    endpoint_rows: list[dict[str, object]],
    *,
    split_by_patient: dict[str, str] | None = None,
    clinical_rows: list[dict[str, object]] | None = None,
    bc_diagnosis_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    blood_path = tmp_path / "blood.csv"
    endpoint_path = tmp_path / "endpoint.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(blood_rows).to_csv(blood_path, index=False)
    patient_ids = [str(row["id"]) for row in endpoint_rows]
    split_by_patient = split_by_patient or {patient_id: "train" for patient_id in patient_ids}
    pd.DataFrame(endpoints_with_split_years(endpoint_rows, split_by_patient)).to_csv(endpoint_path, index=False)
    pd.DataFrame(
        {"id": patient_ids, "split": [split_by_patient.get(patient_id, "train") for patient_id in patient_ids]}
    ).to_csv(split_path, index=False)
    cfg = {
        "split": "train",
        "bloodtest_csv_path": str(blood_path),
        "endpoint_csv_path": str(endpoint_path),
        "split_csv_path": str(split_path),
        "num_most_common_blood_tests": 4,
        "input_recurrence_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "endpoint_ddfs_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "recurrence_endpoint_name": "distant_recurrence",
        "censoring_horizon_days": 3650,
        "random_seed": 42,
        "set_all_bloodtest_values_to_one": False,
    }
    if clinical_rows is not None:
        clinical_path = tmp_path / "clinical.csv"
        pd.DataFrame(clinical_rows).to_csv(clinical_path, index=False)
        cfg["diagnosis_procedure_csv_path"] = str(clinical_path)
    if bc_diagnosis_rows is not None:
        bc_diagnosis_path = tmp_path / "bc_diagnosis_days.csv"
        pd.DataFrame(bc_diagnosis_rows).to_csv(bc_diagnosis_path, index=False)
        cfg["bc_diagnosis_days_csv_path"] = str(bc_diagnosis_path)
    return cfg


def endpoints_with_split_years(
    rows: list[dict[str, object]], splits: dict[str, str],
) -> list[dict[str, object]]:
    """Supply temporally valid synthetic years only when a test leaves the year unspecified."""
    years = {"train": 2016, "val": 2020, "test": 2022}
    return [
        {**row, "bc_diagnosis_year": years[splits.get(str(row["id"]), "train")]}
        if row["bc_diagnosis_year"] is None else row
        for row in rows
    ]


def single_patient_outcome_dataset(
    tmp_path: Path,
    *,
    blood_days: tuple[int, ...] = (40,),
    recurrence_day: int | None = None,
    recurrence_code: str = "C787",
    death_day: int | None = None,
    other_cancer_day: int | None = None,
    other_cancer_enabled: bool = False,
) -> PatientBloodTestDataset:
    """Build the common one-patient dataset used by censoring tests."""
    clinical_rows = None if other_cancer_day is None else [clinical_row("P0", other_cancer_day, code="C80")]
    outcome = endpoint_row(
        "P0",
        diagnosis_to_ddfs="" if recurrence_day is None else recurrence_day,
        ddfs_code="" if recurrence_day is None else recurrence_code,
        diagnosis_to_death="" if death_day is None else death_day,
        last_entry_days=100,
    )
    cfg = config(
        tmp_path,
        [blood_row("P0", day, test_name="A", unit="u") for day in blood_days],
        [outcome],
        clinical_rows=clinical_rows,
    )
    cfg.update(selected_blood_tests=["A|||u"], min_landmark_day=31)
    if other_cancer_enabled:
        cfg.update(OTHER_CANCER_CONFIG)
    return PatientBloodTestDataset(cfg)


def panel_a_measurement_exclusion_config(tmp_path: Path) -> dict[str, object]:
    return {
        **config(
            tmp_path,
            [
                blood_row("P0", -10, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=2.0, unit="u"),
                blood_row("P0", 20, test_name="A", result=3.0, unit="u"),
                blood_row("P0", 20, test_name="B", result=4.0, unit="u"),
                blood_row("P0", 40, test_name="A", result=7.0, unit="u"),
                blood_row("P0", 40, test_name="B", result=8.0, unit="u"),
                blood_row("P0", 60, test_name="A", result=9.0, unit="u"),
                blood_row("P0", 60, test_name="B", result=10.0, unit="u"),
                blood_row("P0", 120, test_name="A", result=11.0, unit="u"),
                blood_row("discarded", -10, test_name="A", result=5.0, unit="u"),
            ],
            [
                endpoint_row(
                    "P0",
                    diagnosis_to_ddfs=60,
                    ddfs_code="C787",
                    last_entry_days=120,
                ),
                endpoint_row("discarded", last_entry_days=100),
            ],
            bc_diagnosis_rows=[
                bc_diagnosis_row("P0", 10),
                bc_diagnosis_row("P0", 20),
                bc_diagnosis_row("P0", 40),
                bc_diagnosis_row("P0", 60),
            ],
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "bc_diagnosis_window_days_before": 0,
        "bc_diagnosis_window_days_after": 0,
        "min_blood_test_types_per_day": 2,
        "min_landmark_day": 31,
    }


def test_landmark_table_change_features_use_last_same_day_measurement(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=10.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=16.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=18.0, unit="u"),
                blood_row("P0", 20, test_name="A", result=24.0, unit="u"),
                blood_row("P0", 5, test_name="B", result=2.0, unit="u"),
                blood_row("P0", 20, test_name="B", result=8.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "include_blood_test_change": True,
        "blood_test_change_min_days": 7,
        "blood_test_change_max_days": 20,
        "num_most_common_blood_tests": 2,
    }
    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)
    feature_index = {name: index for index, name in enumerate(table.feature_names)}
    day_20 = int(np.flatnonzero(table.landmark_days == 20)[0])

    assert table.features[day_20, feature_index["A|||u"]] == 24.0
    assert table.features[day_20, feature_index["B|||u"]] == 8.0
    assert table.features[day_20, feature_index["A|||u change"]] == 6.0
    assert table.features[day_20, feature_index["B|||u change"]] == 6.0


def test_configured_blood_tests_replace_most_common_selection(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 0, test_name="B", result=2.0, unit="u"),
                blood_row("P0", 0, test_name="B", result=3.0, unit="u"),
                blood_row("P0", 0, test_name="C", result=4.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "num_most_common_blood_tests": 1,
        "selected_blood_tests": ["C|||u", "A|||u"],
    }

    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)

    assert table.feature_names == ("C|||u", "A|||u")
    assert table.features.tolist() == [[4.0, 1.0]]


def test_configured_blood_tests_reject_unknown_keys(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 0, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "selected_blood_tests": ["missing|||u"],
    }

    with pytest.raises(ValueError, match="unknown blood-test keys: missing"):
        PatientBloodTestDataset(cfg)


def test_frequency_selection_deduplicates_same_patient_day_and_type_first(
    tmp_path: Path,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 0, test_name="A", result=2.0, unit="u"),
                blood_row("P0", 0, test_name="A", result=3.0, unit="u"),
                blood_row("P0", 0, test_name="B", result=4.0, unit="u"),
                blood_row("P0", 10, test_name="B", result=5.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "num_most_common_blood_tests": 1,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)

    assert dataset.selected_test_keys == ("B|||u",)
    assert table.landmark_days.tolist() == [0, 10]
    assert table.features.tolist() == [[4.0], [5.0]]


def test_frequency_selection_precedes_other_cancer_exclusion(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("excluded", 40, test_name="A", result=1.0, unit="u"),
                blood_row("excluded", 50, test_name="A", result=2.0, unit="u"),
                blood_row("included", 40, test_name="B", result=3.0, unit="u"),
            ],
            [
                endpoint_row("excluded", last_entry_days=100),
                endpoint_row("included", last_entry_days=100),
            ],
            clinical_rows=[clinical_row("excluded", 30, code="C44.9")],
        ),
        "num_most_common_blood_tests": 1,
        "min_landmark_day": 31,
        **OTHER_CANCER_CONFIG,
    }

    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        dataset = PatientBloodTestDataset(cfg)

    assert dataset.selected_test_keys == ("A|||u",)
    assert dataset.discarded_patient_items_by_id["excluded"].discard_reasons == (
        "other cancer before first eligible landmark",
    )
    stage_keys = tuple(stage.key for stage in dataset.prepared_data.cohort_stages)
    assert stage_keys.index(STAGE_EARLY_OTHER_CANCER) == stage_keys.index("early_recurrence") - 1
    assert "other-cancer exclusions before first eligible landmark (day < 31): patients=1" in caplog.messages
    assert "other-cancer exclusion ICD-10 category C44: patients=1; descriptions=diagnosis" in caplog.messages


def test_later_other_cancer_right_censors_same_day_recurrence_and_landmarks(
    tmp_path: Path,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        blood_days=(40, 50, 60),
        recurrence_day=50,
        other_cancer_day=50,
        other_cancer_enabled=True,
    )
    item = dataset.items[0]
    table = dataset.to_landmark_table(bootstrap=False)

    assert item.right_censoring_day == 50
    assert item.right_censoring_reason == "other_cancer"
    assert item.valid_landmark_days == (40,)
    assert table.durations.tolist() == [10]
    assert table.events.tolist() == [False]
    assert table.target_end_reasons.tolist() == ["other_cancer"]


def test_disabled_other_cancer_handling_has_no_cohort_or_censoring_semantics(
    tmp_path: Path,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        blood_days=(40, 60),
        other_cancer_day=50,
    )
    item = dataset.items[0]

    assert item.right_censoring_day == 100
    assert item.right_censoring_reason == "last_observation"
    assert item.valid_landmark_days == (40, 60)
    assert STAGE_EARLY_OTHER_CANCER not in tuple(stage.key for stage in dataset.prepared_data.cohort_stages)


def test_post_death_outcomes_move_to_death_with_other_cancer_precedence(tmp_path: Path) -> None:
    recurrence_dataset = single_patient_outcome_dataset(tmp_path, recurrence_day=80, death_day=70)
    cancer_dataset = single_patient_outcome_dataset(
        tmp_path,
        recurrence_day=80,
        death_day=70,
        other_cancer_day=90,
        other_cancer_enabled=True,
    )
    recurrence_item = recurrence_dataset.items[0]
    recurrence_table = recurrence_dataset.to_landmark_table(bootstrap=False)
    assert (
        recurrence_item.input_recurrence_day,
        recurrence_item.endpoint_day,
        recurrence_item.distant_recurrence_events[0].days_since_diagnosis,
        recurrence_table.events.tolist(),
        recurrence_table.target_end_reasons.tolist(),
    ) == (70, 70, 70, [True], ["configured_recurrence"])

    cancer_item = cancer_dataset.items[0]
    cancer_table = cancer_dataset.to_landmark_table(bootstrap=False)
    assert (
        cancer_item.endpoint_day,
        cancer_item.right_censoring_day,
        cancer_item.right_censoring_reason,
        cancer_item.clinical_events[0].days_since_diagnosis,
        cancer_table.events.tolist(),
        cancer_table.target_end_reasons.tolist(),
    ) == (70, 70, "other_cancer", 70, [False], ["other_cancer"])


def test_post_death_raw_records_are_excluded_from_model_data(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P0", 40, test_name="A", unit="u"),
            blood_row("P0", 70, test_name="A", unit="u"),
            blood_row("P0", 90, test_name="A", unit="u"),
        ],
        [
            endpoint_row(
                "P0",
                diagnosis_to_ddfs=80,
                ddfs_code="C787",
                diagnosis_to_death=70,
                last_entry_days=70,
            )
        ],
        clinical_rows=[clinical_row("P0", 80, code="C787", description="Liver metastasis")],
    )
    cfg.update(
        selected_blood_tests=["A|||u"],
        min_landmark_day=31,
        build_patient_timeline_tensors=True,
    )

    dataset_config = PatientDatasetConfig.from_mapping(cfg)
    sources = load_patient_data_sources(dataset_config)
    dataset = PatientBloodTestDataset(dataset_config, data_sources=sources)
    item = dataset.items[0]
    table = dataset.to_landmark_table(bootstrap=False)
    excluded_df = dataset.prepared_data.excluded_measurement_df

    assert dataset.prepared_data.measurement_df["days_since_diagnosis"].tolist() == [40]
    assert item.valid_landmark_days == (40,)
    assert table.landmark_days.tolist() == [40]
    assert item.input_recurrence_day == 70
    assert item.endpoint_day == 70
    assert item.clinical_events[0].days_since_diagnosis == 80
    assert sources.endpoint_patients["P0"].recurrence_events[0].days_since_diagnosis == 80
    assert sources.clinical_events_by_patient["P0"][0].days_since_diagnosis == 80
    assert excluded_df["days_since_diagnosis"].tolist() == [70, 90]
    assert excluded_df[MEASUREMENT_EXCLUSION_STAGE_COLUMN].tolist() == [
        STAGE_OBSERVATION_WINDOW,
        STAGE_OBSERVATION_WINDOW,
    ]


def test_final_measurement_boundary_validator_rejects_retained_boundary_row() -> None:
    measurement_df = pd.DataFrame(
        [{"patient_id": "P0", "days_since_diagnosis": 70}]
    )
    status = PatientEndpointStatus(
        input_recurrence_event=None,
        endpoint_event=None,
        non_endpoint_events=(),
        discard_reason=None,
        right_censoring_day=70,
        right_censoring_reason="death",
    )

    with pytest.raises(
        ValueError,
        match=r"patient P0 on day 70.*observation endpoint day 70",
    ):
        validate_final_measurement_boundaries(measurement_df, {"P0": status})


@pytest.mark.parametrize(
    ("censoring_day", "reason", "expected"),
    [
        (42, "death", "death"),
        (42, "last_observation", "last_observation"),
        (42, "non_endpoint_recurrence", "non_endpoint_recurrence"),
        (100, "last_observation", "censoring_horizon"),
    ],
)
def test_landmark_targets_report_each_right_censoring_reason(
    tmp_path: Path,
    censoring_day: int,
    reason: str,
    expected: str,
) -> None:
    dataset = single_patient_outcome_dataset(tmp_path)
    item = replace(dataset.items[0], right_censoring_day=censoring_day, right_censoring_reason=reason)
    target = patient_targets(item, np.asarray([40]), replace(dataset.config, censoring_horizon_days=5))

    assert target[3].tolist() == [expected]


def test_removed_landmark_missing_fraction_config_is_rejected(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "max_landmark_missing_fraction": 0.5,
    }

    with pytest.raises(ValueError, match="min_blood_test_types_per_day"):
        PatientBloodTestDataset(cfg)


def test_min_blood_test_types_per_day_keeps_days_at_inclusive_threshold(
    tmp_path: Path,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 0, test_name="A", result=2.0, unit="u"),
                blood_row("P0", 0, test_name="B", result=3.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=4.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
        "min_blood_test_types_per_day": 2,
    }

    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)

    assert table.landmark_days.tolist() == [0]
    np.testing.assert_allclose(
        table.features,
        np.asarray([[2.0, 3.0]], dtype=np.float32),
    )
    assert table.metadata["min_blood_test_types_per_day"] == 2


@pytest.mark.parametrize("threshold", [0, -1])
def test_min_blood_test_types_per_day_rejects_invalid_thresholds(
    tmp_path: Path,
    threshold: float,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 0, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "min_blood_test_types_per_day": threshold,
    }

    with pytest.raises(ValueError, match="min_blood_test_types_per_day"):
        PatientBloodTestDataset(cfg)


def test_min_blood_test_types_per_day_changes_input_fingerprint(tmp_path: Path) -> None:
    base_cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 0, test_name="B", result=2.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
    }
    default_dataset = PatientBloodTestDataset(base_cfg)
    filtered_dataset = PatientBloodTestDataset({**base_cfg, "min_blood_test_types_per_day": 2})

    assert default_dataset.prepared_data.input_fingerprint != (filtered_dataset.prepared_data.input_fingerprint)


def test_bc_diagnosis_window_filter_keeps_inclusive_rows_and_reports_counts(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", -7, test_name="A", result=1.0, unit="u"),
                blood_row("P0", -8, test_name="B", result=2.0, unit="u"),
                blood_row("P0", 0, test_name="A", result=3.0, unit="u"),
                blood_row("P0", 7, test_name="A", result=4.0, unit="u"),
                blood_row("P0", 8, test_name="B", result=5.0, unit="u"),
                blood_row("P1", 0, test_name="A", result=6.0, unit="u"),
                blood_row("P2", 10, test_name="C", result=7.0, unit="u"),
                blood_row("P2", 20, test_name="C", result=8.0, unit="u"),
                blood_row("P2", 25, test_name="D", result=9.0, unit="u"),
            ],
            [
                endpoint_row("P0", last_entry_days=100),
                endpoint_row("P1", last_entry_days=100),
                endpoint_row("P2", last_entry_days=100),
            ],
            split_by_patient={"P0": "train", "P1": "train", "P2": "val"},
            bc_diagnosis_rows=[
                bc_diagnosis_row("P0", 0),
                bc_diagnosis_row("P0", 0),
                bc_diagnosis_row("P2", 17),
            ],
        ),
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "bc_diagnosis_window_days_before": 7,
        "bc_diagnosis_window_days_after": 7,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)

    assert dataset.selected_test_keys == ("A|||u", "B|||u")
    assert table.landmark_days.tolist() == [0, 7]
    np.testing.assert_allclose(
        table.features,
        np.asarray([[3.0, np.nan], [4.0, np.nan]], dtype=np.float32),
        equal_nan=True,
    )
    assert dataset.discarded_patient_items_by_id["P1"].discard_reasons == ("no valid landmark days",)

    counts = table.metadata["bc_diagnosis_window_filter_counts"]
    assert counts["enabled"] is True
    assert counts["global"]["input_bloodtest_rows"] == 6
    assert counts["global"]["kept_bloodtest_rows"] == 3
    assert counts["global"]["dropped_bloodtest_rows"] == 3
    assert counts["global"]["patients_without_c50_days"] == 1
    assert counts["global"]["patients_without_kept_bloodtests"] == 1
    assert counts["by_split"]["train"]["input_bloodtest_rows"] == 6
    assert counts["by_split"]["train"]["kept_bloodtest_rows"] == 3
    assert counts["by_split"]["train"]["dropped_bloodtest_rows"] == 3
    assert counts["by_split"]["train"]["patients_without_c50_days"] == 1
    assert counts["by_split"]["val"]["input_bloodtest_rows"] == 0
    assert counts["by_split"]["val"]["kept_bloodtest_rows"] == 0
    assert counts["by_split"]["val"]["dropped_bloodtest_rows"] == 0


def test_bc_diagnosis_window_filter_toggle_off_preserves_blood_tests(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 100, test_name="B", result=2.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=200)],
            bc_diagnosis_rows=[bc_diagnosis_row("P0", 0)],
        ),
        "restrict_bloodtests_to_bc_diagnosis_window": False,
    }

    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)

    assert table.feature_names == ("A|||u", "B|||u")
    assert table.landmark_days.tolist() == [0, 100]
    assert table.metadata["bc_diagnosis_window_filter_counts"]["enabled"] is False


def test_timeline_exclusions_record_first_panel_a_measurement_stage(
    tmp_path: Path,
) -> None:
    cfg = panel_a_measurement_exclusion_config(tmp_path)

    dataset = PatientBloodTestDataset(cfg)
    excluded_df = dataset.prepared_data.excluded_measurement_df

    assert dataset.patient_ids == ("P0",)
    assert set(excluded_df["patient_id"]) == {"P0"}
    assert excluded_df["days_since_diagnosis"].tolist() == [-10, 10, 20, 20, 60, 60, 120]
    assert excluded_df[MEASUREMENT_EXCLUSION_STAGE_COLUMN].tolist() == [
        STAGE_BC_DIAGNOSIS_WINDOW,
        STAGE_EARLY_MODEL_MEASUREMENTS,
        STAGE_EARLY_MODEL_MEASUREMENTS,
        STAGE_EARLY_MODEL_MEASUREMENTS,
        STAGE_OBSERVATION_WINDOW,
        STAGE_OBSERVATION_WINDOW,
        STAGE_BC_DIAGNOSIS_WINDOW,
    ]
    assert excluded_df["plot_value"].tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]
    assert dataset.prepared_data.measurement_df["days_since_diagnosis"].tolist() == [40, 40]
    assert dataset.prepared_data.cohort_stages[-3].key == STAGE_MIN_BLOOD_TEST_TYPES
    assert dataset.prepared_data.cohort_stages[-2].key == STAGE_EARLY_MODEL_MEASUREMENTS
    assert dataset.prepared_data.cohort_stages[-1].key == STAGE_SPARSE_MODEL_MEASUREMENTS


def test_min_diagnosis_year_defaults_to_disabled(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
        [endpoint_row("P0", diagnosis_year=1999, last_entry_days=100)],
    )

    dataset = PatientBloodTestDataset(cfg)

    assert dataset.patient_ids == ("P0",)
    assert dataset.config.min_diagnosis_year is None


def test_min_diagnosis_year_filters_patients_before_measurements(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("old", 0, test_name="A", result=1.0, unit="u"),
                blood_row("new", 0, test_name="A", result=2.0, unit="u"),
            ],
            [
                endpoint_row("old", diagnosis_year=2010, last_entry_days=100),
                endpoint_row("new", diagnosis_year=2015, last_entry_days=100),
            ],
            bc_diagnosis_rows=[
                bc_diagnosis_row("old", 0),
                bc_diagnosis_row("new", 0),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "min_diagnosis_year": 2012,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)

    assert dataset.patient_ids == ("new",)
    assert dataset.discarded_patient_items_by_id["old"].discard_reasons == ("diagnosis before configured minimum year",)
    assert dataset.discarded_patient_items_by_id["old"].valid_landmark_days == ()
    assert table.metadata["min_diagnosis_year"] == 2012


def test_min_diagnosis_year_changes_input_fingerprint(tmp_path: Path) -> None:
    base_cfg = config(
        tmp_path,
        [blood_row("P0", 0, test_name="A", result=1.0, unit="u")],
        [endpoint_row("P0", diagnosis_year=2015, last_entry_days=100)],
    )
    default_dataset = PatientBloodTestDataset(base_cfg)
    cutoff_dataset = PatientBloodTestDataset({**base_cfg, "min_diagnosis_year": 2012})

    assert default_dataset.prepared_data.input_fingerprint != (cutoff_dataset.prepared_data.input_fingerprint)


def test_required_diagnosis_age_changes_input_fingerprint(tmp_path: Path) -> None:
    age_path = tmp_path / "age.csv"
    pd.DataFrame({"id": ["P0"], "age": [55]}).to_csv(age_path, index=False)
    base_cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "bc_diagnosis_patient_age_csv_path": str(age_path),
    }

    optional_dataset = PatientBloodTestDataset(base_cfg)
    required_dataset = PatientBloodTestDataset({**base_cfg, "require_diagnosis_age": True})

    assert optional_dataset.prepared_data.input_fingerprint != (required_dataset.prepared_data.input_fingerprint)


def test_missing_outcome_time_raises_for_otherwise_eligible_patient(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
        [endpoint_row("P0", last_entry_days="")],
    )

    with pytest.raises(ValueError, match="outcome time.*P0"):
        PatientBloodTestDataset(cfg)


def test_missing_diagnosis_age_is_removed_before_test_selection_and_cohort_flow(
    tmp_path: Path,
) -> None:
    age_path = tmp_path / "age.csv"
    pd.DataFrame({"id": ["with_age"], "age": [55]}).to_csv(age_path, index=False)
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("with_age", 10, test_name="A", result=1.0, unit="u"),
                blood_row("missing_age", 10, test_name="B", result=1.0, unit="u"),
                blood_row("missing_age", 20, test_name="B", result=2.0, unit="u"),
            ],
            [
                endpoint_row("with_age", last_entry_days=100),
                endpoint_row("missing_age", last_entry_days=100),
            ],
        ),
        "include_diagnosis_age": True,
        "require_diagnosis_age": True,
        "bc_diagnosis_patient_age_csv_path": str(age_path),
        "num_most_common_blood_tests": 1,
    }

    dataset = PatientBloodTestDataset(cfg)

    assert dataset.selected_test_keys == ("A|||u",)
    assert dataset.patient_ids == ("with_age",)
    assert dataset.prepared_data.cohort_stages[0].patient_ids == {"with_age"}
    assert dataset.prepared_data.cohort_stages[0].bloodtest_count == 1
    assert dataset.discarded_patient_items_by_id["missing_age"].discard_reasons == ("missing diagnosis age",)


def test_missing_required_data_does_not_raise_for_measurement_excluded_patient(
    tmp_path: Path,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days="")],
        ),
        "include_diagnosis_age": True,
        "require_diagnosis_age": True,
        "min_landmark_day": 31,
    }

    dataset = PatientBloodTestDataset(cfg)

    assert len(dataset) == 0


def test_prior_bc_surgery_exclusion_uses_regex_before_first_c50_day(tmp_path: Path) -> None:
    patient_ids = (
        "drop_before_c50",
        "drop_before_positive_c50",
        "keep_3880",
        "keep_3881",
        "keep_after_c50",
    )
    cfg = {
        **config(
            tmp_path,
            [blood_row(patient_id, 10, test_name="A", result=1.0, unit="u") for patient_id in patient_ids],
            [endpoint_row(patient_id, last_entry_days=100) for patient_id in patient_ids],
            clinical_rows=[
                clinical_row("drop_before_c50", -1, event_type="procedure", code="3800"),
                clinical_row("drop_before_positive_c50", 4, event_type="procedure", code="3899"),
                clinical_row("keep_3880", -1, event_type="procedure", code="3880"),
                clinical_row("keep_3881", -1, event_type="procedure", code="3881"),
                clinical_row("keep_after_c50", 5, event_type="procedure", code="3800"),
            ],
            bc_diagnosis_rows=[
                bc_diagnosis_row("drop_before_c50", 0),
                bc_diagnosis_row("drop_before_positive_c50", 5),
                bc_diagnosis_row("keep_3880", 0),
                bc_diagnosis_row("keep_3881", 0),
                bc_diagnosis_row("keep_after_c50", 5),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "exclude_prior_bc_surgery_patients": True,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)

    assert set(dataset.patient_ids) == {"keep_3880", "keep_3881", "keep_after_c50"}
    assert dataset.discarded_patient_items_by_id["drop_before_c50"].discard_reasons == (
        "prior BC surgery before first BC diagnosis",
    )
    assert dataset.discarded_patient_items_by_id["drop_before_positive_c50"].discard_reasons == (
        "prior BC surgery before first BC diagnosis",
    )
    assert table.metadata["exclude_prior_bc_surgery_patients"] is True
    assert table.metadata["prior_bc_surgery_code_regex"] == r"^38(?!8[01])\d{2}$"


def test_bootstrap_run_controls_deterministic_validation_sampling(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=2.0, unit="u"),
                blood_row("P1", 0, test_name="A", result=3.0, unit="u"),
                blood_row("P1", 20, test_name="A", result=4.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100), endpoint_row("P1", last_entry_days=100)],
            split_by_patient={"P0": "val", "P1": "val"},
        ),
        "split": "val",
        "selected_blood_tests": ["A|||u"],
        "validation_bootstrap_samples": 20,
    }
    dataset = PatientBloodTestDataset(cfg)

    first_run = dataset.to_landmark_table(bootstrap=True, bootstrap_run=0)
    repeated_first_run = dataset.to_landmark_table(bootstrap=True, bootstrap_run=0)
    second_run = dataset.to_landmark_table(bootstrap=True, bootstrap_run=1)

    assert first_run.metadata["bootstrap_run"] == 0
    assert second_run.metadata["bootstrap_run"] == 1
    assert bootstrap_seed(dataset.config, 0) != bootstrap_seed(dataset.config, 1)
    assert first_run.patient_ids.tolist() == repeated_first_run.patient_ids.tolist()
    assert first_run.landmark_days.tolist() == repeated_first_run.landmark_days.tolist()


def test_endpoint_regex_changes_targets_not_input_landmarks(tmp_path: Path) -> None:
    base_cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 50, test_name="A", result=2.0, unit="u"),
            ],
            [
                endpoint_row(
                    "P0",
                    diagnosis_to_ddfs="[40, 70]",
                    ddfs_code='["C787", "C793"]',
                    last_entry_days=100,
                ),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "censoring_horizon_days": 3650,
    }
    liver_table = PatientBloodTestDataset(
        {
            **base_cfg,
            "endpoint_ddfs_code_filter_regex": r"C787",
            "recurrence_endpoint_name": "liver",
        }
    ).to_landmark_table(bootstrap=False)
    brain_table = PatientBloodTestDataset(
        {
            **base_cfg,
            "endpoint_ddfs_code_filter_regex": r"C793",
            "recurrence_endpoint_name": "brain",
        }
    ).to_landmark_table(bootstrap=False)

    assert liver_table.landmark_days.tolist() == brain_table.landmark_days.tolist() == [10]
    assert liver_table.features.tolist() == brain_table.features.tolist() == [[1.0]]
    assert liver_table.events.tolist() == [True]
    assert liver_table.durations.tolist() == [30]
    assert brain_table.events.tolist() == [False]
    assert brain_table.durations.tolist() == [30]


def test_pre_diagnosis_input_recurrence_discards_for_site_specific_target(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [
                endpoint_row(
                    "P0",
                    diagnosis_to_ddfs=-5,
                    ddfs_code="C787",
                    last_entry_days=100,
                ),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "endpoint_ddfs_code_filter_regex": r"C793",
        "recurrence_endpoint_name": "brain",
    }

    dataset = PatientBloodTestDataset(cfg)

    assert len(dataset) == 0
    assert dataset.discarded_patient_items_by_id["P0"].discard_reasons == ("recurrence before BC diagnosis",)


@pytest.mark.parametrize("recurrence_day", [0, 30])
def test_input_recurrence_through_day_30_is_an_explicit_patient_exclusion(
    tmp_path: Path,
    recurrence_day: int,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 40, test_name="A", result=1.0, unit="u")],
            [
                endpoint_row(
                    "P0",
                    diagnosis_to_ddfs=recurrence_day,
                    ddfs_code="C780",
                    last_entry_days=100,
                )
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "min_landmark_day": 31,
    }

    dataset = PatientBloodTestDataset(cfg)

    assert dataset.discarded_patient_items_by_id["P0"].discard_reasons == (
        "distant recurrence within 30 days of BC diagnosis",
    )


def test_training_bootstrap_uses_train_sample_count(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 10, test_name="A", result=2.0, unit="u"),
                blood_row("P1", 0, test_name="A", result=3.0, unit="u"),
                blood_row("P1", 20, test_name="A", result=4.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100), endpoint_row("P1", last_entry_days=100)],
        ),
        "selected_blood_tests": ["A|||u"],
        "train_bootstrap_samples": 7,
        "validation_bootstrap_samples": 20,
    }
    dataset = PatientBloodTestDataset(cfg)

    table = dataset.to_landmark_table(bootstrap=True, bootstrap_run=0)

    assert bootstrap_sample_count(dataset.config, dataset.items) == 7
    assert table.metadata["split"] == "train"
    assert table.metadata["bootstrap"] is True
    assert len(table.landmark_days) == 7


def test_otherwise_eligible_patient_missing_model_split_raises(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("assigned", 40, test_name="A", result=1.0, unit="u"),
                blood_row("unassigned", 40, test_name="A", result=2.0, unit="u"),
            ],
            [
                endpoint_row("assigned", last_entry_days=100),
                endpoint_row("unassigned", last_entry_days=100),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
    }
    pd.DataFrame({"id": ["assigned"], "split": ["train"]}).to_csv(
        cfg["split_csv_path"],
        index=False,
    )

    with pytest.raises(
        ValueError,
        match="otherwise eligible patients are missing a train/val/test split assignment: unassigned",
    ):
        PatientBloodTestDataset(cfg)


def test_default_minimum_landmark_day_removes_prediagnosis_rows_from_final_cohort(
    tmp_path: Path,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", -1, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "selected_blood_tests": ["A|||u"],
    }

    dataset = PatientBloodTestDataset(cfg)

    assert len(dataset) == 0
    assert dataset.prepared_data.cohort_stages[-1].patient_ids == frozenset()
    assert dataset.prepared_data.cohort_stages[-1].bloodtest_count == 0


def test_recent_diagnoses_use_train_patient_prevalence_after_final_eligibility(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P1", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P2", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P3", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P4", 10, test_name="A", result=1.0, unit="u"),
                blood_row("P5", 10, test_name="A", result=1.0, unit="u"),
            ],
            [
                endpoint_row("P0", last_entry_days=100),
                endpoint_row("P1", last_entry_days=100),
                endpoint_row("P2", last_entry_days=100),
                endpoint_row("P3", last_entry_days=100),
                endpoint_row("P4", last_entry_days=10),
                endpoint_row("P5", last_entry_days=10),
            ],
            split_by_patient={
                "P0": "train",
                "P1": "train",
                "P2": "train",
                "P3": "val",
                "P4": "train",
                "P5": "train",
            },
            clinical_rows=[
                clinical_row("P0", 5, code="D2", description="Repeated condition"),
                clinical_row("P0", 6, code="D2", description="Repeated condition"),
                clinical_row("P0", 5, code="C509", description="Breast cancer"),
                clinical_row("P1", 5, code="d1", description="Beta Condition!"),
                clinical_row("P2", 5, code="D1", description="Beta Condition!"),
                clinical_row("P3", 5, code="D3", description="Validation only"),
                clinical_row("P4", 5, code="A0", description="Discarded"),
                clinical_row("P5", 5, code="A0", description="Discarded"),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "include_recent_diagnoses": True,
        "num_most_common_recent_diagnoses": 1,
        "recent_diagnosis_lookback_days": 10,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)
    feature_index = {name: index for index, name in enumerate(table.feature_names)}
    diagnosis_feature = "recent_diagnosis_D1__beta_condition"

    assert dataset.selected_recent_diagnosis_codes == ("D1",)
    assert table.feature_names == ("A|||u", diagnosis_feature)
    assert table.metadata["recent_diagnosis_train_patient_counts"] == {"D1": 2}
    assert table.metadata["recent_diagnosis_descriptions"] == {"D1": "Beta Condition!"}
    assert all("C509" not in name for name in table.feature_names)
    flags = {
        str(patient_id): table.features[row, feature_index[diagnosis_feature]]
        for row, patient_id in enumerate(table.patient_ids)
    }
    assert flags == {"P0": 0.0, "P1": 1.0, "P2": 1.0}


def test_recent_diagnosis_lookback_boundaries_and_same_day_opt_in(tmp_path: Path) -> None:
    base_cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
            clinical_rows=[
                clinical_row("P0", 0, code="A", description="Old edge"),
                clinical_row("P0", 10, code="B", description="Same day"),
                clinical_row("P0", -1, code="C", description="Too old"),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "include_recent_diagnoses": True,
        "num_most_common_recent_diagnoses": 3,
        "recent_diagnosis_lookback_days": 10,
    }

    default_table = PatientBloodTestDataset(base_cfg).to_landmark_table(bootstrap=False)
    assert default_table.landmark_days.tolist() == [10]
    assert default_table.feature_names == ("A|||u", "recent_diagnosis_A__old_edge")
    assert default_table.features.tolist() == [[1.0, 1.0]]

    opt_in_cfg = {**base_cfg, "include_landmark_day_recent_diagnoses": True}
    opt_in_table = PatientBloodTestDataset(opt_in_cfg).to_landmark_table(bootstrap=False)
    assert opt_in_table.landmark_days.tolist() == [10]
    assert opt_in_table.feature_names == (
        "A|||u",
        "recent_diagnosis_A__old_edge",
        "recent_diagnosis_B__same_day",
    )
    assert opt_in_table.features.tolist() == [[1.0, 1.0, 1.0]]


@pytest.mark.parametrize(
    "enabled_option",
    ("include_recent_diagnoses", "exclude_prior_bc_surgery_patients", "exclude_and_censor_other_cancer_patients"),
)
def test_clinical_features_require_clinical_csv(tmp_path: Path, enabled_option: str) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        enabled_option: True,
    }

    with pytest.raises(ValueError, match="diagnosis_procedure_csv_path"):
        PatientBloodTestDataset(cfg)


def test_prior_bc_surgery_exclusion_precedes_missing_post_diagnosis_blood(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", -5, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
            clinical_rows=[clinical_row("P0", -1, event_type="procedure", code="3800")],
            bc_diagnosis_rows=[bc_diagnosis_row("P0", 0)],
        ),
        "exclude_prior_bc_surgery_patients": True,
    }

    dataset = PatientBloodTestDataset(cfg)

    assert dataset.discarded_patient_items_by_id["P0"].discard_reasons == (
        "prior BC surgery before first BC diagnosis",
    )


def test_missing_clinical_csv_is_optional_without_recent_diagnoses(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "diagnosis_procedure_csv_path": str(tmp_path / "missing_clinical.csv"),
    }

    dataset = PatientBloodTestDataset(cfg)

    assert dataset[0].clinical_events == ()
    assert dataset.prepared_data.dropped_clinical_event_counts == {
        "ignored_event_type": 0,
        "missing_day": 0,
        "non_parseable_day": 0,
    }


def test_recent_diagnoses_require_existing_clinical_csv_when_enabled(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "diagnosis_procedure_csv_path": str(tmp_path / "missing_clinical.csv"),
        "include_recent_diagnoses": True,
    }

    with pytest.raises(FileNotFoundError, match="Clinical diagnosis/procedure CSV not found"):
        PatientBloodTestDataset(cfg)


def test_recent_diagnoses_can_use_raw_dips_diagnosis_rows(tmp_path: Path) -> None:
    dips_path = tmp_path / "dips.csv"
    pd.DataFrame(
        [
            dips_row("P0", "5 days", datatype="Diagnoser", code="D1", description="Raw diagnosis"),
            dips_row("P0", "", datatype="Diagnoser", code="D2", description="Missing day"),
            dips_row("P0", "not a day", datatype="Diagnoser", code="D3", description="Bad day"),
            dips_row("P0", "6 days", datatype="Blodprøver", code="BLOOD", description="Not clinical"),
            dips_row("P0", "7 days", datatype="Prosedyrer", code="PROC", description="Raw procedure"),
            dips_row("P0", "8 days", datatype="Diagnoser", code="C509", description="Breast cancer"),
        ]
    ).to_csv(dips_path, index=False)
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 10, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "diagnosis_procedure_csv_path": str(dips_path),
        "selected_blood_tests": ["A|||u"],
        "include_recent_diagnoses": True,
        "num_most_common_recent_diagnoses": 3,
        "recent_diagnosis_lookback_days": 10,
    }

    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)

    assert [(event.days_since_diagnosis, event.event_type, event.code) for event in dataset[0].clinical_events] == [
        (5, "diagnosis", "D1"),
        (7, "procedure", "PROC"),
        (8, "diagnosis", "C509"),
    ]
    assert table.feature_names == ("A|||u", "recent_diagnosis_D1__raw_diagnosis")
    assert table.features.tolist() == [[1.0, 1.0]]
    assert table.metadata["dropped_clinical_event_counts"] == {
        "ignored_event_type": 1,
        "missing_day": 1,
        "non_parseable_day": 1,
    }
