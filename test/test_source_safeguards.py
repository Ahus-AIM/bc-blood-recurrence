from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.extraction.splits import extract_splits
from src.data import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.patient_parsing import load_endpoint_patients, load_split_assignments
from test.test_patient_bloodtest import blood_row, config, endpoint_row


@pytest.mark.parametrize("second_split", ["train", "val", "test"])
def test_split_loader_rejects_repeated_patient_ids(tmp_path: Path, second_split: str) -> None:
    path = tmp_path / "split.csv"
    pd.DataFrame([
        {"id": "P0", "split": "train"},
        {"id": " P0 ", "split": second_split},
    ]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate patient.*P0.*split.csv"):
        load_split_assignments(path)


@pytest.mark.parametrize("second_year", [2010, 2020])
def test_endpoint_loader_rejects_identical_and_conflicting_patient_records(
    tmp_path: Path, second_year: int,
) -> None:
    path = tmp_path / "endpoints.csv"
    pd.DataFrame([
        endpoint_row("P0", diagnosis_year=2010),
        endpoint_row("P0", diagnosis_year=second_year),
    ]).to_csv(path, index=False)

    with pytest.raises(ValueError, match="Duplicate patient.*P0.*endpoints.csv"):
        load_endpoint_patients(path)


@pytest.mark.parametrize(("year", "split"), [(2017, "train"), (2016, "val"), (2022, "val"), (2021, "test")])
def test_final_cohort_rejects_year_split_mismatches(tmp_path: Path, year: int, split: str) -> None:
    cfg = config(tmp_path, [blood_row("P0", 50)], [endpoint_row("P0", diagnosis_year=year)],
                 split_by_patient={"P0": split})
    cfg["selected_blood_tests"] = ["B-Hb|||g/dL"]

    with pytest.raises(ValueError, match=rf"P0.*diagnosis year {year}.*assigned.*{split}.*expected"):
        PatientBloodTestDataset(cfg)


@pytest.mark.parametrize(("year", "split"), [(2004, "train"), (2016, "train"), (2017, "val"),
                                            (2021, "val"), (2022, "test"), (2026, "test")])
def test_split_boundaries_include_the_correct_years(tmp_path: Path, year: int, split: str) -> None:
    cfg = config(tmp_path, [blood_row("P0", 50)], [endpoint_row("P0", diagnosis_year=year)],
                 split_by_patient={"P0": split})
    cfg.update(split=split, selected_blood_tests=["B-Hb|||g/dL"])

    assert PatientBloodTestDataset(cfg).patient_ids == ("P0",)


@pytest.mark.parametrize(("excluded_year", "blood_day"), [(1999, 50), (2020, 10)])
def test_excluded_year_split_mismatch_does_not_stop_the_run(
    tmp_path: Path, excluded_year: int, blood_day: int,
) -> None:
    cfg = config(
        tmp_path, [blood_row("included", 50), blood_row("excluded", blood_day)],
        [endpoint_row("included", diagnosis_year=2010), endpoint_row("excluded", diagnosis_year=excluded_year)],
        split_by_patient={"included": "train", "excluded": "test"},
    )
    cfg.update(min_diagnosis_year=2004, min_landmark_day=31)

    assert PatientBloodTestDataset(cfg).patient_ids == ("included",)


def test_temporal_cutoffs_are_configurable_and_change_the_fingerprint(tmp_path: Path) -> None:
    cfg = config(tmp_path, [blood_row("P0", 50)], [endpoint_row("P0", diagnosis_year=2010)])
    standard = PatientBloodTestDataset(cfg)
    custom = PatientBloodTestDataset({**cfg, "train_end_year": 2019, "test_start_year": 2024})

    assert standard.prepared_data.input_fingerprint != custom.prepared_data.input_fingerprint
    cfg = config(tmp_path, [blood_row("P0", 50)], [endpoint_row("P0", diagnosis_year=2019)])
    assert PatientBloodTestDataset({**cfg, "train_end_year": 2019, "test_start_year": 2024}).patient_ids == ("P0",)


@pytest.mark.parametrize(("train_end", "test_start"), [(-1, 2022), (2022, 2022), (2024, 2022)])
def test_invalid_temporal_cutoffs_are_rejected(train_end: int, test_start: int) -> None:
    with pytest.raises(ValueError, match="train_end_year.*test_start_year"):
        PatientDatasetConfig.from_mapping({
            "bloodtest_csv_path": "blood.csv", "endpoint_csv_path": "endpoints.csv", "split_csv_path": "split.csv",
            "train_end_year": train_end, "test_start_year": test_start,
        })


@pytest.mark.parametrize(("train_end", "test_start"), [(2016, 2022), (2019, 2024)])
def test_split_export_uses_configured_temporal_boundaries(train_end: int, test_start: int) -> None:
    endpoints = pd.DataFrame({
        "id": ["A", "B", "C", "D", "excluded"],
        "bc_diagnosis_year": [train_end, train_end + 1, test_start - 1, test_start, train_end],
        "days_from_diagnosis_to_death": [None] * 5,
        "diagnosis_to_ddfs_in_days": [None] * 5,
    })
    blood = pd.DataFrame({"id": ["A", "B", "C", "D"], "days_since_diagnosis": 50})
    actual = extract_splits(endpoints, blood, {
        "train_end_year": train_end, "test_start_year": test_start, "min_landmark_day": 31,
    })
    assert actual.to_dict("records") == [
        {"id": "A", "split": "train"}, {"id": "B", "split": "val"},
        {"id": "C", "split": "val"}, {"id": "D", "split": "test"},
    ]
