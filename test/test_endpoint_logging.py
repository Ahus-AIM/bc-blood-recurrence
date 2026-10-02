from __future__ import annotations

import logging
from pathlib import Path

import pytest

from src.data import PatientBloodTestDataset
from test.test_patient_bloodtest import OTHER_CANCER_CONFIG, blood_row, clinical_row, config, endpoint_row


def endpoint_logging_config(tmp_path: Path) -> dict[str, object]:
    """Cover tied outcomes, later events, every fold, and excluded patients."""
    endpoints = [
        endpoint_row(
            "synchronous",
            diagnosis_to_ddfs=[70, 70, 70, 70],
            ddfs_code=["C78.7", "C787", "C795", "C793"],
            ddfs_description=["Liver\nmetastasis", "Liver metastasis", "Bone metastasis", "Brain metastasis"],
            diagnosis_to_death=70,
            last_entry_days=70,
        ),
        endpoint_row(
            "recurrence",
            diagnosis_to_ddfs=[100, 300],
            ddfs_code=["C787", "C780"],
            ddfs_description=["Liver metastasis", "Later lung metastasis"],
        ),
        endpoint_row(
            "non_target",
            diagnosis_to_ddfs=[60, 120],
            ddfs_code=["C795", "C787"],
            ddfs_description=["Bone metastasis", "Later liver metastasis"],
            diagnosis_to_death=200,
        ),
        endpoint_row("other", diagnosis_to_ddfs=100, ddfs_code="C787"),
        endpoint_row("administrative", diagnosis_to_death=120, last_entry_days=65),
        endpoint_row("death", diagnosis_to_death=80),
        endpoint_row("early_cancer"),
        endpoint_row("no_landmarks", diagnosis_to_ddfs=100, ddfs_code="C787"),
    ]
    splits = {
        "synchronous": "train", "recurrence": "train", "non_target": "val", "other": "val",
        "administrative": "test", "death": "test", "early_cancer": "train", "no_landmarks": "train",
    }
    clinical = [
        clinical_row("synchronous", 70, code="c34.1", description="Lung cancer"),
        clinical_row("synchronous", 70, code="C34.8", description="Other lung cancer"),
        clinical_row("synchronous", 70, code="C44.9", description="Skin cancer"),
        clinical_row("other", 70, code="C44.9", description="Skin cancer"),
        clinical_row("other", 90, code="C51", description="Later cancer"),
        clinical_row("recurrence", 70, event_type="procedure", code="C34", description="Procedure"),
        clinical_row("early_cancer", 20, code="C80", description="Excluded cancer"),
    ]
    blood = [blood_row(patient_id, day) for patient_id in splits for day in (10, 40, 45)]
    blood = [row for row in blood if row["id"] != "no_landmarks" or row["days_since_diagnosis"] == 10]
    return {
        **config(tmp_path, blood, endpoints, split_by_patient=splits, clinical_rows=clinical),
        **OTHER_CANCER_CONFIG,
        "min_landmark_day": 31,
        "endpoint_ddfs_code_filter_regex": r"C78\.?7\d*",
    }


@pytest.mark.parametrize("horizon", [1, 365])
def test_endpoint_logs_count_eligible_patients_and_same_day_overlaps_by_fold(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    horizon: int,
) -> None:
    cfg = endpoint_logging_config(tmp_path)
    cfg["censoring_horizon_days"] = horizon
    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        dataset = PatientBloodTestDataset(cfg)
    messages = [message for message in caplog.messages if message.startswith("patient endpoint")]

    expected = {
        "overall": (6, 2, 2, 2, 2, 2, 1),
        "train": (2, 2, 1, 1, 1, 1, 1),
        "val": (2, 0, 1, 0, 1, 0, 0),
        "test": (2, 0, 0, 1, 0, 1, 0),
    }
    for group, (total, recurrence, non_target, death, other, last, overlaps) in expected.items():
        assert (
            f"patient endpoints [{group}]: patients={total}; first outcome day; independent of prediction horizon"
        ) in messages
        for kind, count in (
            ("configured_recurrence", recurrence), ("non_endpoint_recurrence", non_target),
            ("death", death), ("other_cancer", other), ("last_observation", last),
        ):
            assert f"patient endpoint [{group}] {kind}: patients={count}" in messages
        assert f"patient endpoint overlap [{group}]: patients with multiple first-day outcomes={overlaps}" in messages

    code_rows = [message for message in messages if "[overall]" in message and "ICD-10" in message]
    assert code_rows == [
        "patient endpoint [overall] configured_recurrence ICD-10 code C787: patients=2; descriptions=Liver metastasis",
        "patient endpoint [overall] non_endpoint_recurrence ICD-10 code C795: patients=2; descriptions=Bone metastasis",
        "patient endpoint [overall] non_endpoint_recurrence ICD-10 code C793: patients=1; descriptions=Brain metastasis",
        "patient endpoint [overall] non_endpoint_recurrence ICD-10 code overlap: patients=1",
        "patient endpoint [overall] other_cancer ICD-10 category C44: patients=2; descriptions=Skin cancer",
        (
            "patient endpoint [overall] other_cancer ICD-10 category C34: patients=1; "
            "descriptions=Lung cancer; Other lung cancer"
        ),
        "patient endpoint [overall] other_cancer ICD-10 category overlap: patients=1",
    ]
    assert not any("Later" in message or "Excluded cancer" in message or "Procedure" in message for message in messages)
    assert "target_recurrence_regex=C78\\.?7\\d*" in messages[0]
    assert dataset.items[0].patient_id == "recurrence"
    table = dataset.to_landmark_table(bootstrap=False)
    assert table.events.tolist() == ([False] * 4 if horizon == 1 else [True, True, False, False])
    assert dataset.prepared_data.items_by_split["test"][0].last_observation_day == 120


@pytest.mark.parametrize("other_cancer_enabled", [False, True])
def test_endpoint_logs_use_post_death_normalization_and_other_cancer_toggle(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    other_cancer_enabled: bool,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 40)],
        [endpoint_row("P0", diagnosis_to_ddfs=80, ddfs_code="C787", diagnosis_to_death=70, last_entry_days=100)],
        clinical_rows=[clinical_row("P0", 90, code="C80", description="Other cancer")],
    )
    cfg.update(OTHER_CANCER_CONFIG)
    cfg["exclude_and_censor_other_cancer_patients"] = other_cancer_enabled
    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        dataset = PatientBloodTestDataset(cfg)

    assert "patient endpoint [overall] configured_recurrence: patients=1" in caplog.messages
    assert "patient endpoint [overall] death: patients=1" in caplog.messages
    assert "patient endpoint [overall] last_observation: patients=0" in caplog.messages
    assert f"patient endpoint [overall] other_cancer: patients={int(other_cancer_enabled)}" in caplog.messages
    assert "patient endpoint overlap [overall]: patients with multiple first-day outcomes=1" in caplog.messages
    assert dataset.to_landmark_table(bootstrap=False).events.tolist() == [not other_cancer_enabled]


def test_endpoint_logs_include_empty_cohort_and_folds(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    cfg = config(tmp_path, [blood_row("P0", 10)], [endpoint_row("P0")])
    cfg["min_landmark_day"] = 31
    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        PatientBloodTestDataset(cfg)

    for group in ("overall", "train", "val", "test"):
        assert (
            f"patient endpoints [{group}]: patients=0; first outcome day; independent of prediction horizon"
        ) in caplog.messages
        assert f"patient endpoint [{group}] death: patients=0" in caplog.messages
        assert f"patient endpoint overlap [{group}]: patients with multiple first-day outcomes=0" in caplog.messages
