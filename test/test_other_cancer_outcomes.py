from __future__ import annotations

import re

import pytest

from src.data.dataset_config import PatientDatasetConfig
from src.data.other_cancer_outcomes import is_other_cancer_diagnosis
from src.data.types import PatientClinicalEvent


OTHER_CANCER_REGEX = re.compile(r"C(?:0[0-9]|[1-4][0-9]|5[1-9]|6[0-9]|7[0-6]|8[0-9]|9[0-7])\d*")


@pytest.mark.parametrize(
    ("event_type", "code", "expected"),
    [
        ("diagnosis", "C00", True), ("diagnosis", "c44.9", True),
        ("diagnosis", "C80", True), ("diagnosis", "C97.1", True),
        ("diagnosis", "C50.9", False), ("diagnosis", "C77", False),
        ("diagnosis", "C78.7", False), ("diagnosis", "C79.5", False),
        ("diagnosis", "D05", False),
        ("procedure", "C44.9", False),
        ("diagnosis", "", False),
    ],
)
def test_other_cancer_classification_uses_structured_invasive_diagnosis_codes(
    event_type: str,
    code: str,
    expected: bool,
) -> None:
    event = PatientClinicalEvent(0, event_type, code, "Lung cancer")
    assert is_other_cancer_diagnosis(event, OTHER_CANCER_REGEX) is expected


def test_enabled_other_cancer_handling_requires_regex() -> None:
    config = {
        "bloodtest_csv_path": "blood.csv",
        "endpoint_csv_path": "endpoint.csv",
        "split_csv_path": "split.csv",
        "diagnosis_procedure_csv_path": "clinical.csv",
        "exclude_and_censor_other_cancer_patients": True,
    }

    with pytest.raises(ValueError, match="other_cancer_code_filter_regex"):
        PatientDatasetConfig.from_mapping(config)
