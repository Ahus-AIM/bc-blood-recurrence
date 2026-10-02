from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.competing_risk_outcomes import competing_risk_outcomes
from test.test_patient_bloodtest import OTHER_CANCER_CONFIG, blood_row, clinical_row, config, endpoint_row


@pytest.mark.parametrize(
    "recurrence_days,recurrence_codes,death,other_cancer,expected",
    [
        ("100,100", "C787,C780", "", None, 1),
        ("100,100", "C787,C780", 100, None, 1),
        ("100,100", "C787,C780", 100, 100, 2),
        ("99,100", "C780,C787", "", None, 2),
        ("100", "C780", "", None, 2),
        ("", "", 100, None, 2),
        ("", "", "", None, 0),
    ],
)
def test_endpoint_specific_competing_outcomes_preserve_same_day_precedence(
    tmp_path: Path,
    recurrence_days: str,
    recurrence_codes: str,
    death: object,
    other_cancer: int | None,
    expected: int,
) -> None:
    settings = config(
        tmp_path,
        [blood_row("P", 50)],
        [
            endpoint_row(
                "P",
                diagnosis_to_ddfs=json.dumps(recurrence_days.split(",")) if recurrence_days else "",
                ddfs_code=json.dumps(recurrence_codes.split(",")) if recurrence_codes else "",
                diagnosis_to_death=death,
                last_entry_days=100,
            )
        ],
        clinical_rows=(
            [clinical_row("P", 1000, code="D1")]
            if other_cancer is None
            else [clinical_row("P", other_cancer, code="C80")]
        ),
    )
    settings.update(**OTHER_CANCER_CONFIG, censoring_horizon_days=180, include_recurrence_day_blood_tests=True)
    canonical = PatientBloodTestDataset(settings)
    landmarks = BenchmarkLandmarks(canonical)
    liver_config = replace(canonical.config, endpoint_ddfs_code_filter_regex="C787")
    liver = competing_risk_outcomes(landmarks.table(liver_config))
    assert liver.causes.tolist() == [expected]
    if expected == 0:
        assert np.isnan(liver.binary_outcomes[0])
    if recurrence_days == "100,100" and other_cancer is None:
        lung = competing_risk_outcomes(landmarks.table(replace(liver_config, endpoint_ddfs_code_filter_regex="C780")))
        assert lung.causes.tolist() == [1]


@pytest.mark.parametrize("duration,expected", [(0, 1), (179, 1), (180, 1), (181, 0)])
def test_landmark_zero_and_180_day_boundaries(tmp_path: Path, duration: int, expected: int) -> None:
    settings = config(
        tmp_path, [blood_row("P", 50)], [endpoint_row("P", diagnosis_to_ddfs=50 + duration, ddfs_code="C787")]
    )
    settings.update(censoring_horizon_days=180, include_recurrence_day_blood_tests=True)
    dataset = PatientBloodTestDataset(settings)
    outcomes = competing_risk_outcomes(dataset.to_landmark_table(bootstrap=False))
    assert outcomes.causes.tolist() == [expected]
    assert outcomes.binary_outcomes.tolist() == [float(expected)]
    assert outcomes.durations.tolist() == [min(duration, 180)]
