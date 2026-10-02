from __future__ import annotations

import logging
import re
from types import SimpleNamespace

import pytest

from src.data.dataset_logging import (
    log_filtered_blood_test_counts,
    log_other_cancer_exclusion_counts,
)
from src.data.types import PatientClinicalEvent


def test_filtered_blood_test_counts_include_zeroes_and_sort_descending(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.dataset_logging")

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_filtered_blood_test_counts(
            logger,
            ("B|||u", "A|||u", "A|||u"),
            ("C|||u", "B|||u", "A|||u"),
        )

    assert caplog.messages == [
        "filtered blood-test counts before imputation: A|||u=2, B|||u=1, C|||u=0"
    ]


def test_other_cancer_exclusions_are_logged_by_first_day_icd_category(
    caplog: pytest.LogCaptureFixture,
) -> None:
    logger = logging.getLogger("test.dataset_logging.other_cancer")
    items = (
        SimpleNamespace(
            patient_id="P0",
            right_censoring_day=10,
            clinical_events=(
                PatientClinicalEvent(10, "diagnosis", "C34.1", "Lung cancer"),
                PatientClinicalEvent(10, "diagnosis", "C34.8", "Other lung cancer"),
                PatientClinicalEvent(10, "diagnosis", "C44.9", "Skin\ncancer"),
                PatientClinicalEvent(20, "diagnosis", "C51", "Later cancer"),
            ),
        ),
        SimpleNamespace(
            patient_id="P1",
            right_censoring_day=5,
            clinical_events=(
                PatientClinicalEvent(5, "diagnosis", "c34.9", "Lung cancer"),
            ),
        ),
    )

    with caplog.at_level(logging.INFO, logger=logger.name):
        log_other_cancer_exclusion_counts(
            logger,
            items,  # type: ignore[arg-type]
            re.compile(r"C(?:34|44)\d*"),
            31,
        )

    assert caplog.messages == [
        "other-cancer exclusions before first eligible landmark (day < 31): patients=2",
        (
            "other-cancer exclusion ICD-10 category C34: patients=2; "
            "descriptions=Lung cancer; Other lung cancer"
        ),
        "other-cancer exclusion ICD-10 category C44: patients=1; descriptions=Skin cancer",
        "other-cancer exclusion category overlap: patients with multiple first-day categories=1",
    ]
