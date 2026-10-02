"""Compact source-history summaries for the included/excluded comparison."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from src.data.cohort_selection import initial_cohort_patient_ids
from src.data.other_cancer_outcomes import compact_icd_code, is_other_cancer_diagnosis

if TYPE_CHECKING:
    from src.data.dataset_config import PatientDatasetConfig
    from src.data.types import PatientDataSources

DISTANT_METASTASIS_REGEX = re.compile(r"C7[89]\d*")
DEFAULT_OTHER_CANCER_REGEX = r"C(?!(?:50|7[7-9]|9[89]))[0-9]{2}\d*"


def other_cancer_pattern(config: PatientDatasetConfig) -> str:
    """Use the configured definition even when model other-cancer handling is disabled."""
    return config.other_cancer_code_filter_regex if config.other_cancer_code_filter_regex else DEFAULT_OTHER_CANCER_REGEX


@dataclass(frozen=True)
class CohortComparisonPatient:
    """One source patient after required-age filtering, before later exclusions.

    ``observation_day`` is last observation capped at death; None means neither
    date is available or the end precedes BC diagnosis. ``death_recorded`` indicates
    a source death date regardless of its timing relative to diagnosis or follow-up.
    ``other_cancer`` and ``metastasis_codes`` cover all recorded history. ``blood_test_count`` counts
    numeric results in the selected panel once per patient/day/type,
    including records outside model follow-up.
    """

    patient_id: str
    diagnosis_year: int
    observation_day: int | None
    death_recorded: bool
    other_cancer: bool | None
    metastasis_codes: tuple[str, ...]
    blood_test_count: int


def build_cohort_comparison_patients(
    sources: PatientDataSources,
    config: PatientDatasetConfig,
    selected_test_keys: tuple[str, ...],
) -> tuple[CohortComparisonPatient, ...]:
    """Summarize the initial training population using the shared required-age filter."""
    patient_ids = initial_cohort_patient_ids(sources, config)
    measurements = sources.blood_df
    counts = measurements.loc[measurements["test_key"].isin(selected_test_keys)].groupby("patient_id").size()
    clinical_available = (
        config.diagnosis_procedure_csv_path is not None and config.diagnosis_procedure_csv_path.is_file()
    )
    other_regex = re.compile(other_cancer_pattern(config))
    rows: list[CohortComparisonPatient] = []
    for patient_id, patient in sources.endpoint_patients.items():
        if patient_id not in patient_ids:
            continue
        end_days = [day for day in (patient.last_observation_day, patient.death_day) if day is not None]
        observation_day = min(end_days) if end_days else None
        if observation_day is not None and observation_day < 0:
            observation_day = None
        clinical = sources.clinical_events_by_patient.get(patient_id, ())
        codes = {compact_icd_code(event.code) for event in patient.recurrence_events}
        rows.append(CohortComparisonPatient(
            patient_id=patient_id,
            diagnosis_year=patient.diagnosis_year,
            observation_day=observation_day,
            death_recorded=patient.death_day is not None,
            other_cancer=(any(is_other_cancer_diagnosis(event, other_regex) for event in clinical)
                          if clinical_available else None),
            metastasis_codes=tuple(sorted(code for code in codes if DISTANT_METASTASIS_REGEX.fullmatch(code))),
            blood_test_count=int(counts.get(patient_id, 0)),
        ))
    return tuple(rows)
