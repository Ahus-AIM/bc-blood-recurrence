from __future__ import annotations

import re

import pandas as pd

from src.data.other_cancer_outcomes import first_other_cancer_day
from src.data.types import PatientCohortStage, PatientDataSources


def append_patient_stage(
    stages: list[PatientCohortStage],
    discard_reasons: dict[str, str],
    key: str,
    previous_ids: frozenset[str],
    retained_ids: frozenset[str],
    measurement_df: pd.DataFrame,
    discard_reason: str,
) -> pd.DataFrame:
    filtered_df = measurements_for_patients(measurement_df, retained_ids)
    stages.append(cohort_stage(key, retained_ids, filtered_df))
    discard_reasons.update(
        {patient_id: discard_reason for patient_id in previous_ids - retained_ids}
    )
    return filtered_df


def append_other_cancer_stage(
    stages: list[PatientCohortStage],
    discard_reasons: dict[str, str],
    patient_ids: frozenset[str],
    measurement_df: pd.DataFrame,
    sources: PatientDataSources,
    code_regex: re.Pattern[str],
    *,
    before_day: int,
    stage_key: str,
    discard_reason: str,
) -> tuple[pd.DataFrame, frozenset[str]]:
    retained_ids = _without_other_cancer_before(
        patient_ids,
        sources,
        code_regex,
        before_day,
    )
    retained_df = append_patient_stage(
        stages,
        discard_reasons,
        stage_key,
        patient_ids,
        retained_ids,
        measurement_df,
        discard_reason,
    )
    return retained_df, retained_ids


def measurements_for_patients(
    measurement_df: pd.DataFrame,
    patient_ids: frozenset[str],
) -> pd.DataFrame:
    return measurement_df.loc[
        measurement_df["patient_id"].isin(patient_ids)
    ].copy()


def cohort_stage(
    key: str,
    patient_ids: frozenset[str],
    measurement_df: pd.DataFrame,
) -> PatientCohortStage:
    return PatientCohortStage(
        key=key,
        patient_ids=patient_ids,
        bloodtest_count=len(measurement_df),
    )


def _without_other_cancer_before(
    patient_ids: frozenset[str],
    sources: PatientDataSources,
    code_regex: re.Pattern[str],
    day: int,
) -> frozenset[str]:
    return frozenset(
        patient_id
        for patient_id in patient_ids
        if not _has_other_cancer_before(patient_id, sources, code_regex, day)
    )


def _has_other_cancer_before(
    patient_id: str,
    sources: PatientDataSources,
    code_regex: re.Pattern[str],
    day: int,
) -> bool:
    clinical_events = sources.clinical_events_by_patient.get(patient_id, ())
    death_day = sources.endpoint_patients[patient_id].death_day
    other_cancer_day = first_other_cancer_day(
        clinical_events,
        code_regex,
        death_day,
    )
    return other_cancer_day is not None and other_cancer_day < day
