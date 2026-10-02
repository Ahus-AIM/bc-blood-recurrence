from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

import pandas as pd

from src.data.bc_diagnosis_window import (
    disabled_filter_counts,
    restrict_blood_tests_to_bc_diagnosis_window,
)
from src.data.dataset_config import PatientDatasetConfig
from src.data.observation_window import measurement_window_mask, observation_end_day
from src.data.patient_cohort_stages import (
    append_other_cancer_stage,
    append_patient_stage,
    cohort_stage,
    measurements_for_patients,
)
from src.data.prior_bc_exclusions import (
    PRIOR_BC_SURGERY_DISCARD_REASON,
    has_prior_bc_surgery,
)
from src.data.recurrence_measurement_history import RecurrenceMeasurementHistory
from src.data.types import (
    PatientCohortStage,
    PatientDataSources,
    PatientEndpointStatus,
    PatientRecord,
)

# Version 8 starts both the flow and comparison after required-age filtering.
COHORT_SELECTION_VERSION = 8

STAGE_INITIAL = "initial"
STAGE_DIAGNOSIS_YEAR = "diagnosis_year"
STAGE_PRIOR_BC_SURGERY = "prior_bc_surgery"
STAGE_PRE_DIAGNOSIS_OTHER_CANCER = "pre_diagnosis_other_cancer"
STAGE_PRE_DIAGNOSIS_RECURRENCE = "pre_diagnosis_recurrence"
STAGE_EARLY_OTHER_CANCER = "early_other_cancer"
STAGE_EARLY_RECURRENCE = "early_recurrence"
STAGE_BC_DIAGNOSIS_WINDOW = "bc_diagnosis_window"
STAGE_MIN_BLOOD_TEST_TYPES = "min_blood_test_types"
STAGE_MIN_LANDMARK_DAY = "min_landmark_day"
STAGE_OBSERVATION_WINDOW = "observation_window"
STAGE_EARLY_MODEL_MEASUREMENTS = "early_model_measurements"
STAGE_SPARSE_MODEL_MEASUREMENTS = "sparse_model_measurements"
MEASUREMENT_EXCLUSION_STAGE_COLUMN = "exclusion_stage"
MODEL_SPLITS = frozenset(("train", "val", "test"))

DISCARD_DIAGNOSIS_BEFORE_MIN_YEAR = "diagnosis before configured minimum year"
DISCARD_MISSING_DIAGNOSIS_AGE = "missing diagnosis age"
DISCARD_PRE_DIAGNOSIS_OTHER_CANCER = "other cancer before BC diagnosis"
DISCARD_PRE_DIAGNOSIS_RECURRENCE = "recurrence before BC diagnosis"
DISCARD_EARLY_OTHER_CANCER = "other cancer before first eligible landmark"
OTHER_CANCER_DISCARD_REASONS = frozenset(
    (DISCARD_PRE_DIAGNOSIS_OTHER_CANCER, DISCARD_EARLY_OTHER_CANCER)
)


@dataclass(frozen=True)
class CohortSelectionResult:
    """Model rows, lazy report history, display exclusions, stages, and cohort diagnostics."""

    measurement_df: pd.DataFrame
    excluded_measurement_df: pd.DataFrame
    stages: tuple[PatientCohortStage, ...]
    discard_reasons_by_patient: dict[str, str]
    bc_diagnosis_window_filter_counts: dict[str, object]
    recurrence_history: RecurrenceMeasurementHistory | None


def select_patient_cohort(
    measurement_df: pd.DataFrame,
    sources: PatientDataSources,
    status_by_patient: Mapping[str, PatientEndpointStatus],
    config: PatientDatasetConfig,
) -> CohortSelectionResult:
    """Apply patient rules, then blood-test rules, in configured cohort order."""
    current_df, stages, discard_reasons = select_patients(measurement_df, sources, status_by_patient, config)
    current_df, excluded_df, bc_window_counts = select_measurements(
        current_df,
        stages,
        sources,
        status_by_patient,
        config,
    )
    current_df, excluded_df, history = select_model_measurements(current_df, excluded_df, stages, config)
    return CohortSelectionResult(
        measurement_df=current_df.reset_index(drop=True),
        excluded_measurement_df=excluded_df,
        stages=tuple(stages),
        discard_reasons_by_patient=discard_reasons,
        bc_diagnosis_window_filter_counts=bc_window_counts,
        recurrence_history=history,
    )


def select_patients(
    measurement_df: pd.DataFrame,
    sources: PatientDataSources,
    status_by_patient: Mapping[str, PatientEndpointStatus],
    config: PatientDatasetConfig,
) -> tuple[pd.DataFrame, list[PatientCohortStage], dict[str, str]]:
    all_patient_ids = frozenset(sources.endpoint_patients)
    patient_ids = initial_cohort_patient_ids(sources, config)
    current_df = measurements_for_patients(measurement_df, patient_ids)
    stages = [cohort_stage(STAGE_INITIAL, patient_ids, current_df)]
    discard_reasons = {patient_id: DISCARD_MISSING_DIAGNOSIS_AGE for patient_id in all_patient_ids - patient_ids}
    other_cancer_regex = (
        re.compile(config.other_cancer_code_filter_regex)
        if config.exclude_and_censor_other_cancer_patients
        else None
    )

    if config.min_diagnosis_year is not None:
        next_ids = diagnosis_year_patient_ids(
            patient_ids,
            sources.endpoint_patients,
            config.min_diagnosis_year,
        )
        current_df = append_patient_stage(
            stages,
            discard_reasons,
            STAGE_DIAGNOSIS_YEAR,
            patient_ids,
            next_ids,
            current_df,
            DISCARD_DIAGNOSIS_BEFORE_MIN_YEAR,
        )
        patient_ids = next_ids

    if other_cancer_regex is not None:
        current_df, patient_ids = append_other_cancer_stage(
            stages, discard_reasons, patient_ids, current_df, sources, other_cancer_regex,
            before_day=0,
            stage_key=STAGE_PRE_DIAGNOSIS_OTHER_CANCER,
            discard_reason=DISCARD_PRE_DIAGNOSIS_OTHER_CANCER,
        )

    if config.exclude_prior_bc_surgery_patients:
        next_ids = without_prior_bc_surgery(
            patient_ids,
            sources,
            re.compile(config.prior_bc_surgery_code_regex),
        )
        current_df = append_patient_stage(
            stages,
            discard_reasons,
            STAGE_PRIOR_BC_SURGERY,
            patient_ids,
            next_ids,
            current_df,
            PRIOR_BC_SURGERY_DISCARD_REASON,
        )
        patient_ids = next_ids

    next_ids = without_recurrence_before(patient_ids, status_by_patient, 0)
    current_df = append_patient_stage(
        stages,
        discard_reasons,
        STAGE_PRE_DIAGNOSIS_RECURRENCE,
        patient_ids,
        next_ids,
        current_df,
        DISCARD_PRE_DIAGNOSIS_RECURRENCE,
    )
    patient_ids = next_ids

    if other_cancer_regex is not None and config.min_landmark_day > 0:
        current_df, patient_ids = append_other_cancer_stage(
            stages, discard_reasons, patient_ids, current_df, sources, other_cancer_regex,
            before_day=config.min_landmark_day,
            stage_key=STAGE_EARLY_OTHER_CANCER,
            discard_reason=DISCARD_EARLY_OTHER_CANCER,
        )

    if config.min_landmark_day > 0:
        next_ids = without_early_recurrence(
            patient_ids,
            status_by_patient,
            config.min_landmark_day,
        )
        current_df = append_patient_stage(
            stages,
            discard_reasons,
            STAGE_EARLY_RECURRENCE,
            patient_ids,
            next_ids,
            current_df,
            early_recurrence_reason(config.min_landmark_day),
        )
        patient_ids = next_ids

    return current_df, stages, discard_reasons


def initial_cohort_patient_ids(
    sources: PatientDataSources,
    config: PatientDatasetConfig,
) -> frozenset[str]:
    patient_ids = frozenset(sources.endpoint_patients)
    if not config.require_diagnosis_age:
        return patient_ids
    return patient_ids & frozenset(sources.age_by_patient)


def select_measurements(
    measurement_df: pd.DataFrame,
    stages: list[PatientCohortStage],
    sources: PatientDataSources,
    status_by_patient: Mapping[str, PatientEndpointStatus],
    config: PatientDatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    excluded_frames: list[pd.DataFrame] = []
    next_df, bc_window_counts = apply_bc_diagnosis_window(measurement_df, sources, config)
    if config.restrict_bloodtests_to_bc_diagnosis_window:
        current_df = append_measurement_stage(
            stages,
            excluded_frames,
            STAGE_BC_DIAGNOSIS_WINDOW,
            measurement_df,
            next_df,
            config.build_patient_timeline_tensors,
        )
    else:
        current_df = next_df

    next_df = retain_landmark_patients(current_df, config)
    current_df = append_measurement_stage(
        stages,
        excluded_frames,
        STAGE_MIN_LANDMARK_DAY,
        current_df,
        next_df,
        config.build_patient_timeline_tensors,
    )

    next_df = filter_observation_window(
        current_df,
        sources.endpoint_patients,
        status_by_patient,
        include_end_day=config.include_recurrence_day_blood_tests,
    )
    next_df = retain_landmark_patients(next_df, config)
    current_df = append_measurement_stage(
        stages,
        excluded_frames,
        STAGE_OBSERVATION_WINDOW,
        current_df,
        next_df,
        config.build_patient_timeline_tensors,
    )
    if config.min_blood_test_types_per_day is not None:
        qualifying_df = filter_sparse_patient_days(
            current_df,
            config.min_blood_test_types_per_day,
            config.min_landmark_day,
        )
        qualifying_df = retain_landmark_patients(qualifying_df, config)
        next_df = measurements_for_patients(current_df, measurement_patient_ids(qualifying_df))
        current_df = append_measurement_stage(
            stages,
            excluded_frames,
            STAGE_MIN_BLOOD_TEST_TYPES,
            current_df,
            next_df,
            config.build_patient_timeline_tensors,
        )

    excluded_df = combined_excluded_measurements(measurement_df, excluded_frames)
    patient_ids = measurement_patient_ids(current_df)
    validate_final_split_assignments(patient_ids, sources.split_by_patient)
    return current_df, excluded_df, bc_window_counts


def retain_landmark_patients(measurement_df: pd.DataFrame, config: PatientDatasetConfig) -> pd.DataFrame:
    """Keep earlier history only for patients with a remaining landmark candidate."""
    landmark_rows = measurement_df["days_since_diagnosis"] >= config.min_landmark_day
    patient_ids = measurement_patient_ids(measurement_df.loc[landmark_rows])
    return measurements_for_patients(measurement_df, patient_ids)


def select_model_measurements(
    study_df: pd.DataFrame,
    excluded_df: pd.DataFrame,
    stages: list[PatientCohortStage],
    config: PatientDatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, RecurrenceMeasurementHistory | None]:
    """Keep report history, then remove early rows and finally sparse model days.

    Both model-only stages preserve the selected patients. The removed source-row
    references make every study measurement available to recurrence analyses,
    without adding sparse days to prediction landmarks or change-feature history.
    """
    model_df = study_df
    excluded_frames: list[pd.DataFrame] = []
    if not config.include_blood_test_change:
        retained = model_df.loc[model_df["days_since_diagnosis"] >= config.min_landmark_day].copy()
        model_df = append_measurement_stage(
            stages, excluded_frames, STAGE_EARLY_MODEL_MEASUREMENTS, model_df, retained,
            config.build_patient_timeline_tensors,
        )
    if config.min_blood_test_types_per_day is not None:
        retained = filter_sparse_patient_days(model_df, config.min_blood_test_types_per_day, config.min_landmark_day)
        model_df = append_measurement_stage(
            stages, excluded_frames, STAGE_SPARSE_MODEL_MEASUREMENTS, model_df, retained,
            config.build_patient_timeline_tensors,
        )
    if measurement_patient_ids(model_df) != measurement_patient_ids(study_df):
        raise ValueError("Model-only measurement exclusions must not remove patients.")
    report_only = removed_measurements(study_df, model_df)
    history = (
        RecurrenceMeasurementHistory.from_measurements(config.bloodtest_csv_path, report_only)
        if not report_only.empty else None
    )
    if excluded_frames:
        excluded_df = pd.concat([excluded_df, *excluded_frames], ignore_index=True)
    return model_df, excluded_df, history


def append_measurement_stage(
    stages: list[PatientCohortStage],
    excluded_frames: list[pd.DataFrame],
    stage_key: str,
    previous_df: pd.DataFrame,
    retained_df: pd.DataFrame,
    record_exclusions: bool,
) -> pd.DataFrame:
    """Record one Panel A measurement stage without changing its filtering semantics."""
    if record_exclusions:
        excluded_df = removed_measurements(previous_df, retained_df)
        if not excluded_df.empty:
            excluded_df[MEASUREMENT_EXCLUSION_STAGE_COLUMN] = stage_key
            excluded_frames.append(excluded_df)
    patient_ids = measurement_patient_ids(retained_df)
    stages.append(cohort_stage(stage_key, patient_ids, retained_df))
    return retained_df


def removed_measurements(
    previous_df: pd.DataFrame,
    retained_df: pd.DataFrame,
) -> pd.DataFrame:
    retained_source_indices = frozenset(int(value) for value in retained_df["source_index"])
    return previous_df.loc[~previous_df["source_index"].isin(retained_source_indices)].copy()


def combined_excluded_measurements(
    measurement_df: pd.DataFrame,
    excluded_frames: list[pd.DataFrame],
) -> pd.DataFrame:
    if not excluded_frames:
        empty_df = measurement_df.head(0).copy()
        empty_df[MEASUREMENT_EXCLUSION_STAGE_COLUMN] = pd.Series(dtype="string")
        return empty_df
    return pd.concat(excluded_frames, ignore_index=True)


def diagnosis_year_patient_ids(
    patient_ids: frozenset[str],
    patients: Mapping[str, PatientRecord],
    min_diagnosis_year: int,
) -> frozenset[str]:
    return frozenset(
        patient_id for patient_id in patient_ids if patients[patient_id].diagnosis_year >= min_diagnosis_year
    )


def without_prior_bc_surgery(
    patient_ids: frozenset[str],
    sources: PatientDataSources,
    code_regex: re.Pattern[str],
) -> frozenset[str]:
    return frozenset(
        patient_id
        for patient_id in patient_ids
        if not has_prior_bc_surgery(
            sources.clinical_events_by_patient.get(patient_id, ()),
            sources.bc_diagnosis_events_by_patient.get(patient_id, ()),
            code_regex,
        )
    )


def early_recurrence_reason(min_landmark_day: int) -> str:
    return f"distant recurrence within {min_landmark_day - 1} days of BC diagnosis"


def validate_final_split_assignments(
    patient_ids: frozenset[str],
    split_by_patient: Mapping[str, str],
) -> None:
    invalid_ids = tuple(
        sorted(patient_id for patient_id in patient_ids if split_by_patient.get(patient_id) not in MODEL_SPLITS)
    )
    if not invalid_ids:
        return
    examples = ", ".join(invalid_ids[:10])
    raise ValueError(
        f"{len(invalid_ids)} otherwise eligible patients are missing a train/val/test split assignment: {examples}"
    )


def without_recurrence_before(
    patient_ids: frozenset[str],
    statuses: Mapping[str, PatientEndpointStatus],
    day: int,
) -> frozenset[str]:
    return frozenset(
        patient_id
        for patient_id in patient_ids
        if statuses[patient_id].input_recurrence_event is None
        or statuses[patient_id].input_recurrence_event.days_since_diagnosis >= day
    )


def without_early_recurrence(
    patient_ids: frozenset[str],
    statuses: Mapping[str, PatientEndpointStatus],
    min_landmark_day: int,
) -> frozenset[str]:
    return frozenset(
        patient_id
        for patient_id in patient_ids
        if statuses[patient_id].input_recurrence_event is None
        or statuses[patient_id].input_recurrence_event.days_since_diagnosis >= min_landmark_day
    )


def measurement_patient_ids(measurement_df: pd.DataFrame) -> frozenset[str]:
    return frozenset(str(patient_id) for patient_id in measurement_df["patient_id"].unique())


def apply_bc_diagnosis_window(
    measurement_df: pd.DataFrame,
    sources: PatientDataSources,
    config: PatientDatasetConfig,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if not config.restrict_bloodtests_to_bc_diagnosis_window:
        return measurement_df, disabled_filter_counts(
            config.bc_diagnosis_window_days_before,
            config.bc_diagnosis_window_days_after,
        )
    if config.bc_diagnosis_days_csv_path is None:
        raise ValueError("`dataset.bc_diagnosis_days_csv_path` is required for BC diagnosis-window filtering.")
    filtered_df, counts = restrict_blood_tests_to_bc_diagnosis_window(
        measurement_df,
        config.bc_diagnosis_days_csv_path,
        sources.split_by_patient,
        days_before=config.bc_diagnosis_window_days_before,
        days_after=config.bc_diagnosis_window_days_after,
    )
    return filtered_df, {
        **counts,
        "bc_diagnosis_days_csv_path": str(config.bc_diagnosis_days_csv_path),
        "bc_diagnosis_window_days_before": config.bc_diagnosis_window_days_before,
        "bc_diagnosis_window_days_after": config.bc_diagnosis_window_days_after,
    }


def filter_sparse_patient_days(
    measurement_df: pd.DataFrame,
    min_blood_test_types_per_day: int,
    min_landmark_day: int,
) -> pd.DataFrame:
    if measurement_df.empty:
        return measurement_df
    day_counts = (
        measurement_df.groupby(
            ["patient_id", "days_since_diagnosis"],
            sort=False,
        )["test_key"]
        .nunique()
        .rename("blood_test_type_count")
        .reset_index()
    )
    kept_days = day_counts.loc[
        (day_counts["days_since_diagnosis"] < min_landmark_day)
        | (day_counts["blood_test_type_count"] >= min_blood_test_types_per_day),
        ["patient_id", "days_since_diagnosis"],
    ]
    return measurement_df.merge(
        kept_days,
        on=["patient_id", "days_since_diagnosis"],
        how="inner",
    )


def filter_observation_window(
    measurement_df: pd.DataFrame,
    patients: Mapping[str, PatientRecord],
    statuses: Mapping[str, PatientEndpointStatus],
    *,
    include_end_day: bool = False,
) -> pd.DataFrame:
    if measurement_df.empty:
        return measurement_df
    mask = measurement_window_mask(
        measurement_df,
        {patient_id: statuses[patient_id] for patient_id in patients},
        include_end_day=include_end_day,
    )
    return measurement_df.loc[mask].copy()


def patient_observation_end(
    status: PatientEndpointStatus,
) -> int | None:
    return observation_end_day(
        None if status.input_recurrence_event is None else status.input_recurrence_event.days_since_diagnosis,
        status.right_censoring_day,
    )
