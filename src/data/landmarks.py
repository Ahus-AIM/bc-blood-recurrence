from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.data.blood_test_features import blood_test_feature_names
from src.data.blood_test_change import BLOOD_TEST_CHANGE_VERSION, blood_test_change_inputs
from src.data.cohort_selection import COHORT_SELECTION_VERSION
from src.data.dataset_config import PatientDatasetConfig
from src.data.observation_window import OBSERVATION_WINDOW_VERSION
from src.data.other_cancer_outcomes import recurrence_precedes_censoring
from src.data.recent_diagnoses import normalized_recent_diagnosis_code, recent_diagnosis_window_mask
from src.data.sample_weighting import (
    RowPlan,
    bootstrap_row_plan,
    landmark_sample_weights,
)
from src.data.timing import timed
from src.data.survival_targets import CENSORING_OFFSET_DAYS, ZERO_DURATION_DAYS, signed_survival_durations
from src.data.types import PatientBloodTestItem, PatientLandmarkTable
from src.time_units import DAYS_PER_YEAR

LOGGER = logging.getLogger(__name__)
TIME_SINCE_DIAGNOSIS_FEATURE_NAME = "time_since_diagnosis_years"


@timed(LOGGER, lambda dataset, **_kwargs: f"build {dataset.config.split} landmark table")
def build_patient_landmark_table(dataset: object, *, bootstrap: bool, bootstrap_run: int = 0) -> PatientLandmarkTable:
    config = dataset.config
    items = tuple(dataset.items)
    selected_test_keys = tuple(dataset.selected_test_keys)
    selected_recent_diagnosis_codes = tuple(dataset.selected_recent_diagnosis_codes)
    selected_recent_diagnosis_feature_names = tuple(dataset.selected_recent_diagnosis_feature_names)
    measurement_df = dataset.prepared_data.measurement_df
    log_landmark_input(config, items, selected_test_keys, selected_recent_diagnosis_codes, bootstrap)
    if bootstrap:
        row_plan, bootstrap_probabilities = bootstrap_row_plan(items, config, bootstrap_run)
    else:
        row_plan = training_row_plan(items)
        bootstrap_probabilities = None
    row_count = sum(len(days) for _item, _positions, days in row_plan)
    feature_names = landmark_feature_names(
        selected_test_keys,
        selected_recent_diagnosis_feature_names,
        config,
    )
    LOGGER.info("landmark rows=%d features=%d", row_count, len(feature_names))
    features = np.full((row_count, len(feature_names)), np.nan, dtype=np.float32)
    durations = np.empty(row_count, dtype=np.int32)
    events = np.empty(row_count, dtype=bool)
    patient_ids = np.empty(row_count, dtype=object)
    landmark_days = np.empty(row_count, dtype=np.int32)
    prior_non_endpoint = np.empty(row_count, dtype=bool)
    target_end_reasons = np.empty(row_count, dtype=object)
    right_censoring_days = np.empty(row_count, dtype=np.float64)
    right_censoring_reasons = np.empty(row_count, dtype=object)

    fill_landmark_metadata_rows(
        features,
        durations,
        events,
        patient_ids,
        landmark_days,
        prior_non_endpoint,
        target_end_reasons,
        right_censoring_days,
        right_censoring_reasons,
        row_plan,
        config,
    )
    fill_observed_value_features(
        features,
        measurement_df,
        patient_ids,
        landmark_days,
        selected_test_keys,
    )
    change_inputs = None
    if config.include_blood_test_change:
        change_inputs = blood_test_change_inputs(
            measurement_df, patient_ids, landmark_days, selected_test_keys, config,
        )
        test_count = len(selected_test_keys)
        features[:, test_count:2 * test_count] = features[:, :test_count] - change_inputs.past_values
    fill_recent_diagnosis_features(
        features,
        row_plan,
        selected_test_keys,
        selected_recent_diagnosis_codes,
        config,
    )
    fill_time_since_diagnosis_feature(
        features,
        landmark_days,
        selected_test_keys,
        selected_recent_diagnosis_codes,
        config,
    )

    weights, weight_interval_days = landmark_sample_weights(row_plan, row_count, config, bootstrap)
    signed_durations = signed_survival_durations(durations, events)
    return PatientLandmarkTable(
        features=features,
        signed_durations=signed_durations,
        durations=durations,
        events=events,
        sample_weights=weights,
        feature_names=feature_names,
        patient_ids=patient_ids,
        landmark_days=landmark_days,
        prior_non_endpoint_recurrence=prior_non_endpoint,
        metadata=landmark_metadata(dataset, row_count, int(events.sum()), bootstrap, bootstrap_run),
        sample_weight_interval_days=weight_interval_days,
        bootstrap_selection_probabilities=bootstrap_probabilities,
        target_end_reasons=target_end_reasons,
        right_censoring_days=right_censoring_days,
        right_censoring_reasons=right_censoring_reasons,
        blood_test_change_inputs=change_inputs,
    )


def log_landmark_input(
    config: PatientDatasetConfig,
    items: tuple[PatientBloodTestItem, ...],
    selected_test_keys: tuple[str, ...],
    selected_recent_diagnosis_codes: tuple[str, ...],
    bootstrap: bool,
) -> None:
    LOGGER.info(
        "landmark input split=%s patients=%d selected_tests=%d selected_recent_diagnoses=%d bootstrap=%s",
        config.split,
        len(items),
        len(selected_test_keys),
        len(selected_recent_diagnosis_codes),
        bootstrap,
    )


def training_row_plan(
    items: tuple[PatientBloodTestItem, ...],
) -> RowPlan:
    plans = []
    row_start = 0
    for item in items:
        days = np.asarray(item.valid_landmark_days, dtype=np.int32)
        positions = np.arange(row_start, row_start + len(days), dtype=np.int64)
        row_start += len(days)
        plans.append((item, positions, days))
    return tuple(plans)


def landmark_feature_names(
    selected_test_keys: tuple[str, ...],
    selected_recent_diagnosis_feature_names: tuple[str, ...],
    config: PatientDatasetConfig,
) -> tuple[str, ...]:
    names = list(blood_test_feature_names(selected_test_keys, config.include_blood_test_change))
    names.extend(selected_recent_diagnosis_feature_names)
    if config.include_time_since_diagnosis:
        names.append(TIME_SINCE_DIAGNOSIS_FEATURE_NAME)
    if config.include_diagnosis_age:
        names.append("age_at_diagnosis")
    return tuple(names)


def fill_landmark_metadata_rows(
    features: np.ndarray,
    durations: np.ndarray,
    events: np.ndarray,
    patient_ids: np.ndarray,
    landmark_days: np.ndarray,
    prior_non_endpoint: np.ndarray,
    target_end_reasons: np.ndarray,
    right_censoring_days: np.ndarray,
    right_censoring_reasons: np.ndarray,
    row_plan: tuple[tuple[PatientBloodTestItem, np.ndarray, np.ndarray], ...],
    config: PatientDatasetConfig,
) -> None:
    for item, positions, days in row_plan:
        duration_values, event_values, prior_values, end_reasons = patient_targets(item, days, config)
        durations[positions] = duration_values
        events[positions] = event_values
        patient_ids[positions] = item.patient_id
        landmark_days[positions] = days
        prior_non_endpoint[positions] = prior_values
        target_end_reasons[positions] = end_reasons
        right_censoring_days[positions] = np.nan if item.right_censoring_day is None else item.right_censoring_day
        right_censoring_reasons[positions] = item.right_censoring_reason
        if config.include_diagnosis_age:
            features[positions, -1] = item.age_at_diagnosis


def fill_observed_value_features(
    features: np.ndarray,
    measurement_df: pd.DataFrame,
    patient_ids: np.ndarray,
    landmark_days: np.ndarray,
    selected_test_keys: tuple[str, ...],
) -> None:
    if measurement_df.empty or len(patient_ids) == 0 or not selected_test_keys:
        return
    landmark_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "days_since_diagnosis": landmark_days,
            "row_pos": np.arange(len(patient_ids), dtype=np.int64),
        }
    )
    patient_id_set = frozenset(patient_ids.tolist())
    selected_df = selected_measurements_for_patients(measurement_df, patient_id_set)
    merged = landmark_df.merge(
        selected_df[["patient_id", "days_since_diagnosis", "test_key", "result"]],
        on=["patient_id", "days_since_diagnosis"],
        how="inner",
        sort=False,
    )
    if merged.empty:
        return
    test_ids = test_id_codes(merged["test_key"], selected_test_keys)
    valid = test_ids >= 0
    row_ids = merged.loc[valid, "row_pos"].to_numpy(dtype=np.int64)
    features[row_ids, test_ids[valid]] = merged.loc[valid, "result"].to_numpy(dtype=np.float32)


def selected_measurements_for_patients(
    measurement_df: pd.DataFrame,
    patient_id_set: frozenset[str],
) -> pd.DataFrame:
    selected_df = measurement_df[measurement_df["patient_id"].isin(patient_id_set)]
    return selected_df.drop_duplicates(
        ["patient_id", "days_since_diagnosis", "test_key"],
        keep="last",
    )


def fill_recent_diagnosis_features(
    features: np.ndarray,
    row_plan: RowPlan,
    selected_test_keys: tuple[str, ...],
    selected_recent_diagnosis_codes: tuple[str, ...],
    config: PatientDatasetConfig,
) -> None:
    if not config.include_recent_diagnoses or not selected_recent_diagnosis_codes:
        return
    feature_start = recent_diagnosis_feature_start(selected_test_keys, config)
    feature_end = feature_start + len(selected_recent_diagnosis_codes)
    features[:, feature_start:feature_end] = 0.0
    code_to_offset = {code: index for index, code in enumerate(selected_recent_diagnosis_codes)}
    for item, positions, days in row_plan:
        fill_item_recent_diagnosis_features(features, positions, days, item, code_to_offset, feature_start, config)


def recent_diagnosis_feature_start(
    selected_test_keys: tuple[str, ...],
    config: PatientDatasetConfig,
) -> int:
    change_feature_count = len(selected_test_keys) if config.include_blood_test_change else 0
    return len(selected_test_keys) + change_feature_count


def fill_item_recent_diagnosis_features(
    features: np.ndarray,
    positions: np.ndarray,
    landmark_days: np.ndarray,
    item: PatientBloodTestItem,
    code_to_offset: dict[str, int],
    feature_start: int,
    config: PatientDatasetConfig,
) -> None:
    if len(positions) == 0:
        return
    for event in item.clinical_events:
        code = normalized_recent_diagnosis_code(event)
        if code is None:
            continue
        offset = code_to_offset.get(code)
        if offset is None:
            continue
        matching_rows = recent_diagnosis_window_mask(landmark_days, event.days_since_diagnosis, config)
        if bool(matching_rows.any()):
            features[positions[matching_rows], feature_start + offset] = 1.0


def fill_time_since_diagnosis_feature(
    features: np.ndarray,
    landmark_days: np.ndarray,
    selected_test_keys: tuple[str, ...],
    selected_recent_diagnosis_codes: tuple[str, ...],
    config: PatientDatasetConfig,
) -> None:
    if not config.include_time_since_diagnosis:
        return
    feature_index = time_since_diagnosis_feature_index(
        selected_test_keys,
        selected_recent_diagnosis_codes,
        config,
    )
    features[:, feature_index] = landmark_days.astype(np.float32) / DAYS_PER_YEAR


def time_since_diagnosis_feature_index(
    selected_test_keys: tuple[str, ...],
    selected_recent_diagnosis_codes: tuple[str, ...],
    config: PatientDatasetConfig,
) -> int:
    return recent_diagnosis_feature_start(selected_test_keys, config) + len(selected_recent_diagnosis_codes)


def test_id_codes(test_keys: pd.Series, selected_test_keys: tuple[str, ...]) -> np.ndarray:
    return pd.Categorical(test_keys, categories=selected_test_keys).codes.astype(np.int16)


def patient_targets(
    item: PatientBloodTestItem,
    landmark_days: np.ndarray,
    config: PatientDatasetConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build landmark targets and their event or censoring reasons."""
    horizon_days = landmark_days + config.censoring_horizon_days
    censor_days = horizon_days.copy()
    censor_reasons = np.full(len(landmark_days), "censoring_horizon", dtype=object)
    if item.right_censoring_day is not None:
        use_right_censoring = item.right_censoring_day <= horizon_days
        censor_days[use_right_censoring] = item.right_censoring_day
        reason = "last_observation" if item.right_censoring_reason is None else item.right_censoring_reason
        censor_reasons[use_right_censoring] = reason
    endpoint_day = item.endpoint_day
    event_observed = np.zeros(len(landmark_days), dtype=bool)
    if endpoint_day is not None and recurrence_precedes_censoring(
        endpoint_day,
        item.right_censoring_day,
        item.right_censoring_reason,
    ):
        after_landmark = (
            endpoint_day >= landmark_days
            if config.include_recurrence_day_blood_tests
            else endpoint_day > landmark_days
        )
        event_observed = after_landmark & (endpoint_day <= horizon_days)
    end_days = np.where(event_observed, endpoint_day, censor_days) if endpoint_day is not None else censor_days
    end_reasons = np.where(event_observed, "configured_recurrence", censor_reasons)
    non_endpoint_days = np.asarray(
        sorted(
            event.days_since_diagnosis
            for event in item.non_endpoint_recurrence_events
            if event.days_since_diagnosis != endpoint_day
        ),
        dtype=np.int32,
    )
    prior_non_endpoint = np.searchsorted(non_endpoint_days, landmark_days, side="left") > 0
    return (
        np.maximum(0, end_days - landmark_days).astype(np.int32),
        event_observed,
        prior_non_endpoint,
        end_reasons,
    )


def landmark_metadata(
    dataset: object,
    row_count: int,
    event_count: int,
    bootstrap: bool,
    bootstrap_run: int,
) -> dict[str, object]:
    return {
        "split": dataset.config.split,
        "row_count": row_count,
        "event_count": event_count,
        "recurrence_endpoint_name": dataset.config.recurrence_endpoint_name,
        "input_recurrence_code_filter_regex": dataset.config.input_recurrence_code_filter_regex,
        "endpoint_ddfs_code_filter_regex": dataset.config.endpoint_ddfs_code_filter_regex,
        "censoring_horizon_days": dataset.config.censoring_horizon_days,
        "min_landmark_day": dataset.config.min_landmark_day,
        "cohort_selection_version": COHORT_SELECTION_VERSION,
        "include_recurrence_day_blood_tests": dataset.config.include_recurrence_day_blood_tests,
        "observation_window_version": OBSERVATION_WINDOW_VERSION,
        "zero_duration_model_days": ZERO_DURATION_DAYS,
        "censoring_offset_model_days": CENSORING_OFFSET_DAYS,
        "min_blood_test_types_per_day": dataset.config.min_blood_test_types_per_day,
        "min_diagnosis_year": dataset.config.min_diagnosis_year,
        "training_sample_weighting": dataset.config.training_sample_weighting,
        "bootstrap_sample_weighting": dataset.config.bootstrap_sample_weighting,
        "include_diagnosis_age": dataset.config.include_diagnosis_age,
        "require_diagnosis_age": dataset.config.require_diagnosis_age,
        "bc_diagnosis_patient_age_csv_path": (
            None
            if dataset.config.bc_diagnosis_patient_age_csv_path is None
            else str(dataset.config.bc_diagnosis_patient_age_csv_path)
        ),
        "exclude_prior_bc_surgery_patients": dataset.config.exclude_prior_bc_surgery_patients,
        "prior_bc_surgery_code_regex": dataset.config.prior_bc_surgery_code_regex,
        "exclude_and_censor_other_cancer_patients": (dataset.config.exclude_and_censor_other_cancer_patients),
        "other_cancer_code_filter_regex": (dataset.config.other_cancer_code_filter_regex),
        "restrict_bloodtests_to_bc_diagnosis_window": (dataset.config.restrict_bloodtests_to_bc_diagnosis_window),
        "bc_diagnosis_days_csv_path": (
            None
            if dataset.config.bc_diagnosis_days_csv_path is None
            else str(dataset.config.bc_diagnosis_days_csv_path)
        ),
        "bc_diagnosis_window_days_before": dataset.config.bc_diagnosis_window_days_before,
        "bc_diagnosis_window_days_after": dataset.config.bc_diagnosis_window_days_after,
        "bc_diagnosis_window_filter_counts": dataset.prepared_data.bc_diagnosis_window_filter_counts,
        "diagnosis_procedure_csv_path": (
            None
            if dataset.config.diagnosis_procedure_csv_path is None
            else str(dataset.config.diagnosis_procedure_csv_path)
        ),
        "num_most_common_blood_tests": dataset.config.num_most_common_blood_tests,
        "selected_blood_tests": dataset.config.selected_blood_tests,
        "include_blood_test_change": dataset.config.include_blood_test_change,
        "blood_test_change_version": BLOOD_TEST_CHANGE_VERSION,
        "blood_test_change_min_days": dataset.config.blood_test_change_min_days,
        "blood_test_change_max_days": dataset.config.blood_test_change_max_days,
        "include_recent_diagnoses": dataset.config.include_recent_diagnoses,
        "num_most_common_recent_diagnoses": dataset.config.num_most_common_recent_diagnoses,
        "recent_diagnosis_lookback_days": dataset.config.recent_diagnosis_lookback_days,
        "include_landmark_day_recent_diagnoses": dataset.config.include_landmark_day_recent_diagnoses,
        "selected_recent_diagnosis_codes": dataset.prepared_data.selected_recent_diagnosis_codes,
        "selected_recent_diagnosis_feature_names": dataset.prepared_data.selected_recent_diagnosis_feature_names,
        "recent_diagnosis_descriptions": dataset.prepared_data.recent_diagnosis_descriptions,
        "recent_diagnosis_train_patient_counts": dataset.prepared_data.recent_diagnosis_train_patient_counts,
        "include_time_since_diagnosis": dataset.config.include_time_since_diagnosis,
        "bootstrap": bootstrap,
        "bootstrap_run": bootstrap_run if bootstrap else None,
        "input_fingerprint": dataset.prepared_data.input_fingerprint,
        "dropped_bloodtest_counts": dataset.prepared_data.dropped_bloodtest_counts,
        "dropped_clinical_event_counts": dataset.prepared_data.dropped_clinical_event_counts,
    }
