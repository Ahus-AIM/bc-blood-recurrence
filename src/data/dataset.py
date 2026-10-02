from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import torch

from src.data.bc_diagnosis_window import load_bc_diagnosis_days
from src.data.cohort_comparison_data import build_cohort_comparison_patients
from src.data.blood_test_change import BLOOD_TEST_CHANGE_VERSION
from src.data.cohort_selection import (
    COHORT_SELECTION_VERSION,
    OTHER_CANCER_DISCARD_REASONS,
    initial_cohort_patient_ids,
    patient_observation_end,
    select_patient_cohort,
)
from src.data.dataset_config import PatientDatasetConfig
from src.data.dataset_logging import (
    log_filtered_blood_test_counts,
    log_other_cancer_exclusion_counts,
)
from src.data.endpoint_logging import log_endpoint_counts
from src.data.icd_codes import matches_icd_code
from src.data.temporal_splits import validate_final_cohort_splits
from src.data.landmarks import build_patient_landmark_table
from src.data.observation_window import OBSERVATION_WINDOW_VERSION, measurement_window_mask
from src.data.other_cancer_outcomes import (
    is_other_cancer_diagnosis,
    normalize_event_dates,
    resolve_endpoint_status,
)
from src.data.patient_parsing import (
    empty_clinical_drop_counts,
    load_age_by_patient,
    load_clinical_events,
    load_endpoint_patients,
    load_split_assignments,
)
from src.data.patient_preprocessing import load_blood_test_source
from src.data.recent_diagnoses import select_recent_diagnosis_features
from src.data.timing import timed
from src.data.types import (
    ItemBuildContext,
    PatientBloodTestItem,
    PatientClinicalEvent,
    PatientDataSources,
    PatientEndpointEvent,
    PatientEndpointStatus,
    PatientMeasurementSummary,
    PatientMeasurementViews,
    PatientRecord,
    PreparedPatientData,
)

LOGGER = logging.getLogger(__name__)
MODEL_SPLITS = ("train", "val", "test")

DISCARD_NO_VALID_LANDMARK_DAYS = "no valid landmark days"
DISTANT_RECURRENCE_CODE_REGEX = re.compile(r"(?:C78\d*)|(?:C79\d*)")


class PatientBloodTestDataset:
    """Split-specific patient timeline dataset backed by shared prepared data."""

    def __init__(
        self,
        config: Mapping[str, object] | PatientDatasetConfig,
        *,
        prepared_data: PreparedPatientData | None = None,
        data_sources: PatientDataSources | None = None,
    ) -> None:
        self.config = config if isinstance(config, PatientDatasetConfig) else PatientDatasetConfig.from_mapping(config)
        if prepared_data is not None and data_sources is not None:
            raise ValueError("Pass either `prepared_data` or `data_sources`, not both.")
        self.prepared_data = (
            prepared_data if prepared_data is not None else prepare_patient_data(self.config, data_sources=data_sources)
        )
        self.selected_test_keys = self.prepared_data.selected_test_keys
        self.selected_recent_diagnosis_codes = self.prepared_data.selected_recent_diagnosis_codes
        self.selected_recent_diagnosis_feature_names = self.prepared_data.selected_recent_diagnosis_feature_names
        self.items = self.prepared_data.items_by_split.get(self.config.split, ())
        self.patient_ids = tuple(item.patient_id for item in self.items)
        self.discarded_patient_items_by_id = self.prepared_data.discarded_patient_items_by_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> PatientBloodTestItem:
        return self.items[index]

    def to_landmark_table(self, *, bootstrap: bool | None = None, bootstrap_run: int = 0):
        use_bootstrap = self.config.split in {"val", "test"} if bootstrap is None else bootstrap
        return build_patient_landmark_table(self, bootstrap=use_bootstrap, bootstrap_run=bootstrap_run)


@timed(LOGGER, "prepare patient data")
def prepare_patient_data(
    config: PatientDatasetConfig,
    *,
    data_sources: PatientDataSources | None = None,
) -> PreparedPatientData:
    """Build endpoint-specific patient data from loaded or configured sources."""
    sources = load_patient_data_sources(config) if data_sources is None else data_sources
    if config.require_diagnosis_age:
        missing_age_count = len(sources.endpoint_patients) - len(initial_cohort_patient_ids(sources, config))
        LOGGER.info("patients removed for missing diagnosis age: %d", missing_age_count)
    status_by_patient = build_endpoint_status(sources, config)
    selected_test_keys = select_train_test_keys(sources, config)
    LOGGER.info("selected %d blood-test types", len(selected_test_keys))
    measurement_df = selected_measurement_frame(sources.blood_df, selected_test_keys, config)
    LOGGER.info("retained %d selected blood-test rows", len(measurement_df))
    cohort = select_patient_cohort(
        measurement_df,
        sources,
        status_by_patient,
        config,
    )
    measurement_df = cohort.measurement_df
    validate_final_measurement_boundaries(
        measurement_df, status_by_patient,
        include_end_day=config.include_recurrence_day_blood_tests,
    )
    status_by_patient = statuses_with_discard_reasons(
        status_by_patient,
        cohort.discard_reasons_by_patient,
    )
    items_by_split, discarded = build_items(
        item_build_context(config, sources, status_by_patient, measurement_df, selected_test_keys)
    )
    excluded_measurement_df = eligible_excluded_measurements(
        cohort.excluded_measurement_df,
        items_by_split,
    )
    if uses_filtered_plot_percentiles(config):
        reference_df = included_measurement_frame(measurement_df, items_by_split)
        measurement_df = measurement_frame_with_reference_plot_values(measurement_df, reference_df)
        excluded_measurement_df = measurement_frame_with_reference_plot_values(
            excluded_measurement_df,
            reference_df,
        )
        items_by_split, discarded = build_items(
            item_build_context(config, sources, status_by_patient, measurement_df, selected_test_keys)
        )
    if config.exclude_and_censor_other_cancer_patients:
        log_other_cancer_exclusion_counts(
            LOGGER,
            (
                item
                for item in discarded.values()
                if not OTHER_CANCER_DISCARD_REASONS.isdisjoint(item.discard_reasons)
            ),
            re.compile(config.other_cancer_code_filter_regex),
            config.min_landmark_day,
        )
    filtered_measurements = included_measurement_frame(measurement_df, items_by_split)
    log_filtered_blood_test_counts(LOGGER, filtered_measurements["test_key"], selected_test_keys)
    validate_final_cohort_outcomes(items_by_split, status_by_patient, sources.endpoint_patients)
    validate_final_cohort_splits(items_by_split, config)
    recent_diagnosis_features = select_recent_diagnosis_features(items_by_split.get("train", ()), config)
    LOGGER.info("selected %d recent diagnosis codes", len(recent_diagnosis_features.codes))
    LOGGER.info(
        "eligible patients: train=%d val=%d test=%d discarded=%d",
        len(items_by_split.get("train", ())),
        len(items_by_split.get("val", ())),
        len(items_by_split.get("test", ())),
        len(discarded),
    )
    log_endpoint_counts(LOGGER, items_by_split, sources.endpoint_patients, config)
    return PreparedPatientData(
        selected_test_keys=selected_test_keys,
        selected_recent_diagnosis_codes=recent_diagnosis_features.codes,
        selected_recent_diagnosis_feature_names=recent_diagnosis_features.feature_names,
        recent_diagnosis_descriptions=recent_diagnosis_features.descriptions,
        recent_diagnosis_train_patient_counts=recent_diagnosis_features.train_patient_counts,
        status_by_patient=status_by_patient,
        items_by_split=items_by_split,
        discarded_patient_items_by_id=discarded,
        measurement_df=measurement_df,
        excluded_measurement_df=excluded_measurement_df,
        dropped_bloodtest_counts=sources.dropped_bloodtest_counts,
        dropped_clinical_event_counts=sources.dropped_clinical_event_counts,
        bc_diagnosis_window_filter_counts=cohort.bc_diagnosis_window_filter_counts,
        cohort_stages=cohort.stages,
        input_fingerprint=input_fingerprint(config),
        recurrence_history=cohort.recurrence_history,
        cohort_comparison_patients=build_cohort_comparison_patients(sources, config, selected_test_keys),
    )


def load_patient_data_sources(config: PatientDatasetConfig) -> PatientDataSources:
    endpoint_patients = load_endpoint_patients(config.endpoint_csv_path)
    LOGGER.info("loaded %d endpoint patients", len(endpoint_patients))
    blood_source = load_blood_test_source(config.bloodtest_csv_path)
    blood_df, dropped_counts = blood_source.measurements, blood_source.dropped_counts
    LOGGER.info("loaded %d clean deduplicated blood-test rows", len(blood_df))
    LOGGER.info("dropped blood-test counts: %s", dropped_counts)
    if not config.split_csv_path.exists():
        raise FileNotFoundError(f"Split CSV not found: {config.split_csv_path}")
    split_assignments = load_split_assignments(config.split_csv_path)
    split_by_patient = {patient_id: split for split, ids in split_assignments.items() for patient_id in ids}
    age_by_patient = load_age_by_patient(config.bc_diagnosis_patient_age_csv_path)
    clinical_events_by_patient, dropped_clinical_counts = load_clinical_events_for_config(config)
    LOGGER.info("loaded %d clinical-event patients", len(clinical_events_by_patient))
    LOGGER.info("dropped clinical-event counts: %s", dropped_clinical_counts)
    bc_diagnosis_events_by_patient = load_bc_diagnosis_events_by_patient(
        config.bc_diagnosis_days_csv_path,
        require_existing=config.exclude_prior_bc_surgery_patients,
    )
    LOGGER.info("loaded %d BC diagnosis-event patients", len(bc_diagnosis_events_by_patient))
    return PatientDataSources(
        endpoint_patients=endpoint_patients,
        blood_df=blood_df,
        split_by_patient=split_by_patient,
        age_by_patient=age_by_patient,
        clinical_events_by_patient=clinical_events_by_patient,
        bc_diagnosis_events_by_patient=bc_diagnosis_events_by_patient,
        dropped_bloodtest_counts=dropped_counts,
        dropped_clinical_event_counts=dropped_clinical_counts,
    )


def validate_final_measurement_boundaries(
    measurement_df: pd.DataFrame,
    status_by_patient: Mapping[str, PatientEndpointStatus],
    *,
    include_end_day: bool = False,
) -> None:
    """Require retained measurements to satisfy the shared observation-window rule."""
    if measurement_df.empty:
        return
    invalid_mask = ~measurement_window_mask(
        measurement_df, status_by_patient, include_end_day=include_end_day,
    )
    if not invalid_mask.any():
        return
    invalid_index = invalid_mask[invalid_mask].index[0]
    invalid_row = measurement_df.loc[invalid_index]
    end_day = patient_observation_end(status_by_patient[str(invalid_row["patient_id"])])
    raise ValueError(
        f"Retained blood test for patient {invalid_row['patient_id']} on day "
        f"{int(invalid_row['days_since_diagnosis'])} violates observation endpoint day "
        f"{end_day} (include_end_day={include_end_day})."
    )


def load_clinical_events_for_config(
    config: PatientDatasetConfig,
) -> tuple[dict[str, tuple[PatientClinicalEvent, ...]], dict[str, int]]:
    csv_path = config.diagnosis_procedure_csv_path
    if csv_path is None:
        return {}, empty_clinical_drop_counts()
    if csv_path.exists():
        return load_clinical_events(csv_path)
    if (
        config.include_recent_diagnoses
        or config.exclude_prior_bc_surgery_patients
        or config.exclude_and_censor_other_cancer_patients
    ):
        raise FileNotFoundError(f"Clinical diagnosis/procedure CSV not found: {csv_path}")
    LOGGER.warning("Clinical diagnosis/procedure CSV not found for timeline metadata: %s", csv_path)
    return {}, empty_clinical_drop_counts()


def load_bc_diagnosis_events_by_patient(
    csv_path: Path | None,
    *,
    require_existing: bool = False,
) -> dict[str, tuple[PatientClinicalEvent, ...]]:
    """Load extracted C50 diagnosis days as timeline-only clinical events."""
    if csv_path is None:
        return {}
    if not csv_path.exists():
        if require_existing:
            raise FileNotFoundError(f"BC diagnosis-days CSV not found: {csv_path}")
        LOGGER.warning("BC diagnosis-days CSV not found for timeline metadata: %s", csv_path)
        return {}
    diagnosis_df = load_bc_diagnosis_days(csv_path)
    grouped: dict[str, list[PatientClinicalEvent]] = {}
    for row in diagnosis_df.to_dict("records"):
        patient_id = str(row["patient_id"])
        grouped.setdefault(patient_id, []).append(
            PatientClinicalEvent(
                days_since_diagnosis=int(row["bc_diagnosis_day"]),
                event_type="bc_diagnosis",
                code=str(row["code"]),
                description=str(row["description"]),
            )
        )
    return {
        patient_id: tuple(sorted(events, key=lambda event: (event.days_since_diagnosis, event.code, event.description)))
        for patient_id, events in grouped.items()
    }


@timed(LOGGER, "build patient eligibility status")
def build_endpoint_status(
    sources: PatientDataSources,
    config: PatientDatasetConfig,
) -> dict[str, PatientEndpointStatus]:
    input_recurrence_regex = re.compile(config.input_recurrence_code_filter_regex)
    endpoint_regex = re.compile(config.endpoint_ddfs_code_filter_regex)
    other_cancer_regex = (
        re.compile(config.other_cancer_code_filter_regex) if config.exclude_and_censor_other_cancer_patients else None
    )
    return {
        patient_id: resolve_endpoint_status(
            patient.recurrence_events,
            sources.clinical_events_by_patient.get(patient_id, ()),
            input_recurrence_regex,
            endpoint_regex,
            other_cancer_regex,
            death_day=patient.death_day,
            last_observation_day=patient.last_observation_day,
        )
        for patient_id, patient in sources.endpoint_patients.items()
    }


def statuses_with_discard_reasons(
    statuses: Mapping[str, PatientEndpointStatus],
    discard_reasons: Mapping[str, str],
) -> dict[str, PatientEndpointStatus]:
    return {
        patient_id: replace(status, discard_reason=discard_reasons.get(patient_id))
        for patient_id, status in statuses.items()
    }


def validate_final_cohort_outcomes(
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]],
    status_by_patient: Mapping[str, PatientEndpointStatus],
    endpoint_patients: Mapping[str, PatientRecord],
) -> None:
    """Require source follow-up and consistent outcome dates in the final cohort."""
    patient_ids = tuple(item.patient_id for split in MODEL_SPLITS for item in items_by_split.get(split, ()))
    missing_outcome = tuple(
        patient_id
        for patient_id in patient_ids
        if not patient_has_outcome_time(
            status_by_patient[patient_id],
        )
    )
    if missing_outcome:
        raise_missing_patient_data("outcome time", missing_outcome)
    for patient_id in patient_ids:
        patient = endpoint_patients[patient_id]
        end_days = (patient.last_observation_day, patient.death_day)
        if any(day is not None and day < 0 for day in end_days):
            raise ValueError(
                f"Patient {patient_id} has an observation end before BC diagnosis "
                f"(last_observation_day={patient.last_observation_day}, death_day={patient.death_day})."
            )
        if patient.last_observation_day is None:
            raise ValueError(f"Patient {patient_id} is missing source last observation.")
        recurrence = status_by_patient[patient_id].input_recurrence_event
        if recurrence is not None and recurrence.days_since_diagnosis > patient.last_observation_day:
            raise ValueError(
                f"Patient {patient_id} has first input recurrence on day {recurrence.days_since_diagnosis} "
                f"after source last observation on day {patient.last_observation_day}."
            )


def patient_has_outcome_time(
    status: PatientEndpointStatus,
) -> bool:
    return any(
        value is not None
        for value in (
            status.input_recurrence_event,
            status.right_censoring_day,
        )
    )


def raise_missing_patient_data(label: str, patient_ids: tuple[str, ...]) -> None:
    examples = ", ".join(patient_ids[:10])
    raise ValueError(f"{len(patient_ids)} otherwise eligible patients are missing required {label}: {examples}")


def distant_recurrence_events(
    events: tuple[PatientEndpointEvent, ...],
) -> tuple[PatientEndpointEvent, ...]:
    return tuple(event for event in events if matches_icd_code(event.code, DISTANT_RECURRENCE_CODE_REGEX))


@timed(LOGGER, "select train-fold blood-test types")
def select_train_test_keys(
    sources: PatientDataSources,
    config: PatientDatasetConfig,
) -> tuple[str, ...]:
    eligible_patient_ids = initial_cohort_patient_ids(sources, config)
    eligible_blood_df = sources.blood_df.loc[sources.blood_df["patient_id"].isin(eligible_patient_ids)]
    if config.selected_blood_tests:
        return configured_test_keys(eligible_blood_df, config.selected_blood_tests)
    train_patient_ids = {
        patient_id
        for patient_id, split in sources.split_by_patient.items()
        if split == "train" and patient_id in eligible_patient_ids
    }
    train_blood_df = eligible_blood_df.loc[eligible_blood_df["patient_id"].isin(train_patient_ids)]
    counts = train_blood_df.groupby("test_key", sort=True).size().sort_values(ascending=False, kind="mergesort")
    return tuple(str(test_key) for test_key in counts.head(config.num_most_common_blood_tests).index)


def configured_test_keys(
    blood_df: pd.DataFrame,
    selected_blood_tests: tuple[str, ...],
) -> tuple[str, ...]:
    available = frozenset(str(test_key) for test_key in blood_df["test_key"].unique())
    missing = tuple(test_key for test_key in selected_blood_tests if test_key not in available)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"`selected_blood_tests` contains unknown blood-test keys: {joined}")
    return selected_blood_tests


@timed(LOGGER, "filter selected blood-test measurements")
def selected_measurement_frame(
    blood_df: pd.DataFrame,
    selected_test_keys: tuple[str, ...],
    config: PatientDatasetConfig,
) -> pd.DataFrame:
    columns = ["patient_id", "days_since_diagnosis", "test_key", "result", "source_index"]
    measurement_df = blood_df.loc[blood_df["test_key"].isin(selected_test_keys), columns].copy()
    if not config.build_patient_timeline_tensors:
        measurement_df["plot_value"] = np.nan
    elif config.set_all_bloodtest_values_to_one:
        measurement_df["plot_value"] = 1.0
    else:
        measurement_df["plot_value"] = percentile_plot_values(measurement_df)
    return measurement_df.sort_values(
        ["patient_id", "days_since_diagnosis", "test_key", "source_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def uses_filtered_plot_percentiles(config: PatientDatasetConfig) -> bool:
    return config.build_patient_timeline_tensors and not config.set_all_bloodtest_values_to_one


def measurement_frame_with_reference_plot_values(
    measurement_df: pd.DataFrame,
    reference_df: pd.DataFrame,
) -> pd.DataFrame:
    measurement_df = measurement_df.copy()
    measurement_df["plot_value"] = percentile_plot_values(measurement_df, reference_df)
    return measurement_df


def eligible_excluded_measurements(
    excluded_measurement_df: pd.DataFrame,
    items_by_split: dict[str, tuple[PatientBloodTestItem, ...]],
) -> pd.DataFrame:
    patient_ids = {item.patient_id for items in items_by_split.values() for item in items}
    eligible_df = excluded_measurement_df.loc[excluded_measurement_df["patient_id"].isin(patient_ids)].copy()
    return eligible_df.sort_values(
        ["patient_id", "days_since_diagnosis", "test_key", "source_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def included_measurement_frame(
    measurement_df: pd.DataFrame,
    items_by_split: dict[str, tuple[PatientBloodTestItem, ...]],
) -> pd.DataFrame:
    valid_day_records = [
        (item.patient_id, day)
        for items in items_by_split.values()
        for item in items
        for day in item.valid_landmark_days
    ]
    if not valid_day_records:
        return measurement_df.head(0)
    valid_days = pd.DataFrame.from_records(
        valid_day_records,
        columns=["patient_id", "days_since_diagnosis"],
    ).drop_duplicates()
    return measurement_df.merge(
        valid_days,
        on=["patient_id", "days_since_diagnosis"],
        how="inner",
    )


def percentile_plot_values(
    measurement_df: pd.DataFrame,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    if measurement_df.empty:
        return pd.Series(dtype=float, index=measurement_df.index)
    reference_df = measurement_df if reference_df is None else reference_df
    percentiles = pd.Series(np.nan, index=measurement_df.index, dtype=float)
    for test_key, test_df in measurement_df.groupby("test_key", sort=False):
        reference_values = np.sort(reference_df.loc[reference_df["test_key"] == test_key, "result"].to_numpy())
        if len(reference_values) == 0:
            continue
        values = test_df["result"].to_numpy()
        percentiles.loc[test_df.index] = np.searchsorted(
            reference_values,
            values,
            side="right",
        ) / float(len(reference_values))
    return percentiles


def item_build_context(
    config: PatientDatasetConfig,
    sources: PatientDataSources,
    status_by_patient: dict[str, PatientEndpointStatus],
    measurement_df: pd.DataFrame,
    selected_test_keys: tuple[str, ...],
) -> ItemBuildContext:
    return ItemBuildContext(
        config=config,
        sources=sources,
        status_by_patient=status_by_patient,
        measurements=patient_measurement_views(measurement_df, config),
        test_key_to_id={test_key: index for index, test_key in enumerate(selected_test_keys)},
    )


@timed(LOGGER, "build patient timeline items")
def build_items(
    context: ItemBuildContext,
) -> tuple[dict[str, tuple[PatientBloodTestItem, ...]], dict[str, PatientBloodTestItem]]:
    item_lists: dict[str, list[PatientBloodTestItem]] = {"train": [], "val": [], "test": []}
    discarded: dict[str, PatientBloodTestItem] = {}
    for patient_id, patient in context.sources.endpoint_patients.items():
        if patient_id not in context.sources.split_by_patient:
            continue
        item = build_item(context, patient_id, patient)
        if item.discard_reasons:
            discarded[patient_id] = item
        else:
            item_lists.setdefault(item.split, []).append(item)
    items_by_split = {
        split: tuple(sorted(items, key=lambda item: (item.diagnosis_year, item.patient_id)))
        for split, items in item_lists.items()
    }
    return items_by_split, discarded


def build_item(
    context: ItemBuildContext,
    patient_id: str,
    patient: PatientRecord,
) -> PatientBloodTestItem:
    status = context.status_by_patient[patient_id]
    events = patient.recurrence_events
    selected_df = patient_measurement_df(context, patient_id)
    measurement_summary = context.measurements.summaries.get(patient_id)
    discard_reason = status.discard_reason
    endpoint_event = status.endpoint_event
    input_recurrence_event = status.input_recurrence_event
    input_recurrence_regex = re.compile(context.config.input_recurrence_code_filter_regex)
    displayed_recurrence_events = normalize_event_dates(
        events,
        patient.death_day,
        lambda event: matches_icd_code(event.code, input_recurrence_regex),
    )
    clinical_events = context.sources.clinical_events_by_patient.get(patient_id, ())
    if context.config.exclude_and_censor_other_cancer_patients:
        other_cancer_regex = re.compile(context.config.other_cancer_code_filter_regex)
        clinical_events = normalize_event_dates(
            clinical_events,
            patient.death_day,
            lambda event: is_other_cancer_diagnosis(event, other_cancer_regex),
        )
    valid_landmark_days = ()
    if discard_reason is None:
        valid_landmark_days = valid_days(selected_df, measurement_summary, context.config.min_landmark_day)
        if not valid_landmark_days:
            discard_reason = DISCARD_NO_VALID_LANDMARK_DAYS
    return PatientBloodTestItem(
        patient_id=patient_id,
        split=context.sources.split_by_patient[patient_id],
        diagnosis_year=patient.diagnosis_year,
        age_at_diagnosis=context.sources.age_by_patient.get(patient_id, math.nan),
        blood_days_since_diagnosis=timeline_tensor(selected_df, "days_since_diagnosis", torch.long, context.config),
        blood_test_type_ids=test_type_tensor(selected_df, context.test_key_to_id, context.config),
        blood_values=timeline_tensor(selected_df, "plot_value", torch.float32, context.config),
        input_recurrence_day=(None if input_recurrence_event is None else input_recurrence_event.days_since_diagnosis),
        endpoint_day=None if endpoint_event is None else endpoint_event.days_since_diagnosis,
        death_day=patient.death_day,
        right_censoring_day=status.right_censoring_day,
        source_last_observation_day=patient.last_observation_day,
        last_observation_day=display_last_observation_day(
            patient,
            displayed_recurrence_events,
            selected_df,
            measurement_summary,
        ),
        distant_recurrence_events=distant_recurrence_events(displayed_recurrence_events),
        non_endpoint_recurrence_events=status.non_endpoint_events,
        bc_diagnosis_events=context.sources.bc_diagnosis_events_by_patient.get(patient_id, ()),
        clinical_events=clinical_events,
        discard_reasons=() if discard_reason is None else (discard_reason,),
        valid_landmark_days=valid_landmark_days,
        right_censoring_reason=status.right_censoring_reason,
    )


def patient_measurement_df(context: ItemBuildContext, patient_id: str) -> pd.DataFrame:
    measurement_df = context.measurements.groups.get(patient_id)
    return empty_measurement_frame() if measurement_df is None else measurement_df


def patient_measurement_views(
    measurement_df: pd.DataFrame,
    config: PatientDatasetConfig,
) -> PatientMeasurementViews:
    if measurement_df.empty:
        return PatientMeasurementViews({}, {})
    if config.build_patient_timeline_tensors:
        groups = {patient_id: group for patient_id, group in measurement_df.groupby("patient_id", sort=False)}
        return PatientMeasurementViews(groups, {})
    day_df = measurement_df[["patient_id", "days_since_diagnosis"]]
    max_day_by_patient = day_df.groupby("patient_id", sort=False)["days_since_diagnosis"].max().to_dict()
    day_count_df = (
        measurement_df.groupby(["patient_id", "days_since_diagnosis"], sort=False)["test_key"]
        .nunique()
        .reset_index(name="test_count")
        .sort_values(["patient_id", "days_since_diagnosis"], kind="mergesort")
    )
    post_days_by_patient = {
        str(patient_id): tuple(int(day) for day in group["days_since_diagnosis"].to_numpy())
        for patient_id, group in day_count_df.groupby("patient_id", sort=False)
    }
    test_counts_by_patient = {
        str(patient_id): {int(row.days_since_diagnosis): int(row.test_count) for row in group.itertuples(index=False)}
        for patient_id, group in day_count_df.groupby("patient_id", sort=False)
    }
    summaries = {
        str(patient_id): PatientMeasurementSummary(
            post_diagnosis_days=post_days_by_patient.get(str(patient_id), ()),
            test_counts_by_day=test_counts_by_patient.get(str(patient_id), {}),
            max_day=int(max_day),
        )
        for patient_id, max_day in max_day_by_patient.items()
    }
    return PatientMeasurementViews({}, summaries)


def valid_days(
    selected_df: pd.DataFrame,
    measurement_summary: PatientMeasurementSummary | None,
    min_landmark_day: int,
) -> tuple[int, ...]:
    day_counts = selected_day_test_counts(selected_df, measurement_summary)
    return tuple(sorted(day for day in day_counts if day >= min_landmark_day))


def selected_day_test_counts(
    selected_df: pd.DataFrame,
    measurement_summary: PatientMeasurementSummary | None,
) -> dict[int, int]:
    if measurement_summary is not None:
        return dict(measurement_summary.test_counts_by_day)
    if selected_df.empty:
        return {}
    counts = selected_df.groupby("days_since_diagnosis", sort=True)["test_key"].nunique()
    return {int(day): int(count) for day, count in counts.items()}


def empty_measurement_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=["patient_id", "days_since_diagnosis", "test_key", "result", "source_index", "plot_value"]
    )


def timeline_tensor(
    df: pd.DataFrame,
    column: str,
    dtype: torch.dtype,
    config: PatientDatasetConfig,
) -> torch.Tensor:
    if df.empty or not config.build_patient_timeline_tensors:
        return torch.empty(0, dtype=dtype)
    return torch.as_tensor(df[column].to_numpy(), dtype=dtype)


def test_type_tensor(
    df: pd.DataFrame,
    test_key_to_id: dict[str, int],
    config: PatientDatasetConfig,
) -> torch.Tensor:
    if df.empty or not config.build_patient_timeline_tensors:
        return torch.empty(0, dtype=torch.long)
    ids = df["test_key"].map(test_key_to_id).to_numpy()
    return torch.as_tensor(ids, dtype=torch.long)


def display_last_observation_day(
    patient: PatientRecord,
    events: tuple[PatientEndpointEvent, ...],
    selected_df: pd.DataFrame,
    measurement_summary: PatientMeasurementSummary | None,
) -> int:
    days = [0]
    days.extend(day for day in (patient.last_observation_day, patient.death_day) if day is not None)
    days.extend(event.days_since_diagnosis for event in events)
    if measurement_summary is not None and measurement_summary.max_day is not None:
        days.append(measurement_summary.max_day)
    elif not selected_df.empty:
        days.append(int(selected_df["days_since_diagnosis"].max()))
    return int(max(days))


def input_fingerprint(config: PatientDatasetConfig) -> str:
    payload = {
        "cohort_selection_version": COHORT_SELECTION_VERSION,
        "observation_window_version": OBSERVATION_WINDOW_VERSION,
        "endpoint": file_fingerprint(config.endpoint_csv_path),
        "bloodtest": file_fingerprint(config.bloodtest_csv_path),
        "split": file_fingerprint(config.split_csv_path),
        "input_recurrence_regex": config.input_recurrence_code_filter_regex,
        "endpoint_regex": config.endpoint_ddfs_code_filter_regex,
        "min_landmark_day": config.min_landmark_day,
        "include_recurrence_day_blood_tests": config.include_recurrence_day_blood_tests,
        "min_blood_test_types_per_day": config.min_blood_test_types_per_day,
        "min_diagnosis_year": config.min_diagnosis_year,
        "train_end_year": config.train_end_year,
        "test_start_year": config.test_start_year,
        "exclude_prior_bc_surgery_patients": config.exclude_prior_bc_surgery_patients,
        "prior_bc_surgery_code_regex": config.prior_bc_surgery_code_regex,
        "exclude_and_censor_other_cancer_patients": (config.exclude_and_censor_other_cancer_patients),
        "other_cancer_code_filter_regex": (
            config.other_cancer_code_filter_regex if config.exclude_and_censor_other_cancer_patients else None
        ),
        "restrict_bloodtests_to_bc_diagnosis_window": config.restrict_bloodtests_to_bc_diagnosis_window,
        "bc_diagnosis_days": file_fingerprint(config.bc_diagnosis_days_csv_path)
        if uses_bc_diagnosis_days_file(config)
        else None,
        "bc_diagnosis_window_days_before": config.bc_diagnosis_window_days_before,
        "bc_diagnosis_window_days_after": config.bc_diagnosis_window_days_after,
        "num_most_common_blood_tests": config.num_most_common_blood_tests,
        "selected_blood_tests": config.selected_blood_tests,
        "include_blood_test_change": config.include_blood_test_change,
        "blood_test_change_min_days": config.blood_test_change_min_days,
        "blood_test_change_max_days": config.blood_test_change_max_days,
        "include_diagnosis_age": config.include_diagnosis_age,
        "require_diagnosis_age": config.require_diagnosis_age,
        "diagnosis_age": file_fingerprint(config.bc_diagnosis_patient_age_csv_path)
        if uses_diagnosis_age_file(config)
        else None,
        "diagnosis_procedure": file_fingerprint(config.diagnosis_procedure_csv_path)
        if uses_diagnosis_procedure_file(config)
        else None,
        "include_recent_diagnoses": config.include_recent_diagnoses,
        "num_most_common_recent_diagnoses": config.num_most_common_recent_diagnoses,
        "recent_diagnosis_lookback_days": config.recent_diagnosis_lookback_days,
        "include_landmark_day_recent_diagnoses": config.include_landmark_day_recent_diagnoses,
        "include_time_since_diagnosis": config.include_time_since_diagnosis,
    }
    if config.include_blood_test_change:
        payload["blood_test_change_version"] = BLOOD_TEST_CHANGE_VERSION
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def uses_bc_diagnosis_days_file(config: PatientDatasetConfig) -> bool:
    if config.bc_diagnosis_days_csv_path is None:
        return False
    return config.restrict_bloodtests_to_bc_diagnosis_window or config.exclude_prior_bc_surgery_patients


def uses_diagnosis_age_file(config: PatientDatasetConfig) -> bool:
    if config.bc_diagnosis_patient_age_csv_path is None:
        return False
    return config.include_diagnosis_age or config.require_diagnosis_age


def uses_diagnosis_procedure_file(config: PatientDatasetConfig) -> bool:
    if config.diagnosis_procedure_csv_path is None:
        return False
    return (
        config.include_recent_diagnoses
        or config.exclude_prior_bc_surgery_patients
        or config.exclude_and_censor_other_cancer_patients
    )


def file_fingerprint(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
