from __future__ import annotations

import re
from dataclasses import dataclass, field, replace

import numpy as np

from src.data.cohort_selection import patient_observation_end
from src.data.dataset import PatientBloodTestDataset, input_fingerprint
from src.data.dataset_config import PatientDatasetConfig
from src.data.landmarks import patient_targets
from src.data.other_cancer_outcomes import matching_recurrence_events, resolve_right_censoring_candidates
from src.data.sample_weighting import bootstrap_row_indices
from src.data.survival_targets import signed_survival_durations
from src.data.types import (
    PatientBloodTestItem,
    PatientDataSources,
    PatientEndpointStatus,
    PatientLandmarkTable,
)


def canonical_benchmark_dataset(
    config: PatientDatasetConfig,
    selected_tests: tuple[str, ...],
    data_sources: PatientDataSources,
) -> PatientBloodTestDataset:
    """Prepare the one cohort and feature definition shared by benchmark targets."""
    return PatientBloodTestDataset(
        replace(
            config,
            selected_blood_tests=selected_tests,
            recurrence_endpoint_name="distant_recurrence",
            endpoint_ddfs_code_filter_regex=config.input_recurrence_code_filter_regex,
        ),
        data_sources=data_sources,
    )


def benchmark_target_dataset(
    canonical: PatientBloodTestDataset,
    config: PatientDatasetConfig,
) -> PatientBloodTestDataset:
    """Return a target-config view without rebuilding the canonical cohort."""
    prepared_data = replace(
        canonical.prepared_data,
        input_fingerprint=input_fingerprint(config),
    )
    return PatientBloodTestDataset(config, prepared_data=prepared_data)


def benchmark_washout_dataset(
    canonical: PatientBloodTestDataset,
    washout_days: int,
) -> PatientBloodTestDataset:
    """Exclude landmarks at most ``washout_days`` before any observed outcome.

    Combined with skipping horizons <= washout, this keeps exactly landmarks
    whose horizon-capped time to event or censoring is > ``washout_days``.
    """
    if washout_days < 0:
        raise ValueError("Washout days must be non-negative.")
    if washout_days == 0:
        return canonical
    statuses = canonical.prepared_data.status_by_patient

    def washed(item: PatientBloodTestItem) -> PatientBloodTestItem:
        boundary_day = patient_observation_end(statuses[item.patient_id])
        if boundary_day is None:
            return item
        days = tuple(day for day in item.valid_landmark_days if boundary_day - day > washout_days)
        return replace(item, valid_landmark_days=days)

    items_by_split = {
        split: tuple(washed(item) for item in items)
        for split, items in canonical.prepared_data.items_by_split.items()
    }
    prepared = replace(canonical.prepared_data, items_by_split=items_by_split)
    return PatientBloodTestDataset(canonical.config, prepared_data=prepared)


@dataclass
class BenchmarkLandmarks:
    """Canonical benchmark landmarks with target labels projected on demand.

    ``dataset`` supplies one cohort and feature matrix for every Endpoint
    Definition. Non-bootstrapped rows and deterministic bootstrap row selections
    are cached; ``table`` changes only Recurrence-Free Survival labels.
    """

    dataset: PatientBloodTestDataset
    _sources: dict[tuple[str, int | None], PatientLandmarkTable] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )
    _fingerprints: dict[PatientDatasetConfig, str] = field(default_factory=dict, init=False, repr=False)
    _bootstrap_indices: dict[tuple[str, int], np.ndarray] = field(default_factory=dict, init=False, repr=False)

    def table(
        self,
        config: PatientDatasetConfig,
        bootstrap_run: int | None = None,
    ) -> PatientLandmarkTable:
        """Return canonical rows with labels for ``config`` and optional draw."""
        validate_target_config(self.dataset.config, config)
        source = self._source(config, bootstrap_run)
        return project_targets(self.dataset, config, source, self._fingerprint(config))

    def _source(
        self,
        config: PatientDatasetConfig,
        bootstrap_run: int | None,
    ) -> PatientLandmarkTable:
        key = (config.split, bootstrap_run)
        source = self._sources.get(key)
        if source is not None:
            return source
        base = self._base_table(config.split)
        if bootstrap_run is None:
            return base
        items = self.dataset.prepared_data.items_by_split.get(config.split, ())
        indices, probabilities = bootstrap_row_indices(items, config, bootstrap_run)
        self._bootstrap_indices[config.split, bootstrap_run] = indices
        source = select_rows(base, indices, probabilities, bootstrap_run)
        self._sources[key] = source
        return source

    def bootstrap_indices(self, config: PatientDatasetConfig, bootstrap_run: int) -> np.ndarray:
        """Return the exact ordered pool-row indices used by a canonical draw."""
        validate_target_config(self.dataset.config, config)
        self._source(config, bootstrap_run)
        return self._bootstrap_indices[config.split, bootstrap_run]

    def _base_table(self, split: str) -> PatientLandmarkTable:
        key = (split, None)
        table = self._sources.get(key)
        if table is None:
            split_dataset = PatientBloodTestDataset(
                replace(self.dataset.config, split=split),
                prepared_data=self.dataset.prepared_data,
            )
            table = split_dataset.to_landmark_table(bootstrap=False)
            self._sources[key] = table
        return table

    def _fingerprint(self, config: PatientDatasetConfig) -> str:
        fingerprint = self._fingerprints.get(config)
        if fingerprint is None:
            fingerprint = input_fingerprint(config)
            self._fingerprints[config] = fingerprint
        return fingerprint


def select_rows(
    table: PatientLandmarkTable,
    indices: np.ndarray,
    probabilities: np.ndarray,
    bootstrap_run: int,
) -> PatientLandmarkTable:
    """Select one bootstrap draw from canonical landmark rows."""
    def selected(values: np.ndarray | None) -> np.ndarray | None:
        return None if values is None else values[indices]

    return replace(
        table,
        features=table.features[indices],
        signed_durations=table.signed_durations[indices],
        durations=table.durations[indices],
        events=table.events[indices],
        sample_weights=None,
        patient_ids=table.patient_ids[indices],
        landmark_days=table.landmark_days[indices],
        prior_non_endpoint_recurrence=table.prior_non_endpoint_recurrence[indices],
        metadata={**table.metadata, "bootstrap": True, "bootstrap_run": bootstrap_run},
        sample_weight_interval_days=None,
        bootstrap_selection_probabilities=probabilities,
        target_end_reasons=selected(table.target_end_reasons),
        right_censoring_days=selected(table.right_censoring_days),
        right_censoring_reasons=selected(table.right_censoring_reasons),
        blood_test_change_inputs=(
            None if table.blood_test_change_inputs is None else table.blood_test_change_inputs.select_rows(indices)
        ),
    )


def project_targets(
    canonical: PatientBloodTestDataset,
    config: PatientDatasetConfig,
    source: PatientLandmarkTable,
    fingerprint: str,
) -> PatientLandmarkTable:
    """Attach one Endpoint Definition's labels to canonical rows and features."""
    items = {item.patient_id: item for item in canonical.prepared_data.items_by_split.get(config.split, ())}
    input_regex = re.compile(config.input_recurrence_code_filter_regex)
    endpoint_regex = re.compile(config.endpoint_ddfs_code_filter_regex)
    row_count = len(source.landmark_days)
    durations = np.empty(row_count, dtype=np.int32)
    events = np.empty(row_count, dtype=bool)
    prior_non_endpoint = np.empty(row_count, dtype=bool)
    end_reasons = np.empty(row_count, dtype=object)
    censor_days = np.empty(row_count, dtype=np.float64)
    censor_reasons = np.empty(row_count, dtype=object)
    for patient_id, positions in positions_by_patient(source.patient_ids).items():
        item = items[patient_id]
        target_item = projected_item(
            item,
            canonical.prepared_data.status_by_patient[patient_id],
            input_regex,
            endpoint_regex,
        )
        values = patient_targets(target_item, source.landmark_days[positions], config)
        durations[positions], events[positions], prior_non_endpoint[positions], end_reasons[positions] = values
        censor_days[positions] = (
            np.nan if target_item.right_censoring_day is None else target_item.right_censoring_day
        )
        censor_reasons[positions] = target_item.right_censoring_reason
    bootstrap = bool(source.metadata.get("bootstrap", False))
    metadata = {
        **source.metadata,
        "split": config.split,
        "row_count": row_count,
        "event_count": int(events.sum()),
        "recurrence_endpoint_name": config.recurrence_endpoint_name,
        "input_recurrence_code_filter_regex": config.input_recurrence_code_filter_regex,
        "endpoint_ddfs_code_filter_regex": config.endpoint_ddfs_code_filter_regex,
        "censoring_horizon_days": config.censoring_horizon_days,
        "bootstrap": bootstrap,
        "bootstrap_run": source.metadata.get("bootstrap_run") if bootstrap else None,
        "input_fingerprint": fingerprint,
    }
    return replace(
        source,
        signed_durations=signed_survival_durations(durations, events),
        durations=durations,
        events=events,
        prior_non_endpoint_recurrence=prior_non_endpoint,
        metadata=metadata,
        target_end_reasons=end_reasons,
        right_censoring_days=censor_days,
        right_censoring_reasons=censor_reasons,
    )


def projected_item(
    item: PatientBloodTestItem,
    canonical_status: PatientEndpointStatus,
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
) -> PatientBloodTestItem:
    """Return one lightweight item view with target-specific outcome fields."""
    input_events = matching_recurrence_events(item.distant_recurrence_events, input_regex)
    endpoint_events = matching_recurrence_events(item.distant_recurrence_events, endpoint_regex)
    if any(event not in input_events for event in endpoint_events):
        raise ValueError("Benchmark endpoint definitions must select only configured input recurrences.")
    endpoint_day = endpoint_events[0].days_since_diagnosis if endpoint_events else None
    non_endpoint = tuple(event for event in input_events if event not in endpoint_events)
    candidates = []
    if canonical_status.right_censoring_day is not None and canonical_status.right_censoring_reason is not None:
        candidates.append((canonical_status.right_censoring_day, canonical_status.right_censoring_reason))
    candidates.extend(
        (event.days_since_diagnosis, "non_endpoint_recurrence")
        for event in non_endpoint
        if event.days_since_diagnosis != endpoint_day
    )
    censoring = resolve_right_censoring_candidates(candidates)
    return replace(
        item,
        endpoint_day=endpoint_day,
        non_endpoint_recurrence_events=non_endpoint,
        right_censoring_day=censoring.day,
        right_censoring_reason=censoring.reason,
    )


def positions_by_patient(patient_ids: np.ndarray) -> dict[str, np.ndarray]:
    positions: dict[str, list[int]] = {}
    for position, patient_id in enumerate(patient_ids):
        positions.setdefault(str(patient_id), []).append(position)
    return {key: np.asarray(value, dtype=np.int64) for key, value in positions.items()}


def validate_target_config(canonical: PatientDatasetConfig, target: PatientDatasetConfig) -> None:
    """Require one canonical cohort and feature definition across benchmark targets."""
    if canonical.endpoint_ddfs_code_filter_regex != canonical.input_recurrence_code_filter_regex:
        raise ValueError("The canonical Endpoint Definition must select every configured input recurrence.")
    unchanged = replace(
        target,
        split=canonical.split,
        recurrence_endpoint_name=canonical.recurrence_endpoint_name,
        endpoint_ddfs_code_filter_regex=canonical.endpoint_ddfs_code_filter_regex,
        censoring_horizon_days=canonical.censoring_horizon_days,
    )
    if unchanged != canonical:
        raise ValueError(
            "Benchmark targets may change only the Temporal Split, Endpoint Definition, Endpoint Name, and horizon."
        )
