from __future__ import annotations

import numpy as np

from src.data.dataset_config import PatientDatasetConfig, PersonSampleWeighting
from src.data.observation_window import measurement_end_day
from src.data.types import PatientBloodTestItem

RowPlan = tuple[tuple[PatientBloodTestItem, np.ndarray, np.ndarray], ...]


def bootstrap_row_plan(
    items: tuple[PatientBloodTestItem, ...],
    config: PatientDatasetConfig,
    bootstrap_run: int,
) -> tuple[RowPlan, np.ndarray]:
    """Plan deterministic bootstrap rows and report each row's selection probability."""
    indices, probabilities = bootstrap_row_indices(items, config, bootstrap_run)
    return row_plan_from_indices(items, indices), probabilities


def bootstrap_row_indices(
    items: tuple[PatientBloodTestItem, ...],
    config: PatientDatasetConfig,
    bootstrap_run: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw canonical landmark-row indices and their selection probabilities."""
    if not items:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    draw_count = bootstrap_sample_count(config, items)
    rng = np.random.default_rng(bootstrap_seed(config, bootstrap_run))
    weighting = config.bootstrap_sample_weighting
    if weighting == "landmark_uniform":
        return landmark_uniform_bootstrap_indices(items, draw_count, rng)
    return person_bootstrap_indices(
        items, weighting, draw_count, rng,
        include_end_day=config.include_recurrence_day_blood_tests,
    )


def person_bootstrap_indices(
    items: tuple[PatientBloodTestItem, ...],
    weighting: PersonSampleWeighting,
    draw_count: int,
    rng: np.random.Generator,
    *,
    include_end_day: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    active_items = tuple((index, item) for index, item in enumerate(items) if item.valid_landmark_days)
    if not active_items:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    item_offsets = np.cumsum([0, *(len(item.valid_landmark_days) for item in items[:-1])])
    item_indices = rng.integers(0, len(active_items), size=draw_count)
    sampled_rows = np.empty(draw_count, dtype=np.int64)
    sampled_probabilities = np.empty(draw_count, dtype=np.float64)
    for item_index in np.unique(item_indices):
        canonical_item_index, item = active_items[int(item_index)]
        positions = np.flatnonzero(item_indices == item_index)
        valid_days = np.asarray(item.valid_landmark_days, dtype=np.int32)
        probabilities = within_patient_landmark_probabilities(
            item,
            valid_days,
            weighting,
            include_end_day=include_end_day,
        )
        choice_probabilities = probabilities if weighting == "person_time_weighted" else None
        selected_indices = rng.choice(
            len(valid_days),
            size=len(positions),
            replace=True,
            p=choice_probabilities,
        )
        sampled_rows[positions] = item_offsets[canonical_item_index] + selected_indices
        sampled_probabilities[positions] = probabilities[selected_indices] / len(active_items)
    return sampled_rows, sampled_probabilities


def landmark_uniform_bootstrap_indices(
    items: tuple[PatientBloodTestItem, ...],
    draw_count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    total_landmarks = sum(len(item.valid_landmark_days) for item in items)
    if total_landmarks == 0:
        return np.asarray([], dtype=np.int64), np.asarray([], dtype=np.float64)
    selected_landmarks = rng.integers(0, total_landmarks, size=draw_count)
    probabilities = np.full(draw_count, 1.0 / total_landmarks, dtype=np.float64)
    return selected_landmarks, probabilities


def row_plan_from_indices(
    items: tuple[PatientBloodTestItem, ...],
    row_indices: np.ndarray,
) -> RowPlan:
    """Map canonical landmark-row indices back to the existing row-plan format."""
    counts = np.asarray([len(item.valid_landmark_days) for item in items], dtype=np.int64)
    ends = np.cumsum(counts)
    selected_items = np.searchsorted(ends, row_indices, side="right")
    starts = np.concatenate((np.asarray([0], dtype=np.int64), ends[:-1]))
    plans = []
    for item_index in np.unique(selected_items):
        positions = np.flatnonzero(selected_items == item_index)
        day_indices = row_indices[positions] - starts[item_index]
        days = np.asarray(items[int(item_index)].valid_landmark_days, dtype=np.int32)[day_indices]
        plans.append((items[int(item_index)], positions.astype(np.int64), days))
    return tuple(plans)


def bootstrap_sample_count(
    config: PatientDatasetConfig,
    items: tuple[PatientBloodTestItem, ...],
) -> int:
    """Size patient-based test draws to the eligible cohort, including after washout."""
    if config.split == "train":
        return config.train_bootstrap_samples
    if config.split == "test":
        if config.bootstrap_sample_weighting != "landmark_uniform":
            return sum(bool(item.valid_landmark_days) for item in items)
        return config.test_bootstrap_samples
    return config.validation_bootstrap_samples


def bootstrap_seed(config: PatientDatasetConfig, bootstrap_run: int) -> int:
    offsets = {"train": 0, "val": 1_000_003, "test": 2_000_003}
    return config.random_seed + offsets.get(config.split, 0) + bootstrap_run * 10_000_019


def landmark_sample_weights(
    row_plan: RowPlan,
    row_count: int,
    config: PatientDatasetConfig,
    bootstrap: bool,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return normalized sample weights and optional raw interval days."""
    if bootstrap:
        return None, None
    if config.split != "train":
        return patient_balanced_uniform_weights(row_plan, row_count), None
    if config.training_sample_weighting == "landmark_uniform":
        return np.ones(row_count, dtype=np.float64), None
    if config.training_sample_weighting == "person_time_weighted":
        return patient_balanced_time_interval_weights(
            row_plan, row_count, include_end_day=config.include_recurrence_day_blood_tests,
        )
    return patient_balanced_uniform_weights(row_plan, row_count), None


def patient_balanced_uniform_weights(
    row_plan: RowPlan,
    row_count: int,
) -> np.ndarray:
    patient_ids = np.empty(row_count, dtype=object)
    for item, positions, _days in row_plan:
        patient_ids[positions] = item.patient_id
    return patient_uniform_weights(patient_ids)


def patient_uniform_weights(patient_ids: np.ndarray) -> np.ndarray:
    """Give each patient equal total weight, normalized to mean row weight one."""
    if len(patient_ids) == 0:
        return np.empty(0, dtype=float)
    patients, inverse, counts = np.unique(patient_ids, return_inverse=True, return_counts=True)
    return len(patient_ids) / (len(patients) * counts[inverse])


def patient_balanced_time_interval_weights(
    row_plan: RowPlan,
    row_count: int,
    *,
    include_end_day: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    active_patients = active_patient_count(row_plan)
    weights = np.empty(row_count, dtype=np.float64)
    interval_days = np.empty(row_count, dtype=np.int32)
    if active_patients == 0:
        return weights, interval_days
    patient_total_weight = row_count / active_patients
    for item, positions, days in row_plan:
        if len(days) == 0:
            continue
        patient_intervals = landmark_interval_days(item, days, include_end_day=include_end_day)
        interval_days[positions] = patient_intervals
        weights[positions] = patient_intervals * (patient_total_weight / patient_intervals.sum())
    return weights, interval_days


def within_patient_landmark_probabilities(
    item: PatientBloodTestItem,
    landmark_days: np.ndarray,
    weighting: PersonSampleWeighting,
    *,
    include_end_day: bool = False,
) -> np.ndarray:
    """Return configured conditional probabilities for one patient's landmarks."""
    if len(landmark_days) == 0:
        return np.asarray([], dtype=np.float64)
    if weighting == "person_uniform":
        return np.full(len(landmark_days), 1.0 / len(landmark_days), dtype=np.float64)
    interval_days = landmark_interval_days(item, landmark_days, include_end_day=include_end_day)
    return interval_days / interval_days.sum()


def active_patient_count(row_plan: RowPlan) -> int:
    return sum(1 for _item, _positions, days in row_plan if len(days) > 0)


def landmark_interval_days(
    item: PatientBloodTestItem,
    landmark_days: np.ndarray,
    *,
    include_end_day: bool = False,
) -> np.ndarray:
    """Forward coverage, including the event or censoring calendar day when enabled."""
    censor_days = tuple(day for day in (item.death_day, item.right_censoring_day) if day is not None)
    censor_day = min(censor_days) if censor_days else None
    observation_end = measurement_end_day(
        item.input_recurrence_day, censor_day,
        include_end_day=include_end_day,
    )
    if observation_end is None:
        raise ValueError(f"Patient {item.patient_id} has no observation endpoint.")
    next_days = np.concatenate((landmark_days[1:], np.asarray([observation_end], dtype=np.int32)))
    boundaries = np.minimum(next_days, observation_end)
    intervals = boundaries - landmark_days
    invalid_positions = np.flatnonzero(intervals <= 0)
    if len(invalid_positions) > 0:
        position = int(invalid_positions[0])
        raise ValueError(
            f"Non-positive landmark time interval for patient {item.patient_id} at landmark day "
            f"{landmark_days[position]} with boundary day {boundaries[position]}."
        )
    return intervals.astype(np.int32)
