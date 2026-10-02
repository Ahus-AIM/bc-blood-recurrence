from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from src.data import PatientLandmarkTable
from src.model.ebm_config import EBM_DEFAULTS


@dataclass(frozen=True)
class PatientGroupedBagPlan:
    """InterpretML outer bags where each patient stays in one split per bag.

    `bags` has shape `(n_rows, outer_bags)`. InterpretML treats `+1` as an
    internal training row and `-1` as an internal validation row. `metadata`
    records JSON-serializable diagnostics for auditing explanation uncertainty.
    """

    bags: np.ndarray
    metadata: dict[str, object]


def patient_grouped_bags_enabled(ebm_config: Mapping[str, object]) -> bool:
    configured = ebm_config.get("patient_grouped_outer_bags")
    if configured is not None:
        return bool(configured)
    return int(ebm_config.get("outer_bags", EBM_DEFAULTS["outer_bags"])) > 1


def build_patient_grouped_bag_plan(
    table: PatientLandmarkTable,
    ebm_config: Mapping[str, object],
) -> PatientGroupedBagPlan:
    patient_ids = np.asarray(table.patient_ids, dtype=str)
    events = np.asarray(table.events, dtype=bool)
    unique_patients, row_patient_indices = np.unique(patient_ids, return_inverse=True)
    event_patients = patient_event_flags(row_patient_indices, events, len(unique_patients))

    outer_bags = int(ebm_config.get("outer_bags", EBM_DEFAULTS["outer_bags"]))
    validation_size = ebm_config.get("validation_size", EBM_DEFAULTS["validation_size"])
    validation_patient_count = validation_count_from_size(validation_size, len(unique_patients))
    max_attempts = int(ebm_config.get("patient_grouped_bag_max_attempts", 10_000))
    random_state = ebm_config.get("random_state", EBM_DEFAULTS["random_state"])

    validate_sampling_inputs(len(unique_patients), int(event_patients.sum()), max_attempts)
    rng = np.random.default_rng(random_state)
    bags = np.ones((len(patient_ids), outer_bags), dtype=np.int8)
    bag_summaries: list[dict[str, object]] = []

    for bag_index in range(outer_bags):
        validation_indices, attempts = sample_validation_patients(
            rng,
            event_patients,
            validation_patient_count,
            max_attempts,
            bag_index,
        )
        patient_values = np.ones(len(unique_patients), dtype=np.int8)
        patient_values[validation_indices] = -1
        bags[:, bag_index] = patient_values[row_patient_indices]
        bag_summaries.append(
            bag_summary(
                bag_index,
                attempts,
                bags[:, bag_index],
                validation_indices,
                event_patients,
            )
        )

    return PatientGroupedBagPlan(
        bags=bags,
        metadata={
            "patient_grouped": True,
            "outer_bags": outer_bags,
            "validation_size": validation_size,
            "validation_size_unit": "patients",
            "validation_patient_count": validation_patient_count,
            "patient_count": len(unique_patients),
            "event_patient_count": int(event_patients.sum()),
            "max_attempts_per_bag": max_attempts,
            "random_state": random_state,
            "total_redraws": int(sum(int(summary["redraws"]) for summary in bag_summaries)),
            "bags": bag_summaries,
        },
    )


def patient_event_flags(
    row_patient_indices: np.ndarray,
    row_events: np.ndarray,
    patient_count: int,
) -> np.ndarray:
    event_patients = np.zeros(patient_count, dtype=bool)
    np.logical_or.at(event_patients, row_patient_indices, row_events)
    return event_patients


def validation_count_from_size(validation_size: object, patient_count: int) -> int:
    if patient_count < 2:
        raise ValueError("Patient-grouped EBM outer bags require at least 2 training patients.")
    if isinstance(validation_size, bool):
        raise TypeError("`ebm.validation_size` must be a positive fraction or integer patient count.")
    if not isinstance(validation_size, int | float | np.integer | np.floating):
        raise TypeError("`ebm.validation_size` must be a positive fraction or integer patient count.")
    size = float(validation_size)
    if size <= 0:
        raise ValueError("`ebm.validation_size` must select at least one validation patient.")
    if size < 1:
        return min(max(math.ceil(size * patient_count), 1), patient_count - 1)
    if not size.is_integer():
        raise ValueError("`ebm.validation_size` values >= 1 must be integer patient counts.")
    count = int(size)
    if count <= 0:
        raise ValueError("`ebm.validation_size` must select at least one validation patient.")
    if count >= patient_count:
        raise ValueError(
            "`ebm.validation_size` exact patient count must leave at least one training patient; "
            f"validation_size={count}, patient_count={patient_count}."
        )
    return count


def validate_sampling_inputs(patient_count: int, event_patient_count: int, max_attempts: int) -> None:
    if event_patient_count < 2:
        raise ValueError(
            "Patient-grouped EBM outer bags require at least 2 event patients so train and validation "
            f"can both contain an event patient; patient_count={patient_count}, "
            f"event_patient_count={event_patient_count}."
        )
    if max_attempts < 1:
        raise ValueError("`ebm.patient_grouped_bag_max_attempts` must be at least 1.")


def sample_validation_patients(
    rng: np.random.Generator,
    event_patients: np.ndarray,
    validation_patient_count: int,
    max_attempts: int,
    bag_index: int,
) -> tuple[np.ndarray, int]:
    patient_count = len(event_patients)
    for attempt in range(1, max_attempts + 1):
        validation_indices = rng.choice(patient_count, size=validation_patient_count, replace=False)
        validation_event_count = int(event_patients[validation_indices].sum())
        train_event_count = int(event_patients.sum()) - validation_event_count
        if validation_event_count > 0 and train_event_count > 0:
            return np.asarray(validation_indices, dtype=np.int64), attempt
    raise ValueError(
        "Could not sample a valid patient-grouped EBM outer bag with event patients in both train "
        f"and validation after {max_attempts} attempts; bag_index={bag_index}, "
        f"patient_count={patient_count}, event_patient_count={int(event_patients.sum())}, "
        f"validation_patient_count={validation_patient_count}."
    )


def bag_summary(
    bag_index: int,
    attempts: int,
    bag: np.ndarray,
    validation_indices: np.ndarray,
    event_patients: np.ndarray,
) -> dict[str, object]:
    validation_patient_mask = np.zeros(len(event_patients), dtype=bool)
    validation_patient_mask[validation_indices] = True
    train_patient_mask = ~validation_patient_mask
    return {
        "bag_index": bag_index,
        "attempts": attempts,
        "redraws": attempts - 1,
        "train_patient_count": int(train_patient_mask.sum()),
        "validation_patient_count": int(validation_patient_mask.sum()),
        "train_row_count": int(np.sum(bag > 0)),
        "validation_row_count": int(np.sum(bag < 0)),
        "train_event_patient_count": int(event_patients[train_patient_mask].sum()),
        "validation_event_patient_count": int(event_patients[validation_patient_mask].sum()),
    }
