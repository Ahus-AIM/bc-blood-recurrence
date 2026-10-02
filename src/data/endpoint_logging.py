from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from src.data.dataset_config import PatientDatasetConfig
from src.data.dataset_logging import log_icd_patient_counts
from src.data.icd_codes import matches_icd_code
from src.data.observation_window import observation_end_day
from src.data.other_cancer_outcomes import compact_icd_code, is_other_cancer_diagnosis
from src.data.types import PatientBloodTestItem, PatientRecord

EndpointKind = Literal[
    "configured_recurrence", "non_endpoint_recurrence", "death", "other_cancer", "last_observation"
]
ENDPOINT_KINDS: tuple[EndpointKind, ...] = (
    "configured_recurrence", "non_endpoint_recurrence", "death", "other_cancer", "last_observation"
)
MODEL_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class EndpointOccurrence:
    """One outcome present on a patient's first outcome day, including tied outcomes.

    `kind` distinguishes recurrence roles and censoring causes. `code` contains
    the full normalized recurrence code or three-character other-cancer category;
    `description` retains source wording. Death and last observation have no code.
    These diagnostic occurrences do not replace the resolved model outcome.
    """

    kind: EndpointKind
    code: str = ""
    description: str = ""


def log_endpoint_counts(
    logger: logging.Logger,
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]],
    endpoint_patients: Mapping[str, PatientRecord],
    config: PatientDatasetConfig,
) -> None:
    """Log eligible patients' first-day outcomes overall and by fold, without a horizon."""
    input_regex = re.compile(config.input_recurrence_code_filter_regex)
    endpoint_regex = re.compile(config.endpoint_ddfs_code_filter_regex)
    other_cancer_regex = (
        re.compile(config.other_cancer_code_filter_regex) if config.exclude_and_censor_other_cancer_patients else None
    )
    observations_by_split = {
        split: {
            item.patient_id: patient_endpoint_occurrences(
                item,
                endpoint_patients[item.patient_id].last_observation_day,
                input_regex,
                endpoint_regex,
                other_cancer_regex,
            )
            for item in items_by_split.get(split, ())
        }
        for split in MODEL_SPLITS
    }
    overall = {
        patient_id: occurrences
        for observations in observations_by_split.values()
        for patient_id, occurrences in observations.items()
    }
    logger.info(
        "patient endpoints: input_recurrence_regex=%s; target_recurrence_regex=%s",
        config.input_recurrence_code_filter_regex,
        config.endpoint_ddfs_code_filter_regex,
    )
    log_endpoint_group(logger, "overall", overall)
    for split, observations in observations_by_split.items():
        log_endpoint_group(logger, split, observations)


def patient_endpoint_occurrences(
    item: PatientBloodTestItem,
    last_observation_day: int | None,
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
    other_cancer_regex: re.Pattern[str] | None,
) -> tuple[EndpointOccurrence, ...]:
    """Collect all normalized outcomes on the shared boundary before resolving ties.

    `last_observation_day` comes from the endpoint source, not the item's display
    extent, which may extend past the clinical observation boundary.
    """
    day = observation_end_day(item.input_recurrence_day, item.right_censoring_day)
    if day is None:
        raise ValueError(f"Patient {item.patient_id} has no recurrence or censoring time.")
    occurrences: list[EndpointOccurrence] = []
    for event in item.distant_recurrence_events:
        if event.days_since_diagnosis != day:
            continue
        kind: EndpointKind
        if matches_icd_code(event.code, endpoint_regex):
            kind = "configured_recurrence"
        elif matches_icd_code(event.code, input_regex):
            kind = "non_endpoint_recurrence"
        else:
            continue
        occurrences.append(EndpointOccurrence(kind, compact_icd_code(event.code), event.description))
    if item.death_day == day:
        occurrences.append(EndpointOccurrence("death"))
    if last_observation_day == day:
        occurrences.append(EndpointOccurrence("last_observation"))
    if other_cancer_regex is not None:
        occurrences.extend(
            EndpointOccurrence("other_cancer", compact_icd_code(event.code)[:3], event.description)
            for event in item.clinical_events
            if event.days_since_diagnosis == day and is_other_cancer_diagnosis(event, other_cancer_regex)
        )
    return tuple(occurrences)


def log_endpoint_group(
    logger: logging.Logger,
    group: str,
    observations: Mapping[str, tuple[EndpointOccurrence, ...]],
) -> None:
    logger.info(
        "patient endpoints [%s]: patients=%d; first outcome day; independent of prediction horizon",
        group,
        len(observations),
    )
    for kind in ENDPOINT_KINDS:
        matching = {
            patient_id: selected
            for patient_id, occurrences in observations.items()
            if (selected := tuple(event for event in occurrences if event.kind == kind))
        }
        logger.info("patient endpoint [%s] %s: patients=%d", group, kind, len(matching))
        if kind in {"configured_recurrence", "non_endpoint_recurrence", "other_cancer"}:
            log_endpoint_codes(logger, group, kind, matching)
    overlap_count = sum(len({event.kind for event in occurrences}) > 1 for occurrences in observations.values())
    logger.info(
        "patient endpoint overlap [%s]: patients with multiple first-day outcomes=%d",
        group,
        overlap_count,
    )


def log_endpoint_codes(
    logger: logging.Logger,
    group: str,
    kind: EndpointKind,
    observations: Mapping[str, tuple[EndpointOccurrence, ...]],
) -> None:
    codes_by_patient = {
        patient_id: {event.code for event in occurrences}
        for patient_id, occurrences in observations.items()
    }
    counts = Counter(code for codes in codes_by_patient.values() for code in codes)
    descriptions: dict[str, set[str]] = {}
    for occurrences in observations.values():
        for event in occurrences:
            descriptions.setdefault(event.code, set()).add(event.description)
    unit = "category" if kind == "other_cancer" else "code"
    log_icd_patient_counts(logger, f"patient endpoint [{group}] {kind} ICD-10 {unit}", counts, descriptions)
    overlap_count = sum(len(codes) > 1 for codes in codes_by_patient.values())
    if overlap_count:
        logger.info(
            "patient endpoint [%s] %s ICD-10 %s overlap: patients=%d",
            group,
            kind,
            unit,
            overlap_count,
        )
