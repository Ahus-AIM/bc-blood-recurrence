from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from typing import TypeVar

from src.data.icd_codes import compact_icd_code as compact_icd_code, matches_icd_code
from src.data.types import (
    PatientClinicalEvent,
    PatientEndpointEvent,
    PatientEndpointStatus,
    RightCensoringReason,
)

DIAGNOSIS_EVENT_TYPE = "diagnosis"
EventT = TypeVar("EventT", PatientClinicalEvent, PatientEndpointEvent)
RIGHT_CENSORING_REASON_PRIORITY: dict[RightCensoringReason, int] = {
    "other_cancer": 0,
    "death": 1,
    "non_endpoint_recurrence": 2,
    "last_observation": 3,
}


@dataclass(frozen=True)
class ResolvedRightCensoring:
    """Earliest patient-level Right-Censoring Time and its deterministic reason."""

    day: int | None
    reason: RightCensoringReason | None


def recurrence_precedes_censoring(
    recurrence_day: int | None,
    censoring_day: int | None,
    censoring_reason: RightCensoringReason | None,
) -> bool:
    """Return whether recurrence is observed under the shared tie rules."""
    return recurrence_day is not None and (
        censoring_day is None
        or recurrence_day < censoring_day
        or (recurrence_day == censoring_day and censoring_reason != "other_cancer")
    )


def is_other_cancer_diagnosis(
    event: PatientClinicalEvent,
    code_regex: re.Pattern[str],
) -> bool:
    """Return whether a clinical event is a configured other invasive cancer."""
    code = compact_icd_code(event.code)
    return (
        event.event_type.strip().lower() == DIAGNOSIS_EVENT_TYPE
        and bool(code)
        and code_regex.fullmatch(code) is not None
    )


def first_other_cancer_day(
    clinical_events: tuple[PatientClinicalEvent, ...],
    code_regex: re.Pattern[str],
    death_day: int | None,
) -> int | None:
    """Return the first qualifying day after shared post-death normalization."""
    matching_days = (
        normalized_event_day(event.days_since_diagnosis, death_day)
        for event in clinical_events
        if is_other_cancer_diagnosis(event, code_regex)
    )
    return min(matching_days, default=None)


def resolve_endpoint_status(
    recurrence_events: tuple[PatientEndpointEvent, ...],
    clinical_events: tuple[PatientClinicalEvent, ...],
    input_recurrence_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
    other_cancer_regex: re.Pattern[str] | None,
    *,
    death_day: int | None,
    last_observation_day: int | None,
) -> PatientEndpointStatus:
    """Resolve recurrence roles and the shared patient-level censor boundary."""
    events = normalize_event_dates(
        recurrence_events,
        death_day,
        lambda event: matches_icd_code(event.code, input_recurrence_regex),
    )
    input_events = matching_recurrence_events(events, input_recurrence_regex)
    endpoint_events = matching_recurrence_events(events, endpoint_regex)
    endpoint_event = endpoint_events[0] if endpoint_events else None
    non_endpoint_events = tuple(event for event in input_events if event not in endpoint_events)
    censoring = resolve_right_censoring(
        death_day,
        last_observation_day,
        clinical_events,
        other_cancer_regex,
        endpoint_day=None if endpoint_event is None else endpoint_event.days_since_diagnosis,
        non_endpoint_events=non_endpoint_events,
    )
    return PatientEndpointStatus(
        input_events[0] if input_events else None,
        endpoint_event,
        non_endpoint_events,
        None,
        censoring.day,
        censoring.reason,
    )


def matching_recurrence_events(
    events: tuple[PatientEndpointEvent, ...],
    code_regex: re.Pattern[str],
) -> tuple[PatientEndpointEvent, ...]:
    return tuple(event for event in events if matches_icd_code(event.code, code_regex))


def normalize_event_dates(
    events: tuple[EventT, ...],
    death_day: int | None,
    should_move: Callable[[EventT], bool],
) -> tuple[EventT, ...]:
    """Move selected events back to death and restore chronological order."""
    normalized = (
        replace(
            event,
            days_since_diagnosis=normalized_event_day(
                event.days_since_diagnosis,
                death_day,
            ),
        )
        if death_day is not None and should_move(event)
        else event
        for event in events
    )
    return tuple(
        sorted(
            normalized,
            key=lambda event: (event.days_since_diagnosis, compact_icd_code(event.code), event.description),
        )
    )


def normalized_event_day(event_day: int, death_day: int | None) -> int:
    """Assign posthumous diagnoses to death day for analysis."""
    return event_day if death_day is None else min(event_day, death_day)


def resolve_right_censoring(
    death_day: int | None,
    last_observation_day: int | None,
    clinical_events: tuple[PatientClinicalEvent, ...],
    other_cancer_regex: re.Pattern[str] | None,
    *,
    endpoint_day: int | None = None,
    non_endpoint_events: tuple[PatientEndpointEvent, ...] = (),
) -> ResolvedRightCensoring:
    """Resolve the earliest patient-level censor day and same-day reason."""
    candidates: list[tuple[int, RightCensoringReason]] = []
    if other_cancer_regex is not None:
        candidates.extend(
            (
                normalized_event_day(event.days_since_diagnosis, death_day),
                "other_cancer",
            )
            for event in clinical_events
            if is_other_cancer_diagnosis(event, other_cancer_regex)
        )
    if death_day is not None:
        candidates.append((death_day, "death"))
    candidates.extend(
        (event.days_since_diagnosis, "non_endpoint_recurrence")
        for event in non_endpoint_events
        if event.days_since_diagnosis != endpoint_day
    )
    if last_observation_day is not None:
        candidates.append((last_observation_day, "last_observation"))
    return resolve_right_censoring_candidates(candidates)


def resolve_right_censoring_candidates(
    candidates: Iterable[tuple[int, RightCensoringReason]],
) -> ResolvedRightCensoring:
    """Resolve candidate times using the project's deterministic same-day order."""
    available = tuple(candidates)
    if not available:
        return ResolvedRightCensoring(None, None)
    day, reason = min(
        available,
        key=lambda candidate: (
            candidate[0],
            RIGHT_CENSORING_REASON_PRIORITY[candidate[1]],
        ),
    )
    return ResolvedRightCensoring(day, reason)
