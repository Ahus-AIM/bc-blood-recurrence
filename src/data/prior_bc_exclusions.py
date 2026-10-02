from __future__ import annotations

import re
from typing import Protocol

DEFAULT_PRIOR_BC_SURGERY_CODE_REGEX = r"^38(?!8[01])\d{2}$"
PRIOR_BC_SURGERY_DISCARD_REASON = "prior BC surgery before first BC diagnosis"


class ClinicalEventLike(Protocol):
    """Minimal clinical-event shape needed for prior-BC exclusion checks."""

    days_since_diagnosis: int
    code: str


def has_prior_bc_surgery(
    clinical_events: tuple[ClinicalEventLike, ...],
    bc_diagnosis_events: tuple[ClinicalEventLike, ...],
    code_regex: re.Pattern[str],
) -> bool:
    first_bc_day = first_bc_diagnosis_day(bc_diagnosis_events)
    return any(is_prior_bc_surgery_event(event, first_bc_day, code_regex) for event in clinical_events)


def first_bc_diagnosis_day(bc_diagnosis_events: tuple[ClinicalEventLike, ...]) -> int:
    if not bc_diagnosis_events:
        return 0
    return min(event.days_since_diagnosis for event in bc_diagnosis_events)


def is_prior_bc_surgery_event(
    event: ClinicalEventLike,
    first_bc_day: int,
    code_regex: re.Pattern[str],
) -> bool:
    if event.days_since_diagnosis >= first_bc_day:
        return False
    return code_regex.search(event.code.strip().upper()) is not None
