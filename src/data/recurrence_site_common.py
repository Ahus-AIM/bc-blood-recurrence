from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from src.data.icd_codes import matches_icd_code


@dataclass(frozen=True)
class RecurrenceSiteRule:
    """Rule for assigning recurrence events to a displayed site.

    `label` is the plot/export category and `code_regex` is matched against the
    compact uppercase alphanumeric representation of each source code.
    """

    label: str
    code_regex: str


def site_label(
    code: str,
    site_rules: tuple[RecurrenceSiteRule, ...],
    other_label: str | None,
) -> str | None:
    for rule in site_rules:
        if code_matches_site_rule(code, rule):
            return rule.label
    return other_label


def code_matches_site_rule(code: str, rule: RecurrenceSiteRule) -> bool:
    return matches_icd_code(code, re.compile(rule.code_regex))


def patients_at_risk(durations: np.ndarray, day: float) -> int:
    return int(np.sum(np.isfinite(durations) & (durations >= day)))


def validate_min_patients_at_risk(min_patients_at_risk: int) -> None:
    if min_patients_at_risk < 1:
        raise ValueError("`min_patients_at_risk` must be at least 1.")
