from __future__ import annotations

import logging
import re
from collections import Counter
from collections.abc import Iterable, Mapping

from src.data.other_cancer_outcomes import (
    compact_icd_code,
    first_other_cancer_day,
    is_other_cancer_diagnosis,
)
from src.data.types import PatientBloodTestItem


def log_filtered_blood_test_counts(
    logger: logging.Logger,
    observed_test_keys: Iterable[str],
    selected_test_keys: tuple[str, ...],
) -> None:
    counts = Counter(observed_test_keys)
    ordered_counts = sorted(
        ((test_key, counts[test_key]) for test_key in selected_test_keys),
        key=lambda item: (-item[1], item[0]),
    )
    summary = ", ".join(f"{test_key}={count}" for test_key, count in ordered_counts)
    logger.info("filtered blood-test counts before imputation: %s", summary)


def log_other_cancer_exclusion_counts(
    logger: logging.Logger,
    discarded_items: Iterable[PatientBloodTestItem],
    code_regex: re.Pattern[str],
    min_landmark_day: int,
) -> None:
    """Log early other-cancer exclusions by three-character ICD-10 category."""
    categories_by_patient: dict[str, set[str]] = {}
    descriptions_by_category: dict[str, set[str]] = {}
    for item in discarded_items:
        first_day = first_other_cancer_day(item.clinical_events, code_regex, None)
        if first_day is None:
            continue
        categories = categories_by_patient.setdefault(item.patient_id, set())
        for event in item.clinical_events:
            if event.days_since_diagnosis != first_day:
                continue
            if not is_other_cancer_diagnosis(event, code_regex):
                continue
            category = compact_icd_code(event.code)[:3]
            categories.add(category)
            description = " ".join(event.description.split())
            if description:
                descriptions_by_category.setdefault(category, set()).add(description)

    logger.info(
        "other-cancer exclusions before first eligible landmark (day < %d): patients=%d",
        min_landmark_day,
        len(categories_by_patient),
    )
    counts = Counter(category for categories in categories_by_patient.values() for category in categories)
    log_icd_patient_counts(logger, "other-cancer exclusion ICD-10 category", counts, descriptions_by_category)

    overlap_count = sum(len(categories) > 1 for categories in categories_by_patient.values())
    if overlap_count:
        logger.info(
            "other-cancer exclusion category overlap: patients with multiple first-day categories=%d",
            overlap_count,
        )


def log_icd_patient_counts(
    logger: logging.Logger,
    label: str,
    counts: Mapping[str, int],
    descriptions_by_code: Mapping[str, Iterable[str]],
) -> None:
    """Log patient counts in descending order with distinct, single-line descriptions."""
    for code, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        descriptions = "; ".join(sorted({
            cleaned
            for description in descriptions_by_code.get(code, ())
            if (cleaned := " ".join(description.split()))
        }))
        description_text = "" if not descriptions else f"; descriptions={descriptions}"
        logger.info("%s %s: patients=%d%s", label, code, count, description_text)
