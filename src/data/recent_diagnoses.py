from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np

from src.data.dataset_config import PatientDatasetConfig
from src.data.types import PatientBloodTestItem, PatientClinicalEvent

DIAGNOSIS_EVENT_TYPES = frozenset({"diagnosis", "diagnose"})
BREAST_CANCER_CODE_REGEX = re.compile(r"C50\d*")
FEATURE_PREFIX = "recent_diagnosis"
DESCRIPTION_SLUG_MAX_LENGTH = 60


@dataclass(frozen=True)
class RecentDiagnosisFeatureSet:
    """Recent diagnosis feature metadata in model-column order.

    `codes` stores normalized diagnosis codes, `feature_names` stores the
    matching model feature names, `descriptions` maps each code to the selected
    full-text description, and `train_patient_counts` records selection counts.
    """

    codes: tuple[str, ...]
    feature_names: tuple[str, ...]
    descriptions: dict[str, str]
    train_patient_counts: dict[str, int]


def empty_recent_diagnosis_feature_set() -> RecentDiagnosisFeatureSet:
    return RecentDiagnosisFeatureSet((), (), {}, {})


def select_recent_diagnosis_features(
    train_items: tuple[PatientBloodTestItem, ...],
    config: PatientDatasetConfig,
) -> RecentDiagnosisFeatureSet:
    if not config.include_recent_diagnoses or config.num_most_common_recent_diagnoses == 0:
        return empty_recent_diagnosis_feature_set()
    patient_counts: Counter[str] = Counter()
    description_counts: defaultdict[str, Counter[str]] = defaultdict(Counter)
    for item in train_items:
        patient_codes, patient_descriptions = recent_patient_codes(item, config)
        patient_counts.update(patient_codes)
        for code, descriptions in patient_descriptions.items():
            description_counts[code].update(descriptions)
    selected_codes = top_recent_diagnosis_codes(patient_counts, config.num_most_common_recent_diagnoses)
    descriptions = {
        code: selected_code_description(description_counts[code])
        for code in selected_codes
    }
    feature_names = tuple(
        recent_diagnosis_feature_name(code, descriptions[code])
        for code in selected_codes
    )
    train_patient_counts = {code: int(patient_counts[code]) for code in selected_codes}
    return RecentDiagnosisFeatureSet(selected_codes, feature_names, descriptions, train_patient_counts)


def recent_patient_codes(
    item: PatientBloodTestItem,
    config: PatientDatasetConfig,
) -> tuple[set[str], dict[str, set[str]]]:
    codes: set[str] = set()
    descriptions: dict[str, set[str]] = defaultdict(set)
    for event in item.clinical_events:
        code = normalized_recent_diagnosis_code(event)
        if code is None:
            continue
        if not diagnosis_matches_any_landmark(event.days_since_diagnosis, item.valid_landmark_days, config):
            continue
        codes.add(code)
        description = event.description.strip()
        if description:
            descriptions[code].add(description)
    return codes, descriptions


def normalized_recent_diagnosis_code(event: PatientClinicalEvent) -> str | None:
    if event.event_type.strip().lower() not in DIAGNOSIS_EVENT_TYPES:
        return None
    code = event.code.strip().upper()
    if not code or BREAST_CANCER_CODE_REGEX.fullmatch(code):
        return None
    return code


def diagnosis_matches_any_landmark(
    diagnosis_day: int,
    landmark_days: tuple[int, ...],
    config: PatientDatasetConfig,
) -> bool:
    if not landmark_days:
        return False
    days = np.asarray(landmark_days, dtype=np.int32)
    return bool(recent_diagnosis_window_mask(days, diagnosis_day, config).any())


def recent_diagnosis_window_mask(
    landmark_days: np.ndarray,
    diagnosis_day: int,
    config: PatientDatasetConfig,
) -> np.ndarray:
    latest_landmark_day = diagnosis_day + config.recent_diagnosis_lookback_days
    inside_lookback = landmark_days <= latest_landmark_day
    if config.include_landmark_day_recent_diagnoses:
        after_diagnosis = landmark_days >= diagnosis_day
    else:
        after_diagnosis = landmark_days > diagnosis_day
    return inside_lookback & after_diagnosis


def top_recent_diagnosis_codes(
    patient_counts: Counter[str],
    count: int,
) -> tuple[str, ...]:
    ranked = sorted(patient_counts.items(), key=lambda item: (-item[1], item[0]))
    return tuple(code for code, _patient_count in ranked[:count])


def selected_code_description(description_counts: Counter[str]) -> str:
    if not description_counts:
        return ""
    ranked = sorted(description_counts.items(), key=lambda item: (-item[1], item[0]))
    return ranked[0][0]


def recent_diagnosis_feature_name(code: str, description: str) -> str:
    code_slug = slugify_code(code)
    description_slug = slugify_description(description)
    if not description_slug:
        return f"{FEATURE_PREFIX}_{code_slug}"
    return f"{FEATURE_PREFIX}_{code_slug}__{description_slug}"


def slugify_code(code: str) -> str:
    slug = re.sub(r"[^0-9A-Z]+", "_", code.upper()).strip("_")
    return slug or "UNKNOWN"


def slugify_description(description: str) -> str:
    slug = re.sub(r"[^0-9a-z]+", "_", description.lower()).strip("_")
    return slug[:DESCRIPTION_SLUG_MAX_LENGTH].rstrip("_")
