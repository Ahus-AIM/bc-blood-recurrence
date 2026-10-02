from __future__ import annotations

import csv
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.data.blood_test_features import BLOOD_TEST_CHANGE_SUFFIX
from src.data.patient_preprocessing import TEST_KEY_SEPARATOR

TRANSLATION_COLUMNS = ("norwegian_test_name", "english_test_name")
NON_BLOOD_TEST_FEATURE_LABELS: Mapping[str, tuple[str, str]] = {
    "age_at_diagnosis": ("Age at diagnosis", "years"),
    "time_since_diagnosis": ("Time since diagnosis", "years"),
    "time_since_diagnosis_years": ("Time since diagnosis", "years"),
}
UNIT_DISPLAY_LABELS: Mapping[str, str] = {
    "g/dl": "g/dL",
    "g/l": "g/L",
    "mg/l": "mg/L",
    "mmol/l": "mmol/L",
    "µmol/l": "µmol/L",
    "umol/l": "µmol/L",
    "u/l": "U/L",
    "U/l": "U/L",
    "x10e9/l": "10⁹/L",
    "10^9/l": "10⁹/L",
    "10⁹/l": "10⁹/L",
    "ml/min/1,73m2": "mL/min per 1.73 m²",
    "ml/min/1.73m2": "mL/min per 1.73 m²",
    "ml/min/1,73m²": "mL/min per 1.73 m²",
    "ml/min/1.73m²": "mL/min per 1.73 m²",
}


def format_blood_test_unit(unit: str) -> str:
    """Format known laboratory units without rescaling values or guessing unknown units.

    Only the litre and micro symbols are normalized for lookup; prefix case
    remains significant. The source's legacy ``x10e9/l`` means billions per litre.
    Plain Unicode keeps unit symbols upright and selectable in vector exports.
    """
    key = "".join(unit.split()).replace("L", "l").replace("μ", "µ")
    return UNIT_DISPLAY_LABELS.get(key, unit)


@dataclass(frozen=True)
class FeatureDisplayLabel:
    """Display labels for one model feature.

    `raw_name` is the model feature key used for data lookup, `title` is the
    translated feature name without a unit, and `unit` retains the original
    source spelling for reports and reference calculations. `display_unit`
    formats figure units; `full_label` combines it with `title`.
    """

    raw_name: str
    title: str
    unit: str

    @property
    def display_unit(self) -> str:
        return format_blood_test_unit(self.unit)

    @property
    def full_label(self) -> str:
        return f"{self.title} ({self.display_unit})" if self.unit else self.title

    @property
    def x_axis_label(self) -> str:
        return self.display_unit if self.unit else self.title

    @property
    def compact_x_axis_label(self) -> str:
        return self.display_unit


def load_test_name_translation(translation_csv_path: Path | None) -> dict[str, str]:
    if translation_csv_path is None or not translation_csv_path.exists():
        return {}
    with translation_csv_path.open(encoding="utf-8", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_columns = [column for column in TRANSLATION_COLUMNS if column not in fieldnames]
        if missing_columns:
            joined = ", ".join(missing_columns)
            raise ValueError(f"Missing required columns in blood-test translation CSV: {joined}.")
        return translation_from_rows(reader)


def translation_from_rows(rows: Iterable[Mapping[str, str | None]]) -> dict[str, str]:
    translation: dict[str, str] = {}
    for row in rows:
        norwegian_name = row.get("norwegian_test_name", "")
        english_name = row.get("english_test_name", "")
        if norwegian_name is None or english_name is None:
            continue
        source = str(norwegian_name).strip()
        target = str(english_name).strip()
        if source and target:
            translation[source] = target
    return translation


def display_name_for_test_key(
    test_key: str,
    test_name_translation: Mapping[str, str],
) -> str:
    return feature_display_label(test_key, test_name_translation).full_label


def display_title_for_test_key(
    test_key: str,
    test_name_translation: Mapping[str, str],
) -> str:
    return feature_display_label(test_key, test_name_translation).title


def feature_display_title(
    feature_name: str,
    test_name_translation: Mapping[str, str],
) -> str:
    return " & ".join(
        feature_display_label(component, test_name_translation).title
        for component in interaction_components(feature_name)
    )


def feature_display_label(
    feature_name: str,
    test_name_translation: Mapping[str, str],
) -> FeatureDisplayLabel:
    raw_name = str(feature_name)
    base_name, is_change = split_change_suffix(raw_name)
    if base_name in NON_BLOOD_TEST_FEATURE_LABELS:
        title, unit = NON_BLOOD_TEST_FEATURE_LABELS[base_name]
        return FeatureDisplayLabel(
            raw_name,
            change_title(title, is_change),
            unit,
        )
    if TEST_KEY_SEPARATOR not in base_name:
        title = test_name_translation.get(base_name, base_name)
        return FeatureDisplayLabel(raw_name, change_title(title, is_change), "")
    test_name, unit = base_name.split(TEST_KEY_SEPARATOR, maxsplit=1)
    title = test_name_translation.get(test_name, test_name)
    return FeatureDisplayLabel(
        raw_name,
        change_title(title, is_change),
        unit,
    )


def split_change_suffix(feature_name: str) -> tuple[str, bool]:
    if not feature_name.endswith(BLOOD_TEST_CHANGE_SUFFIX):
        return feature_name, False
    return feature_name[: -len(BLOOD_TEST_CHANGE_SUFFIX)], True


def interaction_components(feature_name: str) -> tuple[str, ...]:
    return tuple(str(feature_name).split(" & "))


def change_title(title: str, is_change: bool) -> str:
    return f"{title} change" if is_change else title
