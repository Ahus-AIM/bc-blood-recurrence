from __future__ import annotations

import ast
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Iterable

import pandas as pd

from src.data.dataset_config import normalize_split
from src.data.types import PatientClinicalEvent, PatientEndpointEvent, PatientRecord

PATIENT_ID_COLUMNS = ("patient_id", "id")
DIAGNOSIS_YEAR_COLUMNS = ("diagnosis_year", "bc_diagnosis_year")
RECURRENCE_DAY_COLUMNS = ("diagnosis_to_ddfs_in_days", "diagnosis_to_ddfs")
RECURRENCE_CODE_COLUMNS = ("ddfs_code", "code")
RECURRENCE_DESCRIPTION_COLUMNS = ("ddfs_description", "description")
DEATH_DAY_COLUMNS = ("days_from_diagnosis_to_death", "diagnosis_to_death")
LAST_ENTRY_DAY_COLUMNS = ("days_from_diagnosis_to_last_entry", "last_entry_days")
SPLIT_REQUIRED = {"patient_id": PATIENT_ID_COLUMNS, "split": ("split", "fold")}
ENDPOINT_REQUIRED = {"patient_id": PATIENT_ID_COLUMNS, "diagnosis_year": DIAGNOSIS_YEAR_COLUMNS}
ENDPOINT_OPTIONAL = {
    "recurrence_day": RECURRENCE_DAY_COLUMNS,
    "recurrence_code": RECURRENCE_CODE_COLUMNS,
    "recurrence_description": RECURRENCE_DESCRIPTION_COLUMNS,
    "death_day": DEATH_DAY_COLUMNS,
    "last_observation_day": LAST_ENTRY_DAY_COLUMNS,
}
AGE_REQUIRED = {"patient_id": PATIENT_ID_COLUMNS, "age": ("age_at_diagnosis", "diagnosis_age", "age")}
CLINICAL_REQUIRED = {
    "patient_id": PATIENT_ID_COLUMNS,
    "days_since_diagnosis": ("days_since_diagnosis", "day", "Dager_siden_BC_diagnose"),
    "event_type": ("event_type", "type", "Datatype"),
}
CLINICAL_OPTIONAL = {
    "code": ("code", "procedure_code", "diagnosis_code", "Kode"),
    "description": ("description", "text", "name", "Tekst"),
}
CLINICAL_DROP_REASONS = ("ignored_event_type", "missing_day", "non_parseable_day")
CsvRow = Mapping[str, object]
ColumnAliases = Mapping[str, tuple[str, ...]]


def canonical_csv(
    csv_path: Path,
    required: ColumnAliases,
    optional: ColumnAliases | None = None,
) -> pd.DataFrame:
    """Read a CSV and rename supported source aliases to canonical names."""
    df = pd.read_csv(csv_path)
    optional = {} if optional is None else optional
    selected = {name: first_existing_column(df, aliases) for name, aliases in required.items()}
    selected.update({name: optional_existing_column(df, aliases) for name, aliases in optional.items()})
    renamed = df.rename(columns={column: name for name, column in selected.items() if column is not None})
    for name in optional:
        if name not in renamed:
            renamed[name] = None
    return renamed


def load_split_assignments(split_csv_path: Path) -> dict[str, tuple[str, ...]]:
    """Load one authoritative split assignment per patient; reject repeated IDs."""
    split_df = canonical_csv(split_csv_path, SPLIT_REQUIRED)
    validate_unique_patient_ids(split_df, split_csv_path)
    assignments: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for row in split_df.to_dict("records"):
        patient_id = clean_text(row["patient_id"])
        split = normalize_split(str(row["split"]))
        assignments.setdefault(split, []).append(patient_id)
    return {split: tuple(patient_ids) for split, patient_ids in assignments.items()}


def load_endpoint_patients(endpoint_csv_path: Path) -> dict[str, PatientRecord]:
    endpoint_df = canonical_csv(endpoint_csv_path, ENDPOINT_REQUIRED, ENDPOINT_OPTIONAL)
    validate_unique_patient_ids(endpoint_df, endpoint_csv_path)
    return {
        clean_text(row["patient_id"]): endpoint_patient_record(row)
        for row in endpoint_df.to_dict("records")
    }


def validate_unique_patient_ids(frame: pd.DataFrame, path: Path) -> None:
    patient_ids = frame["patient_id"].map(clean_text)
    duplicates = patient_ids[patient_ids.duplicated()].unique()
    if len(duplicates):
        raise ValueError(f"Duplicate patient IDs {', '.join(duplicates[:10])} in {path}.")


def load_age_by_patient(age_csv_path: Path | None) -> dict[str, float]:
    if age_csv_path is None:
        return {}
    age_df = canonical_csv(age_csv_path, AGE_REQUIRED)
    ages: dict[str, float] = {}
    for row in age_df.to_dict("records"):
        patient_id = clean_text(row["patient_id"])
        age = optional_float(row["age"])
        if age is not None:
            ages[patient_id] = age
    return ages


def load_clinical_events(
    clinical_csv_path: Path | None,
) -> tuple[dict[str, tuple[PatientClinicalEvent, ...]], dict[str, int]]:
    if clinical_csv_path is None:
        return {}, empty_clinical_drop_counts()
    clinical_df = canonical_csv(clinical_csv_path, CLINICAL_REQUIRED, CLINICAL_OPTIONAL)
    grouped: dict[str, list[PatientClinicalEvent]] = {}
    dropped_counts = empty_clinical_drop_counts()
    for row in clinical_df.to_dict("records"):
        event, drop_reason = clinical_event(row)
        if drop_reason is not None:
            dropped_counts[drop_reason] += 1
        if event is not None:
            grouped.setdefault(clean_text(row["patient_id"]), []).append(event)
    events_by_patient = {
        patient_id: tuple(sorted(events, key=lambda event: (event.days_since_diagnosis, event.event_type)))
        for patient_id, events in grouped.items()
    }
    return events_by_patient, dropped_counts


def empty_clinical_drop_counts() -> dict[str, int]:
    return dict.fromkeys(CLINICAL_DROP_REASONS, 0)


def endpoint_patient_record(row: CsvRow) -> PatientRecord:
    patient_id = clean_text(row["patient_id"])
    return PatientRecord(
        patient_id=patient_id,
        diagnosis_year=int(row["diagnosis_year"]),
        death_day=optional_int(row["death_day"]),
        last_observation_day=optional_int(row["last_observation_day"]),
        recurrence_events=parse_endpoint_events(
            parse_sequence(row["recurrence_day"]),
            parse_sequence(row["recurrence_code"]),
            parse_sequence(row["recurrence_description"]),
        ),
    )


def clinical_event(row: CsvRow) -> tuple[PatientClinicalEvent | None, str | None]:
    event_type = normalized_clinical_event_type(row["event_type"])
    if event_type is None:
        return None, "ignored_event_type"
    if is_missing(row["days_since_diagnosis"]):
        return None, "missing_day"
    day = optional_clinical_day(row["days_since_diagnosis"])
    if day is None:
        return None, "non_parseable_day"
    return PatientClinicalEvent(
        days_since_diagnosis=day,
        event_type=event_type,
        code=clean_text(row["code"]),
        description=clean_text(row["description"]),
    ), None


def normalized_clinical_event_type(value: object) -> str | None:
    event_type = clean_text(value).lower()
    if event_type in {"diagnosis", "diagnose", "diagnoses", "diagnoser"}:
        return "diagnosis"
    if event_type in {"procedure", "procedures", "prosedyre", "prosedyrer"}:
        return "procedure"
    return None


def clinical_day(value: object) -> int:
    day = optional_clinical_day(value)
    if day is None:
        raise ValueError(f"Could not parse clinical event day: {value}")
    return day


def optional_clinical_day(value: object) -> int | None:
    if is_missing(value):
        return None
    if isinstance(value, pd.Timedelta):
        return int(value.days)
    if isinstance(value, str):
        stripped = value.strip()
        try:
            return int(float(stripped))
        except ValueError:
            elapsed = pd.to_timedelta(stripped, errors="coerce")
            if pd.isna(elapsed):
                return None
            return int(elapsed.days)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def parse_endpoint_events(
    days: tuple[object, ...],
    codes: tuple[object, ...],
    descriptions: tuple[object, ...],
) -> tuple[PatientEndpointEvent, ...]:
    events: list[PatientEndpointEvent] = []
    max_len = max(len(days), len(codes), len(descriptions), 0)
    for index in range(max_len):
        day = sequence_item(days, index)
        if day is None:
            continue
        code = clean_text(sequence_item(codes, index))
        description = clean_text(sequence_item(descriptions, index))
        events.append(PatientEndpointEvent(int(float(day)), code, description))
    return tuple(sorted(events, key=lambda event: (event.days_since_diagnosis, event.code)))


def sequence_item(values: tuple[object, ...], index: int) -> object | None:
    if not values:
        return None
    if index < len(values):
        return values[index]
    return values[-1]


def parse_sequence(value: object) -> tuple[object, ...]:
    if is_missing(value):
        return ()
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null"}:
            return ()
        if stripped.startswith("[") and stripped.endswith("]"):
            parsed = ast.literal_eval(stripped)
            if not isinstance(parsed, list | tuple):
                return (parsed,)
            return tuple(parsed)
        return (stripped,)
    return (value,)


def optional_int(value: object | None) -> int | None:
    if is_missing(value):
        return None
    return int(float(value))


def optional_float(value: object) -> float | None:
    if is_missing(value):
        return None
    return float(value)


def first_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str:
    column = optional_existing_column(df, candidates)
    if column is None:
        joined = ", ".join(candidates)
        raise ValueError(f"Missing required column. Expected one of: {joined}.")
    return column


def optional_existing_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    normalized = {str(column).lower(): str(column) for column in df.columns}
    for candidate in candidates:
        column = normalized.get(candidate.lower())
        if column is not None:
            return column
    return None


def clean_text(value: object | None) -> str:
    if value is None or is_missing(value):
        return ""
    return str(value).strip()


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, list | tuple):
        return False
    if isinstance(value, float):
        return math.isnan(value)
    return bool(pd.isna(value)) if not isinstance(value, str) else False
