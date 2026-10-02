"""Exchange cohort membership and externally calculated age summaries as CSVs."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, TypedDict

import pandas as pd

from src.data.report_return_lookup import ReportReturn, find_report_return, return_directory
from src.manual_age_summary import AgeSummary, ManualAgeSummary, SPLIT_KEYS

if TYPE_CHECKING:
    from src.data.dataset import PatientBloodTestDataset

PACKAGE_DIR = "cohort_age_export"
MEMBERSHIP_FILENAME = "cohort_membership.csv"
AGE_SUMMARIES_FILENAME = "age_summaries.csv"
SOURCE_MEMBERSHIP_COLUMNS = ("id", "population", "split")
EXPORT_COLUMNS = ("id", "fold", "hash")
SUMMARY_COLUMNS = ("fold", "median", "lower_q", "upper_q", "hash")
AGE_FOLDS = ("train", "validation", "test", "overall", "included", "excluded")


class CohortAgeSummary(TypedDict):
    """Validated external quartiles and cohort provenance retained in report snapshots."""

    cohort_hash: str
    cohort_reference: str
    source_note: str
    age_years: dict[str, dict[str, float]]


def cohort_membership(dataset: PatientBloodTestDataset) -> pd.DataFrame:
    """Share the initial source population and final split partition between reports."""
    patients = dataset.prepared_data.cohort_comparison_patients
    stages = dataset.prepared_data.cohort_stages
    source_ids = {patient.patient_id for patient in patients}
    if not stages or source_ids != stages[0].patient_ids or len(source_ids) != len(patients):
        raise ValueError("Comparison summaries must cover the initial cohort; regenerate the prepared dataset.")
    included = {
        item.patient_id: "validation" if split == "val" else split
        for split in ("train", "val", "test")
        for item in dataset.prepared_data.items_by_split[split]
    }
    if not set(included).issubset(source_ids):
        raise ValueError("Included patients must belong to the initial cohort population.")
    return pd.DataFrame([
        {"id": patient.patient_id, "population": "included" if patient.patient_id in included else "excluded",
         "split": included.get(patient.patient_id, "")}
        for patient in patients
    ], columns=SOURCE_MEMBERSHIP_COLUMNS).sort_values("id").reset_index(drop=True)


def find_age_summary(cohort_hash: str) -> ReportReturn[CohortAgeSummary] | None:
    return find_report_return(
        return_directory("ages"), "hash", cohort_hash,
        load=lambda path: load_age_summaries(path, cohort_hash),
        equivalent=lambda left, right: left == right,
    )


def age_display_summary(ages: CohortAgeSummary | None) -> ManualAgeSummary:
    """Round display ages to one decimal; no shared match clears every age cell."""
    if ages is None:
        return ManualAgeSummary({fold: None for fold in SPLIT_KEYS}, "Unavailable", "No matching shared age CSV.")
    summary = ManualAgeSummary.from_mapping(ages)
    return replace(summary, age_years={
        fold: None if value is None else AgeSummary(round(value.median, 1), round(value.q1, 1), round(value.q3, 1))
        for fold, value in summary.age_years.items()
    })


def membership_hash(membership: pd.DataFrame) -> str:
    """Hash the full partition, independent of row order; preserve saved-run hashes.

    Each included patient has exactly one split and belongs to overall/included.
    Each other patient belongs only to excluded, so this partition determines
    every exported (id, fold) pair.
    """
    rows = membership[list(SOURCE_MEMBERSHIP_COLUMNS)].fillna("").astype(str)
    if rows["id"].str.strip().eq("").any() or rows["id"].duplicated().any():
        raise ValueError("Cohort membership requires unique nonempty patient IDs.")
    included = rows["population"].eq("included")
    valid = included & rows["split"].isin(AGE_FOLDS[:3])
    valid |= rows["population"].eq("excluded") & rows["split"].eq("")
    if not valid.all():
        raise ValueError("Invalid cohort population or split membership.")
    content = {"version": 1, "patients": rows.sort_values("id").to_dict("records")}
    encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def build_age_membership(membership: pd.DataFrame, cohort_hash: str) -> pd.DataFrame:
    """Expand each included patient into split, overall and included rows."""
    if membership_hash(membership) != cohort_hash:
        raise ValueError("Cohort membership does not match the saved cohort hash.")
    rows: list[tuple[str, str, str]] = []
    for patient_id, population, split in membership[list(SOURCE_MEMBERSHIP_COLUMNS)].itertuples(index=False, name=None):
        folds = (split, "overall", "included") if population == "included" else ("excluded",)
        rows.extend((str(patient_id), fold, cohort_hash) for fold in folds)
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS).sort_values(["id", "fold"]).reset_index(drop=True)


def read_exchange_csv(path: Path, columns: tuple[str, ...]) -> list[dict[str, str]]:
    """Read literal IDs/hashes and reject missing, duplicate or unexpected columns."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(columns) or set(reader.fieldnames) != set(columns):
            raise ValueError(f"{path.name} must contain exactly these columns: {','.join(columns)}.")
        rows = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path.name}: malformed CSV row {reader.line_num}.")
            rows.append({key: row[key] for key in columns})
    return rows


def load_cohort_membership(path: Path, cohort_hash: str) -> pd.DataFrame:
    """Recover the unique patient partition from current or legacy saved exports."""
    with path.open(encoding="utf-8-sig", newline="") as source:
        columns = next(csv.reader(source), [])
    if set(columns) == set(SOURCE_MEMBERSHIP_COLUMNS):
        membership = pd.DataFrame(read_exchange_csv(path, SOURCE_MEMBERSHIP_COLUMNS), columns=SOURCE_MEMBERSHIP_COLUMNS)
        build_age_membership(membership, cohort_hash)
        return membership
    rows = read_exchange_csv(path, EXPORT_COLUMNS)
    membership = pd.DataFrame([
        {"id": row["id"], "population": "excluded" if row["fold"] == "excluded" else "included",
         "split": "" if row["fold"] == "excluded" else row["fold"]}
        for row in rows if row["fold"] in (*AGE_FOLDS[:3], "excluded")
    ], columns=SOURCE_MEMBERSHIP_COLUMNS)
    expected = build_age_membership(membership, cohort_hash)
    exported = pd.DataFrame(rows, columns=EXPORT_COLUMNS).sort_values(["id", "fold"]).reset_index(drop=True)
    if not exported.equals(expected):
        raise ValueError("Saved membership must contain every (id, fold) pair once, with the matching cohort hash.")
    return membership


def load_age_summaries(path: Path, cohort_hash: str) -> CohortAgeSummary:
    """Validate the complete remote return before either paper output is changed."""
    rows = read_exchange_csv(path, SUMMARY_COLUMNS)
    if len(rows) != len(AGE_FOLDS) or {row["fold"] for row in rows} != set(AGE_FOLDS):
        raise ValueError(f"Age summaries must contain each fold exactly once: {', '.join(AGE_FOLDS)}.")
    if any(row["hash"] != cohort_hash for row in rows):
        raise ValueError("Age summaries do not match this run's cohort membership hash.")
    summaries: dict[str, dict[str, float]] = {}
    for row in rows:
        fold = row["fold"]
        try:
            values = {key: float(row[column]) for key, column in
                      (("median", "median"), ("q1", "lower_q"), ("q3", "upper_q"))}
        except ValueError as error:
            raise ValueError(
                f"{fold}: median, lower_q and upper_q must all be numeric; missing ages are not allowed."
            ) from error
        summaries[fold] = asdict(AgeSummary.from_mapping(values, fold))
    if summaries["overall"] != summaries["included"]:
        raise ValueError("The overall and included age summaries must match.")
    return {
        "cohort_hash": cohort_hash,
        "cohort_reference": f"Included/excluded source cohort {cohort_hash}",
        "source_note": "Externally calculated exact ages at BC diagnosis; CSV return matched by cohort hash.",
        "age_years": {fold: summaries[fold] for fold in AGE_FOLDS},
    }
