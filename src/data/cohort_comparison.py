"""Descriptive, full-history comparison of included and excluded source patients."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.dataset import PatientBloodTestDataset
from src.data.cohort_comparison_data import other_cancer_pattern
from src.data.population_statistics import PopulationStatisticsConfig
from src.data.recurrence_site_common import site_label
from src.reporting import latex_escape
from src.manual_age_summary import display_number
from src.data.cohort_age_exchange import cohort_membership, membership_hash
from src.time_units import DAYS_PER_YEAR

VALUES_FILENAME = "cohort_comparison_values.json"
COMPARISON_VERSION = 4
TABLE_STEM = "cohort_comparison"
POPULATIONS = ("included", "excluded")


def build_comparison_payload(
    dataset: PatientBloodTestDataset,
    config: PopulationStatisticsConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Use the prepared cohort only for membership; summaries use source histories."""
    patients = dataset.prepared_data.cohort_comparison_patients
    membership = cohort_membership(dataset)
    included = set(membership.loc[membership["population"].eq("included"), "id"])
    sites = [rule.label for rule in config.site_rules]
    if config.other_label is not None:
        sites.append(config.other_label)
    if len(set(sites)) != len(sites):
        raise ValueError("Cohort comparison site labels must be unique.")
    columns: dict[str, Any] = {}
    for population in POPULATIONS:
        group = [patient for patient in patients if (patient.patient_id in included) == (population == "included")]
        patient_sites = [
            {site_label(code, config.site_rules, config.other_label) for code in patient.metastasis_codes}
            for patient in group
        ]
        days = [patient.observation_day for patient in group if patient.observation_day is not None]
        columns[population] = {
            "patients": len(group),
            "diagnosis_year": quartiles([patient.diagnosis_year for patient in group]),
            "observation_years": quartiles([day / DAYS_PER_YEAR for day in days]),
            "observation_missing": len(group) - len(days),
            "deaths": sum(patient.death_recorded for patient in group),
            "other_cancer": (None if any(patient.other_cancer is None for patient in group)
                             else sum(bool(patient.other_cancer) for patient in group)),
            "distant_metastasis": sum(bool(patient.metastasis_codes) for patient in group),
            "sites": {site: sum(site in recorded for recorded in patient_sites) for site in sites},
            "blood_tests": sum(patient.blood_test_count for patient in group),
            "blood_tests_per_patient": quartiles([patient.blood_test_count for patient in group]),
        }
    return {
        "version": COMPARISON_VERSION,
        "cohort_hash": membership_hash(membership),
        "input_fingerprint": dataset.prepared_data.input_fingerprint,
        "require_diagnosis_age": dataset.config.require_diagnosis_age,
        "selected_test_keys": list(dataset.selected_test_keys),
        "site_rules": [asdict(rule) for rule in config.site_rules],
        "other_cancer_code_regex": other_cancer_pattern(dataset.config),
        "columns": columns,
        "ages": None,
    }, membership


def quartiles(values: list[int] | list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return dict(zip(("q1", "median", "q3"), map(float, np.percentile(values, [25, 50, 75])), strict=True))


def summary_cell(summary: dict[str, float] | None, *, decimal_places: int | None = None) -> str:
    if summary is None:
        return "—"
    values = [summary[key] for key in ("median", "q1", "q3")]
    rendered = [display_number(value) if decimal_places is None else f"{value:.{decimal_places}f}" for value in values]
    return f"{rendered[0]} ({rendered[1]}–{rendered[2]})"


def percentage_cell(count: int | None, patients: int) -> str:
    if count is None or patients == 0:
        return "—"
    return f"{count:,} ({100 * count / patients:.1f}%)"


def comparison_rows(payload: dict[str, Any]) -> list[tuple[str, tuple[str, ...]]]:
    """Render the included/excluded comparison table rows."""
    columns = [payload["columns"][population] for population in POPULATIONS]
    ages = payload["ages"]
    age_cells = tuple(
        summary_cell(None if ages is None else ages["age_years"][group], decimal_places=1)
        for group in ("overall", "excluded")
    )
    rows = [("Age at BC diagnosis, years", age_cells)]
    rows.extend([
        ("BC diagnosis year", tuple(summary_cell(column["diagnosis_year"]) for column in columns)),
        ("Observation time, years", tuple(
            summary_cell(column["observation_years"], decimal_places=1) for column in columns
        )),
    ])
    for label, key in (("Recorded death, n (%)", "deaths"),
                       ("Registered other cancer, n (%)", "other_cancer"),
                       ("Any registered distant recurrence, n (%)", "distant_metastasis")):
        rows.append((label, tuple(percentage_cell(column[key], column["patients"]) for column in columns)))
    for site in columns[0]["sites"]:
        rows.append((f"  {site}, n (%)", tuple(
            percentage_cell(column["sites"][site], column["patients"]) for column in columns
        )))
    rows.extend([
        ("Blood test results, total", tuple(f"{column['blood_tests']:,}" for column in columns)),
        ("Blood test results per patient", tuple(summary_cell(column["blood_tests_per_patient"]) for column in columns)),
    ])
    return rows


def comparison_notes(payload: dict[str, Any]) -> str:
    notes = (
        "Age, diagnosis year, observation time and blood test results per patient are median (Q1–Q3). "
        "Percentages use all patients in the respective column. Excluded includes every initial-cohort patient absent "
        "from the final study cohort, including zero-test patients. Observation runs from BC "
        "diagnosis to last observation capped at death, without other-cancer or recurrence censoring; "
        "one year is 365 days. Observation ends before BC diagnosis count as missing observation times. "
        "Deaths count patients with a recorded death date anywhere in the source history, "
        "including before BC diagnosis and after model follow-up. "
        "Cancer rows count any registered diagnosis across the full history, "
        "including before BC diagnosis and after death. Patients count once per category; distant recurrence sites overlap. "
        f"Blood counts use the same {len(payload['selected_test_keys'])} selected test types in both columns, "
        "one result per patient/day/test type (name and unit), across all recorded dates including after death. "
        "Missing, nonnumeric and imputed results do not count. Per-patient summaries include zero counts. "
        "Age quartiles are supplied by the external age workflow. "
        "Missing observation times are omitted only from their corresponding quartiles. "
        "An em dash means unavailable."
    )
    if payload["require_diagnosis_age"]:
        notes += " Patients without the required diagnosis age are removed before both comparison columns."
    if payload["ages"] is None:
        notes += " Exact-age summaries await the matching remote age return."
    if any(column["other_cancer"] is None for column in payload["columns"].values()):
        notes += " Other-cancer counts are unavailable because the clinical diagnosis source was unavailable."
    return notes


def validate_comparison_version(payload: dict[str, Any]) -> None:
    """Require current source-history definitions, including recorded death counts."""
    if payload.get("version") != COMPARISON_VERSION:
        raise ValueError(
            "Saved comparison values use an incompatible definition; regenerate the cohort comparison "
            "by resuming the benchmark with the source data."
        )


def write_comparison_outputs(payload: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    validate_comparison_version(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"{population.title()} (N={payload['columns'][population]['patients']:,})" for population in POPULATIONS]
    rows = comparison_rows(payload)
    pd.DataFrame([(label.strip(), *cells) for label, cells in rows], columns=["Characteristic", *columns]).to_csv(
        output_dir / f"{TABLE_STEM}.csv", index=False,
    )
    lines = [r"\begin{table}[htbp]", r"\centering", r"\caption{Included and excluded source populations.}",
             r"\label{tab:cohort_comparison}", r"\begin{tabular}{lrr}", r"\hline",
             " & ".join(map(latex_escape, ["Characteristic", *columns])) + r" \\", r"\hline"]
    for label, cells in rows:
        label_cell = (r"\quad " if label.startswith("  ") else "") + latex_escape(label.strip())
        lines.append(" & ".join([label_cell, *(latex_escape(cell).replace("—", "--").replace("–", "--")
                                                 for cell in cells)]) + r" \\")
    lines.extend([r"\hline", r"\end{tabular}", r"\par\smallskip", r"\begin{minipage}{\linewidth}\footnotesize",
                  latex_escape(comparison_notes(payload)).replace("—", "--").replace("–", "--"),
                  r"\end{minipage}", r"\end{table}"])
    (output_dir / f"{TABLE_STEM}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / VALUES_FILENAME).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return {"csv_path": f"{TABLE_STEM}.csv", "latex_path": f"{TABLE_STEM}.tex",
            "values": {"path": VALUES_FILENAME}, "cohort_hash": payload["cohort_hash"]}
