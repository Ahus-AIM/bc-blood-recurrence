"""Run-local composition and annual curve summaries for the six-panel figure."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from src.data.dataset import PatientBloodTestDataset
from src.data.population_statistics import (
    PopulationStatisticsConfig,
    PopulationStatisticsRow,
    PopulationStatisticsTable,
    SPLIT_KEYS,
    build_population_statistics_table,
)
from src.data.recurrence_site_incidence import (
    first_recurrence_rows,
    first_recurrence_rows_any,
    survival_durations,
)
from src.data.recurrence_site_common import patients_at_risk, validate_min_patients_at_risk
from src.model.blood_test_imputation_statistics import BloodTestImputationStatistics
from src.time_units import DAYS_PER_YEAR


@dataclass(frozen=True)
class AnnualCurveTable:
    """Display cells by site and integer year; unavailable estimates use an em dash."""

    years: tuple[float, ...]
    sites: tuple[str, ...]
    cells: tuple[tuple[str, ...], ...]

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> AnnualCurveTable:
        table = cls(tuple(raw["years"]), tuple(raw["sites"]), tuple(tuple(row) for row in raw["cells"]))
        if len(table.sites) != len(table.cells) or any(len(row) != len(table.years) for row in table.cells):
            raise ValueError("Annual curve table dimensions must match its sites and years.")
        return table


@dataclass(frozen=True)
class CohortFigureData:
    """Prepared population rows, observed diagnosis-year ranges, and annual curve tables.

    Ages are supplied separately through the manual-age config so they can be
    updated without recomputing this run-local snapshot. Annual incidence cells
    sample saved step estimates; survival cells contain pre-time risk counts,
    with small counts bounded independently of the displayed KM steps.
    """

    population: PopulationStatisticsTable
    split_years: tuple[str, ...]
    incidence: AnnualCurveTable
    survival: AnnualCurveTable

    @classmethod
    def from_mapping(cls, raw: dict[str, Any]) -> CohortFigureData:
        population = raw["population"]
        rows = tuple(
            PopulationStatisticsRow(row["label"], tuple(row["cells"]), row["indented"])
            for row in population["rows"]
        )
        if len(raw["split_years"]) != 4 or any(len(row.cells) != 4 for row in rows):
            raise ValueError("Cohort composition requires train, validation, test and overall columns.")
        return cls(
            PopulationStatisticsTable(rows, tuple(population["column_labels"])),
            tuple(raw["split_years"]),
            AnnualCurveTable.from_mapping(raw["incidence"]),
            AnnualCurveTable.from_mapping(raw["survival"]),
        )


def composition_table(
    dataset: PatientBloodTestDataset,
    config: PopulationStatisticsConfig,
    imputation_statistics: BloodTestImputationStatistics | None,
    *,
    endpoint_rows: pd.DataFrame,
    min_days: int,
) -> PopulationStatisticsTable:
    """Show occurrences (unique patients), keeping site percentages occurrence-based."""
    table = build_population_statistics_table(dataset, config, imputation_statistics=imputation_statistics)
    total = next(row.cells for row in table.rows if row.label == "Recurrences")
    unique_counts = recurrence_patient_counts(dataset, endpoint_rows, min_days)
    labels = {
        "Blood tests": "Blood test values",
        "Median observation time, years": "Observation, years",
        "Recurrences": "Site occurrences*",
    }
    rows = []
    site_rows = False
    for row in table.rows:
        cells = tuple(display_cell(cell) for cell in row.cells)
        if row.label == "Recurrences":
            cells = tuple("—" if cell == "—" else f"{cell} ({count:,})"
                          for cell, count in zip(cells, unique_counts, strict=True))
        if site_rows:
            cells = tuple(site_percentage(cell, denominator) for cell, denominator in zip(row.cells, total, strict=True))
        rows.append(PopulationStatisticsRow(labels.get(row.label, row.label), cells, row.indented))
        if row.label == "Recurrences":
            site_rows = True
    return PopulationStatisticsTable(tuple(rows), table.column_labels)


def recurrence_patient_counts(
    dataset: PatientBloodTestDataset, endpoint_rows: pd.DataFrame, min_days: int,
) -> tuple[int, ...]:
    """Count the Any survival population once per split and once across the cohort."""
    if endpoint_rows.empty:
        return (0,) * 4
    rows = first_recurrence_rows_any(endpoint_rows, min_days)
    split_rows = []
    for split in SPLIT_KEYS:
        ids = tuple(item.patient_id for item in dataset.prepared_data.items_by_split.get(split, ()))
        split_rows.append(rows.loc[rows["patient_id"].isin(ids)])
    return tuple(survival_counts_at_years(frame, (0,))[0] for frame in (*split_rows, rows))


def survival_counts_at_years(rows: pd.DataFrame, years: tuple[int, ...]) -> tuple[int, ...]:
    """Use the same finite follow-up and inclusive risk-set boundaries as panel f."""
    durations, _ = survival_durations(rows, "recurrence")
    return tuple(patients_at_risk(durations, year * DAYS_PER_YEAR) for year in years)


def display_cell(cell: str) -> str:
    return "—" if cell == "--" else cell.replace(r"\%", "%").replace("--", "–")


def site_percentage(cell: str, denominator: str) -> str:
    if cell == "--" or denominator == "--":
        return "—"
    total = int(denominator.replace(",", ""))
    if total == 0:
        return f"{cell} (—)"
    percent = 100 * int(cell.replace(",", "")) / total
    return f"{cell} ({percent:.1f}%)"


def diagnosis_year_ranges(dataset: PatientBloodTestDataset) -> tuple[str, ...]:
    """Report actual diagnosis-year ranges of included patients in each saved split."""
    years = [tuple(item.diagnosis_year for item in dataset.prepared_data.items_by_split.get(split, ()))
             for split in SPLIT_KEYS]
    years.append(tuple(year for split in years for year in split))
    return tuple(year_range(split) for split in years)


def year_range(years: tuple[int, ...]) -> str:
    if not years:
        return "—"
    first, last = min(years), max(years)
    return str(first) if first == last else f"{first}–{last}"


def annual_incidence_table(frame: pd.DataFrame, sites: tuple[str, ...], max_year: int) -> AnnualCurveTable:
    years = tuple(range(max_year + 1))
    cells = []
    for site in sites:
        points = frame.loc[frame["site"].eq(site)].sort_values("years_since_diagnosis")
        cells.append(tuple(incidence_at_year(points, year) for year in years))
    return AnnualCurveTable(years, sites, tuple(cells))


def incidence_at_year(points: pd.DataFrame, year: int) -> str:
    """Sample the right-continuous estimate only within its supported time range."""
    if points.empty or year > points["years_since_diagnosis"].max():
        return "—"
    observed = points.loc[points["years_since_diagnosis"] <= year]
    if observed.empty:
        return "—"
    return f"{observed.iloc[-1]['cumulative_incidence_percent']:.1f}"


def annual_survival_table(
    endpoint_rows: pd.DataFrame,
    config: PopulationStatisticsConfig,
    sites: tuple[str, ...],
    *,
    overall_label: str | None,
    max_year: int,
    min_days: int,
    min_patients_at_risk: int,
) -> AnnualCurveTable:
    site_rows = first_recurrence_rows(endpoint_rows, config.site_rules, min_days, config.other_label)
    years = tuple(range(max_year + 1))
    cells = []
    for site in sites:
        rows = (first_recurrence_rows_any(endpoint_rows, min_days) if site == overall_label
                else site_rows.loc[site_rows["site"].eq(site)])
        cells.append(tuple(f"{count:,}" for count in survival_counts_at_years(rows, years)))
    return suppress_risk_counts(AnnualCurveTable(years, sites, tuple(cells)), min_patients_at_risk)


def suppress_risk_counts(table: AnnualCurveTable, threshold: int) -> AnnualCurveTable:
    """Bound small risk sets; the last KM death time does not define count availability.

    Older snapshots may contain ``risk (events)`` cells; only their risk count
    is retained. Previously suppressed values are never reconstructed.
    """
    validate_min_patients_at_risk(threshold)
    cells = tuple(tuple(risk_count_text(cell, threshold) for cell in row) for row in table.cells)
    return AnnualCurveTable(table.years, table.sites, cells)


def risk_count_text(cell: str, threshold: int) -> str:
    count = cell.partition("(")[0].strip().replace(",", "")
    if count == "—":
        return count
    if count.startswith("≤"):
        # A lower threshold cannot recover a count hidden in an earlier export.
        return f"≤{max(threshold, int(count[1:])):,}"
    value = int(count)
    if value <= threshold:
        return f"≤{threshold:,}"
    return f"{value:,}"
