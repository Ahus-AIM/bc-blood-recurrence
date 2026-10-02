from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from src.data.dataset import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.dataset_statistics import split_bloodtest_counts
from src.data.measurement_selection import selected_cohort_measurements
from src.data.recurrence_site_common import site_label
from src.data.recurrence_site_incidence import (
    RecurrenceSiteRule,
    load_item_endpoint_site_rows,
)
from src.data.types import PatientBloodTestItem
from src.reporting import (
    count_percentage_text,
    display_output_path,
    latex_escape,
    report_output_path,
)
from src.time_units import DAYS_PER_YEAR

if TYPE_CHECKING:
    from src.model.blood_test_imputation_statistics import (
        BloodTestImputationStatistics,
        BloodTestValueSourceCounts,
    )

SPLIT_KEYS = ("train", "val", "test")
COLUMN_KEYS = (*SPLIT_KEYS, "overall")
COLUMN_LABELS = ("Train", "Validation", "Test", "Overall")
DEFAULT_SITE_LABELS = ("Bone", "Brain", "Liver", "Lung", "Pleura")
DEFAULT_SITE_REGEXES = (r"C795\d*", r"C793\d*", r"C787\d*", r"C780\d*", r"C782\d*")
DEFAULT_TABLE_CAPTION = (
    "Population statistics. Only recurrence sites registered on the first recurrence day are counted."
)
DEFAULT_TABLE_LABEL = "tab:population_statistics"
DEFAULT_OUTPUT_LATEX_PATH = "population_statistics.tex"


@dataclass(frozen=True)
class PopulationStatisticsConfig:
    """LaTeX output settings and recurrence-site rules for the population table.

    `output_latex_path` is resolved relative to the training output directory
    unless absolute. `site_rules` defines the ordered site rows. `other_label`
    enables a final row for first-day recurrence events not matched by any
    explicit site rule.
    """

    output_latex_path: Path
    table_caption: str
    table_label: str
    site_rules: tuple[RecurrenceSiteRule, ...]
    other_label: str | None

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object] | None,
    ) -> PopulationStatisticsConfig:
        values = {} if config is None else dict(config)
        site_labels = string_tuple(values.get("site_labels", DEFAULT_SITE_LABELS), "site_labels")
        site_regexes = string_tuple(values.get("site_regexes", DEFAULT_SITE_REGEXES), "site_regexes")
        if len(site_labels) != len(site_regexes):
            raise ValueError("`population_statistics.site_labels` and `site_regexes` must have the same length.")
        for regex in site_regexes:
            re.compile(regex)
        return cls(
            output_latex_path=Path(str(values.get("output_latex_path", DEFAULT_OUTPUT_LATEX_PATH))),
            table_caption=str(values.get("table_caption", DEFAULT_TABLE_CAPTION)),
            table_label=str(values.get("table_label", DEFAULT_TABLE_LABEL)),
            site_rules=tuple(
                RecurrenceSiteRule(label, regex) for label, regex in zip(site_labels, site_regexes, strict=True)
            ),
            other_label=optional_text(values.get("other_label", "Other")),
        )


@dataclass(frozen=True)
class ObservationTimeYears:
    """Patient-level observation-time quartiles expressed in years.

    `first_quartile`, `median`, and `third_quartile` summarize the days from BC
    diagnosis to the patient's observation endpoint.
    """

    first_quartile: float
    median: float
    third_quartile: float


@dataclass(frozen=True)
class PopulationStatisticsColumn:
    """Counts and observation-time summary for one table column.

    `patient_count` is the number of eligible patients. `landmark_day_count` is
    the number of valid landmark days across those patients. `bloodtest_count`
    counts all retained selected measurements (including earlier history), plus
    LOCF/MICE current values and MICE-imputed past change inputs when source
    attribution is available.
    `observation_time_years` summarizes time to the first input recurrence or
    resolved right-censoring day.
    `bloodtest_source_counts` attributes those combined values when an
    imputer is available. `recurrence_counts_by_site` counts unique first-day
    recurrence sites.
    """

    patient_count: int
    landmark_day_count: int
    bloodtest_count: int
    observation_time_years: ObservationTimeYears | None
    recurrence_counts_by_site: Mapping[str, int]
    bloodtest_source_counts: BloodTestValueSourceCounts | None = None


@dataclass(frozen=True)
class PopulationStatisticsRow:
    """One display row in the rendered population-statistics LaTeX table."""

    label: str
    cells: tuple[str, ...]
    indented: bool = False


@dataclass(frozen=True)
class PopulationStatisticsTable:
    """Formatted population-statistics rows for train, validation, test, and overall columns."""

    rows: tuple[PopulationStatisticsRow, ...]
    column_labels: tuple[str, ...]


def write_population_statistics_latex(
    dataset: PatientBloodTestDataset,
    config: PopulationStatisticsConfig,
    output_dir: Path,
    *,
    imputation_statistics: BloodTestImputationStatistics | None = None,
) -> dict[str, object]:
    table = build_population_statistics_table(
        dataset,
        config,
        imputation_statistics=imputation_statistics,
    )
    output_path = report_output_path(output_dir, config.output_latex_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(population_statistics_latex(table, config), encoding="utf-8")
    return {"path": display_output_path(output_path, output_dir)}


def build_population_statistics_table(
    dataset: PatientBloodTestDataset,
    config: PopulationStatisticsConfig,
    *,
    imputation_statistics: BloodTestImputationStatistics | None = None,
) -> PopulationStatisticsTable:
    items_by_split = {split: tuple(dataset.prepared_data.items_by_split.get(split, ())) for split in SPLIT_KEYS}
    all_items = tuple(item for split in SPLIT_KEYS for item in items_by_split[split])
    measured_counts = split_bloodtest_counts(
        items_by_split,
        selected_cohort_measurements(dataset.prepared_data),
    )
    columns = {
        **{
            split: population_statistics_column(
                items,
                config,
                dataset.config,
                measured_bloodtest_count=measured_counts[split],
                bloodtest_source_counts=(
                    None if imputation_statistics is None else imputation_statistics.by_split[split]
                ),
            )
            for split, items in items_by_split.items()
        },
        "overall": population_statistics_column(
            all_items,
            config,
            dataset.config,
            measured_bloodtest_count=sum(measured_counts.values()),
            bloodtest_source_counts=(None if imputation_statistics is None else imputation_statistics.overall),
        ),
    }
    return PopulationStatisticsTable(
        rows=population_statistics_rows(columns, site_labels(config)),
        column_labels=COLUMN_LABELS,
    )


def population_statistics_column(
    items: tuple[PatientBloodTestItem, ...],
    config: PopulationStatisticsConfig,
    dataset_config: PatientDatasetConfig,
    *,
    measured_bloodtest_count: int,
    bloodtest_source_counts: BloodTestValueSourceCounts | None = None,
) -> PopulationStatisticsColumn:
    observation_days = [end_day for item in items if (end_day := observation_end_day(item)) is not None]
    endpoint_rows = load_item_endpoint_site_rows(items, dataset_config)
    return PopulationStatisticsColumn(
        patient_count=len(items),
        landmark_day_count=sum(len(item.valid_landmark_days) for item in items),
        bloodtest_count=(
            measured_bloodtest_count if bloodtest_source_counts is None else bloodtest_source_counts.total
        ),
        observation_time_years=summarize_observation_time_years(observation_days),
        recurrence_counts_by_site=recurrence_site_counts(endpoint_rows, config),
        bloodtest_source_counts=bloodtest_source_counts,
    )


def population_statistics_rows(
    columns: Mapping[str, PopulationStatisticsColumn],
    recurrence_site_labels: tuple[str, ...],
) -> tuple[PopulationStatisticsRow, ...]:
    bloodtest_rows = [
        PopulationStatisticsRow(
            "Blood tests",
            count_cells(columns, lambda column: column.bloodtest_count),
        )
    ]
    if any(column.bloodtest_source_counts is not None for column in columns.values()):
        bloodtest_rows.extend(
            (
                PopulationStatisticsRow(
                    "Measured",
                    bloodtest_source_cells(columns, lambda counts: counts.measured),
                    indented=True,
                ),
                PopulationStatisticsRow(
                    "LOCF",
                    bloodtest_source_cells(columns, lambda counts: counts.locf),
                    indented=True,
                ),
                PopulationStatisticsRow(
                    "MICE",
                    bloodtest_source_cells(columns, lambda counts: counts.mice),
                    indented=True,
                ),
            )
        )
    return (
        PopulationStatisticsRow(
            "Patients",
            count_cells(columns, lambda column: column.patient_count),
        ),
        PopulationStatisticsRow(
            "Landmark days",
            count_cells(columns, lambda column: column.landmark_day_count),
        ),
        *bloodtest_rows,
        PopulationStatisticsRow(
            "Median observation time, years",
            tuple(format_observation_time_cell(columns[key]) for key in COLUMN_KEYS),
        ),
        PopulationStatisticsRow(
            "Recurrences",
            count_cells(
                columns,
                lambda column: sum(column.recurrence_counts_by_site.get(site, 0) for site in recurrence_site_labels),
            ),
        ),
        *(
            PopulationStatisticsRow(
                site,
                count_cells(columns, lambda column, site=site: column.recurrence_counts_by_site.get(site, 0)),
                indented=True,
            )
            for site in recurrence_site_labels
        ),
    )


def bloodtest_source_cells(
    columns: Mapping[str, PopulationStatisticsColumn],
    value_getter: Callable[[BloodTestValueSourceCounts], int],
) -> tuple[str, ...]:
    cells = []
    for key in COLUMN_KEYS:
        column = columns[key]
        counts = column.bloodtest_source_counts
        if counts is None or counts.total == 0:
            cells.append("--")
            continue
        text = count_percentage_text(value_getter(counts), counts.total)
        cells.append(text.replace("%", r"\%"))
    return tuple(cells)


def count_cells(
    columns: Mapping[str, PopulationStatisticsColumn],
    value_getter: Callable[[PopulationStatisticsColumn], int],
) -> tuple[str, ...]:
    return tuple(format_count_cell(columns[key], int(value_getter(columns[key]))) for key in COLUMN_KEYS)


def format_count_cell(column: PopulationStatisticsColumn, value: int) -> str:
    if column.patient_count == 0:
        return "--"
    return f"{value:,}"


def format_observation_time_cell(column: PopulationStatisticsColumn) -> str:
    summary = column.observation_time_years
    if column.patient_count == 0 or summary is None:
        return "--"
    return f"{summary.median:.1f} ({summary.first_quartile:.1f}--{summary.third_quartile:.1f})"


def observation_end_day(item: PatientBloodTestItem) -> int | None:
    candidates = (
        item.input_recurrence_day,
        item.right_censoring_day,
    )
    observed_days = [day for day in candidates if day is not None]
    if not observed_days:
        return None
    return int(min(observed_days))


def summarize_observation_time_years(
    observation_days: list[int],
) -> ObservationTimeYears | None:
    if not observation_days:
        return None
    quartiles = np.percentile(np.asarray(observation_days, dtype=float), (25, 50, 75)) / DAYS_PER_YEAR
    return ObservationTimeYears(
        first_quartile=float(quartiles[0]),
        median=float(quartiles[1]),
        third_quartile=float(quartiles[2]),
    )


def recurrence_site_counts(
    endpoint_rows: pd.DataFrame,
    config: PopulationStatisticsConfig,
) -> dict[str, int]:
    """Count unique first-day recurrence sites from prepared endpoint rows."""
    if endpoint_rows.empty:
        return {}
    event_rows = endpoint_rows.loc[endpoint_rows["days_since_diagnosis"].notna()]
    counts = Counter[str]()
    for _, patient_rows in event_rows.groupby("patient_id", sort=False):
        labels = {first_matching_site_label(str(code), config) for code in patient_rows["code"]}
        counts.update(label for label in labels if label is not None)
    return dict(counts)


def first_matching_site_label(
    code: str,
    config: PopulationStatisticsConfig,
) -> str | None:
    return site_label(code, config.site_rules, config.other_label)


def site_labels(config: PopulationStatisticsConfig) -> tuple[str, ...]:
    labels = tuple(rule.label for rule in config.site_rules)
    if config.other_label is None:
        return labels
    return (*labels, config.other_label)


def population_statistics_latex(
    table: PopulationStatisticsTable,
    config: PopulationStatisticsConfig,
) -> str:
    lines = [
        r"\begin{table}[H]",
        r"\centering",
        rf"\caption{{{config.table_caption}}}",
        rf"\label{{{config.table_label}}}",
        r"\begin{tabular}{l|ccc|c}",
        r"\hline",
        "& " + " & ".join(table.column_labels) + r" \\",
        r"\hline",
    ]
    lines.extend(latex_row(row) for row in table.rows)
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table}"))
    return "\n".join(lines) + "\n"


def latex_row(row: PopulationStatisticsRow) -> str:
    label = latex_escape(row.label)
    if row.indented:
        label = rf"\quad {label}"
    return f"{label} & " + " & ".join(row.cells) + r" \\"


def string_tuple(value: object, config_key: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"`population_statistics.{config_key}` must be a list.")
    values = tuple(str(item).strip() for item in value)
    if any(item == "" for item in values):
        raise ValueError(f"`population_statistics.{config_key}` cannot contain empty values.")
    return values


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
