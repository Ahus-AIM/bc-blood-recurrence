from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, replace
from itertools import groupby
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.dataset import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.icd_codes import matches_icd_code
from src.data.landmarks import patient_targets
from src.data.other_cancer_outcomes import (
    matching_recurrence_events,
    recurrence_precedes_censoring,
    resolve_endpoint_status,
)
from src.data.recurrence_site_incidence import RecurrenceSiteRule
from src.data.types import PatientBloodTestItem, PatientEndpointEvent
from src.reporting import display_output_path, latex_escape, recurrence_endpoint_label, report_output_path

SPLIT_KEYS = ("train", "val", "test")
DISPLAY_SPLITS = (
    ("train", "Train"),
    ("val", "Validation"),
    ("test", "Test"),
    ("overall", "Overall"),
)
COUNT_KEYS = (
    "event_landmark_count",
    "landmark_count",
    "event_patient_count",
    "patient_count",
    "recurrence_patient_count",
)
DEFAULT_OUTPUT_CSV_PATH = "landmark_event_counts.csv"
DEFAULT_OUTPUT_LATEX_PATH = "landmark_event_counts.tex"
DEFAULT_TABLE_CAPTION = (
    "Patients with at least one event landmark by recurrence site and censoring "
    "horizon. Dataset-fold labels show first-recurrence patient counts; cells "
    "show event patients and percentages of those fold counts."
)
DEFAULT_TABLE_LABEL = "tab:landmark_event_counts"


@dataclass(frozen=True)
class LandmarkEventCountsHorizon:
    """Censoring horizon used to build landmark-level target-event counts.

    `label` is the display value used in output tables. `days` is the
    post-landmark censoring horizon passed to target construction.
    """

    label: str
    days: int


@dataclass(frozen=True)
class LandmarkEventCountsSite:
    """Recurrence target represented by a label and endpoint-code regex."""

    label: str
    endpoint_regex: str


@dataclass(frozen=True)
class LandmarkEventCountValues:
    """Patient-landmark and patient counts for one site, horizon, and split.

    `landmark_count` is the number of non-bootstrapped valid landmark rows.
    `event_landmark_count` is the subset with the configured target event.
    `patient_count` is the number of patients contributing landmark rows.
    `recurrence_patient_count` is the number of patients whose first input
    recurrence has this row's site.
    `event_patient_count` is the subset with at least one target-event row.
    """

    landmark_count: int
    event_landmark_count: int
    patient_count: int
    recurrence_patient_count: int
    event_patient_count: int


@dataclass(frozen=True)
class LandmarkEventCountsConfig:
    """Output paths, site rules, and horizons for landmark event-count export.

    Relative output paths are resolved under the configured model output
    directory. `include_all_site` adds an all-input-recurrence target row.
    `other_label` adds a target row for input recurrence codes not matching an
    explicit site rule.
    """

    output_csv_path: Path
    output_latex_path: Path
    table_caption: str
    table_label: str
    include_all_site: bool
    all_site_label: str
    other_label: str | None
    site_rules: tuple[RecurrenceSiteRule, ...]
    horizons: tuple[LandmarkEventCountsHorizon, ...]

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object] | None,
        *,
        output_dir: Path,
    ) -> LandmarkEventCountsConfig:
        values = {} if config is None else dict(config)
        site_labels = string_tuple(values.get("site_labels", ()), "site_labels")
        site_regexes = string_tuple(values.get("site_regexes", ()), "site_regexes")
        validate_site_rules(site_labels, site_regexes)
        return cls(
            output_csv_path=report_output_path(
                output_dir,
                values.get("output_csv_path", DEFAULT_OUTPUT_CSV_PATH),
            ),
            output_latex_path=report_output_path(
                output_dir,
                values.get("output_latex_path", DEFAULT_OUTPUT_LATEX_PATH),
            ),
            table_caption=str(values.get("table_caption", DEFAULT_TABLE_CAPTION)),
            table_label=str(values.get("table_label", DEFAULT_TABLE_LABEL)),
            include_all_site=bool(values.get("include_all_site", True)),
            all_site_label=non_empty_text(values.get("all_site_label", "All"), "all_site_label"),
            other_label=optional_text(values.get("other_label", "Other")),
            site_rules=tuple(
                RecurrenceSiteRule(label, regex) for label, regex in zip(site_labels, site_regexes, strict=True)
            ),
            horizons=horizon_tuple(values.get("horizons", ())),
        )


def build_landmark_event_count_frame(
    dataset_config: PatientDatasetConfig,
    config: LandmarkEventCountsConfig,
    *,
    prepared_dataset: PatientBloodTestDataset | None = None,
) -> pd.DataFrame:
    """Count every configured target using the prepared cohort and landmark days."""
    dataset = PatientBloodTestDataset(dataset_config) if prepared_dataset is None else prepared_dataset
    input_regex = re.compile(dataset_config.input_recurrence_code_filter_regex)
    rows: list[dict[str, object]] = []
    for site in landmark_event_count_sites(dataset_config, config):
        endpoint_regex = re.compile(site.endpoint_regex)
        for horizon in config.horizons:
            target_config = landmark_event_target_config(dataset_config, site, horizon)
            counts_by_split = split_counts(
                dataset.prepared_data.items_by_split,
                input_regex,
                endpoint_regex,
                target_config,
            )
            rows.append(count_row(site, horizon, counts_by_split))
    return pd.DataFrame(rows)


def write_landmark_event_count_outputs(
    count_df: pd.DataFrame,
    config: LandmarkEventCountsConfig,
    output_dir: Path,
) -> dict[str, object]:
    config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    count_df.to_csv(config.output_csv_path, index=False)
    config.output_latex_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_latex_path.write_text(landmark_event_counts_latex(count_df, config), encoding="utf-8")
    return {
        "csv_path": display_output_path(config.output_csv_path, output_dir),
        "latex_path": display_output_path(config.output_latex_path, output_dir),
    }


def landmark_event_count_sites(
    dataset_config: PatientDatasetConfig,
    config: LandmarkEventCountsConfig,
) -> tuple[LandmarkEventCountsSite, ...]:
    sites: list[LandmarkEventCountsSite] = []
    if config.include_all_site:
        sites.append(
            LandmarkEventCountsSite(
                config.all_site_label,
                dataset_config.input_recurrence_code_filter_regex,
            )
        )
    sites.extend(LandmarkEventCountsSite(rule.label, rule.code_regex) for rule in config.site_rules)
    if config.other_label is not None:
        sites.append(
            LandmarkEventCountsSite(
                config.other_label,
                other_endpoint_regex(dataset_config.input_recurrence_code_filter_regex, config.site_rules),
            )
        )
    return tuple(sites)


def landmark_event_target_config(
    dataset_config: PatientDatasetConfig,
    site: LandmarkEventCountsSite,
    horizon: LandmarkEventCountsHorizon,
) -> PatientDatasetConfig:
    return replace(
        dataset_config,
        recurrence_endpoint_name=endpoint_name(site.label),
        endpoint_ddfs_code_filter_regex=site.endpoint_regex,
        censoring_horizon_days=horizon.days,
    )


def split_counts(
    items_by_split: Mapping[str, tuple[PatientBloodTestItem, ...]],
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
    target_config: PatientDatasetConfig,
) -> dict[str, LandmarkEventCountValues]:
    counts = {
        split: split_landmark_event_count_values(
            items_by_split.get(split, ()),
            input_regex,
            endpoint_regex,
            target_config,
        )
        for split in SPLIT_KEYS
    }
    return {**counts, "overall": overall_counts(tuple(counts.values()))}


def split_landmark_event_count_values(
    items: tuple[PatientBloodTestItem, ...],
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
    target_config: PatientDatasetConfig,
) -> LandmarkEventCountValues:
    landmark_count = 0
    event_landmark_count = 0
    patient_count = 0
    recurrence_patient_count = 0
    event_patient_count = 0
    for item in items:
        landmark_days = np.asarray(item.valid_landmark_days, dtype=np.int32)
        if len(landmark_days) == 0:
            continue
        target_item = recurrence_target_item(
            item,
            input_regex,
            endpoint_regex,
            target_config,
        )
        _durations, events, _prior_non_endpoint, _end_reasons = patient_targets(
            target_item,
            landmark_days,
            target_config,
        )
        patient_count += 1
        landmark_count += len(landmark_days)
        recurrence_patient_count += int(has_first_recurrence_site(target_item, input_regex, endpoint_regex))
        event_landmark_count += int(events.sum())
        event_patient_count += int(events.any())
    return LandmarkEventCountValues(
        landmark_count=landmark_count,
        event_landmark_count=event_landmark_count,
        patient_count=patient_count,
        recurrence_patient_count=recurrence_patient_count,
        event_patient_count=event_patient_count,
    )


def recurrence_target_item(
    item: PatientBloodTestItem,
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
    config: PatientDatasetConfig,
) -> PatientBloodTestItem:
    other_cancer_regex = (
        re.compile(config.other_cancer_code_filter_regex) if config.exclude_and_censor_other_cancer_patients else None
    )
    status = resolve_endpoint_status(
        item.distant_recurrence_events,
        item.clinical_events,
        input_regex,
        endpoint_regex,
        other_cancer_regex,
        death_day=item.death_day,
        last_observation_day=item.source_last_observation_day,
    )
    return replace(
        item,
        endpoint_day=None if status.endpoint_event is None else status.endpoint_event.days_since_diagnosis,
        non_endpoint_recurrence_events=status.non_endpoint_events,
        right_censoring_day=status.right_censoring_day,
        right_censoring_reason=status.right_censoring_reason,
    )


def has_first_recurrence_site(
    item: PatientBloodTestItem,
    input_regex: re.Pattern[str],
    endpoint_regex: re.Pattern[str],
) -> bool:
    first_events = first_input_recurrence_events(item.distant_recurrence_events, input_regex)
    if not any(matches_icd_code(event.code, endpoint_regex) for event in first_events):
        return False
    return recurrence_precedes_censoring(
        first_events[0].days_since_diagnosis,
        item.right_censoring_day,
        item.right_censoring_reason,
    )


def first_input_recurrence_events(
    events: tuple[PatientEndpointEvent, ...],
    input_regex: re.Pattern[str],
) -> tuple[PatientEndpointEvent, ...]:
    input_events = matching_recurrence_events(events, input_regex)
    if not input_events:
        return ()
    first_day = min(event.days_since_diagnosis for event in input_events)
    return tuple(event for event in input_events if event.days_since_diagnosis == first_day)


def overall_counts(counts: tuple[LandmarkEventCountValues, ...]) -> LandmarkEventCountValues:
    return LandmarkEventCountValues(
        landmark_count=sum(count.landmark_count for count in counts),
        event_landmark_count=sum(count.event_landmark_count for count in counts),
        patient_count=sum(count.patient_count for count in counts),
        recurrence_patient_count=sum(count.recurrence_patient_count for count in counts),
        event_patient_count=sum(count.event_patient_count for count in counts),
    )


def count_row(
    site: LandmarkEventCountsSite,
    horizon: LandmarkEventCountsHorizon,
    counts_by_split: Mapping[str, LandmarkEventCountValues],
) -> dict[str, object]:
    row: dict[str, object] = {
        "site_label": site.label,
        "endpoint_ddfs_code_filter_regex": site.endpoint_regex,
        "horizon_label": horizon.label,
        "horizon_days": horizon.days,
    }
    for split in (*SPLIT_KEYS, "overall"):
        counts = counts_by_split[split]
        for key in COUNT_KEYS:
            row[f"{split}_{key}"] = getattr(counts, key)
        row[f"{split}_event_patient_percent"] = event_patient_percent(counts)
    return row


def event_patient_percent(counts: LandmarkEventCountValues) -> float | None:
    if counts.recurrence_patient_count == 0:
        return None
    return 100.0 * counts.event_patient_count / counts.recurrence_patient_count


def landmark_event_counts_latex(
    count_df: pd.DataFrame,
    config: LandmarkEventCountsConfig,
) -> str:
    horizon_headers = " & ".join(latex_escape(horizon.label) for horizon in config.horizons)
    column_spec = "ll|" + "c" * len(config.horizons)
    lines = [
        r"\begin{table*}[h]",
        r"\centering",
        rf"\caption{{{latex_escape(config.table_caption)}}}",
        rf"\label{{{latex_escape(config.table_label)}}}",
        rf"\begin{{tabular}}{{{column_spec}}}",
        r"\hline",
        f"Site & Dataset fold & {horizon_headers} \\\\",
        r"\hline",
    ]
    lines.extend(latex_rows(count_df))
    lines.extend((r"\hline", r"\end{tabular}", r"\end{table*}"))
    return "\n".join(lines) + "\n"


def latex_rows(count_df: pd.DataFrame) -> tuple[str, ...]:
    rows = []
    records = count_df.to_dict("records")
    for _site_label, site_records in groupby(records, key=lambda row: str(row["site_label"])):
        if rows:
            rows.append(r"\hline")
        site_rows = tuple(site_records)
        for index, (split, split_label) in enumerate(DISPLAY_SPLITS):
            site_row_count = len(DISPLAY_SPLITS) if index == 0 else None
            rows.append(
                latex_fold_row(
                    site_rows,
                    split,
                    split_label,
                    site_row_count=site_row_count,
                )
            )
    return tuple(rows)


def latex_fold_row(
    site_rows: tuple[Mapping[str, object], ...],
    split: str,
    split_label: str,
    *,
    site_row_count: int | None = None,
) -> str:
    first_row = site_rows[0]
    cells = [
        "" if site_row_count is None else latex_site_cell(first_row, site_row_count),
        latex_fold_cell(first_row, split, split_label),
        *(latex_count_cell(row, split) for row in site_rows),
    ]
    return " & ".join(cells) + r" \\"


def latex_site_cell(row: Mapping[str, object], row_count: int) -> str:
    site = latex_escape(recurrence_endpoint_label(str(row["site_label"])))
    return rf"\multirow[c]{{{row_count}}}{{*}}{{{site}}}"


def latex_fold_cell(row: Mapping[str, object], split: str, split_label: str) -> str:
    patient_count = int(row[f"{split}_recurrence_patient_count"])
    return f"{latex_escape(split_label)} ({patient_count:,})"


def latex_count_cell(row: Mapping[str, object], split: str) -> str:
    percent = row[f"{split}_event_patient_percent"]
    if percent is None or pd.isna(percent):
        return "--"
    return f"{int(row[f'{split}_event_patient_count']):,} ({float(percent):.1f}\\%)"


def other_endpoint_regex(
    input_recurrence_regex: str,
    site_rules: tuple[RecurrenceSiteRule, ...],
) -> str:
    if not site_rules:
        return input_recurrence_regex
    site_union = "|".join(f"(?:{rule.code_regex})" for rule in site_rules)
    return rf"^(?=.*(?:{input_recurrence_regex}))(?!.*(?:{site_union})).*$"


def endpoint_name(label: str) -> str:
    cleaned = "".join(character.lower() if character.isalnum() else "_" for character in label)
    return f"{cleaned.strip('_') or 'endpoint'}_recurrence"


def horizon_tuple(value: object) -> tuple[LandmarkEventCountsHorizon, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError("`landmark_event_counts.horizons` must be a list.")
    horizons = tuple(horizon_from_mapping(item) for item in value)
    if not horizons:
        raise ValueError("`landmark_event_counts.horizons` cannot be empty.")
    return horizons


def horizon_from_mapping(value: object) -> LandmarkEventCountsHorizon:
    if not isinstance(value, Mapping):
        raise TypeError("Each `landmark_event_counts.horizons` entry must be a mapping.")
    label = non_empty_text(value.get("label"), "horizons.label")
    days = int(value["days"])
    if days < 1:
        raise ValueError("`landmark_event_counts.horizons.days` must be at least 1.")
    return LandmarkEventCountsHorizon(label, days)


def string_tuple(value: object, config_key: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple):
        raise TypeError(f"`landmark_event_counts.{config_key}` must be a list.")
    values = tuple(str(item).strip() for item in value)
    if any(item == "" for item in values):
        raise ValueError(f"`landmark_event_counts.{config_key}` cannot contain empty values.")
    return values


def validate_site_rules(labels: tuple[str, ...], regexes: tuple[str, ...]) -> None:
    if len(labels) != len(regexes):
        raise ValueError("`landmark_event_counts.site_labels` and `site_regexes` must have the same length.")
    for regex in regexes:
        re.compile(regex)


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def non_empty_text(value: object, key: str) -> str:
    if value is None:
        raise KeyError(f"Missing required `landmark_event_counts.{key}`.")
    text = str(value).strip()
    if text == "":
        raise ValueError(f"`landmark_event_counts.{key}` cannot be empty.")
    return text
