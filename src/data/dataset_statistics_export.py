from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data.cohort_selection import COHORT_SELECTION_VERSION
from src.data.cohort_comparison import VALUES_FILENAME, validate_comparison_version
from src.data.dataset_statistics import DatasetStatisticsSummary
from src.data.observation_window import OBSERVATION_WINDOW_VERSION
from src.data.recurrence_site_cumulative_incidence import (
    INCIDENCE_COLUMNS as RECURRENCE_INCIDENCE_COLUMNS,
)
from src.data.recurrence_site_incidence import POST_RECURRENCE_SURVIVAL_VERSION

RECURRENCE_SURVIVAL_COLUMNS = (
    "site",
    "years_since_origin",
    "survival_percent",
)

STATISTICS_UPDATE_KEYS = (
    ("panel_a", "inclusion_box_labels"),
    ("panel_a", "inclusion_box_counts"),
    ("panel_a", "exclusion_box_labels"),
    ("panel_a", "exclusion_box_counts"),
    ("panel_a", "split_box_counts"),
    ("panel_a", "footnotes"),
    ("panel_b", "diagnosis_years"),
    ("panel_b", "series_labels"),
    ("panel_b", "series_counts"),
    ("panel_c", "blood_test_names"),
    ("panel_c", "blood_test_counts"),
    ("panel_c", "blood_test_locf_counts"),
    ("panel_c", "blood_test_mice_counts"),
)


def dataset_statistics_payload(
    summary: DatasetStatisticsSummary,
    *,
    dataset_config: Mapping[str, object] | None = None,
    recurrence_incidence_table: pd.DataFrame | None = None,
    recurrence_survival_table: pd.DataFrame | None = None,
) -> dict[str, object]:
    return {
        "metadata": {
            "selected_test_count": summary.selected_test_count,
            "input_fingerprint": summary.input_fingerprint,
            "dataset_config": comparable_dataset_config(dataset_config),
            "cohort_selection_version": COHORT_SELECTION_VERSION,
            "observation_window_version": OBSERVATION_WINDOW_VERSION,
            "post_recurrence_survival_version": POST_RECURRENCE_SURVIVAL_VERSION,
        },
        "cohort": {
            "included_patient_ids": sorted(summary.stage_ids[-1]),
        },
        "panel_a": {
            "inclusion_box_labels": list(summary.inclusion_box_labels),
            "inclusion_box_counts": list(summary.inclusion_box_counts),
            "exclusion_box_labels": list(summary.exclusion_box_labels),
            "exclusion_box_counts": list(summary.exclusion_box_counts),
            "split_box_counts": list(summary.split_box_counts),
            "footnotes": list(summary.flow_footnotes),
        },
        "panel_b": {
            "diagnosis_years": list(summary.diagnosis_years),
            "series_labels": list(summary.series_labels),
            "series_counts": [list(counts) for counts in summary.series_counts],
        },
        "panel_c": {
            "blood_test_names": list(summary.blood_test_names),
            "blood_test_counts": list(summary.blood_test_counts),
            "blood_test_locf_counts": list(summary.blood_test_locf_counts),
            "blood_test_mice_counts": list(summary.blood_test_mice_counts),
        },
        "panel_d": {
            "recurrence_site_incidence": dataframe_records(
                recurrence_incidence_table,
                RECURRENCE_INCIDENCE_COLUMNS,
            ),
        },
        "panel_e": {
            "first_recurrence_site_survival": dataframe_records(
                recurrence_survival_table,
                RECURRENCE_SURVIVAL_COLUMNS,
            ),
        },
        "tables": {
            "bloodtest_count_table": bloodtest_count_records(summary),
        },
    }


def dataframe_records(
    frame: pd.DataFrame | None,
    columns: tuple[str, ...],
) -> list[dict[str, object]]:
    if frame is None or frame.empty:
        return []
    return [
        {column: row[column] for column in columns}
        for row in frame.loc[:, list(columns)].to_dict("records")
    ]


def bloodtest_count_records(summary: DatasetStatisticsSummary) -> list[dict[str, object]]:
    return [
        {
            "display_name": str(row.display_name),
            "count": int(row.count),
            "locf_count": int(row.locf_count),
            "mice_count": int(row.mice_count),
        }
        for row in summary.bloodtest_count_table.itertuples(index=False)
    ]


def write_dataset_statistics_payload(
    payload: Mapping[str, object],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(dict(payload), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_dataset_statistics_payload(payload_path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(payload_path.read_text(encoding="utf-8"))
    if not isinstance(loaded, Mapping):
        raise ValueError(f"Expected a mapping at the top level of {payload_path}.")
    payload = dict(loaded)
    comparison_path = payload_path.with_name(VALUES_FILENAME)
    metadata = payload.get("metadata", {})
    if metadata.get("cohort_hash") is None and comparison_path.is_file():
        comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
        validate_comparison_version(comparison)
        if comparison["input_fingerprint"] != metadata.get("input_fingerprint"):
            raise ValueError("Dataset statistics and cohort comparison come from different prepared inputs.")
        metadata["cohort_hash"] = comparison["cohort_hash"]
    return payload


def validate_payload_dataset_config(
    payload: Mapping[str, object],
    current_dataset_config: Mapping[str, object],
) -> None:
    metadata = payload.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("Missing `metadata` mapping in dataset-statistics payload.")
    payload_config = metadata.get("dataset_config")
    if payload_config is None:
        raise ValueError(
            "Dataset-statistics payload has no dataset-config snapshot. "
            "Regenerate dataset statistics through model training before plotting."
        )
    if not isinstance(payload_config, Mapping):
        raise ValueError("Expected `metadata.dataset_config` to be a mapping.")
    current_config = comparable_dataset_config(current_dataset_config)
    payload_config = dict(payload_config)
    if dict(payload_config) == current_config:
        if (
            current_dataset_config.get("include_recurrence_day_blood_tests", False)
            and metadata.get("observation_window_version") != OBSERVATION_WINDOW_VERSION
        ):
            raise ValueError(
                "Dataset-statistics payload uses outdated observation-window rules. "
                "Regenerate dataset statistics through model training before plotting."
            )
        if metadata.get("cohort_selection_version") != COHORT_SELECTION_VERSION:
            raise ValueError(
                "Dataset-statistics payload uses outdated cohort-selection order. "
                "Regenerate dataset statistics through model training before plotting."
            )
        return
    changed_keys = sorted(
        set(payload_config) | set(current_config),
        key=str,
    )
    mismatched_keys = [
        key
        for key in changed_keys
        if payload_config.get(key) != current_config.get(key)
    ]
    joined = ", ".join(str(key) for key in mismatched_keys)
    raise ValueError(
        "Dataset-statistics payload was generated with a different dataset config "
        f"({joined}). Select the matching run's training config or regenerate "
        "dataset statistics through model training before plotting."
    )


def comparable_dataset_config(config: Mapping[str, object] | None) -> dict[str, object] | None:
    if config is None:
        return None
    return {
        str(key): comparable_config_value(value)
        for key, value in sorted(config.items(), key=lambda item: str(item[0]))
    }


def comparable_config_value(value: Any) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return comparable_dataset_config(value)
    if isinstance(value, tuple):
        return [comparable_config_value(item) for item in value]
    if isinstance(value, list):
        return [comparable_config_value(item) for item in value]
    return value


def included_patient_ids_from_payload(payload: Mapping[str, object]) -> frozenset[str]:
    cohort = payload.get("cohort")
    if not isinstance(cohort, Mapping):
        raise ValueError("Missing `cohort` mapping in dataset-statistics payload.")
    patient_ids = cohort.get("included_patient_ids")
    if not isinstance(patient_ids, list):
        raise ValueError("Missing list value `cohort.included_patient_ids` in dataset-statistics payload.")
    return frozenset(str(patient_id) for patient_id in patient_ids)


def recurrence_incidence_from_payload(payload: Mapping[str, object]) -> pd.DataFrame:
    return dataframe_from_payload_records(
        payload,
        "panel_d",
        "recurrence_site_incidence",
        RECURRENCE_INCIDENCE_COLUMNS,
    )


def recurrence_survival_from_payload(payload: Mapping[str, object]) -> pd.DataFrame:
    metadata = payload.get("metadata")
    if (
        not isinstance(metadata, Mapping)
        or metadata.get("post_recurrence_survival_version") != POST_RECURRENCE_SURVIVAL_VERSION
    ):
        raise ValueError(
            "Saved post-recurrence survival uses outdated follow-up rules. "
            "Run regeneration task cohort_statistics with source data before redrawing Figure 1."
        )
    return dataframe_from_payload_records(
        payload,
        "panel_e",
        "first_recurrence_site_survival",
        RECURRENCE_SURVIVAL_COLUMNS,
    )


def dataframe_from_payload_records(
    payload: Mapping[str, object],
    section: str,
    key: str,
    columns: tuple[str, ...],
) -> pd.DataFrame:
    rows = payload_records(payload, section, key)
    frame = pd.DataFrame.from_records(rows, columns=columns)
    for column in columns:
        if column not in frame:
            frame[column] = []
    return frame.loc[:, list(columns)]


def payload_records(
    payload: Mapping[str, object],
    section: str,
    key: str,
) -> list[Mapping[str, object]]:
    section_values = payload.get(section)
    if not isinstance(section_values, Mapping):
        raise ValueError(f"Missing `{section}` mapping in dataset-statistics payload.")
    value = section_values.get(key)
    if not isinstance(value, list):
        raise ValueError(f"Missing list value `{section}.{key}` in dataset-statistics payload.")
    if not all(isinstance(row, Mapping) for row in value):
        raise ValueError(f"`{section}.{key}` must contain mapping rows.")
    return value


def dataset_statistics_updates_from_payload(
    payload: Mapping[str, object],
) -> dict[tuple[str, str], list[object]]:
    return {
        (section, key): list(payload_section_value(payload, section, key))
        for section, key in STATISTICS_UPDATE_KEYS
    }


def payload_section_value(
    payload: Mapping[str, object],
    section: str,
    key: str,
) -> list[object]:
    section_values = payload.get(section)
    if not isinstance(section_values, Mapping):
        raise ValueError(f"Missing `{section}` mapping in dataset-statistics payload.")
    value = section_values.get(key, [] if key == "footnotes" else None)
    if not isinstance(value, list):
        raise ValueError(f"Missing list value `{section}.{key}` in dataset-statistics payload.")
    return value


def update_dataset_statistics_yml(
    config_path: Path,
    updates: Mapping[tuple[str, str], list[object]],
) -> None:
    text = config_path.read_text(encoding="utf-8")
    for (section, key), values in updates.items():
        text = replace_yaml_list(text, section, key, values)
    config_path.write_text(text, encoding="utf-8")


def replace_yaml_list(text: str, section: str, key: str, values: list[object]) -> str:
    lines = text.splitlines()
    section_index = yaml_section_index(lines, section)
    section_end = yaml_section_end(lines, section_index)
    key_index = yaml_key_index(lines, section, key, section_index, section_end)
    item_end = key_index + 1
    while item_end < section_end and lines[item_end].startswith("    - "):
        item_end += 1
    replacement = yaml_list_replacement(key, values)
    return "\n".join([*lines[:key_index], *replacement, *lines[item_end:]]) + "\n"


def yaml_section_index(lines: list[str], section: str) -> int:
    section_line = f"{section}:"
    return next(index for index, line in enumerate(lines) if line == section_line)


def yaml_section_end(lines: list[str], section_index: int) -> int:
    for index in range(section_index + 1, len(lines)):
        if lines[index] and not lines[index].startswith(" "):
            return index
    return len(lines)


def yaml_key_index(
    lines: list[str],
    section: str,
    key: str,
    section_index: int,
    section_end: int,
) -> int:
    key_prefix = f"  {key}:"
    try:
        return next(
            index
            for index in range(section_index + 1, section_end)
            if lines[index].startswith(key_prefix)
        )
    except StopIteration as error:
        raise ValueError(f"Missing `{section}.{key}` in dataset-statistics config.") from error


def yaml_list_replacement(key: str, values: list[object]) -> list[str]:
    if not values:
        return [f"  {key}: []"]
    return [f"  {key}:", *(f"    - {yaml_list_value(value)}" for value in values)]


def yaml_list_value(value: object) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, list):
        return json.dumps(value, ensure_ascii=False)
    return json.dumps(str(value), ensure_ascii=False)
