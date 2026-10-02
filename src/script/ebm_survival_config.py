from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from src.config import resolve_path
from src.data.blood_test_display import load_test_name_translation
from src.data.dataset_config import PatientDatasetConfig
from src.model.ebm_plotting_defaults import DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS
from src.model.normal_ranges import NormalRange, load_normal_ranges

TRAINING_CONFIG_FILENAME = "training_config.yml"
METADATA_FILENAME = "metadata.json"
EXPLANATION_TERM_X_AXIS_SCALES = ("linear", "log")
DEFAULT_COMBINED_TERM_HAZARD_RATIO_COLUMNS = 4
OUTPUT_CONFIG_RELATIVE_PATH_KEYS = (
    "output_dir",
    "normal_ranges_csv_path",
    "bloodtest_translation_csv_path",
)
BENCHMARK_CONFIG_RELATIVE_PATH_KEYS = ("output_dir", "bloodtest_translation_csv_path")


def run_output_dir(
    output_config: Mapping[str, object],
    dataset_config: PatientDatasetConfig,
    config_dir: Path,
) -> Path:
    base_dir = resolve_path(str(output_config.get("output_dir", "../../output/ebm_survival")), config_dir)
    endpoint = safe_path_component(dataset_config.recurrence_endpoint_name)
    return base_dir / endpoint


def write_training_config_snapshot(
    raw_config: Mapping[str, Mapping[str, object]],
    dataset_config: PatientDatasetConfig,
    output_dir: Path,
    config_dir: Path,
) -> dict[str, object]:
    """Write the effective training config with config-relative paths resolved."""
    output_path = output_dir / TRAINING_CONFIG_FILENAME
    output_path.write_text(
        yaml.safe_dump(
            training_config_snapshot(raw_config, dataset_config, config_dir),
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {"path": output_path.name}


def training_config_snapshot(
    raw_config: Mapping[str, Mapping[str, object]],
    dataset_config: PatientDatasetConfig,
    config_dir: Path,
) -> dict[str, dict[str, object]]:
    snapshot = {
        str(section): serializable_mapping(values)
        for section, values in raw_config.items()
    }
    snapshot["dataset"] = {
        **serializable_mapping(raw_config.get("dataset", {})),
        **dataset_config_mapping(dataset_config),
    }
    if "output" in snapshot:
        snapshot["output"] = resolved_path_mapping(
            snapshot["output"],
            OUTPUT_CONFIG_RELATIVE_PATH_KEYS,
            config_dir,
        )
    if "benchmark" in snapshot:
        snapshot["benchmark"] = resolved_path_mapping(
            snapshot["benchmark"],
            BENCHMARK_CONFIG_RELATIVE_PATH_KEYS,
            config_dir,
        )
    return snapshot


def dataset_config_mapping(config: PatientDatasetConfig) -> dict[str, object]:
    return {
        field.name: serializable_value(getattr(config, field.name))
        for field in fields(PatientDatasetConfig)
    }


def resolved_path_mapping(
    config: Mapping[str, object],
    path_keys: tuple[str, ...],
    config_dir: Path,
) -> dict[str, object]:
    resolved = dict(config)
    for key in path_keys:
        if key in resolved:
            resolved[key] = resolved_config_path(resolved[key], config_dir)
    return resolved


def resolved_config_path(value: object, config_dir: Path) -> object:
    if value is None or str(value).strip() == "":
        return value
    return str(resolve_path(str(value), config_dir))


def serializable_mapping(config: Mapping[str, object]) -> dict[str, object]:
    return {
        str(key): serializable_value(value)
        for key, value in config.items()
    }


def serializable_value(value: Any) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return serializable_mapping(value)
    if isinstance(value, tuple | list):
        return [serializable_value(item) for item in value]
    return value


def safe_path_component(value: str) -> str:
    cleaned = "".join(character if character.isalnum() or character in {"-", "_"} else "_" for character in value)
    return cleaned or "endpoint"


def update_run_metadata(
    output_dir: Path,
    key: str,
    value: object,
) -> bool:
    """Replace one saved metadata field, returning false when no metadata artifact exists."""
    metadata_path = output_dir / METADATA_FILENAME
    if not metadata_path.is_file():
        return False
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"Metadata file must contain a JSON object: {metadata_path}")
    metadata[key] = value
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return True


def explanation_normal_ranges(
    output_config: Mapping[str, object],
    config_dir: Path,
) -> tuple[dict[str, tuple[NormalRange, ...]], Path | None]:
    csv_path_value = output_config.get("normal_ranges_csv_path")
    if csv_path_value is None or str(csv_path_value).strip() == "":
        return {}, None
    csv_path = resolve_path(str(csv_path_value), config_dir)
    return load_normal_ranges(csv_path), csv_path


def explanation_test_name_translation(
    output_config: Mapping[str, object],
    config_dir: Path,
) -> tuple[dict[str, str], Path | None]:
    csv_path_value = output_config.get("bloodtest_translation_csv_path")
    if csv_path_value is None or str(csv_path_value).strip() == "":
        return {}, None
    csv_path = resolve_path(str(csv_path_value), config_dir)
    return load_test_name_translation(csv_path), csv_path


def explanation_term_x_axis_scale(output_config: Mapping[str, object]) -> str:
    scale = str(output_config.get("explanation_term_x_axis_scale", "linear")).strip().lower()
    if scale not in EXPLANATION_TERM_X_AXIS_SCALES:
        allowed = ", ".join(EXPLANATION_TERM_X_AXIS_SCALES)
        raise ValueError(f"`output.explanation_term_x_axis_scale` must be one of: {allowed}.")
    return scale


def combined_term_hazard_ratio_columns(output_config: Mapping[str, object]) -> int:
    columns = int(
        output_config.get(
            "combined_term_hazard_ratio_columns",
            DEFAULT_COMBINED_TERM_HAZARD_RATIO_COLUMNS,
        )
    )
    if columns < 1:
        raise ValueError("`output.combined_term_hazard_ratio_columns` must be at least 1.")
    return columns


def combined_term_hazard_ratio_show_x_markers(output_config: Mapping[str, object]) -> bool:
    return bool(output_config.get("combined_term_hazard_ratio_show_x_markers", False))


def combined_term_hazard_ratio_show_uncertainty_area(
    output_config: Mapping[str, object],
) -> bool:
    return bool(output_config.get("combined_term_hazard_ratio_show_uncertainty_area", True))


def combined_term_hazard_ratio_y_limits(output_config: Mapping[str, object]) -> tuple[float, float]:
    lower = float(
        output_config.get(
            "combined_term_hazard_ratio_y_min",
            DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS[0],
        )
    )
    upper = float(
        output_config.get(
            "combined_term_hazard_ratio_y_max",
            DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS[1],
        )
    )
    if lower <= 0.0:
        raise ValueError("`output.combined_term_hazard_ratio_y_min` must be positive.")
    if upper <= lower:
        raise ValueError(
            "`output.combined_term_hazard_ratio_y_max` must be greater than "
            "`output.combined_term_hazard_ratio_y_min`."
        )
    return lower, upper
