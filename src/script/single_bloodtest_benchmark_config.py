from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.config import load_yml, resolve_path
from src.script.ebm_survival_config import safe_path_component

INHERITED_SECTIONS = (
    "dataset",
    "ebm",
    "imputation",
    "output",
    "population_statistics",
    "landmark_event_counts",
)


def load_config(config_path: Path) -> tuple[dict[str, dict[str, object]], Path]:
    """Merge single-blood-test benchmark settings over the base EBM config."""
    benchmark_config = load_yml(config_path)
    config_dir = config_path.parent
    base_path = resolve_path(str(benchmark_config["source"]["base_config_path"]), config_dir)
    base_config = load_yml(base_path)
    raw_config = {
        section: {
            **base_config.get(section, {}),
            **benchmark_config.get(section, {}),
        }
        for section in INHERITED_SECTIONS
    }
    raw_config["benchmark"] = dict(benchmark_config["benchmark"])
    return raw_config, config_dir


def mapping_sequence(value: object, key: str) -> tuple[Mapping[str, object], ...]:
    """Validate a nonempty benchmark axis while preserving configured order."""
    if not isinstance(value, list | tuple) or not all(isinstance(item, Mapping) for item in value):
        raise TypeError(f"`benchmark.{key}` must be a list of mappings.")
    if not value:
        raise ValueError(f"`benchmark.{key}` cannot be empty.")
    return tuple(value)


def task_output_dir(
    output_dir: Path,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
) -> Path:
    return output_dir / safe_path_component(str(site["label"])) / safe_path_component(str(horizon["label"]))


def benchmark_model_output_dir(
    output_dir: Path,
    model_key: str,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
) -> Path:
    return task_output_dir(output_dir, site, horizon) / safe_path_component(model_key)
