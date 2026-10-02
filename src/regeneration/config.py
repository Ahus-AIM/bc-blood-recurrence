"""The single public configuration for study regeneration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from src.config import load_yml, resolve_path

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "regenerate.yml"

# Producers already write their figures and tables; redraw tasks reuse saved values.
ALL_TASKS = (
    "cohort_statistics",
    "landmark_counts",
    "recurrence_statistics",
    "model_inputs",
    "cindex",
    "washout_counts",
    "washout",
    "brier",
    "explanations",
    "importance",
    "top_features",
    "training_risk_reference",
    "temporal_risk",
)
REDRAW_TASKS = (
    "dataset_figure",
    "recurrence_figures",
    "benchmark_table",
    "washout_table",
    "brier_table",
    "temporal_risk_figure",
    "age_returns",
    "age_membership",
)
TASK_ORDER = (*ALL_TASKS, *REDRAW_TASKS, "paper_bundle")


@dataclass(frozen=True)
class PaperBundleConfig:
    """Collect manuscript outputs for one horizon without regenerating them.

    ``enabled`` adds collection after each successful task. ``output_dir`` is
    relative to the benchmark directory unless absolute. ``horizon`` fixes the
    manuscript's horizon independently of regeneration's model filters.
    """

    enabled: bool = False
    output_dir: str = "paper_bundle"
    horizon: str = "6m"


@dataclass(frozen=True)
class RegenerationConfig:
    """Select one benchmark, tasks and optional cell filters.

    Source paths resolve against the YAML directory. A null benchmark directory uses
    the benchmark config's output directory. Null filters select every saved
    cell on the configured axes. Companion fitting is a separate explicit opt-in.
    ``paper_bundle`` controls manuscript copies and their recorded horizon.
    """

    benchmark_config_path: Path
    benchmark_dir: Path | None
    tasks: tuple[str, ...]
    sites: tuple[str, ...] | None
    horizons: tuple[str, ...] | None
    models: tuple[str, ...] | None
    allow_competing_model_fit: bool = False
    paper_bundle: PaperBundleConfig = field(default_factory=PaperBundleConfig)


def selection(value: object, name: str) -> tuple[str, ...] | None:
    if value is None:
        return None
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item.strip() for item in value):
        raise ValueError(f"`{name}` must be null or a nonempty list of names.")
    if len(value) != len(set(value)):
        raise ValueError(f"`{name}` contains duplicate names.")
    return tuple(value)


def load_regeneration_config(path: Path = CONFIG_PATH) -> RegenerationConfig:
    raw = load_yml(path)
    unknown_sections = set(raw) - {"source", "regenerate", "paper_bundle"}
    if unknown_sections:
        raise ValueError(f"Unknown regeneration sections: {sorted(unknown_sections)}")
    source = raw["source"]
    settings = raw["regenerate"]
    unknown_keys = (set(source) - {"benchmark_config_path", "benchmark_dir"}) | (
        set(settings) - {"tasks", "sites", "horizons", "models", "allow_competing_model_fit"}
    )
    if unknown_keys:
        raise ValueError(f"Unknown regeneration settings: {sorted(unknown_keys)}")
    requested = settings.get("tasks", "all")
    tasks = ALL_TASKS if requested == "all" else selection(requested, "regenerate.tasks")
    if tasks is None or set(tasks) - set(TASK_ORDER):
        raise ValueError(f"Choose `all` or task names from: {', '.join(TASK_ORDER)}")
    allow_fit = settings.get("allow_competing_model_fit", False)
    if not isinstance(allow_fit, bool):
        raise TypeError("`allow_competing_model_fit` must be a YAML boolean.")
    directory = source.get("benchmark_dir")
    return RegenerationConfig(
        resolve_path(str(source["benchmark_config_path"]), path.parent),
        None if directory is None else resolve_path(str(directory), path.parent),
        tuple(task for task in TASK_ORDER if task in tasks),
        selection(settings.get("sites"), "sites"),
        selection(settings.get("horizons"), "horizons"),
        selection(settings.get("models"), "models"),
        allow_fit,
        paper_bundle_config(raw.get("paper_bundle", {})),
    )


def paper_bundle_config(raw: object) -> PaperBundleConfig:
    if not isinstance(raw, Mapping):
        raise TypeError("`paper_bundle` must be a mapping.")
    if unknown := set(raw) - {"enabled", "output_dir", "horizon"}:
        raise ValueError(f"Unknown paper bundle settings: {sorted(unknown)}")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        raise TypeError("`paper_bundle.enabled` must be a YAML boolean.")
    directory = raw.get("output_dir", "paper_bundle")
    horizon = raw.get("horizon", "6m")
    for name, value in (("output_dir", directory), ("horizon", horizon)):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"`paper_bundle.{name}` must be a nonempty string.")
    return PaperBundleConfig(enabled, directory, horizon)
