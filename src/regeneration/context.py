"""Resolve saved cells once and share compatible cohort and imputation work."""

from __future__ import annotations

from collections.abc import Mapping
import json
from dataclasses import dataclass, field, replace
from functools import cached_property
from pathlib import Path

from interpret.glassbox import ExplainableBoostingRegressor

from src.config import load_yml, resolve_path
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import benchmark_target_dataset
from src.data.dataset_config import PatientDatasetConfig
from src.model.imputation import FittedFeatureImputer
from src.reporting import report_output_path
from src.regeneration.config import RegenerationConfig
from src.script.saved_ebm_artifacts import load_feature_imputer, load_model
from src.script.single_bloodtest_benchmark_config import mapping_sequence
from src.script.single_bloodtest_benchmark_reporting import (
    ALL_MODEL_KEY,
    CLINICAL_BASELINE_MODEL_KEY,
    load_saved_summary,
)
from src.script.single_bloodtest_washout import configured_washout_days, saved_model_dir, saved_washout_landmarks
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache

CellKey = tuple[str, str, str]
MODEL_ALIASES = {"all_blood_tests": ALL_MODEL_KEY, "clinical_baseline": CLINICAL_BASELINE_MODEL_KEY}


def cell_key(result: Mapping[str, object]) -> CellKey:
    return str(result["site_label"]), str(result["horizon_label"]), str(result["model_key"])


@dataclass(frozen=True)
class SavedCell:
    """One saved site/horizon/model directory and its aggregate result records.

    Configuration is loaded only when a task needs it, keeping table-only work
    independent of patient sources, imputers and model files.
    """

    path: Path
    results: tuple[dict[str, object], ...]

    @property
    def key(self) -> CellKey:
        return cell_key(self.results[0])

    @cached_property
    def raw(self) -> dict[str, dict[str, object]]:
        return load_yml(self.path / "training_config.yml")

    @cached_property
    def config(self) -> PatientDatasetConfig:
        config = PatientDatasetConfig.from_mapping(self.raw["dataset"], config_dir=self.path)
        source = self.results[0]
        if config.censoring_horizon_days != source["horizon_days"]:
            raise ValueError(f"Saved horizon differs from the benchmark: {self.path}")
        if config.endpoint_ddfs_code_filter_regex != source["endpoint_ddfs_code_filter_regex"]:
            raise ValueError(f"Saved endpoint differs from the benchmark: {self.path}")
        return replace(config, split="train")

    @cached_property
    def model(self) -> ExplainableBoostingRegressor:
        if any(row["status"] == "failed" for row in self.results):
            raise ValueError(f"Cannot regenerate a failed model from stale artifacts: {self.path}")
        metadata_path = self.path / "metadata.json"
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            if metadata.get("status") != "complete":
                raise ValueError(f"Saved model is not complete: {self.path}")
        return load_model(self.path / "model.pkl")


@dataclass
class RegenerationContext:
    """Saved benchmark inventory plus lazy caches for the selected work.

    ``results`` always retains unselected cells so partial evaluation cannot
    truncate the benchmark. ``benchmark`` retains the full report axes; selected
    axes are available separately for composite figures. Sources and imputers
    are loaded lazily, and every dataset uses a saved training configuration.
    """

    settings: RegenerationConfig
    benchmark: dict[str, object]
    output_dir: Path
    config_dir: Path
    selected_tests: tuple[str, ...]
    results: list[dict[str, object]]
    cells: tuple[SavedCell, ...]
    selected: tuple[SavedCell, ...]
    _imputers: dict[Path, FittedFeatureImputer] = field(default_factory=dict, repr=False)

    @cached_property
    def presentation(self) -> dict[str, dict[str, object]]:
        raw = load_yml(self.settings.benchmark_config_path)
        source = raw.get("source", {})
        path = source.get("base_config_path")
        if path is None:
            return load_yml(Path(__file__).resolve().parents[1] / "config" / "ebm_survival.yml")
        return load_yml(resolve_path(str(path), self.config_dir))

    @cached_property
    def pools(self) -> dict[int, SurvivalBenchmarkFeatureCache]:
        sources = [{**cell.results[0], "output_dir": str(cell.path)} for cell in self.selected]
        return saved_washout_landmarks(sources, configured_washout_days(self.benchmark))

    def dataset(self, cell: SavedCell) -> PatientBloodTestDataset:
        # table() validates that cohort and predictor definitions match the shared pool.
        self.pools[0].landmarks.table(cell.config)
        return benchmark_target_dataset(self.pools[0].landmarks.dataset, cell.config)

    def imputer(self, cell: SavedCell) -> FittedFeatureImputer:
        if cell.path not in self._imputers:
            table = self.pools[0].landmarks.table(cell.config)
            self._imputers[cell.path] = load_feature_imputer(cell.raw, cell.path, table)
        return self._imputers[cell.path]

    def evaluation_cells(self) -> tuple[SavedCell, ...]:
        """Include each selected cell's clinical reference for paired comparisons."""
        needed = {cell.key for cell in self.selected}
        needed.update((site, horizon, CLINICAL_BASELINE_MODEL_KEY) for site, horizon, _model in tuple(needed))
        available = {cell.key for cell in self.cells}
        if missing := needed - available:
            raise ValueError(f"Missing clinical reference cells: {sorted(missing)}")
        return tuple(
            sorted(
                (cell for cell in self.cells if cell.key in needed),
                key=lambda cell: (cell.key[2] != CLINICAL_BASELINE_MODEL_KEY, cell.key),
            )
        )

    def task_groups(self) -> tuple[tuple[SavedCell, ...], ...]:
        groups: dict[tuple[str, str], list[SavedCell]] = {}
        for cell in self.selected:
            groups.setdefault(cell.key[:2], []).append(cell)
        return tuple(tuple(cells) for cells in groups.values())

    def figure_benchmark(self) -> dict[str, object]:
        return {
            **self.benchmark,
            "sites": [
                site
                for site in mapping_sequence(self.benchmark["sites"], "sites")
                if any(cell.key[0] == site["label"] for cell in self.selected)
            ],
            "horizons": [
                horizon
                for horizon in mapping_sequence(self.benchmark["horizons"], "horizons")
                if any(cell.key[1] == horizon["label"] for cell in self.selected)
            ],
        }


def load_context(settings: RegenerationConfig) -> RegenerationContext:
    config_dir = settings.benchmark_config_path.parent
    benchmark = load_yml(settings.benchmark_config_path)["benchmark"]
    output_dir = (
        resolve_path(str(benchmark["output_dir"]), config_dir)
        if settings.benchmark_dir is None
        else settings.benchmark_dir
    )
    selected_tests, results = load_saved_summary(
        report_output_path(output_dir, benchmark.get("output_metadata_path", "metadata.json"))
    )
    axes = {
        name: {str(item["label"]) for item in mapping_sequence(benchmark[name], name)} for name in ("sites", "horizons")
    }
    groups: dict[CellKey, list[dict[str, object]]] = {}
    for result in results:
        groups.setdefault(cell_key(result), []).append(result)
    cells = tuple(SavedCell(saved_model_dir(output_dir, values[0]), tuple(values)) for values in groups.values())
    models = None if settings.models is None else tuple(MODEL_ALIASES.get(name, name) for name in settings.models)
    filters = (settings.sites, settings.horizons, models)
    choices = (axes["sites"], axes["horizons"], {cell.key[2] for cell in cells})
    for name, requested, available in zip(("sites", "horizons", "models"), filters, choices, strict=True):
        if requested is not None and (unknown := set(requested) - available):
            raise ValueError(f"Unknown {name}: {sorted(unknown)}; available: {sorted(available)}")
    selected = tuple(
        cell
        for cell in cells
        if cell.key[0] in axes["sites"]
        and cell.key[1] in axes["horizons"]
        and all(requested is None or label in requested for requested, label in zip(filters, cell.key, strict=True))
    )
    if not selected:
        raise ValueError("No saved benchmark cells match the regeneration selection.")
    for name, requested, index in (("sites", settings.sites, 0), ("horizons", settings.horizons, 1)):
        if requested is not None and (missing := set(requested) - {cell.key[index] for cell in selected}):
            raise ValueError(f"No saved cells for selected {name}: {sorted(missing)}")
    return RegenerationContext(settings, benchmark, output_dir, config_dir, selected_tests, results, cells, selected)
