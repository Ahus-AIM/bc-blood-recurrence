"""Re-evaluate selected saved models and preserve unrelated aggregate results."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from pathlib import Path

from src.model.survival_evaluation import EVALUATION_SPLITS, evaluate_prepared_split, primary_metric
from src.regeneration.benchmark_table import regenerate_table
from src.regeneration.context import RegenerationContext, SavedCell
from src.reporting import report_output_path
from src.script.brier_reporting import write_brier_outputs
from src.script.ebm_survival_config import update_run_metadata
from src.script.saved_ebm_artifacts import model_feature_names
from src.script.single_bloodtest_benchmark_reporting import (
    load_saved_summary,
    merge_result_records,
    write_summary_outputs,
)
from src.script.single_bloodtest_brier import brier_config, run_brier_benchmark
from src.script.single_bloodtest_washout import evaluate_washout_cell, washout_config, write_washout_outputs
from src.script.single_bloodtest_washout_counts import write_washout_cohort_outputs
from src.script.survival_benchmark_artifacts import COMPLETION_FILENAME


def saved_results(path: Path) -> list[dict[str, object]]:
    """Read prior aggregates when present; a malformed file is an error."""
    return load_saved_summary(path)[1] if path.is_file() else []


def evaluate_cell(context: RegenerationContext, cell: SavedCell) -> list[dict[str, object]]:
    model = cell.model
    imputer = context.imputer(cell)
    cache = context.pools[0].evaluation_cache(cell.config, imputer)
    by_split = {str(row["split"]): row for row in cell.results}
    updated = []
    for split, label in EVALUATION_SPLITS.items():
        if split == "test" and not bool(cell.raw["output"].get("evaluate_test", False)):
            continue
        if label not in by_split:
            raise ValueError(f"Missing saved {label} result: {cell.path}")
        score_path = (
            cell.path / f"{label}_risk_scores.csv" if bool(context.benchmark.get("write_risk_scores", True)) else None
        )
        summary = evaluate_prepared_split(model, cache.get(split), score_path, feature_names=model_feature_names(model))
        status = "complete" if math.isfinite(float(summary["harrell_c_index"])) else "not_estimable"
        updated.append(
            {
                **by_split[label],
                **summary,
                "status": status,
                "not_estimable_reason": None if status == "complete" else "no comparable patient pairs",
            }
        )
        update_run_metadata(cell.path, label, summary)
        if split == "val":
            update_run_metadata(cell.path, "primary_metric", primary_metric(summary))
    # Resuming training must not restore the old evaluation from its completion marker.
    marker = cell.path / COMPLETION_FILENAME
    if marker.is_file():
        payload = json.loads(marker.read_text(encoding="utf-8"))
        payload["results"] = merge_result_records(payload["results"], updated)
        marker.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return updated


def cindex(context: RegenerationContext) -> None:
    for cell in context.evaluation_cells():
        context.results = merge_result_records(context.results, evaluate_cell(context, cell))
    write_summary_outputs(
        context.benchmark,
        context.output_dir,
        context.selected_tests,
        context.results,
        context.config_dir,
    )


def test_source(context: RegenerationContext, cell: SavedCell) -> dict[str, object]:
    row = next(
        (
            row
            for row in context.results
            if row["split"] == "test" and (row["site_label"], row["horizon_label"], row["model_key"]) == cell.key
        ),
        None,
    )
    if row is None:
        raise ValueError(f"No saved test evaluation for {cell.path}")
    return {**row, "output_dir": str(cell.path)}


def washout(context: RegenerationContext) -> tuple[Path, ...]:
    settings = washout_config(context.benchmark)
    path = report_output_path(context.output_dir, settings.get("output_metadata_path", "washout/metadata.json"))
    results = saved_results(path)
    for cell in context.evaluation_cells():
        source = test_source(context, cell)
        if source["status"] == "failed":
            raise ValueError(f"Cannot regenerate a failed model: {cell.path}")
        results = merge_result_records(results, evaluate_washout_cell(source, context.pools, fail_fast=True))
    count_paths = washout_counts(context)
    paths = write_washout_outputs(context.benchmark, context.selected_tests, results, context.output_dir, context.config_dir)
    return (*count_paths, *(Path(path) for path in paths.values()))


def washout_counts(context: RegenerationContext) -> tuple[Path, ...]:
    """Count the full benchmark's sampling pools without evaluating models."""
    paths = write_washout_cohort_outputs(context.benchmark, context.pools, context.output_dir)
    return tuple(Path(path) for path in paths.values())


def brier(context: RegenerationContext) -> tuple[Path, ...]:
    sources = [
        test_source(context, cell) for cell in context.evaluation_cells() if cell.results[0]["horizon_days"] == 180
    ]
    if not sources:
        raise ValueError("Brier requires selected saved 180-day models.")
    destination = brier_destination(context)
    run_brier_benchmark(
        context.benchmark,
        context.selected_tests,
        sources,
        context.output_dir,
        context.config_dir,
        context.pools,
        allow_competing_model_fit=context.settings.allow_competing_model_fit,
        fail_fast=True,
        previous_results=saved_results(destination / "metadata.json"),
    )
    return tuple(destination / f"complete_case_brier_washout.{extension}" for extension in ("csv", "tex"))


def brier_destination(context: RegenerationContext) -> Path:
    return report_output_path(context.output_dir, brier_config(context.benchmark).get("output_dir", "brier"))


def benchmark_table(context: RegenerationContext) -> None:
    regenerate_table(context.benchmark, context.output_dir, context.config_dir)


def washout_table(context: RegenerationContext) -> tuple[Path, ...]:
    settings = washout_config(context.benchmark)
    path = report_output_path(context.output_dir, settings.get("output_metadata_path", "washout/metadata.json"))
    tests, results = load_saved_summary(path)
    paths = write_washout_outputs(context.benchmark, tests, results, context.output_dir, context.config_dir)
    return tuple(Path(path) for path in paths.values())


def brier_table(context: RegenerationContext) -> tuple[Path, ...]:
    destination = brier_destination(context)
    tests, results = load_saved_summary(destination / "metadata.json")
    benchmark: Mapping[str, object] = {
        **context.benchmark,
        "horizons": [item for item in context.benchmark["horizons"] if item["days"] == 180],
    }
    paths = write_brier_outputs(benchmark, tests, results, destination, context.config_dir)
    return tuple(Path(path) for path in paths.values())
