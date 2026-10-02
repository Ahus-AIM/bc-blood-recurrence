"""Execute selected regeneration tasks in dependency order, stopping on failure."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from src.regeneration import cohort_reports, evaluation, model_reports
from src.regeneration.artifact_run import ArtifactRun, generated_artifacts, load_artifact_run
from src.regeneration.config import ALL_TASKS, CONFIG_PATH, TASK_ORDER, load_regeneration_config
from src.regeneration.context import RegenerationContext, load_context
from src.regeneration.paper_artifacts import bundle_destination, paper_artifacts
from src.regeneration.paper_bundle import collect_paper_bundle, update_bundle_manifest
from src.script.logging_config import configure_logging
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY

LOGGER = logging.getLogger(__name__)
# Tasks that generate manuscript files return their actual output paths.
Task = Callable[[RegenerationContext], tuple[Path, ...] | None]
TASKS: dict[str, Task] = {
    "cohort_statistics": cohort_reports.cohort_statistics,
    "landmark_counts": cohort_reports.landmark_counts,
    "recurrence_statistics": cohort_reports.recurrence_statistics,
    "model_inputs": model_reports.model_inputs,
    "cindex": evaluation.cindex,
    "washout": evaluation.washout,
    "washout_counts": evaluation.washout_counts,
    "brier": evaluation.brier,
    "explanations": model_reports.explanations,
    "importance": model_reports.importance,
    "top_features": model_reports.top_features,
    "training_risk_reference": model_reports.training_risk_reference,
    "temporal_risk": model_reports.temporal_risk,
    "dataset_figure": cohort_reports.dataset_figure,
    "recurrence_figures": cohort_reports.recurrence_figures,
    "benchmark_table": evaluation.benchmark_table,
    "washout_table": evaluation.washout_table,
    "brier_table": evaluation.brier_table,
    "temporal_risk_figure": model_reports.temporal_risk_figure,
    "age_returns": cohort_reports.age_returns,
    "age_membership": cohort_reports.age_membership,
}
PRODUCERS = {
    "dataset_figure": "cohort_statistics",
    "age_returns": "cohort_statistics",
    "age_membership": "cohort_statistics",
    "recurrence_figures": "recurrence_statistics",
    "benchmark_table": "cindex",
    "washout_table": "washout",
    "washout_counts": "washout",
    "brier_table": "brier",
    "temporal_risk_figure": "temporal_risk",
}


def task_plan(context: RegenerationContext) -> tuple[str, ...]:
    """Avoid duplicate rendering; restrict `all` to applicable saved model tasks."""
    requested = set(context.settings.tasks)
    if context.settings.tasks == ALL_TASKS:
        if not any(cell.results[0]["horizon_days"] == 180 for cell in context.selected):
            requested.remove("brier")
        if not any(cell.key[2] == ALL_MODEL_KEY for cell in context.selected):
            requested.difference_update(("importance", "top_features"))
    if context.settings.paper_bundle.enabled:
        requested.add("paper_bundle")
    return tuple(task for task in TASK_ORDER if task in requested and PRODUCERS.get(task) not in requested)


def run_regeneration(config_path: Path = CONFIG_PATH) -> Path:
    context = load_context(load_regeneration_config(config_path))
    configure_logging(context.output_dir / "regenerate.log")
    tasks = task_plan(context)
    producers = tuple(task for task in tasks if task != "paper_bundle")
    collect = "paper_bundle" in tasks
    artifacts = paper_artifacts(context)
    if collect:
        bundle_destination(context, artifacts)
    if producers:
        run = ArtifactRun(context.output_dir, context.settings.paper_bundle.horizon)
        run.save()
    else:
        run = load_artifact_run(context)
    path = context.output_dir / "regeneration.json"
    task_statuses = {task: "pending" for task in tasks}
    summary: dict[str, object] = {
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path.resolve()),
        "benchmark_dir": str(context.output_dir),
        "cells": [list(cell.key) for cell in context.selected],
        "allow_competing_model_fit": context.settings.allow_competing_model_fit,
        "tasks": task_statuses,
        "status": "running",
        "artifact_run_id": run.run_id,
    }
    write_summary(path, summary)
    active_task = "paper_bundle"
    try:
        for task in producers:
            active_task = task
            LOGGER.info("Regenerating %s (%d selected models)", task, len(context.selected))
            task_statuses[task] = "running"
            write_summary(path, summary)
            written = TASKS[task](context)
            files = generated_artifacts(context.output_dir, task, artifacts, () if written is None else written)
            run.record(task, files)
            task_statuses[task] = "complete"
            write_summary(path, summary)
            if collect:
                active_task = "paper_bundle"
                task_statuses[active_task] = "running"
                collect_paper_bundle(context, run, files)
        if producers:
            run.status = "complete"
            run.save()
        if collect:
            active_task = "paper_bundle"
            if producers:
                update_bundle_manifest(context, run)
            else:
                collect_paper_bundle(context, run, tuple(run.files))
            task_statuses[active_task] = "complete"
    except Exception as error:
        if producers:
            run.status, run.failed_task = "failed", active_task
            run.save()
        summary.update(status="failed", error=f"{type(error).__name__}: {error}")
        task_statuses[active_task] = "failed"
        write_summary(path, summary)
        if collect:
            try:
                update_bundle_manifest(context, run)
            except Exception:
                LOGGER.exception("Could not update the bundle manifest after failure")
        LOGGER.exception("Regeneration stopped at %s", active_task)
        raise
    summary.update(status="complete", finished_utc=datetime.now(timezone.utc).isoformat())
    write_summary(path, summary)
    LOGGER.info("Regenerated %d tasks; summary: %s", len(tasks), path)
    return path


def write_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
