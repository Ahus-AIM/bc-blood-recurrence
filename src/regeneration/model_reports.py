"""Figures and prediction references drawn from selected saved EBM models."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.model.imputation_config import ImputationConfig
from src.model.risk_score_calibration import write_training_risk_reference
from src.regeneration.context import RegenerationContext, SavedCell
from src.regeneration.importance import regenerate_importance
from src.reporting import report_output_path
from src.script.ebm_survival_config import update_run_metadata
from src.script.ebm_training_artifacts import write_configured_ebm_explanations
from src.script.plot_temporal_risk import (
    TemporalRiskConfig,
    build_temporal_risk_statistics,
    load_test_risk_scores,
    write_temporal_risk_outputs,
)
from src.script.saved_ebm_artifacts import predict_risk_scores
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY
from src.script.single_bloodtest_ebm_top_features import write_benchmark_ebm_top_features
from src.script.survival_benchmark_artifacts import file_sha256
from src.script.survival_model_input import build_model_input_artifact, write_model_input_artifact


def explanations(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for cell in context.selected:
        model = cell.model
        artifacts = write_configured_ebm_explanations(
            model,
            cell.path,
            cell.config,
            cell.raw["output"],
            cell.path,
            imputation_enabled=ImputationConfig.from_mapping(cell.raw.get("imputation")).enabled,
            fail_fast=True,
        )
        update_run_metadata(cell.path, "explanations", artifacts)
        paths.extend(cell.path / value for key, value in artifacts.items()
                     if key.endswith(("_png", "_svg")) and isinstance(value, str))
    return tuple(paths)


def require_all_tests(context: RegenerationContext) -> None:
    if not any(cell.key[2] == ALL_MODEL_KEY for cell in context.selected):
        raise ValueError("Importance/top-feature composites require models: [all_blood_tests] or models: null.")


def importance(context: RegenerationContext) -> tuple[Path, ...]:
    require_all_tests(context)
    return regenerate_importance(context.figure_benchmark(), context.output_dir, context.results, fail_fast=True)


def top_features(context: RegenerationContext) -> tuple[Path, ...]:
    require_all_tests(context)
    artifacts = write_benchmark_ebm_top_features(
        context.figure_benchmark(),
        context.output_dir,
        context.results,
        fail_fast=True,
    )
    return tuple(report_output_path(context.output_dir, item[extension])
                 for item in artifacts for extension in ("png", "svg"))


def model_inputs(context: RegenerationContext) -> None:
    for group in context.task_groups():
        cell = group[0]
        tables = {
            split: context.pools[0].landmarks.table(replace(cell.config, split=split))
            for split in ("train", "val", "test")
        }
        artifact = build_model_input_artifact(tables, context.imputer(cell))
        path = cell.path.parent / "model_input.csv"
        metadata = {**write_model_input_artifact(artifact, path), "sha256": file_sha256(path), "shared_by_task": True}
        for destination in group:
            update_run_metadata(destination.path, "model_input", {**metadata, "path": "../model_input.csv"})


def training_risk_reference(context: RegenerationContext) -> None:
    for cell in context.selected:
        table = context.pools[0].landmarks.table(cell.config)
        model = cell.model
        scores = predict_risk_scores(model, context.imputer(cell), table)
        metadata = write_training_risk_reference(table, scores, cell.path)
        update_run_metadata(cell.path, "training_risk_reference", metadata)


def temporal_config(context: RegenerationContext, cell: SavedCell) -> TemporalRiskConfig:
    values = cell.raw["temporal_risk"] if "temporal_risk" in cell.raw else context.presentation.get("temporal_risk", {})
    return TemporalRiskConfig.from_mapping(values, output_dir=cell.path)


def temporal_risk(context: RegenerationContext) -> None:
    for cell in context.selected:
        config = temporal_config(context, cell)
        scores = load_test_risk_scores(cell.path / "test_risk_scores.csv")
        frame = build_temporal_risk_statistics(
            cell.model,
            cell.raw,
            cell.config,
            config,
            scores,
            cell.path,
            prepared_dataset=context.dataset(cell),
        )
        write_temporal_risk_outputs(frame, config, cell.config)


def temporal_risk_figure(context: RegenerationContext) -> None:
    for cell in context.selected:
        config = temporal_config(context, cell)
        write_temporal_risk_outputs(pd.read_csv(config.output_csv_path), config, cell.config)
