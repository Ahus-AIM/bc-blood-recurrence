"""Shared cohort reports and saved-value redraws for selected model folders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, replace
import json
from pathlib import Path
from typing import cast

import pandas as pd

from src.data.population_statistics import PopulationStatisticsConfig, write_population_statistics_latex
from src.data.recurrence_blood_test_config import load_recurrence_blood_test_config
from src.model.blood_test_imputation_statistics import build_blood_test_imputation_statistics
from src.regeneration.context import RegenerationContext, SavedCell
from src.script.ebm_survival_config import serializable_mapping, update_run_metadata
from src.script.export_cohort_age_membership import export_saved_cohort_age_membership
from src.script.plot.dataset_statistics import render_saved_dataset_figure
from src.script.plot.recurrence_blood_test_insights import write_recurrence_blood_test_figure
from src.script.survival_benchmark_cache import artifact_paths, copy_relative_artifact
from src.script.survival_dataset_artifacts import write_eligible_patient_folds_csv, write_training_landmark_event_counts
from src.script.training_cohort_comparison import write_training_cohort_comparison
from src.script.training_dataset_statistics import write_training_dataset_statistics_png
from src.script.training_recurrence_blood_test_insights import (
    CONFIG_PATH,
    write_training_recurrence_blood_test_insights,
)
from src.script.update_cohort_comparison import refresh_cohort_comparison


def share_reports(group: tuple[SavedCell, ...], metadata: Mapping[str, object]) -> tuple[Path, ...]:
    """Render once per site/horizon, then update only selected model folders."""
    source = group[0].path
    paths = tuple(artifact_paths(metadata))
    for cell in group:
        if cell.path != source:
            for path in paths:
                copy_relative_artifact(source, cell.path, path)
        for key, value in metadata.items():
            update_run_metadata(cell.path, key, value)
    return tuple(cell.path / path for cell in group for path in paths)


def cohort_statistics(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for group in context.task_groups():
        cell = group[0]
        dataset = context.dataset(cell)
        population = PopulationStatisticsConfig.from_mapping(cell.raw.get("population_statistics"))
        statistics = build_blood_test_imputation_statistics(dataset, context.imputer(cell))
        metadata = {
            "eligible_patient_folds": write_eligible_patient_folds_csv(dataset, cell.path),
            "cohort_comparison": write_training_cohort_comparison(dataset, cell.path, population),
            "population_statistics": write_population_statistics_latex(
                dataset,
                population,
                cell.path,
                imputation_statistics=statistics,
            ),
            "dataset_statistics": write_training_dataset_statistics_png(
                dataset,
                cell.path,
                population_config=population,
                imputation_statistics=statistics,
            ),
        }
        paths.extend(share_reports(group, metadata))
    return tuple(paths)


def landmark_counts(context: RegenerationContext) -> None:
    for group in context.task_groups():
        cell = group[0]
        metadata = write_training_landmark_event_counts(
            context.dataset(cell),
            cell.path,
            cell.raw.get("landmark_event_counts"),
        )
        share_reports(group, {"landmark_event_counts": metadata})


def recurrence_statistics(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for group in context.task_groups():
        cell = group[0]
        metadata = write_training_recurrence_blood_test_insights(context.dataset(cell), cell.path)
        paths.extend(share_reports(group, {"recurrence_blood_test_insights": metadata}))
    return tuple(paths)


def dataset_figure(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for cell in context.selected:
        paths.extend(render_saved_dataset_figure(cell.path))
    return tuple(paths)


def recurrence_figures(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for cell in context.selected:
        config = load_recurrence_blood_test_config(
            CONFIG_PATH,
            source_config_path=cell.path / "training_config.yml",
            dataset_config=cell.config,
        )
        config = replace(config, output_dir=cell.path)
        report_path = cell.path / "recurrence_blood_test_insights.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        saved = report["config"]
        current = serializable_mapping(asdict(config))
        analysis_keys = (
            "windows",
            "site_rules",
            "include_overall",
            "overall_label",
            "other_label",
            "difference_significant_digits",
            "reference_ranges",
        )
        if changed := [key for key in analysis_keys if saved.get(key) != current[key]]:
            raise ValueError(
                f"Recurrence analysis settings changed ({', '.join(changed)}); "
                "run recurrence_statistics before redrawing."
            )
        frame = pd.read_csv(cell.path / "recurrence_blood_test_insights.csv", float_precision="round_trip")
        frame["unit"] = frame["unit"].fillna("")
        metadata = write_recurrence_blood_test_figure(frame, config)
        report["figure"] = metadata
        for key in ("plot", "fdr_alpha", "decimals"):
            saved[key] = current[key]
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        paths.append(report_path)
        paths.extend(cell.path / name for name in cast(list[str], metadata.get("files", [])))
    return tuple(paths)


def age_returns(context: RegenerationContext) -> tuple[Path, ...]:
    paths: list[Path] = []
    for cell in context.selected:
        paths.extend(refresh_cohort_comparison(cell.path))
    return tuple(paths)


def age_membership(context: RegenerationContext) -> None:
    for cell in context.selected:
        export_saved_cohort_age_membership(cell.path)
