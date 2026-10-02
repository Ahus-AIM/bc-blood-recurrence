from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.blood_test_change import require_matching_change_inputs
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.dataset_config import PatientDatasetConfig
from src.data.timing import timed
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    feature_indices,
    transform_landmark_table,
)
from src.model.imputation_metadata import imputation_result_counts
from src.model.imputation_projection import feature_subset_imputation, select_imputation_rows
from src.model.survival_metrics import harrell_c_index_excluding_same_patient

LOGGER = logging.getLogger(__name__)
EVALUATION_SPLITS = {
    "train": "train_bootstrap",
    "val": "validation",
    "test": "test",
}
PreparedPoolFeatures = tuple[PatientLandmarkTable, FeatureImputationResult]


@dataclass(frozen=True)
class PreparedEvaluationTable:
    """One full-schema bootstrap table and its shared imputation result.

    ``table`` retains Recurrence-Free Survival labels and row metadata,
    ``imputation`` contains the full fitted-imputer schema, and
    ``bootstrap_run`` identifies the draw.
    """

    table: PatientLandmarkTable
    imputation: FeatureImputationResult | None
    bootstrap_run: int


@dataclass(frozen=True)
class PreparedSplitEvaluation:
    """One Temporal Split with all attempted bootstrap runs and its draw size."""

    split: str
    runs: tuple[PreparedEvaluationTable, ...]
    bootstrap_sample_count: int


@dataclass
class SharedEvaluationFeatureCache:
    """Imputed bootstrap features shared by compatible evaluation caches.

    Runs are stored by Temporal Split. Their feature rows and imputation masks
    are Censoring-Horizon-independent; each consuming cache supplies its own
    Recurrence-Free Survival labels and metadata after verifying that
    Patient-Landmark Samples still match exactly.
    """

    _runs_by_split: dict[str, tuple[PreparedEvaluationTable, ...]] = field(
        default_factory=dict,
        init=False,
        repr=False,
    )

    def get(self, split: str) -> tuple[PreparedEvaluationTable, ...] | None:
        """Return previously prepared feature runs for ``split`` when present."""
        return self._runs_by_split.get(split)

    def store(self, split: str, runs: tuple[PreparedEvaluationTable, ...]) -> None:
        """Retain the first compatible task's prepared feature runs for ``split``."""
        self._runs_by_split.setdefault(split, runs)


@dataclass
class SurvivalEvaluationCache:
    """Lazily prepare each split once for reuse by comparable fitted models.

    ``landmarks`` owns deterministic canonical rows and target labels.
    ``feature_imputer`` transforms the full shared feature schema before any
    model-specific subset is selected, and ``shared_features`` optionally
    shares those transformations across compatible tasks. ``pool_features``
    supplies already transformed full splits, when available from model inputs.
    """

    dataset_config: PatientDatasetConfig
    feature_imputer: FittedFeatureImputer
    landmarks: BenchmarkLandmarks
    shared_features: SharedEvaluationFeatureCache | None = None
    pool_features: Mapping[str, PreparedPoolFeatures] = field(default_factory=dict)
    _prepared_by_split: dict[str, PreparedSplitEvaluation] = field(default_factory=dict, init=False, repr=False)

    def get(self, split: str) -> PreparedSplitEvaluation:
        """Return one cached split, preparing it on first access."""
        prepared = self._prepared_by_split.get(split)
        if prepared is None:
            shared_runs = None if self.shared_features is None else self.shared_features.get(split)
            prepared = prepare_split_evaluation(
                split,
                self.dataset_config,
                self.feature_imputer,
                self.landmarks,
                shared_runs=shared_runs,
                pool_features=self.pool_features.get(split),
            )
            self._prepared_by_split[split] = prepared
            if self.shared_features is not None and shared_runs is None:
                self.shared_features.store(split, prepared.runs)
        return prepared


@timed(LOGGER, lambda config, *_args, **_kwargs: f"build {EVALUATION_SPLITS[config.split]} landmark table")
def landmark_table(
    config: PatientDatasetConfig,
    prepared_dataset: PatientBloodTestDataset,
    *,
    bootstrap: bool,
    bootstrap_run: int = 0,
) -> PatientLandmarkTable:
    """Build one split from an already prepared cohort."""
    dataset = PatientBloodTestDataset(config, prepared_data=prepared_dataset.prepared_data)
    return dataset.to_landmark_table(bootstrap=bootstrap, bootstrap_run=bootstrap_run)


@timed(LOGGER, lambda _model, split, *_args, **_kwargs: f"evaluate {EVALUATION_SPLITS[split]} bootstrap runs")
def evaluate_split(
    model: ExplainableBoostingRegressor,
    split: str,
    dataset_config: PatientDatasetConfig,
    prepared_dataset: PatientBloodTestDataset,
    feature_imputer: FittedFeatureImputer,
    output_path: Path | None,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    split_config = replace(dataset_config, split=split)
    run_count = bootstrap_run_count(split_config)
    reference_table = None
    if feature_imputer.requires_landmark_context:
        reference_table = feature_subset_table(
            landmark_table(split_config, prepared_dataset, bootstrap=False),
            feature_imputer.feature_names,
        )

    summaries: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    zero_event_run_count = 0
    sample_count = 0
    for bootstrap_run in range(run_count):
        table = feature_subset_table(
            landmark_table(
                split_config,
                prepared_dataset,
                bootstrap=True,
                bootstrap_run=bootstrap_run,
            ),
            feature_imputer.feature_names,
        )
        sample_count = len(table.durations)
        if not bool(table.events.any()):
            zero_event_run_count += 1
            continue
        summary, scores = evaluate_table(
            model,
            table,
            feature_imputer,
            bootstrap_run,
            reference_table=reference_table,
            feature_names=feature_names,
        )
        summaries.append(summary)
        score_frames.append(scores)

    if output_path is not None and score_frames:
        pd.concat(score_frames, ignore_index=True).to_csv(output_path, index=False)
    summary = summarize_bootstrap_runs(summaries, sample_count)
    return {
        **summary,
        "bootstrap_runs_attempted": run_count,
        "zero_event_bootstrap_runs_discarded": zero_event_run_count,
    }


@timed(
    LOGGER,
    lambda split, *_args, **_kwargs: f"prepare cached {EVALUATION_SPLITS[split]} bootstrap inputs",
)
def prepare_split_evaluation(
    split: str,
    dataset_config: PatientDatasetConfig,
    feature_imputer: FittedFeatureImputer,
    landmarks: BenchmarkLandmarks,
    *,
    shared_runs: tuple[PreparedEvaluationTable, ...] | None = None,
    pool_features: PreparedPoolFeatures | None = None,
) -> PreparedSplitEvaluation:
    """Impute the full pool once and select the unchanged canonical bootstrap draws."""
    split_config = replace(dataset_config, split=split)
    run_count = bootstrap_run_count(split_config)
    if shared_runs is not None and len(shared_runs) != run_count:
        raise ValueError("Shared evaluation feature run count does not match the configured bootstrap runs.")
    pool_imputation = None
    if shared_runs is None:
        pool_table = feature_subset_table(landmarks.table(split_config), feature_imputer.feature_names)
        if pool_features is not None:
            require_matching_landmark_feature_rows(pool_table, pool_features[0])
            pool_imputation = pool_features[1]
        elif len(pool_table.durations):
            pool_imputation = transform_landmark_table(feature_imputer, pool_table, reference_table=pool_table)
    runs = tuple(
        prepare_or_reuse_evaluation_table(
            landmarks.table(split_config, bootstrap_run),
            feature_imputer,
            bootstrap_run,
            pool_imputation=pool_imputation,
            row_indices=None if shared_runs is not None else landmarks.bootstrap_indices(split_config, bootstrap_run),
            shared=None if shared_runs is None else shared_runs[bootstrap_run],
        )
        for bootstrap_run in range(run_count)
    )
    sample_count = len(runs[0].table.durations) if runs else 0
    return PreparedSplitEvaluation(split, runs, sample_count)


def prepare_or_reuse_evaluation_table(
    table: PatientLandmarkTable,
    feature_imputer: FittedFeatureImputer,
    bootstrap_run: int,
    *,
    pool_imputation: FeatureImputationResult | None,
    row_indices: np.ndarray | None,
    shared: PreparedEvaluationTable | None,
) -> PreparedEvaluationTable:
    """Prepare once or attach current Recurrence-Free Survival labels to shared features."""
    if shared is None:
        imputation = None
        if len(table.durations):
            if pool_imputation is None or row_indices is None:
                raise ValueError("Nonempty bootstrap draws require prepared pool features and row indices.")
            imputation = select_imputation_rows(pool_imputation, row_indices)
        return PreparedEvaluationTable(
            feature_subset_table(table, feature_imputer.feature_names),
            imputation,
            bootstrap_run,
        )
    if shared.bootstrap_run != bootstrap_run:
        raise ValueError("Shared evaluation features belong to a different bootstrap run.")
    current_table = feature_subset_table(table, shared.table.feature_names)
    require_matching_landmark_feature_rows(current_table, shared.table)
    return PreparedEvaluationTable(
        replace(current_table, features=shared.table.features),
        shared.imputation,
        bootstrap_run,
    )


@timed(LOGGER, lambda _model, prepared, *_args, **_kwargs: f"evaluate cached {EVALUATION_SPLITS[prepared.split]}")
def evaluate_prepared_split(
    model: ExplainableBoostingRegressor,
    prepared: PreparedSplitEvaluation,
    output_path: Path | None,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Evaluate one model without rebuilding or reimputing bootstrap draws."""
    retained_runs = tuple(run for run in prepared.runs if bool(run.table.events.any()))
    evaluated = tuple(evaluate_prepared_table(model, run, feature_names=feature_names) for run in retained_runs)
    summaries = [summary for summary, _scores in evaluated]
    if output_path is not None and evaluated:
        pd.concat([scores for _summary, scores in evaluated], ignore_index=True).to_csv(output_path, index=False)
    summary = summarize_bootstrap_runs(summaries, prepared.bootstrap_sample_count)
    return {
        **summary,
        "bootstrap_runs_attempted": len(prepared.runs),
        "zero_event_bootstrap_runs_discarded": len(prepared.runs) - len(retained_runs),
    }


def feature_subset_table(
    table: PatientLandmarkTable,
    feature_names: tuple[str, ...] | None,
) -> PatientLandmarkTable:
    if feature_names is None or feature_names == table.feature_names:
        return table
    indices = feature_indices(table.feature_names, feature_names)
    return replace(
        table,
        features=table.features[:, indices],
        feature_names=feature_names,
        metadata={**table.metadata, "feature_subset": feature_names},
    )


def require_matching_landmark_feature_rows(
    current: PatientLandmarkTable,
    shared: PatientLandmarkTable,
) -> None:
    """Fail unless two compatible task tables have exactly the same feature rows."""
    if current.feature_names != shared.feature_names:
        raise ValueError("Shared landmark feature names differ between compatible tasks.")
    if not np.array_equal(current.patient_ids, shared.patient_ids):
        raise ValueError("Shared landmark patients differ between compatible tasks.")
    if not np.array_equal(current.landmark_days, shared.landmark_days):
        raise ValueError("Shared Landmark Days differ between compatible tasks.")
    if not np.array_equal(current.features, shared.features, equal_nan=True):
        raise ValueError("Shared landmark feature values differ between compatible tasks.")
    require_matching_change_inputs(current.blood_test_change_inputs, shared.blood_test_change_inputs)


def evaluate_table(
    model: ExplainableBoostingRegressor,
    table: PatientLandmarkTable,
    feature_imputer: FittedFeatureImputer,
    bootstrap_run: int,
    *,
    reference_table: PatientLandmarkTable | None = None,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    prepared = prepare_evaluation_table(
        table,
        feature_imputer,
        bootstrap_run,
        reference_table=reference_table,
    )
    return evaluate_prepared_table(model, prepared, feature_names=feature_names)


def prepare_evaluation_table(
    table: PatientLandmarkTable,
    feature_imputer: FittedFeatureImputer,
    bootstrap_run: int,
    *,
    reference_table: PatientLandmarkTable | None = None,
) -> PreparedEvaluationTable:
    """Impute one bootstrap table using the complete shared feature schema."""
    imputation_table = feature_subset_table(table, feature_imputer.feature_names)
    imputation_reference = (
        None if reference_table is None else feature_subset_table(reference_table, feature_imputer.feature_names)
    )
    imputation_result = None
    if len(imputation_table.signed_durations):
        imputation_result = transform_landmark_table(
            feature_imputer,
            imputation_table,
            reference_table=imputation_reference,
        )
    return PreparedEvaluationTable(imputation_table, imputation_result, bootstrap_run)


def evaluate_prepared_table(
    model: ExplainableBoostingRegressor,
    prepared: PreparedEvaluationTable,
    *,
    feature_names: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Score one prepared table after selecting the model feature subset."""
    model_table = feature_subset_table(prepared.table, feature_names)
    imputation_result = None
    risk_scores = np.asarray([], dtype=float)
    c_index = float("nan")
    if prepared.imputation is not None:
        imputation_result = feature_subset_imputation(
            prepared.imputation,
            prepared.table.feature_names,
            model_table.feature_names,
        )
        risk_scores = model.predict(imputation_result.features)
        c_index = harrell_c_index_excluding_same_patient(
            model_table.durations,
            model_table.events,
            risk_scores,
            model_table.patient_ids,
        )
    summary = {
        **table_summary(model_table),
        "bootstrap_run": prepared.bootstrap_run,
        "harrell_c_index": c_index,
        **imputation_result_counts(imputation_result),
    }
    scores = pd.DataFrame(
        {
            "bootstrap_run": prepared.bootstrap_run,
            "patient_id": model_table.patient_ids,
            "landmark_day": model_table.landmark_days,
            "bootstrap_selection_probability": bootstrap_selection_probabilities(model_table),
            "duration_days": model_table.durations,
            "event_observed": model_table.events,
            "risk_score": risk_scores,
        }
    )
    return summary, scores


def bootstrap_selection_probabilities(table: PatientLandmarkTable) -> np.ndarray:
    probabilities = table.bootstrap_selection_probabilities
    if probabilities is None:
        return np.full(len(table.landmark_days), np.nan, dtype=np.float64)
    return probabilities


def bootstrap_run_count(config: PatientDatasetConfig) -> int:
    count, field = {
        "train": (config.train_bootstrap_runs, "train_bootstrap_runs"),
        "val": (config.validation_bootstrap_runs, "validation_bootstrap_runs"),
        "test": (config.test_bootstrap_runs, "test_bootstrap_runs"),
    }[config.split]
    if count < 1:
        raise ValueError(f"`dataset.{field}` must be at least 1.")
    return count


def summarize_bootstrap_runs(
    run_summaries: list[dict[str, object]],
    sample_count: int,
) -> dict[str, object]:
    c_indexes = np.asarray(
        [float(summary["harrell_c_index"]) for summary in run_summaries],
        dtype=float,
    )
    c_indexes = c_indexes[np.isfinite(c_indexes)]
    lower, upper = (float("nan"), float("nan"))
    if len(c_indexes):
        lower, upper = (float(value) for value in np.percentile(c_indexes, [2.5, 97.5]))
    means = {
        f"{key}_mean": (
            float(np.mean([float(summary[key]) for summary in run_summaries])) if run_summaries else float("nan")
        )
        for key in ("row_count", "event_count", "patient_count")
    }
    return {
        "bootstrap_runs": len(run_summaries),
        "bootstrap_sample_count": sample_count,
        **means,
        "harrell_c_index": float(np.mean(c_indexes)) if len(c_indexes) else float("nan"),
        "harrell_c_index_valid_runs": int(len(c_indexes)),
        "harrell_c_index_ci95_lower": lower,
        "harrell_c_index_ci95_upper": upper,
        "harrell_c_index_std": float(np.std(c_indexes, ddof=1)) if len(c_indexes) > 1 else float("nan"),
        "runs": run_summaries,
    }


def primary_metric(validation_summary: Mapping[str, object]) -> dict[str, object]:
    return {
        "name": "harrell_c_index",
        "definition": (
            "mean sample-level Harrell's concordance index across bootstrap runs excluding same-patient pairs"
        ),
        "split": "validation",
        "value": validation_summary["harrell_c_index"],
        "ci95_lower": validation_summary["harrell_c_index_ci95_lower"],
        "ci95_upper": validation_summary["harrell_c_index_ci95_upper"],
    }


def report_c_indexes(metadata: Mapping[str, object]) -> None:
    for split in EVALUATION_SPLITS.values():
        summary = metadata.get(split)
        if isinstance(summary, Mapping) and "harrell_c_index" in summary:
            LOGGER.info(
                "%s harrell_c_index=%s ci95=[%s, %s] bootstrap_runs=%s",
                split,
                format_metric(summary["harrell_c_index"]),
                format_metric(summary.get("harrell_c_index_ci95_lower")),
                format_metric(summary.get("harrell_c_index_ci95_upper")),
                summary.get("bootstrap_runs"),
            )


def format_metric(value: object) -> str:
    if not isinstance(value, int | float | np.floating):
        return "nan"
    metric = float(value)
    return "nan" if np.isnan(metric) else f"{metric:.6f}"


def table_summary(table: PatientLandmarkTable) -> dict[str, object]:
    return {
        "row_count": int(len(table.signed_durations)),
        "event_count": int(np.sum(table.events)),
        "patient_count": int(len(set(table.patient_ids.tolist()))),
    }
