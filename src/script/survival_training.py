from __future__ import annotations

import json
import logging
import pickle
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.timing import timed
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    transform_landmark_table,
)
from src.model.imputation_config import ImputationConfig
from src.model.blood_test_change_imputation import fit_landmark_imputer
from src.model.imputation_metadata import imputation_result_counts
from src.model.imputation_projection import feature_subset_imputation
from src.model.risk_score_calibration import write_training_risk_reference
from src.model.ebm_bags import PatientGroupedBagPlan, patient_grouped_bags_enabled
from src.model.run_failure import failure_details
from src.model.ebm_fitting import fit_ebm_model
from src.model.survival_evaluation import (
    EVALUATION_SPLITS,
    SharedEvaluationFeatureCache,
    SurvivalEvaluationCache,
    evaluate_prepared_split,
    evaluate_split,
    feature_subset_table,
    primary_metric,
    report_c_indexes,
    require_matching_landmark_feature_rows,
    table_summary,
)
from src.script.ebm_training_artifacts import write_configured_ebm_explanations

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PreparedSurvivalTraining:
    """Shared inputs for fitting comparable survival models.

    ``dataset`` owns the prepared cohort reused for bootstrap evaluation.
    ``train_table`` fixes the model feature order, Recurrence-Free Survival
    labels, and weights.
    ``imputer`` is fitted on this table or a shared superset, while
    ``train_imputation`` is its model-feature projection for the estimator.
    ``patient_grouped_bag_plan`` optionally fixes one EBM plan across feature
    subsets of the same Endpoint Definition and Censoring Horizon.
    """

    dataset: PatientBloodTestDataset
    train_table: PatientLandmarkTable
    imputer: FittedFeatureImputer
    train_imputation: FeatureImputationResult
    patient_grouped_bag_plan: PatientGroupedBagPlan | None = None


@dataclass
class SharedSurvivalFeatureCache:
    """Training and evaluation features shared across compatible benchmark tasks.

    ``evaluation_features`` owns Endpoint-Definition-independent bootstrap transformations.
    The private prepared training value owns the fitted imputer and transformed
    non-bootstrapped training features from the first estimable task. Later
    tasks may change only Recurrence-Free Survival labels while retaining the
    feature schema and canonical Patient-Landmark Samples.
    """

    evaluation_features: SharedEvaluationFeatureCache = field(default_factory=SharedEvaluationFeatureCache)
    _prepared: PreparedSurvivalTraining | None = field(default=None, init=False, repr=False)

    def prepare(
        self,
        dataset: PatientBloodTestDataset,
        train_table: PatientLandmarkTable,
        imputation_config: Mapping[str, object] | None,
        *,
        nonnegative_feature_names: tuple[str, ...] = (),
    ) -> PreparedSurvivalTraining | None:
        """Prepare once, then attach later Recurrence-Free Survival labels."""
        if len(train_table.signed_durations) == 0:
            return None
        if self._prepared is None:
            self._prepared = prepare_survival_training(
                dataset,
                train_table,
                imputation_config,
                nonnegative_feature_names=nonnegative_feature_names,
            )
            return self._prepared
        requested_config = ImputationConfig.from_mapping(imputation_config)
        if requested_config != self._prepared.imputer.config:
            raise ValueError("Shared training imputation config differs between compatible tasks.")
        if nonnegative_feature_names != self._prepared.imputer.nonnegative_feature_names:
            raise ValueError("Shared nonnegative feature names differ between compatible tasks.")
        return reuse_survival_training_features(dataset, train_table, self._prepared)


@dataclass(frozen=True)
class FittedSurvivalTraining:
    """One EBM fitted and evaluated from shared prepared inputs.

    ``prepared`` identifies the exact input data and ``model`` is the fitted EBM.
    ``model_metadata`` records its library/config, and ``split_summaries``
    contains the requested bootstrap evaluations.
    """

    prepared: PreparedSurvivalTraining
    model: ExplainableBoostingRegressor
    model_metadata: dict[str, object]
    split_summaries: dict[str, dict[str, object]]


@dataclass(frozen=True)
class SurvivalRunArtifacts:
    """Artifact settings for one survival run.

    ``output_dir`` is the run directory; ``output_config`` and ``config_dir``
    configure EBM explanations. ``base_metadata`` describes pre-fit artifacts.
    Every fitted model saves its non-bootstrapped training risk reference for
    dashboard percentiles.
    """

    output_dir: Path
    output_config: Mapping[str, object]
    config_dir: Path
    base_metadata: Mapping[str, object]


def prepare_survival_training(
    dataset: PatientBloodTestDataset,
    train_table: PatientLandmarkTable,
    imputation_config: Mapping[str, object] | None,
    *,
    feature_names: tuple[str, ...] | None = None,
    nonnegative_feature_names: tuple[str, ...] = (),
) -> PreparedSurvivalTraining:
    """Fit one training-only imputer and freeze the inputs reused by EBM fits."""
    model_table = feature_subset_table(train_table, feature_names)
    imputer = fit_landmark_imputer(
        model_table,
        ImputationConfig.from_mapping(imputation_config),
        nonnegative_feature_names=nonnegative_feature_names,
    )
    train_imputation = transform_landmark_table(imputer, model_table)
    return PreparedSurvivalTraining(dataset, model_table, imputer, train_imputation)


def reuse_survival_training_features(
    dataset: PatientBloodTestDataset,
    train_table: PatientLandmarkTable,
    shared: PreparedSurvivalTraining,
) -> PreparedSurvivalTraining:
    """Attach task-specific Recurrence-Free Survival labels to shared features."""
    model_table = feature_subset_table(train_table, shared.train_table.feature_names)
    require_matching_landmark_feature_rows(model_table, shared.train_table)
    return PreparedSurvivalTraining(
        dataset,
        replace(model_table, features=shared.train_table.features),
        shared.imputer,
        shared.train_imputation,
    )


def feature_subset_prepared_training(
    prepared: PreparedSurvivalTraining,
    feature_names: tuple[str, ...],
) -> PreparedSurvivalTraining:
    """Reuse a full fitted imputer while exposing only one model feature set."""
    model_table = feature_subset_table(prepared.train_table, feature_names)
    model_imputation = feature_subset_imputation(
        prepared.train_imputation,
        prepared.train_table.feature_names,
        feature_names,
    )
    return PreparedSurvivalTraining(
        prepared.dataset,
        model_table,
        prepared.imputer,
        model_imputation,
        prepared.patient_grouped_bag_plan,
    )


def training_not_estimable_reason(
    train_table: PatientLandmarkTable,
    ebm_config: Mapping[str, object] | None = None,
) -> str | None:
    """Check cohort and EBM bag feasibility; sweeps defer bag checks to each trial."""
    if len(train_table.signed_durations) == 0:
        return "training split has no patient-landmark samples"
    event_rows = np.flatnonzero(train_table.events)
    if len(event_rows) == 0:
        return "training split has no configured endpoint recurrence events"
    if ebm_config is None:
        return None
    grouped_bags = patient_grouped_bags_enabled(ebm_config)
    event_patients = set(train_table.patient_ids[event_rows].tolist())
    if grouped_bags and len(event_patients) < 2:
        return "training split has fewer than two configured endpoint recurrence patients"
    return None


def train_ebm_model(
    prepared: PreparedSurvivalTraining,
    model_config: Mapping[str, object],
    evaluation_paths: Mapping[str, Path | None],
    *,
    evaluation_cache: SurvivalEvaluationCache | None = None,
) -> FittedSurvivalTraining:
    """Fit one EBM and evaluate every requested Temporal Split."""
    model, model_metadata = fit_ebm_model(
        prepared.train_table,
        prepared.train_imputation.features,
        model_config,
        patient_grouped_bag_plan=prepared.patient_grouped_bag_plan,
    )
    summaries = {
        EVALUATION_SPLITS[split]: evaluate_survival_split(
            model,
            prepared,
            split,
            output_path,
            evaluation_cache,
        )
        for split, output_path in evaluation_paths.items()
    }
    return FittedSurvivalTraining(
        prepared,
        model,
        model_metadata,
        summaries,
    )


def evaluate_survival_split(
    model: ExplainableBoostingRegressor,
    prepared: PreparedSurvivalTraining,
    split: str,
    output_path: Path | None,
    evaluation_cache: SurvivalEvaluationCache | None,
) -> dict[str, object]:
    """Evaluate one split from either direct or task-shared prepared inputs."""
    if evaluation_cache is not None:
        return evaluate_prepared_split(
            model,
            evaluation_cache.get(split),
            output_path,
            feature_names=prepared.train_table.feature_names,
        )
    return evaluate_split(
        model,
        split,
        prepared.dataset.config,
        prepared.dataset,
        prepared.imputer,
        output_path,
        feature_names=prepared.train_table.feature_names,
    )


def write_fitted_survival_run(
    fitted: FittedSurvivalTraining,
    artifacts: SurvivalRunArtifacts,
) -> dict[str, object]:
    """Write fitted EBM outputs, explanations, and the training risk reference."""
    metadata = fitted_run_metadata(fitted, artifacts.base_metadata)
    model = fitted.model
    if not isinstance(model, ExplainableBoostingRegressor):
        raise TypeError("EBM fitting must return ExplainableBoostingRegressor.")
    output_config = artifacts.output_config
    if bool(output_config.get("write_explanations", True)):
        metadata["explanations"] = write_configured_ebm_explanations(
            model,
            artifacts.output_dir,
            fitted.prepared.dataset.config,
            output_config,
            artifacts.config_dir,
            imputation_enabled=fitted.prepared.imputer.config.enabled,
        )
    metadata["training_risk_reference"] = write_training_risk_reference(
        fitted.prepared.train_table,
        model.predict(fitted.prepared.train_imputation.features),
        artifacts.output_dir,
    )
    write_outputs(model, fitted.prepared.imputer, metadata, artifacts.output_dir)
    report_c_indexes(metadata)
    return metadata


def fitted_run_metadata(
    fitted: FittedSurvivalTraining,
    base_metadata: Mapping[str, object],
) -> dict[str, object]:
    prepared = fitted.prepared
    imputation_metadata = dict(prepared.imputer.metadata)
    if prepared.imputer.config.enabled:
        imputation_metadata["artifact"] = "imputer.pkl"
        counts = imputation_result_counts(prepared.train_imputation)
        imputation_metadata.update(
            {f"training_{key}": value for key, value in counts.items() if key != "imputed_cell_count"}
        )
        imputation_metadata["training_fallback_method"] = prepared.train_imputation.fallback_method
    metadata = {
        "status": "complete",
        "model_kind": "ebm",
        "model": dict(fitted.model_metadata),
        "imputation": imputation_metadata,
        **run_metadata_base(prepared.train_table, base_metadata),
        **{key: dict(value) for key, value in fitted.split_summaries.items()},
    }
    saved_model_metadata = dict(fitted.model_metadata)
    model_config = saved_model_metadata.get("config", {})
    if not isinstance(model_config, Mapping):
        raise TypeError("Fitted EBM metadata config must be a mapping.")
    saved_model_metadata = {**dict(model_config), **saved_model_metadata}
    metadata["ebm"] = saved_model_metadata
    if "validation" in fitted.split_summaries:
        metadata["primary_metric"] = primary_metric(fitted.split_summaries["validation"])
    return metadata


def write_not_estimable_run(
    train_table: PatientLandmarkTable,
    output_dir: Path,
    reason: str,
    base_metadata: Mapping[str, object],
    imputer: FittedFeatureImputer | None = None,
) -> dict[str, object]:
    """Write the common metadata shape for a recognized non-estimable run."""
    return write_unfitted_run(
        train_table,
        output_dir,
        {"status": "not_estimable", "not_estimable_reason": reason},
        base_metadata,
        imputer,
    )


def write_failed_run(
    train_table: PatientLandmarkTable,
    output_dir: Path,
    error: Exception,
    base_metadata: Mapping[str, object],
    imputer: FittedFeatureImputer | None = None,
) -> dict[str, object]:
    """Write metadata for a failed benchmark model run."""
    return write_unfitted_run(
        train_table,
        output_dir,
        {"status": "failed", **failure_details(error)},
        base_metadata,
        imputer,
    )


def write_unfitted_run(
    train_table: PatientLandmarkTable,
    output_dir: Path,
    status_metadata: Mapping[str, object],
    base_metadata: Mapping[str, object],
    imputer: FittedFeatureImputer | None,
) -> dict[str, object]:
    metadata = {
        **dict(status_metadata),
        **run_metadata_base(train_table, base_metadata),
    }
    if imputer is not None:
        metadata["imputation"] = dict(imputer.metadata)
    write_metadata(metadata, output_dir)
    return metadata


def run_metadata_base(
    train_table: PatientLandmarkTable,
    base_metadata: Mapping[str, object],
) -> dict[str, object]:
    """Build the dataset and artifact envelope shared by every run status."""
    return {
        "train": table_summary(train_table),
        "dataset": train_table.metadata,
        "feature_names": train_table.feature_names,
        **dict(base_metadata),
    }


@timed(LOGGER, "write training outputs")
def write_outputs(
    model: ExplainableBoostingRegressor,
    feature_imputer: FittedFeatureImputer,
    metadata: dict[str, object],
    output_dir: Path,
) -> None:
    with (output_dir / "model.pkl").open("wb") as model_file:
        pickle.dump(model, model_file)
    if feature_imputer.config.enabled:
        with (output_dir / "imputer.pkl").open("wb") as imputer_file:
            pickle.dump(feature_imputer, imputer_file)
    write_metadata(metadata, output_dir)


def write_metadata(
    metadata: Mapping[str, object],
    output_dir: Path,
) -> None:
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True, default=json_default),
        encoding="utf-8",
    )


def json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable.")
