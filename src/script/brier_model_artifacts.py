from __future__ import annotations

import json
import logging
import pickle
import platform
from collections.abc import Mapping
from dataclasses import asdict, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor
from joblib import hash as joblib_hash

from src.config import load_yml
from src.data import PatientLandmarkTable
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.competing_risk_outcomes import (
    HORIZON_DAYS,
    competing_risk_outcomes,
    competing_training_table,
)
from src.data.dataset_config import PatientDatasetConfig
from src.data.sample_weighting import patient_uniform_weights
from src.model.brier_diagnostics import risk_set_diagnostics, risk_set_summary, score_diagnostics
from src.model.cause_specific_probability import (
    BreslowBaseline,
    centered_relative_risks,
    cumulative_incidence,
    log_relative_risks,
    prediction_scale,
    weighted_breslow,
)
from src.model.ebm_fitting import fit_ebm_model
from src.model.imputation import feature_indices
from src.model.survival_evaluation import feature_subset_table
from src.script.brier_diagnostics import BrierDiagnostics
from src.script.ebm_survival_config import TRAINING_CONFIG_FILENAME
from src.script.saved_ebm_artifacts import load_feature_imputer, load_model, model_feature_names
from src.script.survival_benchmark_artifacts import file_sha256
from src.script.survival_benchmark_cache import fitted_imputer_fingerprint
from src.script.survival_model_input import SharedModelInputFeatureCache

LOGGER = logging.getLogger(__name__)


def saved_cell_config(model_dir: Path) -> tuple[dict[str, dict[str, object]], PatientDatasetConfig]:
    raw = load_yml(model_dir / TRAINING_CONFIG_FILENAME)
    config = PatientDatasetConfig.from_mapping(raw["dataset"], config_dir=model_dir)
    if config.censoring_horizon_days != HORIZON_DAYS:
        raise ValueError("Brier evaluation currently supports saved 180-day models only.")
    if config.bootstrap_sample_weighting != "person_uniform":
        raise ValueError("Brier evaluation requires the patient-then-one-landmark C-index sampling mode.")
    return raw, config


def prepare_brier_cell(
    model_dir: Path,
    destination: Path,
    landmarks: BenchmarkLandmarks,
    cache: dict[str, SharedModelInputFeatureCache],
    *,
    allow_competing_model_fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Reuse recurrence model/imputer, fit or load its competitor, then save fixed probabilities."""
    with BrierDiagnostics(destination) as report:
        return prepare_diagnosed_brier_cell(
            model_dir, destination, landmarks, cache, report,
            allow_competing_model_fit=allow_competing_model_fit,
        )


def prepare_diagnosed_brier_cell(
    model_dir: Path,
    destination: Path,
    landmarks: BenchmarkLandmarks,
    cache: dict[str, SharedModelInputFeatureCache],
    report: BrierDiagnostics,
    *,
    allow_competing_model_fit: bool = True,
) -> tuple[pd.DataFrame, dict[str, object]]:
    raw, config = saved_cell_config(model_dir)
    tables = {split: landmarks.table(replace(config, split=split)) for split in ("train", "val", "test")}
    model = load_model(model_dir / "model.pkl")
    feature_names = model_feature_names(model)
    imputer = load_feature_imputer(raw, model_dir, tables["train"])
    fingerprint = fitted_imputer_fingerprint(imputer)
    shared = cache.setdefault(fingerprint, SharedModelInputFeatureCache())
    full_tables = {split: feature_subset_table(table, imputer.feature_names) for split, table in tables.items()}
    prepared = shared.prepare(full_tables, imputer if shared.prepared is None else shared.imputer)
    indices = feature_indices(imputer.feature_names, feature_names)
    features = {split: result.features[:, indices] for split, result in prepared.imputation_results.items()}
    train = feature_subset_table(tables["train"], feature_names)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "source_training_config.yml").write_text(
        (model_dir / TRAINING_CONFIG_FILENAME).read_text(encoding="utf-8"), encoding="utf-8"
    )
    report.record("stage", "fitting_or_loading_competing_model")
    competing_model, competing_metadata = load_or_fit_competitor(
        train, features["train"], raw["ebm"], destination, allow_fit=allow_competing_model_fit,
    )
    weights = patient_uniform_weights(train.patient_ids)
    target_risks, target_center = predict_splits(model, features, weights, report=report, cause="target")
    competing_risks, competing_center = predict_splits(competing_model, features, weights, report=report, cause="competing")
    report.record("stage", "fitting_baselines")
    outcomes = competing_risk_outcomes(train)
    target_baseline = weighted_breslow(outcomes.durations, outcomes.causes == 1, target_risks["train"], weights)
    competing_baseline = weighted_breslow(outcomes.durations, outcomes.causes == 2, competing_risks["train"], weights)
    write_baselines(target_baseline, competing_baseline, destination)
    risk_sets = pd.concat([
        risk_set_diagnostics(outcomes.durations, train.patient_ids, weights, risks["train"], baseline, cause)
        for cause, risks, baseline in (
            ("target", target_risks, target_baseline), ("competing", competing_risks, competing_baseline)
        )
    ], ignore_index=True)
    risk_sets.to_csv(destination / "risk_set_diagnostics.csv", index=False)
    report.record("risk_sets", risk_set_summary(risk_sets))
    report.record("stage", "computing_probabilities")
    frames, invalid_frames = [], []
    for split, table in tables.items():
        frame, invalid = landmark_predictions(
            table, target_baseline, competing_baseline, target_risks[split], competing_risks[split]
        )
        frames.append(frame.assign(split=split))
        invalid_frames.append(invalid.assign(split=split))
    predictions = pd.concat(frames, ignore_index=True)
    populated_diagnostics = [frame for frame in invalid_frames if not frame.empty]
    diagnostics = pd.concat(populated_diagnostics, ignore_index=True) if populated_diagnostics else invalid_frames[0]
    predictions.to_csv(destination / "predictions.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    diagnostics.to_csv(destination / "invalid_hazard_increments.csv", index=False)
    report.predictions(predictions, diagnostics)
    metadata = {
        "source_model": str(model_dir / "model.pkl"),
        "source_model_sha256": file_sha256(model_dir / "model.pkl"),
        "source_config_sha256": file_sha256(model_dir / TRAINING_CONFIG_FILENAME),
        "fitted_imputer_fingerprint": fingerprint,
        "input_fingerprint": train.metadata["input_fingerprint"],
        "feature_names": feature_names,
        "target_prediction_scale": prediction_scale(model),
        "risk_centering": "patient_weighted_mean_training_log_risk",
        "target_log_risk_center": target_center,
        "competing_log_risk_center": competing_center,
        "competing_model": competing_metadata,
        "baseline_weighting": "person_uniform_in_numerator_and_denominator",
        "horizon_days": HORIZON_DAYS,
        "invalid_hazard_increment_count": len(diagnostics),
        "invalid_prediction_count": int(predictions["target_probability_180d"].isna().sum()),
        "probability_method": "product_integral",
        "diagnostics_path": "diagnostics.json",
        "risk_set_diagnostics_path": "risk_set_diagnostics.csv",
        "ipcw": False,
        "runtime_versions": {
            "python": platform.python_version(),
            **{name: version(name) for name in ("numpy", "pandas", "interpret")},
        },
    }
    (destination / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if len(diagnostics):
        LOGGER.error("%s: %d invalid hazard increments; see diagnostics CSV", destination, len(diagnostics))
    report.record("stage", "finished")
    return predictions, metadata


def load_or_fit_competitor(
    table: PatientLandmarkTable,
    features: np.ndarray,
    parameters: Mapping[str, object],
    destination: Path,
    *,
    allow_fit: bool = True,
) -> tuple[ExplainableBoostingRegressor | None, dict[str, object]]:
    train = competing_training_table(table)
    fingerprint = joblib_hash((features, train, dict(parameters)))
    model_path = destination / "competing_model.pkl"
    metadata_path = destination / "competing_model.json"
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("training_fingerprint") == fingerprint:
            if metadata["method"] == "zero_hazard_no_training_competing_events":
                return None, metadata
            if model_path.is_file() and file_sha256(model_path) == metadata.get("model_sha256"):
                model = load_model(model_path)
                if tuple(model.feature_names_in_) != table.feature_names:
                    raise ValueError("Saved competing-model predictors do not match the recurrence model.")
                return model, metadata
    if train.events.any() and not allow_fit:
        raise ValueError(
            f"Missing or outdated Brier competing model: {model_path}. "
            "Set regenerate.allow_competing_model_fit: true in src/config/regenerate.yml to fit it."
        )
    if not train.events.any():
        model = None
        model_path.unlink(missing_ok=True)
        fitted_metadata = {"method": "zero_hazard_no_training_competing_events", "prediction_scale": "relative_risk"}
    else:
        model, fitted_metadata = fit_ebm_model(train, features, parameters)
        model_path.write_bytes(pickle.dumps(model))
        fitted_metadata.update(
            method="survival_cox_ebm", prediction_scale=prediction_scale(model), model_sha256=file_sha256(model_path)
        )
    metadata = {
        **fitted_metadata,
        "training_fingerprint": fingerprint,
        "feature_names": list(table.feature_names),
        "training_weighting": "person_uniform",
        "training_competing_events": int(train.events.sum()),
        "parameters": dict(parameters),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")
    return model, metadata


def predict_splits(
    model: ExplainableBoostingRegressor | None,
    features: Mapping[str, np.ndarray],
    training_weights: np.ndarray,
    *,
    report: BrierDiagnostics | None = None,
    cause: str = "target",
) -> tuple[dict[str, np.ndarray], float]:
    """Predict risks relative to a single training-derived center for this cause."""
    log_risks = {}
    for split, values in features.items():
        if report is not None:
            report.record("stage", f"predicting_{cause}_{split}")
        log_risks[split] = np.zeros(len(values)) if model is None or len(values) == 0 else log_relative_risks(model, values)
    if report is not None:
        report.record(f"{cause}_scores", score_diagnostics(log_risks, training_weights))
        report.record("stage", f"exponentiating_{cause}_scores")
    return centered_relative_risks(log_risks, training_weights)


def write_baselines(target: BreslowBaseline, competing: BreslowBaseline, destination: Path) -> None:
    pd.concat(
        [
            pd.DataFrame(asdict(baseline)).assign(cause=cause)
            for cause, baseline in (("target", target), ("competing", competing))
        ],
        ignore_index=True,
    ).to_csv(destination / "breslow_baselines.csv", index=False)


def landmark_predictions(
    table: PatientLandmarkTable,
    target_baseline: BreslowBaseline,
    competing_baseline: BreslowBaseline,
    target_risks: np.ndarray,
    competing_risks: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    outcomes = competing_risk_outcomes(table)
    probabilities = cumulative_incidence(target_baseline, competing_baseline, target_risks, competing_risks)
    predictions = pd.DataFrame(
        {
            "patient_id": table.patient_ids,
            "landmark_day": table.landmark_days,
            "duration_days": outcomes.durations,
            "outcome_reason": table.target_end_reasons,
            "cause": outcomes.causes,
            "outcome_180d": outcomes.binary_outcomes,
            "target_relative_risk": target_risks,
            "competing_relative_risk": competing_risks,
            "target_probability_180d": probabilities.target,
            "competing_probability_180d": probabilities.competing,
            "event_free_probability_180d": probabilities.survival,
            "max_combined_hazard_jump": probabilities.max_combined_hazard_jump,
            "squared_error": (outcomes.binary_outcomes - probabilities.target) ** 2,
            "patient_weight": patient_uniform_weights(table.patient_ids),
        }
    )
    columns = ["patient_id", "landmark_day", "time_days", "target_increment", "competing_increment", "reason"]
    diagnostics = pd.DataFrame(
        [
            {
                "patient_id": table.patient_ids[entry.row_index],
                "landmark_day": table.landmark_days[entry.row_index],
                "time_days": entry.time_days,
                "target_increment": entry.target_increment,
                "competing_increment": entry.competing_increment,
                "reason": "nonfinite, negative, or combined hazard jump greater than one",
            }
            for entry in probabilities.invalid_hazards
        ],
        columns=columns,
    )
    return predictions, diagnostics
