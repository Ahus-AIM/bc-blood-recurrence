from __future__ import annotations

import pickle
from collections.abc import Mapping
from pathlib import Path

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientLandmarkTable
from src.model.imputation import FittedFeatureImputer, ImputationConfig, transform_landmark_table
from src.model.imputation_projection import feature_subset_imputation
from src.model.survival_evaluation import feature_subset_table


def load_model(model_path: Path) -> ExplainableBoostingRegressor:
    """Load a fitted EBM survival model."""
    if not model_path.is_file():
        raise FileNotFoundError(f"Saved survival model not found: {model_path}")
    model = read_pickled_artifact(model_path)
    if not isinstance(model, ExplainableBoostingRegressor):
        raise TypeError(f"Unsupported saved survival model: {type(model).__name__}.")
    return model


def read_pickled_artifact(path: Path) -> object:
    """Read a saved artifact; callers retain their validation and error-reporting policies."""
    with path.open("rb") as artifact_file:
        return pickle.load(artifact_file)


def model_feature_names(model: ExplainableBoostingRegressor) -> tuple[str, ...]:
    """Return the feature order stored by the fitted model."""
    return tuple(str(name) for name in model.feature_names_in_)


def predict_risk_scores(
    model: ExplainableBoostingRegressor,
    feature_imputer: FittedFeatureImputer,
    table: PatientLandmarkTable,
) -> np.ndarray:
    """Impute the saved full panel before selecting the model's ordered features."""
    imputation_table = feature_subset_table(table, feature_imputer.feature_names)
    imputation_result = transform_landmark_table(feature_imputer, imputation_table)
    model_imputation = feature_subset_imputation(
        imputation_result,
        imputation_table.feature_names,
        model_feature_names(model),
    )
    return np.asarray(model.predict(model_imputation.features), dtype=float)


def load_feature_imputer(
    raw_config: Mapping[str, Mapping[str, object]],
    output_dir: Path,
    landmark_table: PatientLandmarkTable,
) -> FittedFeatureImputer:
    """Load the saved imputer or build the disabled adapter for a landmark feature schema."""
    imputation_config = ImputationConfig.from_mapping(raw_config.get("imputation"))
    if not imputation_config.enabled:
        return FittedFeatureImputer(
            imputation_config,
            landmark_table.feature_names,
            (),
            (),
            None,
            {"enabled": False},
        )
    imputer_path = output_dir / "imputer.pkl"
    if not imputer_path.is_file():
        raise FileNotFoundError(f"Saved feature imputer not found: {imputer_path}")
    imputer = read_pickled_artifact(imputer_path)
    if not isinstance(imputer, FittedFeatureImputer):
        raise TypeError(f"Saved imputer must be FittedFeatureImputer, got {type(imputer).__name__}.")
    return imputer
