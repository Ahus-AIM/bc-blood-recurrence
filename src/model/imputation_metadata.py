from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from sklearn.impute import IterativeImputer

from src.model.imputation_config import ImputationConfig, MEDIAN_FALLBACK_METHOD

if TYPE_CHECKING:
    from src.model.imputation import FeatureImputationResult

def imputation_result_counts(
    result: FeatureImputationResult | None,
) -> dict[str, int]:
    """Count cells filled or changed by each imputation stage."""
    masks = {
        "imputed_cell_count": None if result is None else result.imputed_mask,
        "negative_imputed_values_clipped": None if result is None else result.clipped_mask,
        "locf_cell_count": None if result is None else result.carried_forward_mask,
        "locf_expired_cell_count": None if result is None else result.locf_expired_mask,
        "fallback_cell_count": None if result is None else result.fallback_mask,
        "median_fallback_cell_count": None if result is None else result.median_fallback_mask,
        "mice_fallback_cell_count": None if result is None else result.mice_fallback_mask,
        "derived_cell_count": None if result is None else result.derived_mask,
        "past_mice_cell_count": None if result is None else result.past_mice_mask,
    }
    return {
        key: 0 if mask is None else int(mask.sum())
        for key, mask in masks.items()
    }


IMPUTATION_COUNT_KEYS = tuple(imputation_result_counts(None))


def imputation_metadata(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    config: ImputationConfig,
    nonnegative_feature_names: tuple[str, ...],
    imputer: IterativeImputer,
) -> dict[str, object]:
    counts = missing_counts(features, feature_names)
    missing_counts_by_feature = {name: count for name, count in counts.items() if count > 0}
    completed_iterations = int(imputer.n_iter_)
    return {
        "enabled": True,
        "method": config.method,
        "implementation": "sklearn.impute.IterativeImputer",
        "max_iter": config.max_iter,
        "tol": config.tol,
        "completed_iterations": completed_iterations,
        "max_iter_reached": completed_iterations >= config.max_iter,
        "random_state": config.random_state,
        "feature_count": len(feature_names),
        "training_missing_cell_count": int(sum(missing_counts_by_feature.values())),
        "training_missing_feature_count": len(missing_counts_by_feature),
        "training_missing_counts_by_feature": missing_counts_by_feature,
        "negative_imputed_values_clipped_to_zero": True,
        "nonnegative_feature_count": len(nonnegative_feature_names),
        "nonnegative_feature_names": nonnegative_feature_names,
    }


def locf_imputation_metadata(
    raw_features: np.ndarray,
    post_locf_features: np.ndarray,
    feature_names: tuple[str, ...],
    config: ImputationConfig,
    nonnegative_feature_names: tuple[str, ...],
    locf_feature_names: tuple[str, ...],
    training_medians: tuple[float, ...],
    fallback_imputer: IterativeImputer | None,
) -> dict[str, object]:
    raw_counts = missing_counts(raw_features, feature_names)
    remaining_counts = missing_counts(post_locf_features, feature_names)
    raw_missing = {name: count for name, count in raw_counts.items() if count > 0}
    remaining_missing = {name: count for name, count in remaining_counts.items() if count > 0}
    metadata = {
        "enabled": True,
        "method": config.method,
        "implementation": "last observation carried forward from previous valid landmark day",
        "locf_max_age_days": config.locf_max_age_days,
        "fallback_method": config.locf_fallback_method,
        "fallback": locf_fallback_description(config.locf_fallback_method),
        "feature_count": len(feature_names),
        "training_missing_cell_count": int(sum(raw_missing.values())),
        "training_missing_feature_count": len(raw_missing),
        "training_missing_counts_by_feature": raw_missing,
        "training_remaining_missing_after_locf_cell_count": int(sum(remaining_missing.values())),
        "training_remaining_missing_after_locf_feature_count": len(remaining_missing),
        "training_remaining_missing_after_locf_counts_by_feature": remaining_missing,
        "locf_feature_count": len(locf_feature_names),
        "locf_feature_names": locf_feature_names,
        "remaining_missing_values_fail": False,
        "negative_imputed_values_clipped_to_zero": True,
        "nonnegative_feature_count": len(nonnegative_feature_names),
        "nonnegative_feature_names": nonnegative_feature_names,
    }
    if config.locf_fallback_method == MEDIAN_FALLBACK_METHOD:
        metadata.update(median_fallback_metadata(feature_names, locf_feature_names, training_medians))
    elif fallback_imputer is not None:
        completed_iterations = int(fallback_imputer.n_iter_)
        metadata.update(
            {
                "fallback_implementation": "sklearn.impute.IterativeImputer",
                "max_iter": config.max_iter,
                "tol": config.tol,
                "completed_iterations": completed_iterations,
                "max_iter_reached": completed_iterations >= config.max_iter,
                "random_state": config.random_state,
            }
        )
    return metadata


def locf_fallback_description(method: str) -> str:
    if method == MEDIAN_FALLBACK_METHOD:
        return "training median by model feature after LOCF"
    return "MICE fitted on the post-LOCF training feature matrix"


def median_fallback_metadata(
    feature_names: tuple[str, ...],
    locf_feature_names: tuple[str, ...],
    training_medians: tuple[float, ...],
) -> dict[str, object]:
    median_by_feature = {feature_names[index]: training_medians[index] for index in range(len(feature_names))}
    locf_name_set = frozenset(locf_feature_names)
    median_by_locf_feature = {
        name: training_medians[index] for index, name in enumerate(feature_names) if name in locf_name_set
    }
    return {
        "training_median_by_fallback_feature": median_by_feature,
        "training_median_by_locf_feature": median_by_locf_feature,
    }


def missing_counts(features: np.ndarray, feature_names: tuple[str, ...]) -> dict[str, int]:
    counts = np.isnan(features).sum(axis=0)
    return {name: int(counts[index]) for index, name in enumerate(feature_names)}
