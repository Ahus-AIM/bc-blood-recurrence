from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.model.imputation import FeatureImputationResult, feature_indices


def feature_subset_imputation(
    result: FeatureImputationResult,
    feature_names: tuple[str, ...],
    requested_feature_names: tuple[str, ...],
) -> FeatureImputationResult:
    """Project an imputed matrix and all source masks onto model features."""
    if requested_feature_names == feature_names:
        return result
    indices = feature_indices(feature_names, requested_feature_names)
    return replace(
        result,
        features=result.features[:, indices],
        imputed_mask=result.imputed_mask[:, indices],
        clipped_mask=result.clipped_mask[:, indices],
        carried_forward_mask=subset_optional_mask(result.carried_forward_mask, indices),
        locf_expired_mask=subset_optional_mask(result.locf_expired_mask, indices),
        fallback_mask=subset_optional_mask(result.fallback_mask, indices),
        median_fallback_mask=subset_optional_mask(result.median_fallback_mask, indices),
        mice_fallback_mask=subset_optional_mask(result.mice_fallback_mask, indices),
        derived_mask=subset_optional_mask(result.derived_mask, indices),
        past_mice_mask=subset_optional_mask(result.past_mice_mask, indices),
    )


def subset_optional_mask(
    mask: np.ndarray | None,
    indices: tuple[int, ...],
) -> np.ndarray | None:
    return None if mask is None else mask[:, indices]


def select_imputation_rows(result: FeatureImputationResult, indices: np.ndarray | slice) -> FeatureImputationResult:
    """Select rows in every source mask, preserving slice views or repeated bootstrap indices."""

    def selected(mask: np.ndarray | None) -> np.ndarray | None:
        return None if mask is None else mask[indices]

    return replace(
        result,
        features=result.features[indices],
        imputed_mask=result.imputed_mask[indices],
        clipped_mask=result.clipped_mask[indices],
        carried_forward_mask=selected(result.carried_forward_mask),
        locf_expired_mask=selected(result.locf_expired_mask),
        fallback_mask=selected(result.fallback_mask),
        median_fallback_mask=selected(result.median_fallback_mask),
        mice_fallback_mask=selected(result.mice_fallback_mask),
        derived_mask=selected(result.derived_mask),
        past_mice_mask=selected(result.past_mice_mask),
    )
