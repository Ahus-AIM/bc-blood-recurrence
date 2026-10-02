from __future__ import annotations

from dataclasses import dataclass, replace
from typing import overload

import numpy as np

from src.data.blood_test_change import BLOOD_TEST_CHANGE_VERSION
from src.data.blood_test_features import BLOOD_TEST_CHANGE_SUFFIX
from src.data.types import PatientLandmarkTable
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    fit_feature_imputer,
    validate_feature_width,
)
from src.model.imputation_config import ImputationConfig


@dataclass(frozen=True)
class ChangeFeaturePair:
    """Map one test's current and change columns to its stored past observations.

    ``test_key`` identifies the raw test. ``current_column`` and
    ``change_column`` index the model matrix; ``past_column`` indexes the
    landmark table's separate observed past-value matrix.
    """

    test_key: str
    current_column: int
    change_column: int
    past_column: int

    @property
    def past_feature_name(self) -> str:
        return f"{self.test_key} past"


@dataclass(frozen=True, kw_only=True)
class FittedBloodTestChangeImputer(FittedFeatureImputer):
    """Saved model adapter deriving changes after current/past value imputation.

    Inherited fields describe the model's output schema. ``value_imputer``
    owns the training-fitted current/past/context input schema and applies LOCF
    only to current values. ``change_pairs`` maps completed inputs back to the
    model's signed differences. Older saved base imputers keep their original
    direct-change imputation behavior.
    """

    value_imputer: FittedFeatureImputer
    change_pairs: tuple[ChangeFeaturePair, ...]

    def transform(self, features: np.ndarray) -> FeatureImputationResult:
        validate_feature_width(features, self.feature_names)
        raise ValueError("Blood-test changes require stored past observations; use `transform_table`.")

    def transform_table(
        self,
        table: PatientLandmarkTable,
        *,
        reference_table: PatientLandmarkTable | None = None,
    ) -> FeatureImputationResult:
        if table.feature_names != self.feature_names:
            raise ValueError("Landmark feature names differ from the fitted blood-test change schema.")
        values = change_value_table(table, self.change_pairs)
        reference = None if reference_table is None else change_value_table(reference_table, self.change_pairs)
        completed = self.value_imputer.transform_table(values, reference_table=reference)
        return derive_completed_changes(completed, self.change_pairs)


def fit_landmark_imputer(
    table: PatientLandmarkTable,
    config: ImputationConfig,
    *,
    nonnegative_feature_names: tuple[str, ...] = (),
) -> FittedFeatureImputer:
    """Fit on raw values and context, deriving optional changes only after completion."""
    pairs = change_feature_pairs(table) if config.enabled else ()
    if not pairs:
        return fit_feature_imputer(
            table.features,
            table.feature_names,
            config,
            nonnegative_feature_names=nonnegative_feature_names,
            landmark_table=table,
        )
    values = change_value_table(table, pairs)
    past_names = tuple(pair.past_feature_name for pair in pairs)
    value_imputer = fit_feature_imputer(
        values.features,
        values.feature_names,
        config,
        nonnegative_feature_names=(*nonnegative_feature_names, *past_names),
        locf_feature_names=nonnegative_feature_names,
        landmark_table=values,
    )
    return FittedBloodTestChangeImputer(
        config=config,
        feature_names=table.feature_names,
        nonnegative_feature_names=nonnegative_feature_names,
        nonnegative_feature_indices=tuple(table.feature_names.index(name) for name in nonnegative_feature_names),
        imputer=value_imputer.imputer,
        metadata={
            **value_imputer.metadata,
            "blood_test_change_version": BLOOD_TEST_CHANGE_VERSION,
            "change_calculation": "completed current value minus completed past value",
            "imputation_feature_names": values.feature_names,
            "derived_change_feature_names": tuple(table.feature_names[pair.change_column] for pair in pairs),
            "past_values_use_locf": False,
        },
        training_medians=value_imputer.training_medians,
        locf_feature_names=value_imputer.locf_feature_names,
        locf_feature_indices=value_imputer.locf_feature_indices,
        value_imputer=value_imputer,
        change_pairs=pairs,
    )


def change_feature_pairs(table: PatientLandmarkTable) -> tuple[ChangeFeaturePair, ...]:
    inputs = table.blood_test_change_inputs
    if inputs is None:
        if table.metadata.get("include_blood_test_change", False) and any(
            name.endswith(BLOOD_TEST_CHANGE_SUFFIX) for name in table.feature_names
        ):
            raise ValueError("Blood-test change fitting requires stored past observations; rebuild the landmark table.")
        return ()
    columns = {name: index for index, name in enumerate(table.feature_names)}
    pairs = []
    for past_column, key in enumerate(inputs.test_keys):
        change_name = f"{key}{BLOOD_TEST_CHANGE_SUFFIX}"
        if change_name not in columns:
            continue
        if key not in columns:
            raise ValueError(f"Imputing {change_name} requires its current-value feature {key}.")
        pairs.append(ChangeFeaturePair(key, columns[key], columns[change_name], past_column))
    return tuple(pairs)


def change_value_table(
    table: PatientLandmarkTable,
    pairs: tuple[ChangeFeaturePair, ...],
) -> PatientLandmarkTable:
    """Replace raw differences with observed past values for joint imputation."""
    inputs = table.blood_test_change_inputs
    if inputs is None or inputs.past_values.shape != (len(table.features), len(inputs.test_keys)):
        raise ValueError("Blood-test change imputation requires past observations aligned with landmark rows.")
    features = table.features.copy()
    names = list(table.feature_names)
    for pair in pairs:
        if inputs.test_keys[pair.past_column] != pair.test_key:
            raise ValueError("Past blood-test columns differ from the fitted change schema.")
        features[:, pair.change_column] = inputs.past_values[:, pair.past_column]
        names[pair.change_column] = pair.past_feature_name
    return replace(table, features=features, feature_names=tuple(names), blood_test_change_inputs=None)


def derive_completed_changes(
    completed: FeatureImputationResult,
    pairs: tuple[ChangeFeaturePair, ...],
) -> FeatureImputationResult:
    """Subtract completed inputs and label derived values separately from direct imputations."""
    features = completed.features.copy()
    imputed = completed.imputed_mask.copy()
    derived = np.zeros_like(imputed)
    for pair in pairs:
        current, change = pair.current_column, pair.change_column
        features[:, change] = completed.features[:, current] - completed.features[:, change]
        imputed[:, change] = completed.imputed_mask[:, current] | completed.imputed_mask[:, change]
        derived[:, change] = imputed[:, change]
    columns = tuple(pair.change_column for pair in pairs)
    return replace(
        completed,
        features=features,
        imputed_mask=imputed,
        derived_mask=derived,
        past_mice_mask=past_mice_imputation_mask(completed, columns),
        clipped_mask=clear_change_columns(completed.clipped_mask, columns),
        carried_forward_mask=clear_change_columns(completed.carried_forward_mask, columns),
        locf_expired_mask=clear_change_columns(completed.locf_expired_mask, columns),
        fallback_mask=clear_change_columns(completed.fallback_mask, columns),
        median_fallback_mask=clear_change_columns(completed.median_fallback_mask, columns),
        mice_fallback_mask=clear_change_columns(completed.mice_fallback_mask, columns),
    )


def past_mice_imputation_mask(
    completed: FeatureImputationResult,
    change_columns: tuple[int, ...],
) -> np.ndarray:
    """Keep the past-input MICE provenance before its columns become differences."""
    mice_mask = completed.imputed_mask if completed.fallback_method is None else completed.mice_fallback_mask
    past_mask = np.zeros_like(completed.imputed_mask)
    if mice_mask is not None:
        past_mask[:, change_columns] = mice_mask[:, change_columns]
    return past_mask


@overload
def clear_change_columns(mask: np.ndarray, columns: tuple[int, ...]) -> np.ndarray: ...


@overload
def clear_change_columns(mask: None, columns: tuple[int, ...]) -> None: ...


def clear_change_columns(mask: np.ndarray | None, columns: tuple[int, ...]) -> np.ndarray | None:
    if mask is None:
        return None
    result = mask.copy()
    result[:, columns] = False
    return result
