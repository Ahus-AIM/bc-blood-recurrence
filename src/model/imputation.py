from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from src.data.types import PatientLandmarkTable
from src.model.imputation_config import (
    LOCF_METHOD,
    MEDIAN_FALLBACK_METHOD,
    MICE_METHOD,
    ImputationConfig,
)
from src.model.imputation_metadata import imputation_metadata, locf_imputation_metadata, missing_counts


@dataclass(frozen=True)
class FeatureImputationResult:
    """Completed features and source masks aligned with model rows and columns.

    ``derived_mask`` marks changes using imputed inputs. ``past_mice_mask``
    marks change columns whose past input was MICE-imputed, preserving that
    additional value for reporting separately from current-value imputation.
    """

    features: np.ndarray
    imputed_mask: np.ndarray
    clipped_mask: np.ndarray
    carried_forward_mask: np.ndarray | None = None
    locf_expired_mask: np.ndarray | None = None
    fallback_mask: np.ndarray | None = None
    median_fallback_mask: np.ndarray | None = None
    mice_fallback_mask: np.ndarray | None = None
    fallback_method: str | None = None
    derived_mask: np.ndarray | None = None
    past_mice_mask: np.ndarray | None = None


@dataclass(frozen=True)
class FittedFeatureImputer:
    """Fitted imputer, feature schema, clipping columns, metadata, and LOCF fallback state."""

    config: ImputationConfig
    feature_names: tuple[str, ...]
    nonnegative_feature_names: tuple[str, ...]
    nonnegative_feature_indices: tuple[int, ...]
    imputer: IterativeImputer | None
    metadata: dict[str, object]
    training_medians: tuple[float, ...] = ()
    locf_feature_names: tuple[str, ...] = ()
    locf_feature_indices: tuple[int, ...] = ()

    @property
    def requires_landmark_context(self) -> bool:
        return self.config.enabled and self.config.method == LOCF_METHOD

    def transform(self, features: np.ndarray) -> FeatureImputationResult:
        validate_feature_width(features, self.feature_names)
        missing_mask = np.isnan(features)
        clipped_mask = np.zeros_like(missing_mask)
        if not self.config.enabled:
            return FeatureImputationResult(
                features=features,
                imputed_mask=np.zeros_like(missing_mask),
                clipped_mask=clipped_mask,
            )
        if self.config.method == LOCF_METHOD:
            raise ValueError("LOCF imputation requires patient ids and landmark days; use `transform_table`.")
        return transform_mice_features(
            features,
            self.imputer,
            missing_mask,
            self.nonnegative_feature_indices,
            "MICE imputation",
        )

    def transform_table(
        self,
        table: PatientLandmarkTable,
        *,
        reference_table: PatientLandmarkTable | None = None,
    ) -> FeatureImputationResult:
        validate_feature_width(table.features, table.feature_names)
        validate_feature_names(table.feature_names, self.feature_names)
        if not self.requires_landmark_context:
            return self.transform(table.features)
        reference = table if reference_table is None else reference_table
        validate_feature_width(reference.features, reference.feature_names)
        validate_feature_names(reference.feature_names, self.feature_names)
        locf_result = transform_locf_table(
            table,
            reference,
            self.locf_feature_indices,
            self.nonnegative_feature_indices,
            self.config.locf_max_age_days,
        )
        return apply_locf_fallback(self, locf_result)


def fit_feature_imputer(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    config: ImputationConfig,
    *,
    nonnegative_feature_names: tuple[str, ...] = (),
    locf_feature_names: tuple[str, ...] | None = None,
    landmark_table: PatientLandmarkTable | None = None,
) -> FittedFeatureImputer:
    validate_feature_width(features, feature_names)
    nonnegative_feature_indices = feature_indices(feature_names, nonnegative_feature_names)
    if not config.enabled:
        return FittedFeatureImputer(
            config,
            feature_names,
            nonnegative_feature_names,
            nonnegative_feature_indices,
            None,
            {"enabled": False},
        )
    locf_feature_names = nonnegative_feature_names if locf_feature_names is None else locf_feature_names
    locf_feature_indices = feature_indices(feature_names, locf_feature_names)
    if config.method == LOCF_METHOD:
        return fit_locf_feature_imputer(
            features,
            feature_names,
            config,
            nonnegative_feature_names,
            nonnegative_feature_indices,
            locf_feature_names,
            locf_feature_indices,
            landmark_table,
        )
    validate_fit_features(features, feature_names, "MICE imputation")
    imputer = fit_iterative_imputer(features, config)
    metadata = imputation_metadata(features, feature_names, config, nonnegative_feature_names, imputer)
    return FittedFeatureImputer(
        config, feature_names, nonnegative_feature_names, nonnegative_feature_indices, imputer, metadata
    )


def fit_locf_feature_imputer(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    config: ImputationConfig,
    nonnegative_feature_names: tuple[str, ...],
    nonnegative_feature_indices: tuple[int, ...],
    locf_feature_names: tuple[str, ...],
    locf_feature_indices: tuple[int, ...],
    landmark_table: PatientLandmarkTable | None,
) -> FittedFeatureImputer:
    if landmark_table is None:
        raise ValueError("LOCF imputation fitting requires the non-bootstrapped training landmark table.")
    validate_feature_width(landmark_table.features, landmark_table.feature_names)
    validate_feature_names(landmark_table.feature_names, feature_names)
    validate_locf_fit_features(features, feature_names, locf_feature_indices)
    locf_result = transform_locf_table(
        landmark_table,
        landmark_table,
        locf_feature_indices,
        nonnegative_feature_indices,
        config.locf_max_age_days,
    )
    if config.locf_fallback_method == MEDIAN_FALLBACK_METHOD:
        medians = training_feature_medians(locf_result.features, feature_names, "Median fallback")
        metadata = locf_imputation_metadata(
            features,
            locf_result.features,
            feature_names,
            config,
            nonnegative_feature_names,
            locf_feature_names,
            medians,
            None,
        )
        return FittedFeatureImputer(
            config,
            feature_names,
            nonnegative_feature_names,
            nonnegative_feature_indices,
            None,
            metadata,
            medians,
            locf_feature_names,
            locf_feature_indices,
        )
    validate_fit_features(locf_result.features, feature_names, "MICE fallback after LOCF")
    imputer = fit_iterative_imputer(locf_result.features, config)
    metadata = locf_imputation_metadata(
        features,
        locf_result.features,
        feature_names,
        config,
        nonnegative_feature_names,
        locf_feature_names,
        (),
        imputer,
    )
    return FittedFeatureImputer(
        config, feature_names, nonnegative_feature_names, nonnegative_feature_indices, imputer, metadata, (),
        locf_feature_names, locf_feature_indices
    )


def validate_feature_width(features: np.ndarray, feature_names: tuple[str, ...]) -> None:
    if features.ndim != 2:
        raise ValueError("Feature matrix must be two-dimensional.")
    if features.shape[1] != len(feature_names):
        raise ValueError(
            "Feature matrix column count must match feature names; "
            f"columns={features.shape[1]}, feature_names={len(feature_names)}."
        )


def validate_feature_names(actual: tuple[str, ...], expected: tuple[str, ...]) -> None:
    if actual != expected:
        raise ValueError("Landmark table feature names must match the fitted imputer feature names.")


def validate_fit_features(features: np.ndarray, feature_names: tuple[str, ...], label: str) -> None:
    validate_feature_matrix_for_fit(features, label)
    all_missing = tuple(
        name for name, count in missing_counts(features, feature_names).items() if count == len(features)
    )
    if all_missing:
        raise ValueError(f"{label} cannot fit features with all training values missing: {', '.join(all_missing)}.")


def validate_locf_fit_features(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    locf_feature_indices: tuple[int, ...],
) -> None:
    validate_feature_matrix_for_fit(features, "LOCF imputation")
    all_missing = tuple(
        feature_names[index]
        for index in locf_feature_indices
        if int(np.isnan(features[:, index]).sum()) == len(features)
    )
    if all_missing:
        joined = ", ".join(all_missing)
        raise ValueError(
            f"LOCF imputation cannot fit raw blood-test features with all training values missing: {joined}."
        )


def validate_feature_matrix_for_fit(features: np.ndarray, label: str) -> None:
    if features.shape[0] == 0:
        raise ValueError(f"{label} requires at least one training row.")
    if bool(np.isinf(features).any()):
        raise ValueError(f"{label} does not support infinite feature values.")


def validate_transformed_features(features: np.ndarray, label: str) -> None:
    if bool(np.isnan(features).any()):
        raise ValueError(f"{label} produced missing feature values.")
    if bool(np.isinf(features).any()):
        raise ValueError(f"{label} produced infinite feature values.")


def training_feature_medians(
    features: np.ndarray,
    feature_names: tuple[str, ...],
    label: str,
) -> tuple[float, ...]:
    validate_fit_features(features, feature_names, label)
    return tuple(float(value) for value in np.nanmedian(features, axis=0))


def fit_iterative_imputer(features: np.ndarray, config: ImputationConfig) -> IterativeImputer:
    imputer = IterativeImputer(max_iter=config.max_iter, tol=config.tol, random_state=config.random_state)
    imputer.fit(np.asarray(features, dtype=np.float64))
    return imputer


def transform_mice_features(
    features: np.ndarray,
    imputer: IterativeImputer | None,
    imputed_mask: np.ndarray,
    nonnegative_feature_indices: tuple[int, ...],
    label: str,
) -> FeatureImputationResult:
    if imputer is None:
        raise ValueError("Enabled MICE imputation is missing a fitted imputer.")
    transformed = imputer.transform(np.asarray(features, dtype=np.float64)).astype(np.float32)
    clipped_mask = clip_negative_imputed_features(transformed, imputed_mask, nonnegative_feature_indices)
    validate_transformed_features(transformed, label)
    return FeatureImputationResult(features=transformed, imputed_mask=imputed_mask, clipped_mask=clipped_mask)


def transform_locf_table(
    table: PatientLandmarkTable,
    reference_table: PatientLandmarkTable,
    locf_feature_indices: tuple[int, ...],
    nonnegative_feature_indices: tuple[int, ...],
    max_age_days: int | None,
) -> FeatureImputationResult:
    validate_table_context(table)
    validate_table_context(reference_table)
    transformed = np.asarray(table.features, dtype=np.float32).copy()
    missing_mask = np.isnan(table.features)
    carried_forward_mask = np.zeros_like(missing_mask)
    expired_mask = np.zeros_like(missing_mask)
    for column in locf_feature_indices:
        fill_locf_column(transformed, table, reference_table, missing_mask, carried_forward_mask, expired_mask, column,
                         max_age_days)
    clipped_mask = clip_negative_imputed_features(transformed, carried_forward_mask, nonnegative_feature_indices)
    if bool(np.isinf(transformed).any()):
        raise ValueError("LOCF imputation produced infinite feature values.")
    return FeatureImputationResult(
        features=transformed,
        imputed_mask=carried_forward_mask.copy(),
        clipped_mask=clipped_mask,
        carried_forward_mask=carried_forward_mask,
        locf_expired_mask=expired_mask,
    )


def validate_table_context(table: PatientLandmarkTable) -> None:
    row_count = table.features.shape[0]
    if len(table.patient_ids) != row_count:
        raise ValueError("Landmark table patient id count must match the feature row count.")
    if len(table.landmark_days) != row_count:
        raise ValueError("Landmark table landmark-day count must match the feature row count.")


def fill_locf_column(
    transformed: np.ndarray,
    table: PatientLandmarkTable,
    reference_table: PatientLandmarkTable,
    missing_mask: np.ndarray,
    carried_forward_mask: np.ndarray,
    expired_mask: np.ndarray,
    column: int,
    max_age_days: int | None,
) -> None:
    missing_rows = np.flatnonzero(missing_mask[:, column])
    if len(missing_rows) == 0:
        return
    history = locf_history_by_patient(reference_table, column)
    for row in missing_rows:
        patient_id = str(table.patient_ids[row])
        landmark_day = int(table.landmark_days[row])
        carried_value, expired = carried_forward_value(history.get(patient_id), landmark_day, max_age_days)
        if expired:
            expired_mask[row, column] = True
        if carried_value is not None:
            transformed[row, column] = carried_value
            carried_forward_mask[row, column] = True


def locf_history_by_patient(
    reference_table: PatientLandmarkTable,
    column: int,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    observed_rows = np.flatnonzero(~np.isnan(reference_table.features[:, column]))
    grouped: dict[str, list[tuple[int, float]]] = {}
    for row in observed_rows:
        patient_id = str(reference_table.patient_ids[row])
        grouped.setdefault(patient_id, []).append(
            (int(reference_table.landmark_days[row]), float(reference_table.features[row, column]))
        )
    return {patient_id: locf_patient_history(records) for patient_id, records in grouped.items()}


def locf_patient_history(records: list[tuple[int, float]]) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(records, key=lambda record: record[0])
    days = np.asarray([day for day, _value in ordered], dtype=np.int32)
    values = np.asarray([value for _day, value in ordered], dtype=np.float32)
    return days, values


def carried_forward_value(
    history: tuple[np.ndarray, np.ndarray] | None,
    landmark_day: int,
    max_age_days: int | None,
) -> tuple[float | None, bool]:
    if history is None:
        return None, False
    days, values = history
    prior_position = int(np.searchsorted(days, landmark_day, side="left")) - 1
    if prior_position < 0:
        return None, False
    age_days = int(landmark_day) - int(days[prior_position])
    if max_age_days is not None and age_days > max_age_days:
        return None, True
    return float(values[prior_position]), False


def apply_locf_fallback(
    feature_imputer: FittedFeatureImputer,
    locf_result: FeatureImputationResult,
) -> FeatureImputationResult:
    if feature_imputer.config.locf_fallback_method == MEDIAN_FALLBACK_METHOD:
        return apply_median_fallback(feature_imputer, locf_result)
    return apply_mice_fallback(feature_imputer, locf_result)


def apply_median_fallback(
    feature_imputer: FittedFeatureImputer,
    locf_result: FeatureImputationResult,
) -> FeatureImputationResult:
    transformed = locf_result.features.copy()
    fallback_mask = np.isnan(transformed)
    medians = np.asarray(feature_imputer.training_medians, dtype=np.float32)
    if len(medians) != transformed.shape[1]:
        raise ValueError("Median fallback is missing fitted training medians for every feature.")
    rows, columns = np.nonzero(fallback_mask)
    transformed[rows, columns] = medians[columns]
    clipped_mask = locf_result.clipped_mask.copy()
    clipped_mask |= clip_negative_imputed_features(
        transformed, fallback_mask, feature_imputer.nonnegative_feature_indices
    )
    validate_transformed_features(transformed, "LOCF median fallback")
    return locf_fallback_result(locf_result, transformed, clipped_mask, fallback_mask, MEDIAN_FALLBACK_METHOD)


def apply_mice_fallback(
    feature_imputer: FittedFeatureImputer,
    locf_result: FeatureImputationResult,
) -> FeatureImputationResult:
    fallback_mask = np.isnan(locf_result.features)
    if feature_imputer.imputer is None:
        raise ValueError("LOCF MICE fallback is missing a fitted imputer.")
    transformed = feature_imputer.imputer.transform(np.asarray(locf_result.features, dtype=np.float64)).astype(
        np.float32
    )
    clipped_mask = locf_result.clipped_mask.copy()
    clipped_mask |= clip_negative_imputed_features(
        transformed, fallback_mask, feature_imputer.nonnegative_feature_indices
    )
    validate_transformed_features(transformed, "LOCF MICE fallback")
    return locf_fallback_result(locf_result, transformed, clipped_mask, fallback_mask, MICE_METHOD)


def locf_fallback_result(
    locf_result: FeatureImputationResult,
    features: np.ndarray,
    clipped_mask: np.ndarray,
    fallback_mask: np.ndarray,
    fallback_method: str,
) -> FeatureImputationResult:
    median_mask = fallback_mask.copy() if fallback_method == MEDIAN_FALLBACK_METHOD else None
    mice_mask = fallback_mask.copy() if fallback_method == MICE_METHOD else None
    return FeatureImputationResult(
        features=features,
        imputed_mask=locf_result.imputed_mask | fallback_mask,
        clipped_mask=clipped_mask,
        carried_forward_mask=locf_result.carried_forward_mask,
        locf_expired_mask=locf_result.locf_expired_mask,
        fallback_mask=fallback_mask,
        median_fallback_mask=median_mask,
        mice_fallback_mask=mice_mask,
        fallback_method=fallback_method,
    )


def clip_negative_imputed_features(
    features: np.ndarray,
    imputed_mask: np.ndarray,
    nonnegative_feature_indices: tuple[int, ...],
) -> np.ndarray:
    clipped_mask = np.zeros_like(imputed_mask)
    if not nonnegative_feature_indices:
        return clipped_mask
    for column in nonnegative_feature_indices:
        column_clipped = imputed_mask[:, column] & (features[:, column] < 0.0)
        if bool(column_clipped.any()):
            features[column_clipped, column] = 0.0
            clipped_mask[column_clipped, column] = True
    return clipped_mask


def feature_indices(
    feature_names: tuple[str, ...],
    requested_feature_names: tuple[str, ...],
) -> tuple[int, ...]:
    name_to_index = {name: index for index, name in enumerate(feature_names)}
    missing = tuple(name for name in requested_feature_names if name not in name_to_index)
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Requested features are missing from the feature schema: {joined}.")
    return tuple(name_to_index[name] for name in requested_feature_names)


def transform_landmark_table(
    feature_imputer: object | None,
    table: PatientLandmarkTable,
    *,
    reference_table: PatientLandmarkTable | None = None,
) -> FeatureImputationResult:
    missing_mask = np.isnan(table.features)
    if feature_imputer is None:
        return FeatureImputationResult(
            features=table.features,
            imputed_mask=np.zeros_like(missing_mask),
            clipped_mask=np.zeros_like(missing_mask),
        )
    transform_table = getattr(feature_imputer, "transform_table", None)
    if callable(transform_table):
        return transform_table(table, reference_table=reference_table)
    transform = getattr(feature_imputer, "transform", None)
    if not callable(transform):
        raise TypeError("Feature imputer must provide a callable transform method.")
    return transform(table.features)
