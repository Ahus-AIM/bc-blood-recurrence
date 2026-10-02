from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PatientLandmarkTable
from src.data.timing import timed
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    transform_landmark_table,
)
from src.model.imputation_metadata import IMPUTATION_COUNT_KEYS, imputation_result_counts
from src.model.survival_evaluation import require_matching_landmark_feature_rows, table_summary

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelInputArtifact:
    """Model-input rows built once for one task.

    ``frame`` contains the complete train, validation, and test rows. ``metadata``
    describes their feature, imputation, and split counts and is reused while
    ``frame`` is written independently to each requested output directory.
    """

    frame: pd.DataFrame
    metadata: dict[str, object]


@dataclass(frozen=True)
class PreparedModelInputFeatures:
    """Current target tables paired with shared non-bootstrapped imputations.

    ``tables`` retain task-specific targets and metadata. The corresponding
    ``imputation_results`` contain target-independent transformed features.
    """

    tables: dict[str, PatientLandmarkTable]
    imputation_results: dict[str, FeatureImputationResult]


@dataclass
class SharedModelInputFeatureCache:
    """Reuse train, validation, and test feature imputations across compatible tasks.

    The first call stores raw feature rows and transformations. Later calls use
    the same fitted imputer and canonical patient-landmark features while
    retaining their own target columns.
    """

    _prepared: PreparedModelInputFeatures | None = field(default=None, init=False, repr=False)
    _imputer: FittedFeatureImputer | None = field(default=None, init=False, repr=False)

    @property
    def prepared(self) -> PreparedModelInputFeatures | None:
        """Return the first task's cached feature tables and transformations."""
        return self._prepared

    @property
    def imputer(self) -> FittedFeatureImputer | None:
        """Return the fitted imputer that produced the cached full-split features."""
        return self._imputer

    def prepare(
        self,
        split_tables: Mapping[str, PatientLandmarkTable],
        imputer: FittedFeatureImputer | None,
        *,
        train_imputation: FeatureImputationResult | None = None,
    ) -> PreparedModelInputFeatures:
        """Transform the first task or attach later targets to cached features."""
        if self._prepared is None:
            results = {
                split: (
                    train_imputation
                    if split == "train" and train_imputation is not None
                    else transform_landmark_table(imputer if len(table.patient_ids) else None, table)
                )
                for split, table in split_tables.items()
            }
            self._prepared = PreparedModelInputFeatures(dict(split_tables), results)
            self._imputer = imputer
            return self._prepared
        if imputer is not self._imputer:
            raise ValueError("Shared model-input features require the same fitted imputer.")
        if set(split_tables) != set(self._prepared.tables):
            raise ValueError("Shared model-input Temporal Splits differ between horizons.")
        current_tables = {}
        for split, table in split_tables.items():
            shared_table = self._prepared.tables[split]
            require_matching_landmark_feature_rows(table, shared_table)
            current_tables[split] = replace(table, features=shared_table.features)
        return PreparedModelInputFeatures(
            current_tables,
            self._prepared.imputation_results,
        )


def build_model_input_artifact(
    split_tables: Mapping[str, PatientLandmarkTable],
    imputer: FittedFeatureImputer | None,
    *,
    feature_cache: SharedModelInputFeatureCache | None = None,
    train_imputation: FeatureImputationResult | None = None,
) -> ModelInputArtifact:
    """Build every non-bootstrapped split once for reuse across model folders."""
    train_table = split_tables["train"]
    prepared_features = (
        SharedModelInputFeatureCache().prepare(
            split_tables,
            imputer,
            train_imputation=train_imputation,
        )
        if feature_cache is None
        else feature_cache.prepare(
            split_tables,
            imputer,
            train_imputation=train_imputation,
        )
    )
    split_tables = prepared_features.tables
    imputation_results = prepared_features.imputation_results
    frame = pd.concat(
        [model_input_frame(split, table, imputation_results[split]) for split, table in split_tables.items()],
        ignore_index=True,
    )
    return ModelInputArtifact(
        frame,
        model_input_metadata(train_table, split_tables, imputation_results),
    )


@timed(LOGGER, "write model input CSV")
def write_model_input_artifact(
    artifact: ModelInputArtifact,
    output_path: Path,
) -> dict[str, object]:
    artifact.frame.to_csv(output_path, index=False)
    return {"path": output_path.name, **artifact.metadata}


def model_input_metadata(
    train_table: PatientLandmarkTable,
    split_tables: Mapping[str, PatientLandmarkTable],
    results: Mapping[str, FeatureImputationResult],
) -> dict[str, object]:
    counts = {split: imputation_result_counts(result) for split, result in results.items()}
    return {
        "row_count": int(sum(len(table.signed_durations) for table in split_tables.values())),
        "feature_count": int(len(train_table.feature_names)),
        **{
            f"{key}_by_split": {split: values[key] for split, values in counts.items()} for key in IMPUTATION_COUNT_KEYS
        },
        "splits": {split: table_summary(table) for split, table in split_tables.items()},
    }


def model_input_frame(
    split: str,
    table: PatientLandmarkTable,
    imputation_result: FeatureImputationResult | None = None,
) -> pd.DataFrame:
    row_count = len(table.signed_durations)
    sample_weights = table.sample_weights if split == "train" else None
    weight_interval_days = table.sample_weight_interval_days if split == "train" else None
    base_frame = pd.DataFrame(
        {
            "split": np.full(row_count, split, dtype=object),
            "patient_id": table.patient_ids,
            "landmark_day": table.landmark_days,
            "duration_days": table.durations,
            "event_observed": table.events,
            "right_censoring_day": model_input_optional_values(
                table.right_censoring_days,
                row_count,
            ),
            "right_censoring_reason": model_input_optional_values(
                table.right_censoring_reasons,
                row_count,
            ),
            "target_end_reason": model_input_optional_values(
                table.target_end_reasons,
                row_count,
            ),
            "signed_duration_days": table.signed_durations,
            "prior_non_endpoint_recurrence": table.prior_non_endpoint_recurrence,
            "sample_weight": model_input_optional_values(sample_weights, row_count),
            "sample_weight_interval_days": model_input_optional_values(weight_interval_days, row_count),
        }
    )
    return pd.concat(
        [base_frame, model_input_feature_frame(table, imputation_result)],
        axis=1,
    )


def model_input_feature_frame(
    table: PatientLandmarkTable,
    imputation_result: FeatureImputationResult | None,
) -> pd.DataFrame:
    if imputation_result is None:
        return pd.DataFrame(table.features, columns=table.feature_names)
    values = imputation_result.features.astype(object)
    imputed_rows, imputed_columns = np.nonzero(imputation_result.imputed_mask)
    for row, column in zip(imputed_rows, imputed_columns, strict=True):
        value = float(imputation_result.features[row, column])
        suffix = imputed_value_suffix(imputation_result, row, column)
        values[row, column] = f"{value:.10g} {suffix}"
    return pd.DataFrame(values, columns=table.feature_names)


def imputed_value_suffix(
    result: FeatureImputationResult,
    row: int,
    column: int,
) -> str:
    source = imputation_source_label(result, row, column)
    return f"{source} clipped" if bool(result.clipped_mask[row, column]) else source


def imputation_source_label(
    result: FeatureImputationResult,
    row: int,
    column: int,
) -> str:
    if mask_value(result.derived_mask, row, column):
        return "derived from imputed values"
    if mask_value(result.carried_forward_mask, row, column):
        return "locf"
    if mask_value(result.median_fallback_mask, row, column):
        return "fallback median"
    if mask_value(result.mice_fallback_mask, row, column):
        return "fallback mice"
    return "mice"


def mask_value(mask: np.ndarray | None, row: int, column: int) -> bool:
    return False if mask is None else bool(mask[row, column])


def model_input_optional_values(
    values: np.ndarray | None,
    row_count: int,
) -> np.ndarray:
    if values is None:
        return np.full(row_count, np.nan, dtype=np.float64)
    return values
