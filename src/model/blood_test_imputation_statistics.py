from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

import numpy as np

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.blood_test_features import BLOOD_TEST_CHANGE_SUFFIX
from src.data.measurement_selection import selected_cohort_measurements
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    feature_indices,
    transform_landmark_table,
)
from src.model.imputation_config import MICE_METHOD

TEMPORAL_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class BloodTestValueSourceCounts:
    """Reported blood-test values from measurement, LOCF, or MICE.

    `measured` includes retained earlier history and observed landmark values;
    `locf` counts filled current values. `mice` counts filled current values
    and each filled past input used for a change, once per landmark and test.
    Observed past values are not counted again when reused for a change.
    """

    measured: int = 0
    locf: int = 0
    mice: int = 0

    @property
    def total(self) -> int:
        return self.measured + self.locf + self.mice

    def as_dict(self) -> dict[str, int]:
        return {
            "measured": self.measured,
            "locf": self.locf,
            "mice": self.mice,
        }

    def __add__(
        self,
        other: BloodTestValueSourceCounts,
    ) -> BloodTestValueSourceCounts:
        return BloodTestValueSourceCounts(
            measured=self.measured + other.measured,
            locf=self.locf + other.locf,
            mice=self.mice + other.mice,
        )


@dataclass(frozen=True)
class BloodTestImputationStatistics:
    """Value-source counts by Selected Blood-Test Type and Temporal Split.

    `by_test_key` combines train, validation, and test for each selected key;
    `by_split` combines all selected keys within a split; `overall` combines
    every reported value, including retained earlier measurements.
    """

    by_test_key: dict[str, BloodTestValueSourceCounts]
    by_split: dict[str, BloodTestValueSourceCounts]
    overall: BloodTestValueSourceCounts


def build_blood_test_imputation_statistics(
    dataset: PatientBloodTestDataset,
    imputer: FittedFeatureImputer,
) -> BloodTestImputationStatistics:
    """Count retained measurements and completed current/past inputs without bootstrapping."""
    split_tables = {}
    imputation_results = {}
    for split in TEMPORAL_SPLITS:
        table = PatientBloodTestDataset(
            replace(dataset.config, split=split),
            prepared_data=dataset.prepared_data,
        ).to_landmark_table(bootstrap=False)
        split_tables[split] = table
        if len(table.signed_durations):
            imputation_results[split] = transform_landmark_table(imputer, table)
    return build_blood_test_imputation_statistics_from_tables(
        dataset,
        imputer,
        split_tables,
        imputation_results,
    )


def build_blood_test_imputation_statistics_from_tables(
    dataset: PatientBloodTestDataset,
    imputer: FittedFeatureImputer,
    split_tables: Mapping[str, PatientLandmarkTable],
    imputation_results: Mapping[str, FeatureImputationResult],
) -> BloodTestImputationStatistics:
    """Combine retained history with sources from non-bootstrapped split imputations."""
    by_test_key = {test_key: BloodTestValueSourceCounts() for test_key in dataset.selected_test_keys}
    by_split = {}
    for split in TEMPORAL_SPLITS:
        table = split_tables[split]
        if len(table.signed_durations) == 0:
            by_split[split] = BloodTestValueSourceCounts()
            continue
        result = imputation_results[split]
        split_counts = BloodTestValueSourceCounts()
        indices = feature_indices(table.feature_names, dataset.selected_test_keys)
        columns = {name: index for index, name in enumerate(table.feature_names)}
        for test_key, feature_index in zip(
            dataset.selected_test_keys,
            indices,
            strict=True,
        ):
            counts = feature_source_counts(
                table.features[:, feature_index],
                result,
                feature_index,
                imputer,
            )
            change_column = columns.get(f"{test_key}{BLOOD_TEST_CHANGE_SUFFIX}")
            if change_column is not None:
                counts += BloodTestValueSourceCounts(mice=mask_column_count(result.past_mice_mask, change_column))
            by_test_key[test_key] = by_test_key[test_key] + counts
            split_counts += counts
        by_split[split] = split_counts
    overall = sum(by_split.values(), start=BloodTestValueSourceCounts())
    statistics = BloodTestImputationStatistics(by_test_key, by_split, overall)
    return include_earlier_measurement_counts(dataset, statistics)


def include_earlier_measurement_counts(
    dataset: PatientBloodTestDataset,
    statistics: BloodTestImputationStatistics,
) -> BloodTestImputationStatistics:
    """Add observed history once to report counts, without altering fitted imputations."""
    if not dataset.config.include_blood_test_change:
        return statistics
    measurements = selected_cohort_measurements(dataset.prepared_data)
    earlier = measurements.loc[measurements["days_since_diagnosis"] < dataset.config.min_landmark_day]
    by_test = earlier["test_key"].value_counts()
    split_by_patient = {
        item.patient_id: split
        for split in TEMPORAL_SPLITS
        for item in dataset.prepared_data.items_by_split.get(split, ())
    }
    by_split = earlier["patient_id"].map(split_by_patient).value_counts()
    return BloodTestImputationStatistics(
        by_test_key={
            key: counts + BloodTestValueSourceCounts(measured=int(by_test.get(key, 0)))
            for key, counts in statistics.by_test_key.items()
        },
        by_split={
            split: counts + BloodTestValueSourceCounts(measured=int(by_split.get(split, 0)))
            for split, counts in statistics.by_split.items()
        },
        overall=statistics.overall + BloodTestValueSourceCounts(measured=len(earlier)),
    )


def feature_source_counts(
    raw_values: np.ndarray,
    result: FeatureImputationResult,
    feature_index: int,
    imputer: FittedFeatureImputer,
) -> BloodTestValueSourceCounts:
    measured_count = int((~np.isnan(raw_values)).sum())
    locf_count = mask_column_count(result.carried_forward_mask, feature_index)
    mice_mask = (
        result.imputed_mask
        if imputer.config.enabled and imputer.config.method == MICE_METHOD
        else result.mice_fallback_mask
    )
    return BloodTestValueSourceCounts(
        measured=measured_count,
        locf=locf_count,
        mice=mask_column_count(mice_mask, feature_index),
    )


def mask_column_count(mask: np.ndarray | None, feature_index: int) -> int:
    return 0 if mask is None else int(mask[:, feature_index].sum())
