from __future__ import annotations

from dataclasses import replace
from typing import cast

import numpy as np
import pytest

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.dataset_config import PatientDatasetConfig
from src.model import survival_evaluation
from src.model.imputation import (
    FeatureImputationResult,
    FittedFeatureImputer,
    fit_feature_imputer,
    transform_landmark_table,
)
from src.model.imputation_config import ImputationConfig
from src.model.survival_evaluation import (
    PreparedSplitEvaluation,
    SharedEvaluationFeatureCache,
    SurvivalEvaluationCache,
    evaluate_prepared_split,
    prepare_evaluation_table,
)


class FirstFeatureRiskModel:
    """Minimal test model returning the first feature as recurrence risk."""

    def predict(self, features: np.ndarray) -> np.ndarray:
        return features[:, 0]


def test_prepared_evaluation_discards_zero_event_bootstraps_without_replacement() -> None:
    event_table = reporting_landmark_table()
    zero_event_table = replace(
        event_table,
        events=np.zeros(2, dtype=bool),
        signed_durations=-event_table.durations.astype(np.float64),
    )
    imputer = fitted_imputer(event_table)
    prepared = PreparedSplitEvaluation(
        split="test",
        runs=(
            prepare_evaluation_table(zero_event_table, imputer, 0),
            prepare_evaluation_table(event_table, imputer, 1),
        ),
        bootstrap_sample_count=2,
    )

    summary = evaluate_prepared_split(FirstFeatureRiskModel(), prepared, None)

    assert summary["bootstrap_runs"] == 1
    assert summary["bootstrap_runs_attempted"] == 2
    assert summary["zero_event_bootstrap_runs_discarded"] == 1
    assert [run["bootstrap_run"] for run in summary["runs"]] == [1]


def test_direct_evaluation_discards_zero_event_bootstraps_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    event_table = reporting_landmark_table()
    zero_event_table = replace(
        event_table,
        events=np.zeros(2, dtype=bool),
        signed_durations=-event_table.durations.astype(np.float64),
    )

    def build_table(
        _config: PatientDatasetConfig,
        _dataset: PatientBloodTestDataset,
        *,
        bootstrap: bool,
        bootstrap_run: int = 0,
    ) -> PatientLandmarkTable:
        assert bootstrap
        return zero_event_table if bootstrap_run == 0 else event_table

    monkeypatch.setattr(survival_evaluation, "landmark_table", build_table)
    summary = survival_evaluation.evaluate_split(
        FirstFeatureRiskModel(),
        "test",
        bootstrap_config(run_count=2),
        cast(PatientBloodTestDataset, object()),
        fitted_imputer(event_table),
        None,
    )

    assert summary["bootstrap_runs"] == 1
    assert summary["bootstrap_runs_attempted"] == 2
    assert summary["zero_event_bootstrap_runs_discarded"] == 1
    assert [run["bootstrap_run"] for run in summary["runs"]] == [1]


def test_evaluation_reuses_imputed_features_across_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    first_table = reporting_landmark_table()
    second_table = replace(
        first_table,
        durations=np.asarray([20, 30], dtype=np.int32),
        signed_durations=np.asarray([20.0, -30.0], dtype=np.float64),
    )
    first_config = replace(bootstrap_config(run_count=1), censoring_horizon_days=10)
    second_config = replace(first_config, censoring_horizon_days=30)
    imputer = fitted_imputer(first_table)
    transform_calls = 0

    class Landmarks:
        def table(self, config: PatientDatasetConfig, bootstrap_run: int | None = None) -> PatientLandmarkTable:
            assert bootstrap_run in (None, 0)
            return first_table if config.censoring_horizon_days == 10 else second_table

        def bootstrap_indices(self, _config: PatientDatasetConfig, _bootstrap_run: int) -> np.ndarray:
            return np.arange(len(first_table.durations))

    def transform(
        feature_imputer: object | None,
        table: PatientLandmarkTable,
        *,
        reference_table: PatientLandmarkTable | None = None,
    ) -> FeatureImputationResult:
        nonlocal transform_calls
        transform_calls += 1
        return transform_landmark_table(feature_imputer, table, reference_table=reference_table)

    monkeypatch.setattr(survival_evaluation, "transform_landmark_table", transform)
    shared_features = SharedEvaluationFeatureCache()
    landmarks = cast(BenchmarkLandmarks, Landmarks())
    first = SurvivalEvaluationCache(first_config, imputer, landmarks, shared_features).get("test")
    second = SurvivalEvaluationCache(second_config, imputer, landmarks, shared_features).get("test")

    assert transform_calls == 1
    assert second.runs[0].imputation is first.runs[0].imputation
    assert second.runs[0].table.durations.tolist() == [20, 30]


def fitted_imputer(table: PatientLandmarkTable) -> FittedFeatureImputer:
    return fit_feature_imputer(
        table.features,
        table.feature_names,
        ImputationConfig.from_mapping({"enabled": False}),
    )


def bootstrap_config(*, run_count: int) -> PatientDatasetConfig:
    return PatientDatasetConfig.from_mapping(
        {
            "bloodtest_csv_path": "blood.csv",
            "endpoint_csv_path": "endpoint.csv",
            "split_csv_path": "split.csv",
            "selected_blood_tests": ["A|||u", "B|||u"],
            "test_bootstrap_runs": run_count,
            "test_bootstrap_samples": 2,
        }
    )


def reporting_landmark_table() -> PatientLandmarkTable:
    durations = np.asarray([10, 20], dtype=np.int32)
    events = np.asarray([True, False], dtype=bool)
    return PatientLandmarkTable(
        features=np.asarray([[1.5, np.nan], [2.5, 1.0]], dtype=np.float32),
        signed_durations=np.where(events, durations, -durations).astype(np.float64),
        durations=durations,
        events=events,
        sample_weights=np.asarray([2.0, 3.0], dtype=np.float64),
        feature_names=("A|||u", "B|||u"),
        patient_ids=np.asarray(["P0", "P1"], dtype=object),
        landmark_days=np.asarray([3, 4], dtype=np.int32),
        prior_non_endpoint_recurrence=np.asarray([False, True], dtype=bool),
        metadata={},
    )
