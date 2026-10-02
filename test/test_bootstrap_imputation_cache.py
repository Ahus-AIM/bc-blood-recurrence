from __future__ import annotations

import pickle
from dataclasses import fields, replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_washout_dataset
from src.model import survival_evaluation
from src.model.imputation import FeatureImputationResult, transform_landmark_table
from src.model.survival_evaluation import (
    SurvivalEvaluationCache,
    evaluate_prepared_split,
    evaluate_prepared_table,
    prepare_evaluation_table,
)
from src.script import single_bloodtest_washout
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.single_bloodtest_washout import evaluate_washout_cell, washout_landmarks
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache
from src.script.survival_training import prepare_survival_training
from test.test_patient_bloodtest import blood_row, config, endpoint_row


class RiskModel:
    """Deterministic risk scores sensitive to current values and derived changes."""

    feature_names_in_ = ("A|||u", "A|||u change")

    def predict(self, features: np.ndarray) -> np.ndarray:
        return np.nan_to_num(features) @ np.arange(1, features.shape[1] + 1)


def imputation_dataset(tmp_path: Path, weighting: str = "person_uniform") -> PatientBloodTestDataset:
    """Supply repeated landmarks with observed, carried, expired and missing values."""
    splits = {f"{split}{index}": split for split in ("train", "val", "test") for index in range(3)}
    measurements = [
        blood_row(patient, day, test_name=test, result=2 + index * 1.3 + day / (17 + column), unit="u")
        for index, patient in enumerate(splits)
        for day in (5, 15, 35, 55, 75)
        for column, test in enumerate(("A", "B"))
        if day <= 15 or (day // 5 + index + column) % 3 != 0
    ]
    endpoints = [
        endpoint_row(
            patient,
            diagnosis_to_ddfs=90 if index % 3 < 2 else "",
            ddfs_code=("C787", "C780", "")[index % 3],
            last_entry_days=150,
        )
        for index, patient in enumerate(splits)
    ]
    settings = config(tmp_path, measurements, endpoints, split_by_patient=splits)
    settings.update(
        selected_blood_tests=["A|||u", "B|||u"],
        min_blood_test_types_per_day=1,
        min_landmark_day=31,
        include_blood_test_change=True,
        blood_test_change_min_days=7,
        blood_test_change_max_days=60,
        include_time_since_diagnosis=True,
        bootstrap_sample_weighting=weighting,
        test_bootstrap_runs=4,
        test_bootstrap_samples=12,
    )
    return PatientBloodTestDataset(settings)


def assert_same_imputation(actual: FeatureImputationResult | None, expected: FeatureImputationResult | None) -> None:
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    for field in fields(expected):
        value = getattr(expected, field.name)
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(getattr(actual, field.name), value)
        else:
            assert getattr(actual, field.name) == value


@pytest.mark.parametrize(
    "method,fallback", [(None, "median"), ("mice", "median"), ("locf", "median"), ("locf", "mice")]
)
@pytest.mark.parametrize("weighting", ["person_uniform", "person_time_weighted", "landmark_uniform"])
@pytest.mark.parametrize("washout_days", [0, 30])
def test_pool_imputation_matches_each_draw_without_repeating_transforms(
    tmp_path: Path,
    method: str | None,
    fallback: str,
    weighting: str,
    washout_days: int,
) -> None:
    dataset = imputation_dataset(tmp_path, weighting)
    trained = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {
            "enabled": method is not None,
            "method": "mice" if method is None else method,
            "locf_fallback_method": fallback,
            "locf_max_age_days": 20,
            "max_iter": 10,
        },
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    landmarks = BenchmarkLandmarks(benchmark_washout_dataset(dataset, washout_days))
    target = replace(dataset.config, split="test", endpoint_ddfs_code_filter_regex="C787", censoring_horizon_days=100)
    reference = landmarks.table(target)
    direct = tuple(
        prepare_evaluation_table(landmarks.table(target, run), trained.imputer, run, reference_table=reference)
        for run in range(target.test_bootstrap_runs)
    )
    with patch.object(survival_evaluation, "transform_landmark_table", wraps=transform_landmark_table) as transform:
        cached = SurvivalEvaluationCache(target, trained.imputer, landmarks).get("test")

    for actual, expected in zip(cached.runs, direct, strict=True):
        assert_same_imputation(actual.imputation, expected.imputation)
        for field in ("patient_ids", "landmark_days", "durations", "events", "bootstrap_selection_probabilities"):
            np.testing.assert_array_equal(getattr(actual.table, field), getattr(expected.table, field))
        actual_summary, actual_scores = evaluate_prepared_table(
            RiskModel(), actual, feature_names=RiskModel.feature_names_in_
        )
        expected_summary, expected_scores = evaluate_prepared_table(
            RiskModel(), expected, feature_names=RiskModel.feature_names_in_
        )
        pd.testing.assert_frame_equal(actual_scores, expected_scores)
        pd.testing.assert_series_equal(pd.Series(actual_summary), pd.Series(expected_summary))
    assert transform.call_count == 1


def test_washout_reuses_prepared_draws_across_models_and_keeps_target_labels(tmp_path: Path) -> None:
    dataset = imputation_dataset(tmp_path)
    imputation_config = {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 10}
    trained = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        imputation_config,
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    write_training_config_snapshot({"imputation": imputation_config}, dataset.config, tmp_path, tmp_path)
    pools = washout_landmarks(dataset, (0, 30))
    source = {
        "model_key": "A|||u",
        "site_label": "Liver",
        "horizon_label": "100d",
        "horizon_days": 100,
        "endpoint_ddfs_code_filter_regex": "C787",
        "split": "test",
        "status": "complete",
        "output_dir": str(tmp_path),
    }
    with patch.object(single_bloodtest_washout, "load_model", return_value=RiskModel()):
        with patch.object(single_bloodtest_washout, "load_feature_imputer", return_value=trained.imputer):
            with patch.object(
                single_bloodtest_washout, "evaluate_prepared_split", wraps=evaluate_prepared_split
            ) as evaluate:
                first = evaluate_washout_cell(source, pools)
                second = evaluate_washout_cell({**source, "model_key": "another model"}, pools)
                changed = evaluate_washout_cell({**source, "endpoint_ddfs_code_filter_regex": "C780"}, pools)

    assert all(result["status"] == "complete" for result in (*first, *second, *changed))
    assert [result["runs"] for result in first] == [result["runs"] for result in second]
    for index in range(2):
        assert evaluate.call_args_list[index].args[1] is evaluate.call_args_list[index + 2].args[1]
        assert evaluate.call_args_list[index].args[1] is not evaluate.call_args_list[index + 4].args[1]


def test_model_input_features_and_zero_washout_share_the_training_evaluation_cache(tmp_path: Path) -> None:
    dataset = imputation_dataset(tmp_path)
    features = SurvivalBenchmarkFeatureCache(BenchmarkLandmarks(dataset))
    trained = features.prepare_training(
        dataset,
        features.landmarks.table(dataset.config),
        {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 10},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    assert trained is not None
    features.model_input_artifact(dataset.config, trained)
    cached = features.evaluation_cache(dataset.config, trained.imputer)
    pools = washout_landmarks(dataset, (0, 30), zero_cache=features)
    assert pools[0] is features
    assert pools[30] is not features
    saved_imputer = pickle.loads(pickle.dumps(trained.imputer))
    with patch.object(survival_evaluation, "transform_landmark_table", wraps=transform_landmark_table) as transform:
        main_test = cached.get("test")
        washout_test = pools[0].evaluation_cache(replace(dataset.config, split="test"), saved_imputer).get("test")
        cached.get("train")
        cached.get("val")
    assert transform.call_count == 0
    assert main_test is washout_test

    # Even the same settings/schema cannot share values from differently fitted estimators.
    different_imputer = pickle.loads(pickle.dumps(trained.imputer))
    different_imputer.imputer.imputation_sequence_[0].estimator.coef_[0] += 1
    different = pools[0].evaluation_cache(dataset.config, different_imputer)
    assert different is not cached
    with patch.object(survival_evaluation, "transform_landmark_table", wraps=transform_landmark_table) as transform:
        different.get("test")
    assert transform.call_count == 1


def test_empty_washout_draws_skip_imputation_and_retain_zero_event_diagnostics(tmp_path: Path) -> None:
    dataset = imputation_dataset(tmp_path)
    trained = prepare_survival_training(dataset, dataset.to_landmark_table(bootstrap=False), {"enabled": False})
    landmarks = BenchmarkLandmarks(benchmark_washout_dataset(dataset, 1000))
    with patch.object(survival_evaluation, "transform_landmark_table", wraps=transform_landmark_table) as transform:
        prepared = SurvivalEvaluationCache(dataset.config, trained.imputer, landmarks).get("test")
    assert transform.call_count == 0
    assert all(run.imputation is None for run in prepared.runs)
    summary = evaluate_prepared_split(RiskModel(), prepared, None, feature_names=RiskModel.feature_names_in_)
    assert summary["bootstrap_runs_attempted"] == dataset.config.test_bootstrap_runs
    assert summary["zero_event_bootstrap_runs_discarded"] == dataset.config.test_bootstrap_runs
    assert summary["bootstrap_runs"] == 0
