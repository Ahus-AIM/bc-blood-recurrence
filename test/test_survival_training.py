from __future__ import annotations

import numpy as np

from src.data import PatientLandmarkTable
from src.script.survival_training import prepare_survival_training, training_not_estimable_reason
from src.script.train_single_bloodtest_benchmark import load_benchmark_config


def test_task_is_not_estimable_for_grouped_ebm_with_one_event_patient() -> None:
    raw_config, _config_dir = load_benchmark_config()
    table = PatientLandmarkTable(
        features=np.asarray([[1.0], [2.0], [3.0]]),
        signed_durations=np.asarray([10.0, -20.0, -30.0]),
        durations=np.asarray([10.0, 20.0, 30.0]),
        events=np.asarray([True, False, False]),
        sample_weights=np.ones(3),
        feature_names=("A|||u",),
        patient_ids=np.asarray(["event", "censored-1", "censored-2"]),
        landmark_days=np.asarray([1, 2, 3]),
        prior_non_endpoint_recurrence=np.zeros(3, dtype=bool),
        metadata={},
    )

    reason = training_not_estimable_reason(table, raw_config["ebm"])

    assert reason == "training split has fewer than two configured endpoint recurrence patients"


def test_prepared_training_preserves_shared_targets_features_and_weights() -> None:
    table = PatientLandmarkTable(
        features=np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        signed_durations=np.asarray([10.0, -20.0]),
        durations=np.asarray([10.0, 20.0]),
        events=np.asarray([True, False]),
        sample_weights=np.asarray([0.5, 1.5]),
        feature_names=("blood_a", "blood_b"),
        patient_ids=np.asarray(["event", "censored"]),
        landmark_days=np.asarray([1, 2]),
        prior_non_endpoint_recurrence=np.zeros(2, dtype=bool),
        metadata={},
    )

    prepared = prepare_survival_training(
        dataset=object(),  # type: ignore[arg-type]
        train_table=table,
        imputation_config={"enabled": False},
        nonnegative_feature_names=table.feature_names,
    )

    assert prepared.train_table is table
    assert prepared.train_imputation.features is table.features
    assert prepared.train_table.signed_durations is table.signed_durations
    assert prepared.train_table.sample_weights is table.sample_weights
