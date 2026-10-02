from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_washout_dataset
from src.model.survival_evaluation import evaluate_prepared_split, evaluate_split
from src.script.brier_bootstrap import prepare_brier_washouts
from src.script.single_bloodtest_washout import washout_landmarks
from src.script.survival_training import prepare_survival_training
from test.test_bootstrap_imputation_cache import RiskModel, imputation_dataset


@pytest.mark.parametrize(("washout_days", "patient_count"), [(0, 3), (30, 3), (60, 1), (120, 0)])
@pytest.mark.parametrize("saved_sample_count", [1, 1372])
def test_test_draws_use_the_remaining_patient_count(
    tmp_path: Path, washout_days: int, patient_count: int, saved_sample_count: int,
) -> None:
    original = imputation_dataset(tmp_path)
    config = replace(original.config, test_bootstrap_samples=saved_sample_count, censoring_horizon_days=180)
    dataset = PatientBloodTestDataset(config, prepared_data=original.prepared_data)
    pools = washout_landmarks(dataset, (washout_days,))
    pool = pools[washout_days]
    target = replace(config, split="test")
    eligible = pool.landmarks.table(target)
    assert len(np.unique(eligible.patient_ids)) == patient_count
    brier = prepare_brier_washouts(pools, target, tmp_path / "draws")[washout_days]

    for run in range(config.test_bootstrap_runs):
        draw = pool.landmarks.table(target, run)
        assert len(draw.patient_ids) == patient_count
        assert set(draw.patient_ids) <= set(eligible.patient_ids)
        np.testing.assert_array_equal(draw.patient_ids, eligible.patient_ids[brier.draws[run]])
        np.testing.assert_array_equal(draw.landmark_days, eligible.landmark_days[brier.draws[run]])
        liver = pool.landmarks.table(replace(target, endpoint_ddfs_code_filter_regex="C787"), run)
        np.testing.assert_array_equal(draw.patient_ids, liver.patient_ids)
        np.testing.assert_array_equal(draw.landmark_days, liver.landmark_days)
        for patient, probability in zip(draw.patient_ids, draw.bootstrap_selection_probabilities, strict=True):
            assert probability == pytest.approx(1 / patient_count / np.sum(eligible.patient_ids == patient))

    trained = prepare_survival_training(dataset, dataset.to_landmark_table(bootstrap=False), {"enabled": False})
    cached = pool.evaluation_cache(target, trained.imputer).get("test")
    cached_summary = evaluate_prepared_split(RiskModel(), cached, None)
    direct_summary = evaluate_split(
        RiskModel(), "test", target, pool.landmarks.dataset, trained.imputer, None,
    )
    for summary in (cached_summary, direct_summary):
        assert summary["bootstrap_sample_count"] == patient_count
        assert summary["bootstrap_runs_attempted"] == config.test_bootstrap_runs
    np.testing.assert_equal(cached_summary["harrell_c_index"], direct_summary["harrell_c_index"])


@pytest.mark.parametrize("split", ["train", "val"])
def test_training_and_validation_keep_configured_draw_sizes(tmp_path: Path, split: str) -> None:
    original = imputation_dataset(tmp_path)
    config = replace(original.config, train_bootstrap_samples=7, validation_bootstrap_samples=11)
    dataset = PatientBloodTestDataset(config, prepared_data=original.prepared_data)
    pool = BenchmarkLandmarks(benchmark_washout_dataset(dataset, 60))

    draw = pool.table(replace(config, split=split), 0)

    assert len(draw.patient_ids) == (7 if split == "train" else 11)


def test_test_sampling_still_uses_replacement(tmp_path: Path) -> None:
    dataset = imputation_dataset(tmp_path)
    pool = BenchmarkLandmarks(dataset)
    target = replace(dataset.config, split="test")
    draws = [pool.table(target, run) for run in range(20)]

    assert all(len(draw.patient_ids) == 3 for draw in draws)
    assert any(len(np.unique(draw.patient_ids)) < 3 for draw in draws)


@pytest.mark.parametrize(("weighting", "draw_count"), [("person_time_weighted", 3), ("landmark_uniform", 12)])
def test_optional_sampling_modes_use_their_respective_sizes(
    tmp_path: Path, weighting: str, draw_count: int,
) -> None:
    dataset = imputation_dataset(tmp_path, weighting)
    pool = BenchmarkLandmarks(dataset)

    draw = pool.table(replace(dataset.config, split="test"), 0)

    assert len(draw.patient_ids) == draw_count
