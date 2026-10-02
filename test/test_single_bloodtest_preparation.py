from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest

from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.dataset_config import PatientDatasetConfig
from src.data.types import PatientDataSources
from src.model.ebm_bags import PatientGroupedBagPlan
from src.script import train_single_bloodtest_benchmark
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache
from src.script.survival_training import (
    feature_subset_prepared_training,
    prepare_survival_training,
)
from src.script.train_single_bloodtest_benchmark import (
    SingleBloodtestTask,
)


def test_feature_subsets_reuse_one_ebm_bag_plan() -> None:
    prepared = prepare_survival_training(
        cast(PatientBloodTestDataset, object()),
        landmark_table(),
        {"enabled": False},
    )
    bag_plan = PatientGroupedBagPlan(
        bags=np.asarray([[1], [-1]], dtype=np.int8),
        metadata={"outer_bags": 1},
    )
    prepared = replace(prepared, patient_grouped_bag_plan=bag_plan)

    first = feature_subset_prepared_training(prepared, ("A|||u",))
    second = feature_subset_prepared_training(prepared, ("B|||u",))

    assert first.patient_grouped_bag_plan is bag_plan
    assert second.patient_grouped_bag_plan is bag_plan


def test_tasks_share_training_features_across_endpoint_sites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_config: dict[str, dict[str, object]] = {
        "benchmark": {
            "sites": [
                {"label": "All", "endpoint_ddfs_code_filter_regex": ".*"},
                {"label": "Liver", "endpoint_ddfs_code_filter_regex": r"C787\d*"},
            ],
            "horizons": [{"label": "1y", "days": 365}],
        }
    }
    caches: list[SurvivalBenchmarkFeatureCache] = []
    canonical = cast(
        PatientBloodTestDataset,
        SimpleNamespace(
            config=PatientDatasetConfig.from_mapping(
                {
                    "bloodtest_csv_path": "blood.csv",
                    "endpoint_csv_path": "endpoint.csv",
                    "split_csv_path": "split.csv",
                }
            )
        ),
    )
    monkeypatch.setattr(
        train_single_bloodtest_benchmark,
        "canonical_benchmark_dataset",
        lambda *_args: canonical,
    )
    monkeypatch.setattr(
        train_single_bloodtest_benchmark,
        "benchmark_target_dataset",
        lambda *_args: cast(PatientBloodTestDataset, object()),
    )

    def prepare_task(
        _raw_config: Mapping[str, Mapping[str, object]],
        _site_dataset: PatientBloodTestDataset,
        _site: Mapping[str, object],
        _horizon: Mapping[str, object],
        _output_dir: Path,
        _config_dir: Path,
        feature_cache: SurvivalBenchmarkFeatureCache,
    ) -> SingleBloodtestTask:
        caches.append(feature_cache)
        return cast(SingleBloodtestTask, object())

    monkeypatch.setattr(train_single_bloodtest_benchmark, "prepare_task", prepare_task)

    train_single_bloodtest_benchmark.run_benchmark(
        raw_config,
        cast(PatientDatasetConfig, object()),
        ("A|||u",),
        [],
        tmp_path,
        tmp_path,
        cast(PatientDataSources, object()),
    )

    assert len(caches) == 2
    assert caches[0] is caches[1]


def landmark_table() -> PatientLandmarkTable:
    durations = np.asarray([10, 20], dtype=np.int32)
    events = np.asarray([True, False], dtype=bool)
    return PatientLandmarkTable(
        features=np.asarray([[1.0, 2.0], [4.0, 5.0]], dtype=np.float32),
        signed_durations=np.where(events, durations, -durations).astype(np.float64),
        durations=durations,
        events=events,
        sample_weights=np.asarray([1.0, 1.0], dtype=np.float64),
        feature_names=("A|||u", "B|||u"),
        patient_ids=np.asarray(["P0", "P1"], dtype=object),
        landmark_days=np.asarray([3, 4], dtype=np.int32),
        prior_non_endpoint_recurrence=np.asarray([False, True], dtype=bool),
        metadata={},
    )
