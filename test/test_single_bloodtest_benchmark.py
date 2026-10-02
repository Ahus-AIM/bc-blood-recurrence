from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from src.config import load_yml
from src.model import survival_evaluation
from src.script import survival_benchmark_cache
from src.script import survival_model_input
from src.script import train_single_bloodtest_benchmark
from src.data import PatientLandmarkTable
from src.data.dataset_config import PatientDatasetConfig
from src.data.landmarks import TIME_SINCE_DIAGNOSIS_FEATURE_NAME
from src.model.imputation import FittedFeatureImputer, ImputationConfig
from src.script.ebm_survival_config import TRAINING_CONFIG_FILENAME
from src.script.single_bloodtest_benchmark_reporting import (
    CLINICAL_BASELINE_MODEL_KEY,
    bloodtest_rows,
    model_context_features,
)
from src.script.saved_ebm_artifacts import predict_risk_scores
from src.script.survival_model_input import SharedModelInputFeatureCache
from src.script.survival_training import (
    SharedSurvivalFeatureCache,
    feature_subset_prepared_training,
    prepare_survival_training,
)
from src.script.train_single_bloodtest_benchmark import (
    benchmark_dataset_config,
    load_benchmark_config,
)
from src.model.survival_evaluation import feature_subset_table


def test_benchmark_config_inherits_context_features_and_disables_unrelated_features() -> None:
    raw_config, config_dir = load_benchmark_config()

    config = benchmark_dataset_config(raw_config, config_dir)

    assert config.split == "train"
    assert config.include_diagnosis_age is True
    assert config.require_diagnosis_age is True
    assert config.include_time_since_diagnosis is True
    assert config.include_blood_test_change is False
    assert config.include_recent_diagnoses is False
    assert raw_config["output"]["write_explanations"] is True
    assert raw_config["landmark_event_counts"]["site_labels"] == ["Bone", "Brain", "Liver", "Lung"]
    assert [horizon["days"] for horizon in raw_config["landmark_event_counts"]["horizons"]] == [30, 90, 180, 365, 730]
    assert raw_config["benchmark"]["washout"]["days"] == [0, 30, 60, 90]
    assert "bold indicates the highest mean C-index" in raw_config["benchmark"]["table_caption"]
    assert "underline the second-highest distinct mean C-index" in raw_config["benchmark"][
        "table_caption"
    ]
    assert "one-sided 95\\% lower bootstrap bound is above 0.5" in raw_config["benchmark"][
        "table_caption"
    ]
    assert "paired C-index difference from the clinical baseline" in raw_config[
        "benchmark"
    ]["table_caption"]


@pytest.mark.parametrize("include_changes", [False, True])
def test_bloodtest_rows_add_clinical_baseline_and_context_to_every_model(
    tmp_path: Path, include_changes: bool,
) -> None:
    raw_config, config_dir = load_benchmark_config()
    dataset_config = replace(benchmark_dataset_config(raw_config, config_dir), include_blood_test_change=include_changes)

    rows = bloodtest_rows(
        ("A|||u", "B|||u"),
        raw_config["benchmark"],
        tmp_path,
        context_features=model_context_features(dataset_config),
        include_blood_test_change=dataset_config.include_blood_test_change,
    )

    context = (TIME_SINCE_DIAGNOSIS_FEATURE_NAME, "age_at_diagnosis")
    assert rows[0] == {
        "key": CLINICAL_BASELINE_MODEL_KEY,
        "label": "Clinical baseline",
        "features": context,
        "kind": "clinical_baseline",
        "blood_test_key": None,
    }
    if include_changes:
        assert rows[1]["features"] == ("A|||u", "A|||u change", *context)
        assert rows[2]["features"] == ("B|||u", "B|||u change", *context)
        assert rows[3]["features"] == ("A|||u", "B|||u", "A|||u change", "B|||u change", *context)
    else:
        assert rows[1]["features"] == ("A|||u", *context)
        assert rows[2]["features"] == ("B|||u", *context)
        assert rows[3]["features"] == ("A|||u", "B|||u", *context)


def test_subset_table_keeps_rows_targets_and_requested_feature_order() -> None:
    table = landmark_table()

    subset = feature_subset_table(table, ("C|||u", "A|||u"))

    assert subset.feature_names == ("C|||u", "A|||u")
    assert subset.features.tolist() == [[3.0, 1.0], [6.0, 4.0]]
    assert subset.signed_durations.tolist() == [10.0, -20.0]
    assert subset.events.tolist() == [True, False]
    assert subset.sample_weights is table.sample_weights


def test_prepared_feature_subset_reuses_full_fitted_imputer() -> None:
    prepared = prepare_survival_training(
        object(),
        landmark_table(),
        {"enabled": False},
    )

    subset = feature_subset_prepared_training(prepared, ("B|||u",))

    assert subset.imputer is prepared.imputer
    assert subset.train_table.feature_names == ("B|||u",)
    assert subset.train_imputation.features.tolist() == [[2.0], [5.0]]
    assert subset.train_imputation.imputed_mask.shape == (2, 1)


def test_survival_feature_cache_reuses_training_imputation_across_horizons() -> None:
    first_table = landmark_table()
    second_table = replace(
        first_table,
        durations=np.asarray([30, 40], dtype=np.int32),
        signed_durations=np.asarray([30.0, -40.0], dtype=np.float64),
    )
    cache = SharedSurvivalFeatureCache()

    first = cache.prepare(object(), first_table, {"enabled": False})
    second = cache.prepare(object(), second_table, {"enabled": False})

    assert first is not None
    assert second is not None
    assert second.imputer is first.imputer
    assert second.train_imputation is first.train_imputation
    assert second.train_table.durations.tolist() == [30, 40]


def test_survival_feature_cache_rejects_changed_rows_between_horizons() -> None:
    first_table = landmark_table()
    changed_features = first_table.features.copy()
    changed_features[0, 0] = 99.0
    cache = SharedSurvivalFeatureCache()
    cache.prepare(object(), first_table, {"enabled": False})

    with pytest.raises(ValueError, match="feature values differ"):
        cache.prepare(
            object(),
            replace(first_table, features=changed_features),
            {"enabled": False},
        )


def test_model_input_feature_cache_reuses_nonbootstrap_imputations_across_horizons(
    monkeypatch,
) -> None:
    first_table = landmark_table()
    second_table = replace(
        first_table,
        durations=np.asarray([30, 40], dtype=np.int32),
        signed_durations=np.asarray([30.0, -40.0], dtype=np.float64),
    )
    prepared = prepare_survival_training(object(), first_table, {"enabled": False})
    transform_calls = 0
    original_transform = survival_model_input.transform_landmark_table

    def transform(*args, **kwargs):
        nonlocal transform_calls
        transform_calls += 1
        return original_transform(*args, **kwargs)

    monkeypatch.setattr(survival_model_input, "transform_landmark_table", transform)
    cache = SharedModelInputFeatureCache()
    first = cache.prepare(
        {"train": first_table, "val": first_table, "test": first_table},
        prepared.imputer,
        train_imputation=prepared.train_imputation,
    )
    second = cache.prepare(
        {"train": second_table, "val": second_table, "test": second_table},
        prepared.imputer,
        train_imputation=prepared.train_imputation,
    )

    assert transform_calls == 2
    assert second.imputation_results is first.imputation_results
    assert second.tables["test"].durations.tolist() == [30, 40]


def test_evaluate_split_accepts_feature_subset_keyword(monkeypatch) -> None:
    raw_config, config_dir = load_benchmark_config()
    dataset_config = benchmark_dataset_config(raw_config, config_dir)
    captured: dict[str, object] = {}

    def split_landmark_table(split_config, _prepared_dataset, *, bootstrap, bootstrap_run=0):
        assert split_config.split == "test"
        assert bootstrap is True
        assert bootstrap_run == 0
        return landmark_table()

    def evaluate_table(
        _model,
        table,
        _feature_imputer,
        bootstrap_run,
        *,
        reference_table=None,
        feature_names=None,
    ):
        captured["feature_names"] = table.feature_names
        captured["model_feature_names"] = feature_names
        assert bootstrap_run == 0
        assert reference_table is None
        return {"harrell_c_index": 0.5}, pd.DataFrame({"risk_score": [0.1]})

    monkeypatch.setattr(survival_evaluation, "bootstrap_run_count", lambda _config: 1)
    monkeypatch.setattr(survival_evaluation, "landmark_table", split_landmark_table)
    monkeypatch.setattr(survival_evaluation, "evaluate_table", evaluate_table)
    monkeypatch.setattr(
        survival_evaluation,
        "summarize_bootstrap_runs",
        lambda run_summaries, _sample_count: run_summaries[0],
    )

    summary = survival_evaluation.evaluate_split(
        object(),
        "test",
        dataset_config,
        object(),
        SimpleNamespace(requires_landmark_context=False, feature_names=landmark_table().feature_names),
        None,
        feature_names=("B|||u",),
    )

    assert summary["harrell_c_index"] == 0.5
    assert captured["feature_names"] == landmark_table().feature_names
    assert captured["model_feature_names"] == ("B|||u",)


def test_evaluate_split_passes_reference_table_for_contextual_imputation(monkeypatch) -> None:
    raw_config, config_dir = load_benchmark_config()
    dataset_config = benchmark_dataset_config(raw_config, config_dir)
    captured: dict[str, object] = {}

    def split_landmark_table(split_config, _prepared_dataset, *, bootstrap, bootstrap_run=0):
        assert split_config.split == "test"
        assert bootstrap_run == 0
        captured[f"bootstrap_{bootstrap}"] = True
        return landmark_table()

    def evaluate_table(
        _model,
        table,
        _feature_imputer,
        bootstrap_run,
        *,
        reference_table=None,
        feature_names=None,
    ):
        assert bootstrap_run == 0
        captured["table_features"] = table.feature_names
        captured["reference_features"] = None if reference_table is None else reference_table.feature_names
        captured["model_features"] = feature_names
        return {"harrell_c_index": 0.5}, pd.DataFrame({"risk_score": [0.1]})

    contextual_imputer = type(
        "ContextualImputer",
        (),
        {"requires_landmark_context": True, "feature_names": landmark_table().feature_names},
    )()
    monkeypatch.setattr(survival_evaluation, "bootstrap_run_count", lambda _config: 1)
    monkeypatch.setattr(survival_evaluation, "landmark_table", split_landmark_table)
    monkeypatch.setattr(survival_evaluation, "evaluate_table", evaluate_table)
    monkeypatch.setattr(
        survival_evaluation,
        "summarize_bootstrap_runs",
        lambda run_summaries, _sample_count: run_summaries[0],
    )

    summary = survival_evaluation.evaluate_split(
        object(),
        "test",
        dataset_config,
        object(),
        contextual_imputer,
        None,
        feature_names=("B|||u",),
    )

    assert summary["harrell_c_index"] == 0.5
    assert captured == {
        "bootstrap_False": True,
        "bootstrap_True": True,
        "table_features": landmark_table().feature_names,
        "reference_features": landmark_table().feature_names,
        "model_features": ("B|||u",),
    }


def test_evaluate_table_imputes_full_schema_before_model_subset() -> None:
    table = landmark_table()
    imputer = FittedFeatureImputer(
        ImputationConfig.from_mapping({"enabled": False}),
        table.feature_names,
        (),
        (),
        None,
        {"enabled": False},
    )
    captured: dict[str, np.ndarray] = {}

    def predict(features: np.ndarray) -> np.ndarray:
        captured["features"] = features
        return np.asarray([0.1, 0.2])

    survival_evaluation.evaluate_table(
        SimpleNamespace(predict=predict),
        table,
        imputer,
        0,
        feature_names=("B|||u",),
    )

    assert captured["features"].tolist() == [[2.0], [5.0]]


def test_saved_ebm_scoring_uses_model_subset_after_full_imputation() -> None:
    table = landmark_table()
    imputer = FittedFeatureImputer(
        ImputationConfig.from_mapping({"enabled": False}),
        table.feature_names,
        (),
        (),
        None,
        {"enabled": False},
    )
    model = SimpleNamespace(
        feature_names_in_=["B|||u"],
        predict=lambda features: features[:, 0],
    )

    scores = predict_risk_scores(model, imputer, table)

    assert scores.tolist() == [2.0, 5.0]


def test_cell_artifacts_snapshot_keeps_shared_blood_test_panel(tmp_path: Path, monkeypatch) -> None:
    raw_config = benchmark_raw_config(tmp_path)
    dataset_config = PatientDatasetConfig.from_mapping(raw_config["dataset"])
    dataset = type("Dataset", (), {"config": dataset_config})()
    row = {
        "key": "A|||u",
        "label": "A",
        "features": ("A|||u",),
        "kind": "single_blood_test",
        "blood_test_key": "A|||u",
    }
    site = {"label": "Liver", "endpoint_ddfs_code_filter_regex": r"C787\d*"}
    horizon = {"label": "1y", "days": 365}
    task = train_single_bloodtest_benchmark.SingleBloodtestTask(
        dataset,
        landmark_table(),
        None,
        site,
        horizon,
        {"path": "../model_input.csv", "sha256": "abc"},
        "abc",
        "not estimable",
    )
    artifact_cache = survival_benchmark_cache.TaskDatasetArtifactCache(
        raw_config,
        dataset,  # type: ignore[arg-type]
        tmp_path,
        tmp_path / "shared",
        None,
    )
    task = replace(task, artifact_cache=artifact_cache)
    monkeypatch.setattr(
        survival_benchmark_cache,
        "write_survival_dataset_artifacts",
        lambda *_args, **_kwargs: {},
    )
    model_dir = tmp_path / "benchmark" / "Liver" / "1y" / "A___u"
    model_dir.mkdir(parents=True)

    metadata = train_single_bloodtest_benchmark.prepare_cell_artifacts(
        task,
        row,
        model_dir,
    )

    snapshot_path = model_dir / TRAINING_CONFIG_FILENAME
    snapshot = load_yml(snapshot_path)
    assert snapshot["dataset"]["selected_blood_tests"] == ["A|||u", "B|||u"]
    assert snapshot["dataset"]["bloodtest_csv_path"] == str(tmp_path / "blood.csv")
    assert metadata["benchmark_model"]["feature_names"] == ["A|||u"]


def test_task_dataset_artifacts_are_rendered_once_and_copied_to_each_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    calls: list[str] = []

    def write_config(_raw_config, _dataset_config, output_dir, _config_dir):
        calls.append("training_config")
        (output_dir / "training_config.yml").write_text("config", encoding="utf-8")
        return {"path": "training_config.yml"}

    def write_dataset_artifacts(*_args, **_kwargs):
        output_dir = _args[1]
        calls.append("dataset_artifacts")
        (output_dir / "dataset_statistics.png").write_bytes(b"figure")
        (output_dir / "population_statistics.tex").write_text("table", encoding="utf-8")
        return {
            "dataset_statistics": {"path": "dataset_statistics.png"},
            "population_statistics": {"path": "population_statistics.tex"},
        }

    monkeypatch.setattr(
        survival_benchmark_cache,
        "write_training_config_snapshot",
        write_config,
    )
    monkeypatch.setattr(
        survival_benchmark_cache,
        "write_survival_dataset_artifacts",
        write_dataset_artifacts,
    )
    cache = survival_benchmark_cache.TaskDatasetArtifactCache(
        {},
        SimpleNamespace(config=object()),
        tmp_path,
        tmp_path / "shared",
        None,
    )

    first = cache.materialize(tmp_path / "model_a")
    second = cache.materialize(tmp_path / "model_b")

    assert calls == ["training_config", "dataset_artifacts"]
    assert first == second
    assert (tmp_path / "model_a" / "training_config.yml").read_text(encoding="utf-8") == "config"
    assert (tmp_path / "model_b" / "dataset_statistics.png").read_bytes() == b"figure"
    for model_name in ("model_a", "model_b"):
        assert (tmp_path / model_name / "population_statistics.tex").read_text(encoding="utf-8") == "table"


def test_task_model_input_is_serialized_once_and_referenced_by_each_model(
    tmp_path: Path,
    monkeypatch,
) -> None:
    writes = 0

    def write_model_input(_artifact, output_path):
        nonlocal writes
        writes += 1
        output_path.write_text("feature\n1\n", encoding="utf-8")
        return {"path": output_path.name, "row_count": 1}

    monkeypatch.setattr(survival_benchmark_cache, "write_model_input_artifact", write_model_input)
    cache = survival_benchmark_cache.TaskModelInputCache(
        SimpleNamespace(),  # type: ignore[arg-type]
        tmp_path / "shared" / "model_input.csv",
    )

    first_metadata = cache.shared_metadata("../model_input.csv", shared_by_task=True)
    second_metadata = cache.shared_metadata("../model_input.csv", shared_by_task=True)

    assert writes == 1
    assert first_metadata == second_metadata
    assert first_metadata["path"] == "../model_input.csv"
    assert first_metadata["sha256"] == cache.fingerprint
    assert first_metadata["shared_by_task"] is True
    assert cache.shared_path.read_text(encoding="utf-8") == "feature\n1\n"


def test_fit_cell_evaluates_all_splits_from_shared_preparation(tmp_path: Path, monkeypatch) -> None:
    table = landmark_table()
    shared = prepare_survival_training(object(), table, {"enabled": False})
    evaluation_cache = object()
    row = {
        "key": "B|||u",
        "label": "B",
        "features": ("B|||u",),
        "kind": "single_blood_test",
        "blood_test_key": "B|||u",
    }
    site = {"label": "All", "endpoint_ddfs_code_filter_regex": ".*"}
    horizon = {"label": "1y", "days": 365}
    task = train_single_bloodtest_benchmark.SingleBloodtestTask(
        object(),
        table,
        shared,
        site,
        horizon,
        {},
        "abc",
        None,
        evaluation_cache=evaluation_cache,  # type: ignore[arg-type]
    )
    captured: dict[str, object] = {}

    def train(prepared, _model_config, evaluation_paths, *, evaluation_cache=None):
        captured["prepared"] = prepared
        captured["splits"] = tuple(evaluation_paths)
        captured["evaluation_cache"] = evaluation_cache
        return SimpleNamespace(
            split_summaries={
                "train_bootstrap": summary(),
                "validation": summary(),
                "test": summary(),
            }
        )

    monkeypatch.setattr(train_single_bloodtest_benchmark, "train_ebm_model", train)
    monkeypatch.setattr(train_single_bloodtest_benchmark, "write_fitted_survival_run", lambda *_args: None)

    results = train_single_bloodtest_benchmark.fit_cell(
        {"ebm": {}, "output": {}, "benchmark": {"write_risk_scores": False}},
        task,
        row,
        tmp_path,
        {},
        tmp_path,
    )

    assert captured["prepared"].imputer is shared.imputer
    assert captured["prepared"].train_table.feature_names == ("B|||u",)
    assert captured["splits"] == ("train", "val", "test")
    assert captured["evaluation_cache"] is evaluation_cache
    assert [result["split"] for result in results] == ["train_bootstrap", "validation", "test"]


def test_train_cell_catches_generic_failure_without_completion_marker(tmp_path: Path, monkeypatch) -> None:
    row = {
        "key": "A|||u",
        "label": "A",
        "features": ("A|||u",),
        "kind": "single_blood_test",
        "blood_test_key": "A|||u",
    }
    site = {"label": "All", "endpoint_ddfs_code_filter_regex": ".*"}
    horizon = {"label": "1y", "days": 365}
    task = train_single_bloodtest_benchmark.SingleBloodtestTask(
        object(), landmark_table(), None, site, horizon, {}, "abc", None
    )
    monkeypatch.setattr(
        train_single_bloodtest_benchmark,
        "cell_run_fingerprint",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(train_single_bloodtest_benchmark, "write_cell_failure", lambda *_args: None)
    stale_marker = tmp_path / "All" / "1y" / "A___u" / "benchmark_completion.json"
    stale_marker.parent.mkdir(parents=True)
    stale_marker.write_text("stale", encoding="utf-8")

    results = train_single_bloodtest_benchmark.train_cell(
        {"benchmark": {"resume": True}},
        task,
        row,
        tmp_path,
        tmp_path,
    )

    assert [result["status"] for result in results] == ["failed", "failed", "failed"]
    assert {result["split"] for result in results} == {"train_bootstrap", "validation", "test"}
    assert not stale_marker.exists()


def benchmark_raw_config(tmp_path: Path) -> dict[str, dict[str, object]]:
    return {
        "dataset": {
            "bloodtest_csv_path": str(tmp_path / "blood.csv"),
            "endpoint_csv_path": str(tmp_path / "endpoint.csv"),
            "split_csv_path": str(tmp_path / "split.csv"),
            "selected_blood_tests": ["A|||u", "B|||u"],
        },
        "ebm": {},
        "imputation": {"enabled": False},
        "benchmark": {"write_risk_scores": False},
    }


def summary() -> dict[str, object]:
    return {
        "harrell_c_index": 0.5,
        "harrell_c_index_ci95_lower": 0.4,
        "harrell_c_index_ci95_upper": 0.6,
        "harrell_c_index_valid_runs": 1,
        "bootstrap_runs": 1,
        "bootstrap_sample_count": 2,
        "event_count_mean": 1.0,
        "patient_count_mean": 2.0,
        "row_count_mean": 2.0,
    }


def landmark_table() -> PatientLandmarkTable:
    durations = np.asarray([10, 20], dtype=np.int32)
    events = np.asarray([True, False], dtype=bool)
    return PatientLandmarkTable(
        features=np.asarray([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32),
        signed_durations=np.where(events, durations, -durations).astype(np.float64),
        durations=durations,
        events=events,
        sample_weights=np.asarray([1.0, 1.0], dtype=np.float64),
        feature_names=("A|||u", "B|||u", "C|||u"),
        patient_ids=np.asarray(["P0", "P1"], dtype=object),
        landmark_days=np.asarray([3, 4], dtype=np.int32),
        prior_non_endpoint_recurrence=np.asarray([False, True], dtype=bool),
        metadata={},
    )
