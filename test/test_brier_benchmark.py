from __future__ import annotations

import json
import pickle
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import yaml
from interpret.glassbox import ExplainableBoostingRegressor

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.dataset import load_patient_data_sources
from src.regeneration.runner import run_regeneration
from test.regeneration_helpers import regeneration_config
from src.model.complete_case_brier import METRIC
from src.model.ebm_fitting import fit_ebm_model
from src.model.imputation import transform_landmark_table
from src.model.survival_evaluation import evaluate_prepared_split, feature_subset_table
from src.script import brier_model_artifacts, survival_benchmark_cache, survival_model_input
from src.script import train_single_bloodtest_benchmark as training
from src.script.brier_bootstrap import prepare_brier_washouts
from src.script.brier_model_artifacts import load_or_fit_competitor, prepare_brier_cell
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.single_bloodtest_benchmark_reporting import (
    CLINICAL_BASELINE_MODEL_KEY,
    bloodtest_rows,
    model_context_features,
)
from src.script.single_bloodtest_brier import run_brier_benchmark
from src.script.single_bloodtest_washout import saved_model_dir, washout_landmarks
from src.script.survival_training import prepare_survival_training
from test.test_bootstrap_imputation_cache import RiskModel, imputation_dataset


def brier_dataset(tmp_path: Path) -> PatientBloodTestDataset:
    original = imputation_dataset(tmp_path)
    config = replace(original.config, censoring_horizon_days=180, test_bootstrap_runs=20, test_bootstrap_samples=1)
    return PatientBloodTestDataset(config, prepared_data=original.prepared_data)


def saved_models(tmp_path: Path) -> tuple[PatientBloodTestDataset, list[dict[str, object]], dict[str, object]]:
    dataset = brier_dataset(tmp_path)
    landmarks = BenchmarkLandmarks(dataset)
    target_config = replace(dataset.config, endpoint_ddfs_code_filter_regex="C787")
    train = landmarks.table(target_config)
    imputation = {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 10}
    parameters = {"outer_bags": 1, "validation_size": 0, "max_rounds": 0, "n_jobs": 1, "random_state": 42}
    prepared = prepare_survival_training(
        dataset, train, imputation, nonnegative_feature_names=dataset.selected_test_keys
    )
    sources = []
    for key, kind, indices in (
        (CLINICAL_BASELINE_MODEL_KEY, "clinical_baseline", [len(train.feature_names) - 1]),
        ("A|||u", "single_blood_test", [0, len(train.feature_names) - 1]),
    ):
        names = tuple(train.feature_names[index] for index in indices)
        table = feature_subset_table(train, names)
        model, _ = fit_ebm_model(table, prepared.train_imputation.features[:, indices], parameters)
        source = {
            "model_key": key,
            "model_label": key,
            "model_kind": kind,
            "site_label": "Liver",
            "horizon_label": "6m",
            "horizon_days": 180,
            "split": "test",
            "status": "complete",
            "endpoint_ddfs_code_filter_regex": "C787",
            "feature_names": list(names),
        }
        model_dir = saved_model_dir(tmp_path, source)
        model_dir.mkdir(parents=True)
        source["output_dir"] = str(model_dir)
        (model_dir / "model.pkl").write_bytes(pickle.dumps(model))
        (model_dir / "imputer.pkl").write_bytes(pickle.dumps(prepared.imputer))
        write_training_config_snapshot(
            {"imputation": imputation, "ebm": parameters}, target_config, model_dir, tmp_path
        )
        sources.append(source)
    benchmark = {
        "sites": [{"label": "Liver", "endpoint_ddfs_code_filter_regex": "C787"}],
        "horizons": [{"label": "6m", "days": 180}],
        "washout": {"days": [0, 30, 60, 90, 180]},
        "include_all_blood_tests": False,
        "brier": {"enabled": True, "competing_method": "matching_cox_ebm", "output_dir": "brier"},
    }
    return dataset, sources, benchmark


def test_brier_draws_are_exactly_the_cindex_rows_including_zero_event_draws(tmp_path: Path) -> None:
    dataset = brier_dataset(tmp_path)
    pools = washout_landmarks(dataset, (0, 30, 60, 90))
    target = replace(dataset.config, split="test", endpoint_ddfs_code_filter_regex="C787")
    plan = prepare_brier_washouts(pools, target, tmp_path / "draws")
    prepared = prepare_survival_training(dataset, dataset.to_landmark_table(bootstrap=False), {"enabled": False})
    zero_event_count = 0
    for day, washout in plan.items():
        cindex_draws = pools[day].evaluation_cache(target, prepared.imputer).get("test")
        exported = pd.read_csv(tmp_path / "draws" / f"bootstrap_draws_{day}d.csv.gz")
        for run, indices in enumerate(washout.draws):
            cindex = cindex_draws.runs[run].table
            np.testing.assert_array_equal(washout.table.patient_ids[indices], cindex.patient_ids)
            np.testing.assert_array_equal(washout.table.landmark_days[indices], cindex.landmark_days)
            actual = exported.loc[exported["bootstrap_run"] == run]
            np.testing.assert_array_equal(actual["patient_id"], cindex.patient_ids)
            np.testing.assert_array_equal(actual["landmark_day"], cindex.landmark_days)
            np.testing.assert_allclose(
                actual["bootstrap_selection_probability"], cindex.bootstrap_selection_probabilities
            )
            zero_event_count += not cindex.events.any()
        scores = evaluate_prepared_split(RiskModel(), cindex_draws, None)
        assert scores["bootstrap_runs_attempted"] == len(washout.draws)
        eligible = pd.read_csv(tmp_path / "draws" / f"eligible_landmarks_{day}d.csv.gz")
        totals = eligible.groupby("patient_id")["eligible_patient_weight"].sum()
        if len(totals):
            np.testing.assert_allclose(totals, totals.iloc[0])
    assert zero_event_count > 0


@pytest.mark.parametrize("precomputed_features", [False, True])
def test_saved_model_pipeline_reuses_models_imputers_draws_and_clinical_reference(
    tmp_path: Path, precomputed_features: bool,
) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    pools = washout_landmarks(dataset, (0, 30, 60, 90, 180))
    original_models = [(Path(source["output_dir"]) / "model.pkl").read_bytes() for source in sources]
    if precomputed_features:
        model_dir = Path(sources[0]["output_dir"])
        raw, config = brier_model_artifacts.saved_cell_config(model_dir)
        tables = {split: pools[0].landmarks.table(replace(config, split=split)) for split in ("train", "val", "test")}
        imputer = brier_model_artifacts.load_feature_imputer(raw, model_dir, tables["train"])
        pools[0].model_input_features.prepare(tables, imputer)
    with (
        patch.object(brier_model_artifacts, "fit_ebm_model", wraps=fit_ebm_model) as fit,
        patch.object(survival_model_input, "transform_landmark_table", wraps=transform_landmark_table) as transform,
    ):
        results = run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    assert fit.call_count == 2
    assert transform.call_count == (0 if precomputed_features else 3)
    for call, source in zip(fit.call_args_list, sources, strict=True):
        table = call.args[0]
        assert all(str(patient).startswith("train") for patient in table.patient_ids)
        assert table.feature_names == tuple(source["feature_names"])
        np.testing.assert_array_equal(
            table.events, np.isin(table.target_end_reasons, ["death", "other_cancer", "non_endpoint_recurrence"])
        )
        sums = pd.Series(table.sample_weights).groupby(table.patient_ids).sum()
        np.testing.assert_allclose(sums, sums.iloc[0])
    assert len(results) == 10
    assert {result["status"] for result in results} <= {"complete", "not_estimable", "skipped"}
    ordinary = [result for result in results if result["washout_days"] == 0]
    assert all(result["bootstrap_runs"] == 20 for result in ordinary)
    assert all(result["zero_event_bootstrap_runs_discarded"] == 0 for result in ordinary)
    assert any(run["target_event_count"] == 0 and run[METRIC] is not None for run in ordinary[0]["runs"])
    destination = tmp_path / "brier"
    configuration = yaml.safe_load((destination / "configuration.yml").read_text())
    assert configuration["evaluation"]["bootstrap_samples_by_washout"] == {0: 3, 30: 3, 60: 1, 90: 1}
    reference = pd.read_csv(destination / "Liver" / "6m" / "reference_predictions.csv.gz")
    assert set(reference["split"]) == {"train", "val", "test"}
    assert reference.loc[reference["outcome_180d"].isna(), "squared_error"].isna().all()
    metadata = json.loads((destination / "metadata.json").read_text())
    assert metadata["ipcw"] is False
    complete = [result for result in metadata["results"] if result["status"] == "complete"]
    assert complete
    assert all(result["paired_reference_run_count"] > 0 for result in complete)
    latex = (destination / "complete_case_brier_washout.tex").read_text()
    assert "Complete-case Brier score" in latex
    assert "may be biased by censoring" in latex
    assert r"\begin{sidewaystable*}[p]" in latex
    assert "0d & 30d & 60d & 90d & 180d" in latex
    snapshots = {path.relative_to(destination): pd.read_csv(path) for path in destination.rglob("*.csv.gz")}
    with patch.object(brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected refit")):
        repeated = run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    assert repeated == results
    for path, frame in snapshots.items():
        pd.testing.assert_frame_equal(pd.read_csv(destination / path), frame)
    for source, original in zip(sources, original_models, strict=True):
        assert (Path(source["output_dir"]) / "model.pkl").read_bytes() == original


def test_fixed_full_pool_features_match_cindex_features_after_washout(tmp_path: Path) -> None:
    dataset, sources, _ = saved_models(tmp_path)
    model_dir = Path(sources[1]["output_dir"])
    raw, config = brier_model_artifacts.saved_cell_config(model_dir)
    pools = washout_landmarks(dataset, (0, 30, 60))
    tables = {
        "train": pools[0].landmarks.table(config),
        "test": pools[0].landmarks.table(replace(config, split="test")),
    }
    imputer = brier_model_artifacts.load_feature_imputer(raw, model_dir, tables["train"])
    full = transform_landmark_table(imputer, tables["test"]).features
    identities = pd.MultiIndex.from_arrays([tables["test"].patient_ids, tables["test"].landmark_days])
    for day, pool in pools.items():
        washed = pool.landmarks.table(replace(config, split="test"))
        indices = identities.get_indexer(pd.MultiIndex.from_arrays([washed.patient_ids, washed.landmark_days]))
        if len(indices):
            np.testing.assert_allclose(transform_landmark_table(imputer, washed).features, full[indices])
        assert day == 0 or np.all(washed.durations > day)


def test_saved_brier_cell_centers_large_model_offsets_and_records_both_centers(tmp_path: Path) -> None:
    dataset, sources, _ = saved_models(tmp_path)
    model_dir = Path(sources[0]["output_dir"])
    destination = tmp_path / "centered"
    landmarks = BenchmarkLandmarks(dataset)
    original, metadata = prepare_brier_cell(model_dir, destination, landmarks, {})
    load_model = brier_model_artifacts.load_model

    def shifted_model(path: Path) -> ExplainableBoostingRegressor:
        model = load_model(path)
        model.intercept_ += 1000.0 if path.name == "model.pkl" else -1000.0
        return model

    with patch.object(brier_model_artifacts, "load_model", side_effect=shifted_model):
        centered, centered_metadata = prepare_brier_cell(model_dir, destination, landmarks, {})
    pd.testing.assert_frame_equal(centered, original, rtol=1e-12, atol=1e-12)
    assert centered_metadata["risk_centering"] == "patient_weighted_mean_training_log_risk"
    for cause, offset in (("target", 1000.0), ("competing", -1000.0)):
        key = f"{cause}_log_risk_center"
        assert centered_metadata[key] == pytest.approx(metadata[key] + offset)
    saved_metadata = json.loads((destination / "metadata.json").read_text())
    assert saved_metadata == json.loads(json.dumps(centered_metadata))


def test_unestimable_cindex_does_not_block_brier_and_competitor_cache_invalidates_on_training_changes(
    tmp_path: Path,
) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    sources[0]["status"] = "not_estimable"
    pools = washout_landmarks(dataset, (0, 30, 60, 90, 180))
    results = run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    assert results[0]["status"] == "complete"
    destination = tmp_path / "brier" / "Liver" / "6m" / CLINICAL_BASELINE_MODEL_KEY
    saved = json.loads((destination / "competing_model.json").read_text())
    saved["training_fingerprint"] = "old training data"
    (destination / "competing_model.json").write_text(json.dumps(saved))
    with patch.object(brier_model_artifacts, "fit_ebm_model", wraps=fit_ebm_model) as fit:
        prepare_brier_cell(Path(sources[0]["output_dir"]), destination, pools[0].landmarks, {})
    assert fit.call_count == 1


def test_unified_command_reuses_saved_benchmark_sources(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    config_path = tmp_path / "brier.yml"
    config_path.write_text(json.dumps({"benchmark": {**benchmark, "output_dir": str(tmp_path)}}))
    (tmp_path / "metadata.json").write_text(
        json.dumps({"selected_blood_tests": dataset.selected_test_keys, "results": sources})
    )
    unified_config = regeneration_config(tmp_path, benchmark, ["brier"], allow_competing_model_fit=True)
    run_regeneration(unified_config)
    results = json.loads((tmp_path / "brier" / "metadata.json").read_text())["results"]
    assert results[0]["status"] == "complete"


def test_training_then_retrospective_and_resumed_runs_produce_identical_brier_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = brier_dataset(tmp_path)
    config = replace(
        dataset.config,
        train_bootstrap_runs=2,
        validation_bootstrap_runs=2,
        train_bootstrap_samples=10,
        validation_bootstrap_samples=10,
        test_bootstrap_samples=10,
    )
    output_dir = tmp_path / "benchmark"
    output_dir.mkdir()
    benchmark = {
        "output_dir": str(output_dir),
        "resume": True,
        "write_risk_scores": False,
        "sites": [{"label": "Liver", "endpoint_ddfs_code_filter_regex": "C787"}],
        "horizons": [{"label": "6m", "days": 180}],
        "washout": {
            "days": [0, 30],
            "output_csv_path": "washout/cindex.csv",
            "output_latex_path": "washout/cindex.tex",
            "output_metadata_path": "washout/metadata.json",
        },
        "include_all_blood_tests": False,
        "table_caption": "Synthetic C-index",
        "table_label": "tab:synthetic",
        "brier": {"enabled": True, "output_dir": "brier", "competing_method": "matching_cox_ebm"},
    }
    raw = {
        "benchmark": benchmark,
        "dataset": {},
        "ebm": {"outer_bags": 1, "validation_size": 0, "max_rounds": 0, "n_jobs": 1, "random_state": 42},
        "imputation": {"enabled": True, "method": "locf", "locf_fallback_method": "mice", "max_iter": 10},
        "output": {"write_explanations": False},
    }
    # Skip unrelated cohort figures; model fitting, persistence, evaluation and both Brier entry paths are real.
    monkeypatch.setattr(survival_benchmark_cache, "write_survival_dataset_artifacts", lambda *_args, **_kwargs: {})
    rows = bloodtest_rows(
        dataset.selected_test_keys,
        benchmark,
        tmp_path,
        context_features=model_context_features(config),
        include_blood_test_change=config.include_blood_test_change,
    )
    sources = load_patient_data_sources(config)
    count_path = output_dir / "washout" / "cohort_counts.json"
    original_train = training.train_ebm_model

    def train_with_count_report(*args, **kwargs):
        assert count_path.is_file()
        return original_train(*args, **kwargs)

    monkeypatch.setattr(training, "train_ebm_model", train_with_count_report)
    training.run_benchmark(raw, config, dataset.selected_test_keys, rows, output_dir, tmp_path, sources)
    original_counts = count_path.read_bytes()
    assert len(json.loads(original_counts)["results"]) == 2
    metadata_path = output_dir / "brier" / "metadata.json"
    automatic = json.loads(metadata_path.read_text())
    assert len(automatic["results"]) == 6
    assert all(result["status"] == "complete" for result in automatic["results"])
    originals = {
        path: path.read_bytes()
        for path in output_dir.rglob("*")
        if path.is_file() and "brier" not in path.relative_to(output_dir).parts
    }
    config_path = tmp_path / "retrospective.yml"
    config_path.write_text(json.dumps({"benchmark": benchmark}))
    unified_config = regeneration_config(tmp_path, benchmark, ["brier"])
    with patch.object(
        brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected competing-model refit")
    ):
        run_regeneration(unified_config)
    assert json.loads(metadata_path.read_text()) == automatic
    assert all(path.read_bytes() == content for path, content in originals.items())
    count_path.unlink()
    with patch.object(training, "train_ebm_model", side_effect=AssertionError("Unexpected recurrence-model refit")):
        with patch.object(
            brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected competing-model refit")
        ):
            training.run_benchmark(raw, config, dataset.selected_test_keys, rows, output_dir, tmp_path, sources)
    assert json.loads(metadata_path.read_text()) == automatic
    assert count_path.read_bytes() == original_counts


def test_absent_training_competing_events_have_explicit_zero_hazard(tmp_path: Path) -> None:
    dataset = brier_dataset(tmp_path)
    train = BenchmarkLandmarks(dataset).table(dataset.config)
    model, metadata = load_or_fit_competitor(train, train.features, {}, tmp_path)
    assert model is None
    assert metadata["method"] == "zero_hazard_no_training_competing_events"
    assert metadata["training_competing_events"] == 0
    assert not (tmp_path / "competing_model.pkl").exists()


def test_invalid_hazards_are_exported_and_invalidate_affected_washout_scores(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    predict = brier_model_artifacts.predict_splits

    def extreme_test_risks(model, features, training_weights, **kwargs):
        risks, center = predict(model, features, training_weights, **kwargs)
        risks["test"] = np.full(len(features["test"]), 100.0)
        return risks, center

    pools = washout_landmarks(dataset, (0, 30, 60, 90, 180))
    with patch.object(brier_model_artifacts, "predict_splits", side_effect=extreme_test_risks):
        results = run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    zero = next(result for result in results if result["washout_days"] == 0)
    assert zero["status"] == "invalid_predictions"
    assert zero[METRIC] is None
    assert zero["invalid_hazard_increment_count"] > 0
    diagnostics = pd.read_csv(
        tmp_path / "brier" / "Liver" / "6m" / CLINICAL_BASELINE_MODEL_KEY / "invalid_hazard_increments.csv"
    )
    assert set(diagnostics["split"]) == {"test"}
    assert ((diagnostics["target_increment"] + diagnostics["competing_increment"]) > 1).all()
    summary = json.loads((
        tmp_path / "brier" / "Liver" / "6m" / CLINICAL_BASELINE_MODEL_KEY / "diagnostics.json"
    ).read_text())
    assert summary["status"] == "invalid_predictions"
    assert summary["splits"]["test"]["invalid_probability_count"] > 0
