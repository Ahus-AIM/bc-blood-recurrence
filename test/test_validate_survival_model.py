from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_target_dataset, canonical_benchmark_dataset
from src.data.dataset import load_patient_data_sources
from src.data.dataset_config import PatientDatasetConfig
from src.model import survival_evaluation
from src.model.survival_evaluation import SurvivalEvaluationCache
from src.script import survival_training, validate_survival_model
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.survival_dataset_artifacts import write_eligible_patient_folds_csv
from src.script.survival_training import (
    feature_subset_prepared_training,
    fitted_run_metadata,
    prepare_survival_training,
    train_ebm_model,
    write_outputs,
)
from test.test_patient_bloodtest import blood_row, config as make_dataset_config, endpoint_row


@pytest.mark.parametrize(
    ("benchmark", "imputation"),
    [(False, True), (False, False), (True, True)],
)
@pytest.mark.parametrize("weighting", ["person_uniform", "person_time_weighted", "landmark_uniform"])
def test_saved_run_replays_training_test_bootstraps_without_fitting_or_writing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    benchmark: bool,
    imputation: bool,
    weighting: str,
) -> None:
    run_dir, expected = saved_run(tmp_path, benchmark, imputation, weighting)
    before = {path: path.read_bytes() for path in run_dir.iterdir()}
    evaluated_splits: list[str] = []
    original_evaluate = survival_evaluation.evaluate_prepared_table

    def evaluate(model, prepared, **kwargs):
        evaluated_splits.append(prepared.table.metadata["split"])
        return original_evaluate(model, prepared, **kwargs)

    def unexpected_fit(*_args, **_kwargs):
        pytest.fail("Saved-model evaluation must not fit a model or imputer.")

    monkeypatch.setattr(survival_evaluation, "evaluate_prepared_table", evaluate)
    monkeypatch.setattr(survival_training, "fit_ebm_model", unexpected_fit)
    monkeypatch.setattr(survival_training, "fit_landmark_imputer", unexpected_fit)

    with caplog.at_level(logging.INFO):
        actual = validate_survival_model.validate_saved_model(run_dir)

    assert json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True)
    assert evaluated_splits and set(evaluated_splits) == {"test"}
    assert {path: path.read_bytes() for path in run_dir.iterdir()} == before
    assert "difference=+0" in caplog.text
    assert "verified 6 test patients" in caplog.text
    expected_path = "benchmark evaluation" if benchmark else "direct saved-model evaluation"
    assert expected_path in caplog.text


def test_changed_test_membership_fails_before_prediction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir, _summary = saved_run(tmp_path)
    split_path = tmp_path / "split.csv"
    splits = pd.read_csv(split_path)
    splits.loc[splits["id"] == "P10", "split"] = "val"
    splits.to_csv(split_path, index=False)
    # Keep the new assignment temporally valid to reach the saved-membership guard.
    endpoint_path = tmp_path / "endpoint.csv"
    endpoints = pd.read_csv(endpoint_path)
    endpoints.loc[endpoints["id"] == "P10", "bc_diagnosis_year"] = 2020
    endpoints.to_csv(endpoint_path, index=False)

    def unexpected_evaluation(*_args, **_kwargs):
        pytest.fail("Changed test membership must fail before evaluation.")

    monkeypatch.setattr(validate_survival_model, "evaluate_split", unexpected_evaluation)
    with pytest.raises(ValueError, match="Test patients differ.*1 missing, 0 unexpected"):
        validate_survival_model.validate_saved_model(run_dir)


def test_missing_enabled_imputer_is_not_refitted(tmp_path: Path) -> None:
    run_dir, _summary = saved_run(tmp_path)
    (run_dir / "imputer.pkl").unlink()

    with pytest.raises(FileNotFoundError, match="Saved feature imputer not found"):
        validate_survival_model.validate_saved_model(run_dir)


def test_missing_snapshot_does_not_fall_back_to_active_training_config(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="training_config.yml"):
        validate_survival_model.validate_saved_model(tmp_path)


def test_main_resolves_selected_directory_from_its_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "validate.yml"
    config_path.write_text(yaml.safe_dump({"source": {"run_dir": "../saved model"}}), encoding="utf-8")
    selected: list[Path] = []
    monkeypatch.setattr(validate_survival_model, "CONFIG_PATH", config_path)
    monkeypatch.setattr(validate_survival_model, "configure_logging", lambda: None)
    monkeypatch.setattr(validate_survival_model, "validate_saved_model", selected.append)

    validate_survival_model.main()

    assert selected == [tmp_path / "saved model"]


def saved_run(
    tmp_path: Path,
    benchmark: bool = False,
    imputation: bool = True,
    weighting: str = "person_uniform",
) -> tuple[Path, dict[str, object]]:
    """Train a small site-specific run with missing test values and a saved full-panel imputer."""
    config = synthetic_dataset_config(tmp_path, weighting)
    raw_config = {
        "dataset": {},
        "imputation": {
            "enabled": imputation,
            "method": "locf",
            "locf_max_age_days": 30,
            "locf_fallback_method": "median",
        },
        "output": {"output_dir": "unused_active_output", "evaluate_test": True},
    }
    if benchmark:
        raw_config["benchmark"] = {}
        canonical = canonical_benchmark_dataset(config, config.selected_blood_tests, load_patient_data_sources(config))
        dataset = benchmark_target_dataset(canonical, config)
        landmarks = BenchmarkLandmarks(canonical)
        table = landmarks.table(config)
    else:
        dataset = PatientBloodTestDataset(config)
        table = dataset.to_landmark_table(bootstrap=False)
    prepared = prepare_survival_training(
        dataset, table, raw_config["imputation"], nonnegative_feature_names=dataset.selected_test_keys,
    )
    cache = SurvivalEvaluationCache(config, prepared.imputer, landmarks) if benchmark else None
    prepared = feature_subset_prepared_training(prepared, ("B|||u",))
    model_config = {"outer_bags": 1, "validation_size": 0, "max_rounds": 5}
    fitted = train_ebm_model(prepared, model_config, {"test": None}, evaluation_cache=cache)
    run_dir = tmp_path / "saved model"
    run_dir.mkdir()
    write_training_config_snapshot(raw_config, config, run_dir, tmp_path)
    write_outputs(fitted.model, prepared.imputer, fitted_run_metadata(fitted, {}), run_dir)
    write_eligible_patient_folds_csv(dataset, run_dir)
    return run_dir, fitted.split_summaries["test"]


def synthetic_dataset_config(tmp_path: Path, weighting: str) -> PatientDatasetConfig:
    patients = [f"P{index}" for index in range(16)]
    splits = {patient: "train" if index < 8 else "val" if index < 10 else "test" for index, patient in enumerate(patients)}
    config = make_dataset_config(
        tmp_path,
        [
            blood_row(patient, day, test_name=test, unit="u", result=1.0 + (index * 7 + day) % 11)
            for index, patient in enumerate(patients)
            for day in (35, 50)
            for test in ("A", "B")
            if test == "A" or day == 35 and index % 3 != 0
        ],
        [
            endpoint_row(
                patient,
                diagnosis_to_ddfs=60 + index if index % 3 != 0 else "",
                ddfs_code="C787" if index % 3 == 1 else "C793" if index % 3 == 2 else "",
                last_entry_days=100 + index,
            )
            for index, patient in enumerate(patients)
        ],
        split_by_patient=splits,
    )
    return PatientDatasetConfig.from_mapping({
        **config,
        "selected_blood_tests": ["A|||u", "B|||u"],
        "endpoint_ddfs_code_filter_regex": "C787",
        "recurrence_endpoint_name": "liver",
        "censoring_horizon_days": 30,
        "test_bootstrap_runs": 4,
        "test_bootstrap_samples": 20,
        "bootstrap_sample_weighting": weighting,
    })
