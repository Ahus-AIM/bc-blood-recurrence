from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_target_dataset, canonical_benchmark_dataset
from src.data.dataset import load_patient_data_sources
from src.model import imputation
from src.model.risk_score_calibration import TRAINING_RISK_REFERENCE_FILENAME, write_training_risk_reference
from src.regeneration import risk_reference as reference_script
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script.survival_training import (
    SurvivalRunArtifacts,
    feature_subset_prepared_training,
    prepare_survival_training,
    train_ebm_model,
    write_fitted_survival_run,
)
from test.test_validate_survival_model import synthetic_dataset_config


@pytest.mark.parametrize(
    ("benchmark", "imputation_enabled"),
    [(False, False), (True, True)],
)
def test_training_and_saved_regeneration_use_the_same_training_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    benchmark: bool,
    imputation_enabled: bool,
) -> None:
    config = synthetic_dataset_config(tmp_path, "person_uniform")
    raw_config = {
        "dataset": {},
        "imputation": {"enabled": imputation_enabled, "method": "locf", "locf_fallback_method": "median"},
        "output": {"evaluate_test": False, "write_explanations": False},
    }
    if benchmark:
        raw_config["benchmark"] = {}
        canonical = canonical_benchmark_dataset(config, config.selected_blood_tests, load_patient_data_sources(config))
        dataset = benchmark_target_dataset(canonical, config)
        table = BenchmarkLandmarks(canonical).table(config)
    else:
        dataset = PatientBloodTestDataset(config)
        table = dataset.to_landmark_table(bootstrap=False)
    prepared = prepare_survival_training(dataset, table, raw_config["imputation"])
    prepared = feature_subset_prepared_training(prepared, ("B|||u",))
    model_config = {"outer_bags": 1, "validation_size": 0, "max_rounds": 5, "n_jobs": 1}
    fitted = train_ebm_model(prepared, model_config, {})
    run_dir = tmp_path / "saved model"
    run_dir.mkdir()
    write_training_config_snapshot(raw_config, config, run_dir, tmp_path)
    metadata = write_fitted_survival_run(
        fitted,
        SurvivalRunArtifacts(run_dir, raw_config["output"], tmp_path, {}),
    )
    reference_path = run_dir / TRAINING_RISK_REFERENCE_FILENAME
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    assert list(reference.columns) == ["patient_id", "landmark_day", "risk_score"]
    assert reference["patient_id"].tolist() == table.patient_ids.tolist()
    assert reference["landmark_day"].tolist() == table.landmark_days.tolist()
    np.testing.assert_array_equal(reference["risk_score"], fitted.model.predict(prepared.train_imputation.features))
    assert len(reference) == len(table.patient_ids)
    assert not reference.duplicated(["patient_id", "landmark_day"]).any()
    assert metadata["training_risk_reference"]["landmark_count"] == len(table.patient_ids)
    assert not (run_dir / "test_risk_scores.csv").exists()
    saved_reference = reference_path.read_bytes()
    artifacts = {path.name: path.read_bytes() for path in run_dir.glob("*.pkl")}
    snapshot = (run_dir / "training_config.yml").read_bytes()
    reference_path.unlink()

    def unexpected_fit(*_args: object, **_kwargs: object) -> None:
        pytest.fail("Reference regeneration must reuse the saved imputer.")

    monkeypatch.setattr(imputation, "fit_feature_imputer", unexpected_fit)
    summary = reference_script.regenerate_training_risk_reference(run_dir)

    assert reference_path.read_bytes() == saved_reference
    assert summary == metadata["training_risk_reference"]
    assert (run_dir / "training_config.yml").read_bytes() == snapshot
    assert {path.name: path.read_bytes() for path in run_dir.glob("*.pkl")} == artifacts


def test_bootstrap_duplicates_cannot_be_exported_as_training_reference(tmp_path: Path) -> None:
    dataset = PatientBloodTestDataset(synthetic_dataset_config(tmp_path, "person_uniform"))
    table = dataset.to_landmark_table(bootstrap=False)
    duplicated = replace(
        table, patient_ids=np.full(len(table.patient_ids), "P0"), landmark_days=np.zeros_like(table.landmark_days)
    )

    with pytest.raises(ValueError, match="unique non-bootstrapped"):
        write_training_risk_reference(duplicated, np.zeros(len(table.patient_ids)), tmp_path)
    assert not (tmp_path / TRAINING_RISK_REFERENCE_FILENAME).exists()
