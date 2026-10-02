"""Regenerate a single saved model's training reference without the dashboard dependency."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, canonical_benchmark_dataset
from src.data.dataset import load_patient_data_sources
from src.data.dataset_config import PatientDatasetConfig
from src.model.risk_score_calibration import write_training_risk_reference
from src.script.ebm_survival_config import update_run_metadata
from src.script.saved_ebm_artifacts import load_feature_imputer, load_model, predict_risk_scores

LOGGER = logging.getLogger(__name__)


def regenerate_training_risk_reference(run_dir: Path) -> dict[str, object]:
    """Rebuild the training reference with the original saved model, imputer and cohort."""
    run_dir = run_dir.resolve()
    raw_config = load_yml(run_dir / "training_config.yml")
    dataset_values = dict(raw_config["dataset"])
    normalize_legacy_sample_weighting(dataset_values)
    config = replace(
        PatientDatasetConfig.from_mapping(dataset_values, config_dir=run_dir),
        split="train",
    )
    model = load_model(run_dir / "model.pkl")
    if "benchmark" in raw_config:
        dataset = canonical_benchmark_dataset(config, config.selected_blood_tests, load_patient_data_sources(config))
        table = BenchmarkLandmarks(dataset).table(config)
    else:
        dataset = PatientBloodTestDataset(config)
        table = dataset.to_landmark_table(bootstrap=False)
    imputer = load_feature_imputer(raw_config, run_dir, table)
    scores = predict_risk_scores(model, imputer, table)
    summary = write_training_risk_reference(table, scores, run_dir)
    if not update_run_metadata(run_dir, "training_risk_reference", summary):
        LOGGER.warning("metadata file not found; skipping training risk reference metadata update: %s", run_dir)
    return summary


def normalize_legacy_sample_weighting(dataset_config: dict[str, object]) -> None:
    """Retain the original compatibility mapping for historical saved runs."""
    aliases = {"uniform": "person_uniform", "time_interval": "person_time_weighted"}
    for field_name in ("training_sample_weighting", "bootstrap_sample_weighting"):
        configured = dataset_config.get(field_name)
        if isinstance(configured, str) and configured in aliases:
            dataset_config[field_name] = aliases[configured]
