from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

from src.config import load_yml, resolve_path
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, canonical_benchmark_dataset
from src.data.dataset import input_fingerprint, load_patient_data_sources
from src.data.dataset_config import PatientDatasetConfig
from src.data.patient_parsing import load_split_assignments
from src.model.survival_evaluation import (
    SurvivalEvaluationCache,
    evaluate_prepared_split,
    evaluate_split,
    report_c_indexes,
)
from src.script.ebm_survival_config import METADATA_FILENAME, TRAINING_CONFIG_FILENAME
from src.script.logging_config import configure_logging
from src.script.saved_ebm_artifacts import load_feature_imputer, load_model, model_feature_names
from src.script.survival_dataset_artifacts import ELIGIBLE_PATIENT_FOLDS_FILENAME

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "validate_survival_model.yml"
LOGGER = logging.getLogger(__name__)


def main() -> None:
    configure_logging()
    config = load_yml(CONFIG_PATH)
    run_dir = config["source"]["run_dir"]
    if not isinstance(run_dir, str) or not run_dir.strip():
        raise ValueError("Set `source.run_dir` to a trained model directory.")
    validate_saved_model(resolve_path(run_dir, CONFIG_PATH.parent))


def validate_saved_model(run_dir: Path) -> dict[str, object]:
    """Replay only test bootstraps through the selected run's training evaluation path."""
    run_dir = run_dir.resolve()
    raw_config = load_yml(run_dir / TRAINING_CONFIG_FILENAME)
    config = replace(
        PatientDatasetConfig.from_mapping(raw_config["dataset"], config_dir=run_dir),
        split="test",
    )
    model = load_model(run_dir / "model.pkl")
    feature_names = model_feature_names(model)
    metadata = load_metadata(run_dir)
    LOGGER.info("validating saved model: %s", run_dir / "model.pkl")
    LOGGER.info("training config: %s; test split source: %s", run_dir / TRAINING_CONFIG_FILENAME, config.split_csv_path)
    check_input_fingerprint(config, metadata)

    landmarks = None
    if "benchmark" in raw_config:
        dataset = canonical_benchmark_dataset(config, config.selected_blood_tests, load_patient_data_sources(config))
        landmarks = BenchmarkLandmarks(dataset)
        test_table = landmarks.table(config)
    else:
        dataset = PatientBloodTestDataset(config)
        test_table = dataset.to_landmark_table(bootstrap=False)
    imputer = load_feature_imputer(raw_config, run_dir, test_table)
    check_test_membership(run_dir, dataset)

    if landmarks is not None:
        LOGGER.info("using benchmark evaluation: SurvivalEvaluationCache -> evaluate_prepared_split")
        evaluation_cache = SurvivalEvaluationCache(config, imputer, landmarks)
        summary = evaluate_prepared_split(model, evaluation_cache.get("test"), None, feature_names=feature_names)
    else:
        LOGGER.info("using direct saved-model evaluation: evaluate_split")
        summary = evaluate_split(model, "test", config, dataset, imputer, None, feature_names=feature_names)

    report_c_indexes({"test": summary})
    LOGGER.info(
        "test bootstrap runs: attempted=%s retained=%s zero_event_discarded=%s finite_c_indexes=%s",
        summary["bootstrap_runs_attempted"],
        summary["bootstrap_runs"],
        summary["zero_event_bootstrap_runs_discarded"],
        summary["harrell_c_index_valid_runs"],
    )
    report_saved_comparison(summary, metadata)
    return summary


def load_metadata(run_dir: Path) -> dict[str, object]:
    path = run_dir / METADATA_FILENAME
    if not path.is_file():
        LOGGER.warning("Saved metadata unavailable for comparison: %s", path)
        return {}
    metadata = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(metadata, dict):
        raise TypeError(f"Saved metadata must contain a JSON object: {path}")
    return metadata


def check_input_fingerprint(config: PatientDatasetConfig, metadata: Mapping[str, object]) -> None:
    """Report source/config drift; fingerprints also change when files are moved or touched."""
    saved_dataset = metadata.get("dataset")
    if not isinstance(saved_dataset, Mapping):
        return
    saved = saved_dataset.get("input_fingerprint")
    if saved is None:
        return
    current = input_fingerprint(config)
    if current != saved:
        LOGGER.warning(
            "Input fingerprint differs: saved=%s current=%s; source files, paths, or dataset semantics may have changed.",
            saved,
            current,
        )


def check_test_membership(run_dir: Path, dataset: PatientBloodTestDataset) -> None:
    """Check reconstructed test patients against the run-local eligible folds when available."""
    path = run_dir / ELIGIBLE_PATIENT_FOLDS_FILENAME
    if not path.is_file():
        LOGGER.warning("Saved eligible folds unavailable; using the training config's split CSV: %s", path)
        return
    expected = set(load_split_assignments(path)["test"])
    actual = set(dataset.patient_ids)
    if actual != expected:
        raise ValueError(
            f"Test patients differ from {path}: {len(expected - actual)} missing, {len(actual - expected)} unexpected."
        )
    LOGGER.info("verified %d test patients against %s", len(actual), path)


def report_saved_comparison(summary: Mapping[str, object], metadata: Mapping[str, object]) -> None:
    saved = metadata.get("test")
    if not isinstance(saved, Mapping):
        LOGGER.info("No saved test summary is available for comparison.")
        return
    for key in ("harrell_c_index", "harrell_c_index_ci95_lower", "harrell_c_index_ci95_upper"):
        previous = saved.get(key)
        current = summary[key]
        if isinstance(previous, int | float) and isinstance(current, int | float):
            LOGGER.info("%s: saved=%.12g recalculated=%.12g difference=%+.12g", key, previous, current, current - previous)


if __name__ == "__main__":
    main()
