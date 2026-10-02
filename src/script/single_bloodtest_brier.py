from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path

import yaml

from src.data.competing_risk_outcomes import HORIZON_DAYS
from src.model.complete_case_brier import METRIC
from src.model.run_failure import failure_details
from src.reporting import report_output_path
from src.script.brier_bootstrap import BrierWashout, evaluate_brier_washout, prepare_brier_washouts
from src.script.brier_diagnostics import record_brier_failure
from src.script.brier_model_artifacts import prepare_brier_cell, saved_cell_config
from src.script.brier_reporting import write_brier_outputs
from src.script.ebm_survival_config import safe_path_component
from src.script.single_bloodtest_benchmark_config import mapping_sequence
from src.script.single_bloodtest_benchmark_reporting import (
    CLINICAL_BASELINE_MODEL_KEY, merge_result_records,
)
from src.script.single_bloodtest_washout import configured_washout_days, saved_model_dir, saved_washout_landmarks
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache, fitted_imputer_fingerprint
from src.script.survival_model_input import SharedModelInputFeatureCache

LOGGER = logging.getLogger(__name__)


def brier_config(benchmark: Mapping[str, object]) -> Mapping[str, object]:
    config = benchmark.get("brier", {})
    if not isinstance(config, Mapping):
        raise TypeError("`benchmark.brier` must be a mapping.")
    if config.get("competing_method", "matching_cox_ebm") != "matching_cox_ebm":
        raise ValueError("The competing estimator must be a Cox EBM with matching recurrence-model predictors.")
    return config


def run_brier_benchmark(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    sources: Sequence[Mapping[str, object]],
    output_dir: Path,
    config_dir: Path,
    pools: Mapping[int, SurvivalBenchmarkFeatureCache] | None = None,
    *,
    allow_competing_model_fit: bool = True,
    fail_fast: bool = False,
    previous_results: Sequence[Mapping[str, object]] = (),
) -> list[dict[str, object]]:
    """Evaluate the existing 180-day benchmark cells on their original washout draws."""
    settings = brier_config(benchmark)
    horizons = [
        horizon for horizon in mapping_sequence(benchmark["horizons"], "horizons") if horizon["days"] == HORIZON_DAYS
    ]
    if not horizons:
        return []
    if len(horizons) != 1:
        raise ValueError("Brier requires exactly one configured 180-day horizon.")
    benchmark = {**benchmark, "horizons": horizons}
    cells = [
        dict(source) for source in sources if source.get("split") == "test" and source["horizon_days"] == HORIZON_DAYS
    ]
    for source in cells:
        source["output_dir"] = str(saved_model_dir(output_dir, source))
    available = [
        source
        for source in cells
        if source.get("status") != "failed" and (Path(source["output_dir"]) / "model.pkl").is_file()
    ]
    if not available:
        raise FileNotFoundError("No saved 180-day recurrence models found for Brier evaluation.")
    _, config = saved_cell_config(Path(available[0]["output_dir"]))
    days = configured_washout_days(benchmark)
    if pools is None:
        pools = saved_washout_landmarks(available, days)
    destination = report_output_path(output_dir, settings.get("output_dir", "brier"))
    if destination.resolve() == output_dir.resolve():
        raise ValueError("Brier outputs require a separate directory to preserve recurrence-model artifacts.")
    destination.mkdir(parents=True, exist_ok=True)
    washouts = prepare_brier_washouts(pools, config, destination)
    (destination / "configuration.yml").write_text(
        yaml.safe_dump(
            {
                "benchmark": dict(benchmark),
                "evaluation": {
                    "horizon_days": HORIZON_DAYS,
                    "bootstrap_runs": config.test_bootstrap_runs,
                    "bootstrap_samples_by_washout": {
                        day: washout.sample_count for day, washout in washouts.items()
                    },
                    "random_seed": config.random_seed,
                    "ipcw": False,
                    "invalid_hazard_policy": "save diagnostics; invalidate affected probabilities and washout summaries",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    cache: dict[str, SharedModelInputFeatureCache] = {}
    shared = pools[0].model_input_features
    if shared.prepared is not None and shared.imputer is not None:
        cache[fitted_imputer_fingerprint(shared.imputer)] = shared
    results = []
    for source in cells:
        results.extend(evaluate_brier_cell(
            source, pools[0], cache, washouts, destination, days,
            allow_competing_model_fit=allow_competing_model_fit, fail_fast=fail_fast,
        ))
        write_brier_outputs(
            benchmark, selected_tests, merge_result_records(previous_results, results), destination, config_dir,
        )
    return results


def evaluate_brier_cell(
    source: Mapping[str, object],
    zero_pool: SurvivalBenchmarkFeatureCache,
    cache: dict[str, SharedModelInputFeatureCache],
    washouts: Mapping[int, BrierWashout],
    destination: Path,
    days: tuple[int, ...],
    *,
    allow_competing_model_fit: bool = True,
    fail_fast: bool = False,
) -> list[dict[str, object]]:
    fields = {
        key: source[key]
        for key in (
            "model_key",
            "model_label",
            "model_kind",
            "site_label",
            "horizon_label",
            "horizon_days",
        )
    }
    results = [
        {
            **fields,
            "split": "test",
            "washout_days": day,
            "status": "skipped" if day >= HORIZON_DAYS else "pending",
            "runs": [],
            METRIC: None,
        }
        for day in days
    ]
    cell_dir = destination.joinpath(
        *(
            safe_path_component(str(source[key]))
            for key in (
                "site_label",
                "horizon_label",
                "model_key",
            )
        )
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    reference_path = cell_dir.parent / "reference_predictions.csv.gz"
    if source["model_key"] == CLINICAL_BASELINE_MODEL_KEY:
        reference_path.unlink(missing_ok=True)
    for filename in (
        "predictions.csv.gz", "metadata.json", "breslow_baselines.csv", "invalid_hazard_increments.csv",
        "diagnostics.json", "risk_set_diagnostics.csv",
    ):
        (cell_dir / filename).unlink(missing_ok=True)
    try:
        if source.get("status") == "failed":
            raise ValueError("Recurrence-model training failed; saved artifacts cannot be reused.")
        model_dir = Path(str(source["output_dir"]))
        _, config = saved_cell_config(model_dir)
        if config.endpoint_ddfs_code_filter_regex != source["endpoint_ddfs_code_filter_regex"]:
            raise ValueError("Saved recurrence endpoint does not match the benchmark cell.")
        predictions, metadata = prepare_brier_cell(
            model_dir, cell_dir, zero_pool.landmarks, cache, allow_competing_model_fit=allow_competing_model_fit,
        )
        if source["model_key"] == CLINICAL_BASELINE_MODEL_KEY:
            shutil.copyfile(cell_dir / "predictions.csv.gz", reference_path)
        test_predictions = predictions.loc[predictions["split"] == "test"]
        for result in results:
            if result["status"] == "skipped":
                continue
            day = result["washout_days"]
            result.update(evaluate_brier_washout(test_predictions, washouts[day]))
            result["predictions_path"] = str(cell_dir / "predictions.csv.gz")
            result["reference_predictions_path"] = str(reference_path)
            result["invalid_hazard_increment_count"] = metadata["invalid_hazard_increment_count"]
    except Exception as error:
        LOGGER.exception("Brier evaluation failed: %s", cell_dir)
        record_brier_failure(cell_dir, error)
        details = failure_details(error)
        (cell_dir / "metadata.json").write_text(json.dumps({"status": "failed", **details}, indent=2), encoding="utf-8")
        for result in results:
            if result["status"] != "skipped":
                result.update(status="failed", **details)
        if fail_fast:
            raise
    return results
