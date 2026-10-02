from __future__ import annotations

import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_washout_dataset, canonical_benchmark_dataset
from src.data.dataset import load_patient_data_sources
from src.data.dataset_config import PatientDatasetConfig
from src.model.cindex_benchmark import (
    CINDEX_SUMMARY_KEYS,
    CIndexTableLabel,
    CellFormatter,
    latex_horizon_header,
    write_benchmark_artifacts,
)
from src.model.run_failure import failure_details
from src.model.survival_evaluation import evaluate_prepared_split
from src.reporting import latex_escape, report_output_path
from src.script.ebm_survival_config import TRAINING_CONFIG_FILENAME, safe_path_component
from src.script.saved_ebm_artifacts import load_feature_imputer, load_model
from src.script.single_bloodtest_benchmark_reporting import (
    load_saved_summary,
    rendered_latex_table,
    with_test_inference,
)
from src.script.single_bloodtest_washout_counts import write_washout_cohort_outputs
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache, fitted_imputer_fingerprint
from src.script.single_bloodtest_benchmark_config import mapping_sequence
from src.time_units import DAYS_PER_MONTH

LOGGER = logging.getLogger(__name__)
WashoutState = tuple[Mapping[int, SurvivalBenchmarkFeatureCache], list[dict[str, object]]]


def configured_washout_days(benchmark: Mapping[str, object]) -> tuple[int, ...]:
    config = washout_config(benchmark)
    values = config.get("days", (0, DAYS_PER_MONTH, 2 * DAYS_PER_MONTH, 3 * DAYS_PER_MONTH))
    if not isinstance(values, list | tuple):
        raise TypeError("`benchmark.washout.days` must be a list.")
    days = tuple(int(value) for value in values)
    if not days or days[0] != 0 or tuple(sorted(set(days))) != days:
        raise ValueError("`benchmark.washout.days` must begin at 0 and be unique and ascending.")
    return days


def washout_config(benchmark: Mapping[str, object]) -> Mapping[str, object]:
    config = benchmark.get("washout", {})
    if not isinstance(config, Mapping):
        raise TypeError("`benchmark.washout` must be a mapping.")
    return {
        "output_csv_path": "washout/cindex_benchmark_washout.csv",
        "output_latex_path": "washout/cindex_benchmark_washout.tex",
        "output_metadata_path": "washout/metadata.json",
        **config,
    }


def washout_landmarks(
    canonical: PatientBloodTestDataset,
    days: Sequence[int],
    *,
    zero_cache: SurvivalBenchmarkFeatureCache | None = None,
) -> dict[int, SurvivalBenchmarkFeatureCache]:
    """Keep each positive washout's history separate and reuse the unfiltered cache."""
    return {
        day: (
            zero_cache
            if day == 0 and zero_cache is not None
            else SurvivalBenchmarkFeatureCache(BenchmarkLandmarks(benchmark_washout_dataset(canonical, day)))
        )
        for day in days
    }


def prepare_washout_state(
    canonical: PatientBloodTestDataset,
    benchmark: Mapping[str, object],
    zero_cache: SurvivalBenchmarkFeatureCache,
    output_dir: Path,
) -> WashoutState | None:
    """Prepare shared pools and report their counts before fitting any models."""
    try:
        pools = washout_landmarks(canonical, configured_washout_days(benchmark), zero_cache=zero_cache)
    except Exception:
        LOGGER.exception("automatic washout setup failed; continuing benchmark training")
        return None
    try:
        write_washout_cohort_outputs(benchmark, pools, output_dir)
    except Exception:
        LOGGER.exception("washout cohort-count output failed; continuing benchmark training and evaluation")
    return pools, []


def saved_washout_landmarks(
    results: Sequence[Mapping[str, object]],
    days: Sequence[int],
) -> dict[int, SurvivalBenchmarkFeatureCache]:
    """Build canonical washout pools from a run-local training configuration."""
    paths = (
        Path(output_dir) / TRAINING_CONFIG_FILENAME
        for result in results
        if isinstance(output_dir := result.get("output_dir"), str)
    )
    config_path = next((path for path in paths if path.is_file()), None)
    if config_path is None:
        raise FileNotFoundError("No saved benchmark training configuration was found.")
    raw_config = load_yml(config_path)
    config = PatientDatasetConfig.from_mapping(raw_config["dataset"], config_dir=config_path.parent)
    canonical = canonical_benchmark_dataset(config, config.selected_blood_tests, load_patient_data_sources(config))
    return washout_landmarks(canonical, days)


def evaluate_saved_washout(
    benchmark: Mapping[str, object],
    output_dir: Path,
    config_dir: Path,
) -> list[dict[str, object]]:
    """Re-evaluate every saved test cell and write the washout artifacts."""
    metadata_path = report_output_path(output_dir, benchmark.get("output_metadata_path", "metadata.json"))
    selected_tests, source_results = load_saved_summary(metadata_path)
    source_results = [
        {**source, "output_dir": str(saved_model_dir(output_dir, source))}
        for source in source_results
    ]
    landmarks = saved_washout_landmarks(source_results, configured_washout_days(benchmark))
    results = [
        washout_result
        for source in source_results
        if source.get("split") == "test"
        for washout_result in evaluate_washout_cell(source, landmarks)
    ]
    write_washout_outputs(benchmark, selected_tests, results, output_dir, config_dir)
    return results


def saved_model_dir(output_dir: Path, result: Mapping[str, object]) -> Path:
    return output_dir.joinpath(
        *(safe_path_component(str(result[key])) for key in ("site_label", "horizon_label", "model_key"))
    )


def evaluate_washout_cell(
    source_result: Mapping[str, object],
    landmarks: Mapping[int, SurvivalBenchmarkFeatureCache],
    *,
    fail_fast: bool = False,
) -> list[dict[str, object]]:
    """Evaluate one saved model on every configured washout period."""
    results = [washout_result(source_result, day) for day in landmarks]
    if fail_fast and source_result.get("status") == "failed":
        raise ValueError(f"Cannot re-evaluate a failed model: {source_result['output_dir']}")
    if source_result.get("status") != "complete" and not fail_fast:
        return results
    model_dir = Path(str(source_result["output_dir"]))
    try:
        raw_config = load_yml(model_dir / TRAINING_CONFIG_FILENAME)
        config = replace(
            PatientDatasetConfig.from_mapping(raw_config["dataset"], config_dir=model_dir),
            split="test",
            endpoint_ddfs_code_filter_regex=str(source_result["endpoint_ddfs_code_filter_regex"]),
            censoring_horizon_days=int(source_result["horizon_days"]),
        )
        model = load_model(model_dir / "model.pkl")
        imputer = load_feature_imputer(raw_config, model_dir, landmarks[0].landmarks.table(config))
        imputer_key = fitted_imputer_fingerprint(imputer)
        feature_names = tuple(str(name) for name in model.feature_names_in_)
    except Exception as error:
        if fail_fast:
            raise
        for result in results:
            if result["status"] != "skipped":
                result.update({"status": "failed", **failure_details(error)})
        return results
    for day, result in zip(landmarks, results, strict=True):
        if result["status"] == "skipped":
            continue
        try:
            pool = landmarks[day]
            prepared = pool.evaluation_cache(config, imputer, imputer_key=imputer_key).get("test")
            summary = evaluate_prepared_split(model, prepared, None, feature_names=feature_names)
            status = "complete" if math.isfinite(float(summary["harrell_c_index"])) else "not_estimable"
            result.update(summary, status=status)
            if status == "not_estimable":
                result["not_estimable_reason"] = "no comparable patient pairs"
        except Exception as error:
            if fail_fast:
                raise
            result.update({"status": "failed", **failure_details(error)})
    return results


def record_washout(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    cell_results: Sequence[Mapping[str, object]],
    output_dir: Path,
    config_dir: Path,
    state: WashoutState | None,
) -> None:
    """Evaluate and rewrite washout outputs without interrupting training."""
    if state is None:
        return
    try:
        landmarks, results = state
        source = next(result for result in cell_results if result["split"] == "test")
        results.extend(evaluate_washout_cell(source, landmarks))
        write_washout_outputs(benchmark, selected_tests, results, output_dir, config_dir)
    except Exception:
        LOGGER.exception("automatic washout evaluation or artifact writing failed; continuing")


def washout_result(source: Mapping[str, object], day: int) -> dict[str, object]:
    skipped = day >= int(source["horizon_days"])
    return {
        **source,
        **dict.fromkeys(CINDEX_SUMMARY_KEYS),
        "horizon_label": f"{source['horizon_label']} {day}d",
        "washout_days": day,
        "status": "skipped" if skipped else source["status"],
        "runs": [],
        "bootstrap_runs_attempted": None,
        "zero_event_bootstrap_runs_discarded": None,
    }


def write_washout_outputs(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    output_dir: Path,
    config_dir: Path,
) -> dict[str, str]:
    """Write incremental washout CSV, JSON and LaTeX, returning their paths."""
    washout = washout_config(benchmark)
    configured = {**benchmark, **washout}
    reported = with_test_inference(results)
    flat = [{key: value for key, value in result.items() if key != "runs"} for result in reported]
    return write_benchmark_artifacts(
        output_dir,
        configured,
        flat,
        {"selected_blood_tests": selected_tests, "results": reported},
        washout_table(configured, selected_tests, reported, config_dir),
    )


def washout_table(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: Sequence[Mapping[str, object]],
    config_dir: Path,
    *,
    format_cell: CellFormatter | None = None,
) -> str:
    """Render the benchmark metric with horizon and washout subcolumns."""
    days = configured_washout_days(benchmark)
    horizons = mapping_sequence(benchmark["horizons"], "horizons")
    expanded = {
        **benchmark,
        "horizons": [
            {"label": f"{horizon['label']} {day}d"}
            for horizon in horizons
            for day in days
        ],
    }
    latex = rendered_latex_table(expanded, selected_tests, list(results), config_dir, format_cell=format_cell)
    return format_washout_table(latex, benchmark)


def format_washout_table(latex: str, benchmark: Mapping[str, object]) -> str:
    """Apply the shared site/horizon/washout headers and landscape layout."""
    days = configured_washout_days(benchmark)
    sites = mapping_sequence(benchmark["sites"], "sites")
    horizons = mapping_sequence(benchmark["horizons"], "horizons")
    old_header = latex_horizon_header(
        tuple(CIndexTableLabel(str(site["label"])) for site in sites),
        tuple(CIndexTableLabel(f"{horizon['label']} {day}d") for horizon in horizons for day in days),
    )
    horizon_header = "& " + " & ".join(
        rf"\multicolumn{{{len(days)}}}{{c}}{{{latex_escape(str(horizon['label']))}}}"
        for _site in sites
        for horizon in horizons
    ) + r"\\"
    day_header = "& " + " & ".join(
        f"{day}d" for _site in sites for _horizon in horizons for day in days
    ) + r"\\"
    return (
        latex.replace(old_header, f"{horizon_header}\n{day_header}")
        .replace(r"\begin{table*}[h]", r"\begin{sidewaystable*}[p]")
        .replace(r"\resizebox{\textwidth}{!}{%", r"\resizebox{\textheight}{!}{%")
        .replace(r"\end{table*}", r"\end{sidewaystable*}")
    )
