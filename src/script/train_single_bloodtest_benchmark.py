from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from src.config import resolve_path
from src.data import PatientBloodTestDataset, PatientLandmarkTable
from src.data.benchmark_landmarks import (
    BenchmarkLandmarks,
    benchmark_target_dataset,
    canonical_benchmark_dataset,
)
from src.data.dataset import (
    load_patient_data_sources,
    select_train_test_keys,
)
from src.data.dataset_config import PatientDatasetConfig
from src.data.types import PatientDataSources
from src.model.cindex_benchmark import (
    metric_cell,
    result_records_for_fields,
    unavailable_results_for_fields,
)
from src.model.run_failure import failure_details
from src.model.survival_evaluation import EVALUATION_SPLITS, SurvivalEvaluationCache, feature_subset_table
from src.script.ebm_survival_config import dataset_config_mapping
from src.script.logging_config import DEFAULT_LOG_FILENAME, configure_logging
from src.script.single_bloodtest_benchmark_config import (
    benchmark_model_output_dir,
    load_config,
    mapping_sequence,
    task_output_dir,
)
from src.script.single_bloodtest_ebm_top_features import write_benchmark_ebm_top_features
from src.script.single_bloodtest_brier import brier_config, run_brier_benchmark
from src.script.survival_benchmark_cache import (
    SurvivalBenchmarkFeatureCache,
    TaskDatasetArtifactCache,
    prepare_survival_benchmark_task,
)
from src.script.single_bloodtest_benchmark_reporting import (
    BloodtestModelRow,
    bloodtest_rows,
    model_context_features,
    sort_results,
    write_summary_outputs,
)
from src.script.single_bloodtest_washout import (
    WashoutState,
    prepare_washout_state,
    record_washout,
)
from src.script.single_bloodtest_preparation import (
    single_bloodtest_task_training,
)
from src.script.survival_benchmark_artifacts import (
    load_completed_results,
    mapping_sha256,
    remove_completion_marker,
    write_completion_marker,
)
from src.script.survival_training import (
    PreparedSurvivalTraining,
    SurvivalRunArtifacts,
    feature_subset_prepared_training,
    train_ebm_model,
    write_failed_run,
    write_fitted_survival_run,
    write_not_estimable_run,
)

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "ebm_single_bloodtest_benchmark.yml"
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SingleBloodtestTask:
    """Inputs shared by every feature model in one site/horizon task.

    ``dataset`` and ``train_table`` fix the cohort and targets; ``prepared`` owns
    the full-union imputer. ``artifact_cache`` owns task-identical reports and
    ``evaluation_cache`` lazily shares bootstrap inputs. Other fields name the
    task and describe its shared model-input artifact and estimability.
    """

    dataset: PatientBloodTestDataset
    train_table: PatientLandmarkTable
    prepared: PreparedSurvivalTraining | None
    site: Mapping[str, object]
    horizon: Mapping[str, object]
    model_input_metadata: dict[str, object]
    model_input_fingerprint: str
    not_estimable_reason: str | None
    evaluation_cache: SurvivalEvaluationCache | None = None
    artifact_cache: TaskDatasetArtifactCache | None = None


def main() -> None:
    configure_logging()
    raw_config, config_dir = load_benchmark_config()
    train_from_config(raw_config, config_dir)


def train_from_config(raw_config: dict[str, dict[str, object]], config_dir: Path) -> None:
    """Share the unchanged benchmark workflow between manuscript training and the synthetic demo."""
    benchmark = raw_config["benchmark"]
    output_dir = resolve_output_dir(benchmark, config_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(output_dir / DEFAULT_LOG_FILENAME)
    dataset_config = benchmark_dataset_config(raw_config, config_dir)
    data_sources = load_patient_data_sources(dataset_config)
    selected_tests = select_train_test_keys(data_sources, dataset_config)
    LOGGER.info("selected %d blood-test types", len(selected_tests))
    if not selected_tests:
        raise ValueError("Single-blood-test benchmark requires at least one selected blood-test type.")
    rows = bloodtest_rows(
        selected_tests,
        benchmark,
        config_dir,
        context_features=model_context_features(dataset_config),
        include_blood_test_change=dataset_config.include_blood_test_change,
    )
    run_benchmark(raw_config, dataset_config, selected_tests, rows, output_dir, config_dir, data_sources)
    LOGGER.info("wrote %s", output_dir)


def run_benchmark(
    raw_config: dict[str, dict[str, object]],
    dataset_config: PatientDatasetConfig,
    selected_tests: tuple[str, ...],
    rows: list[BloodtestModelRow],
    output_dir: Path,
    config_dir: Path,
    data_sources: PatientDataSources,
) -> list[dict[str, object]]:
    """Run each EBM cell and rewrite aggregate artifacts after every outcome."""
    results: list[dict[str, object]] = []
    benchmark = raw_config["benchmark"]
    canonical_dataset = canonical_benchmark_dataset(
        dataset_config,
        selected_tests,
        data_sources,
    )
    benchmark_feature_cache = SurvivalBenchmarkFeatureCache(BenchmarkLandmarks(canonical_dataset))
    washout_state = prepare_washout_state(canonical_dataset, benchmark, benchmark_feature_cache, output_dir)

    def record(cell_results: list[dict[str, object]]) -> None:
        record_results(
            benchmark, selected_tests, results, cell_results, output_dir, config_dir,
            washout_state,
        )

    for site in mapping_sequence(benchmark["sites"], "sites"):
        try:
            site_config = replace(
                canonical_dataset.config,
                recurrence_endpoint_name=str(site.get("recurrence_endpoint_name", site["label"])),
                endpoint_ddfs_code_filter_regex=str(site["endpoint_ddfs_code_filter_regex"]),
            )
            site_dataset = benchmark_target_dataset(canonical_dataset, site_config)
        except Exception as error:
            LOGGER.exception("site preparation failed; continuing site=%s", site["label"])
            for horizon in mapping_sequence(benchmark["horizons"], "horizons"):
                for row in rows:
                    record(unavailable_results(row, site, horizon, output_dir, "failed", error))
            continue
        for horizon in mapping_sequence(benchmark["horizons"], "horizons"):
            try:
                task = prepare_task(
                    raw_config,
                    site_dataset,
                    site,
                    horizon,
                    output_dir,
                    config_dir,
                    benchmark_feature_cache,
                )
            except Exception as error:
                LOGGER.exception(
                    "task preparation failed; continuing site=%s horizon=%s",
                    site["label"],
                    horizon["label"],
                )
                for row in rows:
                    record(unavailable_results(row, site, horizon, output_dir, "failed", error))
                continue
            for row in rows:
                record(train_cell(raw_config, task, row, output_dir, config_dir))
    write_benchmark_ebm_top_features(benchmark, output_dir, results)
    if bool(brier_config(benchmark).get("enabled", False)):
        try:
            run_brier_benchmark(
                benchmark, selected_tests, results, output_dir, config_dir,
                None if washout_state is None else washout_state[0],
            )
        except Exception:
            LOGGER.exception("Brier benchmark failed; see logs for the cause")
    return results


def load_benchmark_config() -> tuple[dict[str, dict[str, object]], Path]:
    return load_config(CONFIG_PATH)


def benchmark_dataset_config(
    raw_config: Mapping[str, Mapping[str, object]],
    config_dir: Path,
) -> PatientDatasetConfig:
    return PatientDatasetConfig.from_mapping(
        {
            **raw_config["dataset"],
            "split": "train",
            "require_diagnosis_age": True,
            "include_recent_diagnoses": False,
            "include_landmark_day_recent_diagnoses": False,
        },
        config_dir=config_dir,
    )


def prepare_task(
    raw_config: Mapping[str, Mapping[str, object]],
    site_dataset: PatientBloodTestDataset,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
    output_dir: Path,
    config_dir: Path,
    feature_cache: SurvivalBenchmarkFeatureCache,
) -> SingleBloodtestTask:
    config = replace(site_dataset.config, censoring_horizon_days=int(horizon["days"]))
    task_dir = task_output_dir(output_dir, site, horizon)
    task_dir.mkdir(parents=True, exist_ok=True)
    shared = prepare_survival_benchmark_task(
        raw_config,
        config_dir,
        config,
        site_dataset,
        feature_cache,
        task_dir / ".shared_dataset_artifacts",
        task_dir / "model_input.csv",
        raw_config["ebm"],
    )
    input_metadata = shared.model_input_cache.shared_metadata(
        "../model_input.csv",
        shared_by_task=True,
    )
    prepared = single_bloodtest_task_training(
        shared.prepared,
        shared.not_estimable_reason,
        raw_config["ebm"],
    )
    return SingleBloodtestTask(
        dataset=shared.dataset,
        train_table=shared.train_table,
        prepared=prepared,
        site=site,
        horizon=horizon,
        model_input_metadata=input_metadata,
        model_input_fingerprint=shared.model_input_cache.fingerprint,
        not_estimable_reason=shared.not_estimable_reason,
        evaluation_cache=shared.evaluation_cache,
        artifact_cache=shared.dataset_artifact_cache,
    )


def train_cell(
    raw_config: Mapping[str, Mapping[str, object]],
    task: SingleBloodtestTask,
    row: BloodtestModelRow,
    output_dir: Path,
    config_dir: Path,
) -> list[dict[str, object]]:
    """Run or resume one feature model, recording any exception as a failed cell."""
    model_dir = model_output_dir(output_dir, row, task.site, task.horizon)
    try:
        model_dir.mkdir(parents=True, exist_ok=True)
        run_fingerprint = cell_run_fingerprint(raw_config, task, row)
        base_metadata = prepare_cell_artifacts(task, row, model_dir)
        completed = (
            load_completed_results(model_dir, run_fingerprint, task.model_input_fingerprint)
            if bool(raw_config["benchmark"].get("resume", True))
            else None
        )
        if completed is not None:
            LOGGER.info("resuming completed EBM from %s", model_dir)
            return completed
        remove_completion_marker(model_dir)
        if task.not_estimable_reason is not None:
            results = unavailable_results(
                row,
                task.site,
                task.horizon,
                output_dir,
                "not_estimable",
                task.not_estimable_reason,
            )
            write_not_estimable_run(
                feature_subset_table(task.train_table, row["features"]),
                model_dir,
                task.not_estimable_reason,
                base_metadata,
                None if task.prepared is None else task.prepared.imputer,
            )
        else:
            results = fit_cell(raw_config, task, row, model_dir, base_metadata, config_dir)
        write_completion_marker(model_dir, run_fingerprint, task.model_input_fingerprint, results)
        return results
    except Exception as error:
        remove_completion_marker(model_dir)
        LOGGER.exception(
            "EBM cell failed; continuing test=%s site=%s horizon=%s",
            row["label"],
            task.site["label"],
            task.horizon["label"],
        )
        write_cell_failure(task, row, model_dir, error)
        return unavailable_results(row, task.site, task.horizon, output_dir, "failed", error)


def prepare_cell_artifacts(
    task: SingleBloodtestTask,
    row: BloodtestModelRow,
    model_dir: Path,
) -> dict[str, object]:
    if task.artifact_cache is None:
        raise RuntimeError("Single-blood-test task has no dataset artifact cache.")
    base_metadata = task.artifact_cache.materialize(model_dir)
    base_metadata.update(
        {
            "model_input_csv": task.model_input_metadata,
            "benchmark_model": cell_fields(row, task.site, task.horizon, model_dir),
        }
    )
    return base_metadata


def fit_cell(
    raw_config: Mapping[str, Mapping[str, object]],
    task: SingleBloodtestTask,
    row: BloodtestModelRow,
    model_dir: Path,
    base_metadata: Mapping[str, object],
    config_dir: Path,
) -> list[dict[str, object]]:
    if task.prepared is None:
        raise RuntimeError("Estimable single-blood-test task has no prepared training data.")
    prepared = feature_subset_prepared_training(task.prepared, row["features"])
    fitted = train_ebm_model(
        prepared,
        raw_config["ebm"],
        {split: risk_score_path(raw_config["benchmark"], model_dir, split) for split in EVALUATION_SPLITS},
        evaluation_cache=task.evaluation_cache,
    )
    write_fitted_survival_run(
        fitted,
        SurvivalRunArtifacts(
            output_dir=model_dir,
            output_config=raw_config["output"],
            config_dir=config_dir,
            base_metadata=base_metadata,
        ),
    )
    results = result_records_for_fields(
        cell_fields(row, task.site, task.horizon, model_dir),
        fitted.split_summaries,
    )
    LOGGER.info(
        "%s %s %s test c-index=%s",
        row["label"],
        task.site["label"],
        task.horizon["label"],
        metric_cell(next(result for result in results if result["split"] == "test")),
    )
    return results


def write_cell_failure(
    task: SingleBloodtestTask,
    row: BloodtestModelRow,
    model_dir: Path,
    error: Exception,
) -> None:
    try:
        write_failed_run(
            feature_subset_table(task.train_table, row["features"]),
            model_dir,
            error,
            {"model_input_csv": task.model_input_metadata},
            None if task.prepared is None else task.prepared.imputer,
        )
    except Exception:
        LOGGER.exception("could not write failed-run metadata; continuing output_dir=%s", model_dir)


def unavailable_results(
    row: BloodtestModelRow,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
    output_dir: Path,
    status: Literal["not_estimable", "failed"],
    reason: str | Exception,
) -> list[dict[str, object]]:
    model_dir = model_output_dir(output_dir, row, site, horizon)
    details = (
        {"not_estimable_reason": str(reason)}
        if status == "not_estimable"
        else failure_details(reason if isinstance(reason, Exception) else RuntimeError(reason))
    )
    return unavailable_results_for_fields(
        cell_fields(row, site, horizon, model_dir),
        status,
        details,
    )


def cell_fields(
    row: BloodtestModelRow,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
    model_dir: Path,
) -> dict[str, object]:
    return {
        "model": "ebm",
        "model_key": row["key"],
        "model_label": row["label"],
        "model_kind": row["kind"],
        "blood_test_key": row["blood_test_key"],
        "feature_names": list(row["features"]),
        "site_label": site["label"],
        "endpoint_ddfs_code_filter_regex": site["endpoint_ddfs_code_filter_regex"],
        "horizon_label": horizon["label"],
        "horizon_days": horizon["days"],
        "output_dir": str(model_dir),
    }


def record_results(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    cell_results: list[dict[str, object]],
    output_dir: Path,
    config_dir: Path,
    washout_state: WashoutState | None = None,
) -> None:
    results.extend(cell_results)
    results[:] = sort_results(benchmark, selected_tests, results, config_dir)
    write_summary_outputs(benchmark, output_dir, selected_tests, results, config_dir)
    record_washout(benchmark, selected_tests, cell_results, output_dir, config_dir, washout_state)


def cell_run_fingerprint(
    raw_config: Mapping[str, Mapping[str, object]],
    task: SingleBloodtestTask,
    row: BloodtestModelRow,
) -> str:
    return mapping_sha256(
        {
            "model": "ebm",
            "model_config": raw_config["ebm"],
            "model_feature_names": row["features"],
            "dataset_config": dataset_config_mapping(task.dataset.config),
            "dataset_input_fingerprint": task.dataset.prepared_data.input_fingerprint,
            "imputation": raw_config["imputation"],
            "output": raw_config["output"],
            "population_statistics": raw_config.get("population_statistics", {}),
            "write_risk_scores": raw_config["benchmark"].get("write_risk_scores", True),
        }
    )


def model_output_dir(
    output_dir: Path,
    row: BloodtestModelRow,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
) -> Path:
    return benchmark_model_output_dir(output_dir, row["key"], site, horizon)


def risk_score_path(benchmark: Mapping[str, object], model_dir: Path, split: str) -> Path | None:
    if not bool(benchmark.get("write_risk_scores", True)):
        return None
    return model_dir / f"{EVALUATION_SPLITS[split]}_risk_scores.csv"


def resolve_output_dir(benchmark: Mapping[str, object], config_dir: Path) -> Path:
    return resolve_path(str(benchmark.get("output_dir", "../../output/ebm_single_bloodtest_benchmark")), config_dir)


if __name__ == "__main__":
    main()
