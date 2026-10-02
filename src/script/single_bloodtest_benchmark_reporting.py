from __future__ import annotations

import json
import logging
import math
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np

from src.data.blood_test_display import display_title_for_test_key, load_test_name_translation
from src.data.blood_test_features import blood_test_feature_names
from src.data.dataset_config import PatientDatasetConfig, optional_path
from src.data.landmarks import TIME_SINCE_DIAGNOSIS_FEATURE_NAME
from src.model.cindex_benchmark import (
    CIndexTableLabel,
    CIndexTableRow,
    CellFormatter,
    latex_cindex_table,
    metric_values,
    write_benchmark_artifacts,
)
from src.model.survival_evaluation import EVALUATION_SPLITS
from src.script.single_bloodtest_benchmark_config import mapping_sequence

ALL_MODEL_KEY = "__all_blood_tests__"
CLINICAL_BASELINE_MODEL_KEY = "__clinical_baseline__"
CHANCE_C_INDEX = 0.5
ONE_SIDED_CI95_LOWER_PERCENTILE = 5.0
LOGGER = logging.getLogger(__name__)


class BloodtestModelRow(TypedDict):
    """Stable key, label, exact feature subset, row kind, and optional test key."""

    key: str
    label: str
    features: tuple[str, ...]
    kind: Literal["clinical_baseline", "single_blood_test", "all_blood_tests"]
    blood_test_key: str | None


def bloodtest_rows(
    selected_tests: tuple[str, ...],
    benchmark: Mapping[str, object],
    config_dir: Path,
    *,
    context_features: tuple[str, ...] = (),
    include_blood_test_change: bool = False,
) -> list[BloodtestModelRow]:
    translations = load_test_name_translation(
        optional_path(benchmark.get("bloodtest_translation_csv_path"), config_dir)
    )
    rows: list[BloodtestModelRow] = [
        {
            "key": CLINICAL_BASELINE_MODEL_KEY,
            "label": str(benchmark.get("clinical_baseline_label", "Clinical baseline")),
            "features": context_features,
            "kind": "clinical_baseline",
            "blood_test_key": None,
        }
    ]
    rows.extend(
        {
            "key": test_key,
            "label": display_title_for_test_key(test_key, translations),
            "features": (*blood_test_feature_names((test_key,), include_blood_test_change), *context_features),
            "kind": "single_blood_test",
            "blood_test_key": test_key,
        }
        for test_key in selected_tests
    )
    if bool(benchmark.get("include_all_blood_tests", True)):
        all_tests_label = str(benchmark.get("all_blood_tests_label", "All blood tests"))
        rows.append(
            {
                "key": ALL_MODEL_KEY,
                "label": "All blood tests" if all_tests_label.casefold() == "all" else all_tests_label,
                "features": (*blood_test_feature_names(selected_tests, include_blood_test_change), *context_features),
                "kind": "all_blood_tests",
                "blood_test_key": None,
            }
        )
    return rows


def model_context_features(config: PatientDatasetConfig) -> tuple[str, ...]:
    """Return enabled patient-context features in landmark-table order."""
    features = []
    if config.include_time_since_diagnosis:
        features.append(TIME_SINCE_DIAGNOSIS_FEATURE_NAME)
    if config.include_diagnosis_age:
        features.append("age_at_diagnosis")
    return tuple(features)


def sort_results(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    config_dir: Path,
) -> list[dict[str, object]]:
    rows = bloodtest_rows(selected_tests, benchmark, config_dir)
    orders = (
        {str(row["key"]): index for index, row in enumerate(rows)},
        {str(site["label"]): index for index, site in enumerate(mapping_sequence(benchmark["sites"], "sites"))},
        {
            str(horizon["label"]): index
            for index, horizon in enumerate(mapping_sequence(benchmark["horizons"], "horizons"))
        },
        {split: index for index, split in enumerate(EVALUATION_SPLITS.values())},
    )
    keys = ("model_key", "site_label", "horizon_label", "split")
    return sorted(
        results,
        key=lambda result: tuple(order[str(result[key])] for order, key in zip(orders, keys, strict=True)),
    )


def write_summary_outputs(
    benchmark: Mapping[str, object],
    output_dir: Path,
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    config_dir: Path,
) -> None:
    reported_results = with_test_inference(results)
    write_benchmark_artifacts(
        output_dir,
        benchmark,
        reported_results,
        {"selected_blood_tests": selected_tests, "results": reported_results},
        rendered_latex_table(
            benchmark,
            selected_tests,
            reported_results,
            config_dir,
        ),
    )


def load_saved_summary(
    metadata_path: Path,
) -> tuple[tuple[str, ...], list[dict[str, object]]]:
    """Load selected tests and result records from aggregate metadata."""
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError(f"Benchmark metadata must contain a mapping: {metadata_path}")
    selected_tests = payload.get("selected_blood_tests")
    results = payload.get("results")
    if not isinstance(selected_tests, list) or not all(isinstance(value, str) for value in selected_tests):
        raise TypeError(f"Benchmark metadata has invalid selected_blood_tests: {metadata_path}")
    if not isinstance(results, list) or not all(isinstance(value, Mapping) for value in results):
        raise TypeError(f"Benchmark metadata has invalid results: {metadata_path}")
    return tuple(selected_tests), [dict(value) for value in results]


def merge_result_records(
    previous: Sequence[Mapping[str, object]], updated: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Replace selected cells/splits/washouts without truncating the saved benchmark."""
    fields = ("site_label", "horizon_label", "model_key", "split", "washout_days")
    merged = {tuple(row.get(key) for key in fields): dict(row) for row in previous}
    merged.update({tuple(row.get(key) for key in fields): dict(row) for row in updated})
    return list(merged.values())


def latex_table(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    config_dir: Path,
) -> str:
    return rendered_latex_table(
        benchmark,
        selected_tests,
        with_test_inference(results),
        config_dir,
    )


def rendered_latex_table(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: list[dict[str, object]],
    config_dir: Path,
    *,
    format_cell: CellFormatter | None = None,
) -> str:
    """Render baseline, blood tests sorted A-Z by label, then the all-tests row."""
    sites = tuple(CIndexTableLabel(str(site["label"])) for site in mapping_sequence(benchmark["sites"], "sites"))
    horizons = tuple(
        CIndexTableLabel(str(horizon["label"])) for horizon in mapping_sequence(benchmark["horizons"], "horizons")
    )
    rows = bloodtest_rows(selected_tests, benchmark, config_dir)
    row_kind_order = {"clinical_baseline": 0, "single_blood_test": 1, "all_blood_tests": 2}
    rows.sort(key=lambda row: (row_kind_order[row["kind"]], row["label"].casefold()))
    result_by_cell = {
        (str(result["model_key"]), str(result["site_label"]), str(result["horizon_label"])): result
        for result in results
        if result.get("split", "test") == "test"
    }
    table_rows = tuple(
        CIndexTableRow(
            key=str(row["key"]),
            label=str(row["label"]),
            separator_before=row["kind"] == "all_blood_tests",
        )
        for row in rows
    )
    return latex_cindex_table(benchmark, sites, horizons, table_rows, result_by_cell, format_cell=format_cell)


def with_test_inference(
    results: list[dict[str, object]],
) -> list[dict[str, object]]:
    """Add test-only rank emphasis and one-sided inference fields."""
    reported = [dict(result) for result in results]
    test_results = [result for result in reported if result.get("split") == "test"]
    runs_by_cell = {
        result_cell_key(result): test_bootstrap_c_indexes(result)
        for result in test_results
    }
    for result in test_results:
        runs = runs_by_cell[result_cell_key(result)]
        result.update(difference_inference_fields("chance", tuple(value - CHANCE_C_INDEX for value in runs.values())))
        baseline_runs = runs_by_cell.get(baseline_cell_key(result), {})
        paired_differences = (
            ()
            if result.get("model_key") == CLINICAL_BASELINE_MODEL_KEY
            else paired_metric_differences(runs, baseline_runs)
        )
        result.update(difference_inference_fields("baseline", paired_differences))
        validate_inference_evidence(result, runs, paired_differences)
        result["table_italic"] = passes_italic_rules(result)
    add_rank_emphasis(test_results)
    return reported


def validate_inference_evidence(
    result: Mapping[str, object],
    runs: Mapping[int, float],
    paired_differences: tuple[float, ...],
) -> None:
    """Fail when a displayed result lacks evidence needed for italics."""
    if metric_values(result) is None or result.get("status") != "complete":
        return
    context = inference_context(result)
    if not runs:
        raise ValueError(f"Complete test result has no finite bootstrap C-index values: {context}.")
    if result.get("model_key") == CLINICAL_BASELINE_MODEL_KEY:
        return
    if not paired_differences:
        raise ValueError(
            "Complete test result has no matching finite clinical-baseline "
            f"bootstrap runs: {context}."
        )


def inference_context(result: Mapping[str, object]) -> str:
    return (
        f"model={result.get('model_key')}, site={result.get('site_label')}, "
        f"horizon={result.get('horizon_label')}"
    )


def passes_italic_rules(result: Mapping[str, object]) -> bool:
    if result.get("model_key") == CLINICAL_BASELINE_MODEL_KEY:
        return False
    return positive_lower_bound(result, "chance") and positive_lower_bound(result, "baseline")


def add_rank_emphasis(
    test_results: list[dict[str, object]],
    *,
    metric: str = "harrell_c_index",
    lower_is_better: bool = False,
    column_fields: tuple[str, ...] = ("site_label", "horizon_label"),
) -> None:
    """Mark best and second-best distinct full-precision means within each column."""
    results_by_column: dict[tuple[str, ...], list[dict[str, object]]] = {}
    for result in test_results:
        result["table_bold"] = False
        result["table_underline"] = False
        if result.get("status") == "complete" and metric_values(result, metric) is not None:
            key = tuple(str(result[field]) for field in column_fields)
            results_by_column.setdefault(key, []).append(result)

    for column_results in results_by_column.values():
        ranked_values = sorted(
            {float(result[metric]) for result in column_results},
            reverse=not lower_is_better,
        )
        set_rank_emphasis(column_results, ranked_values, metric=metric)


def set_rank_emphasis(
    results: list[dict[str, object]],
    ranked_values: list[float],
    *,
    metric: str = "harrell_c_index",
) -> None:
    best = ranked_values[0]
    second_best = ranked_values[1] if len(ranked_values) > 1 else None
    for result in results:
        value = float(result[metric])
        result["table_bold"] = value == best
        result["table_underline"] = second_best is not None and value == second_best


def result_cell_key(result: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        str(result["model_key"]),
        str(result["site_label"]),
        str(result["horizon_label"]),
    )


def baseline_cell_key(result: Mapping[str, object]) -> tuple[str, str, str]:
    return (
        CLINICAL_BASELINE_MODEL_KEY,
        str(result["site_label"]),
        str(result["horizon_label"]),
    )


def test_bootstrap_c_indexes(result: Mapping[str, object]) -> dict[int, float]:
    if result.get("status") != "complete":
        return {}
    if isinstance(result.get("runs"), list):
        return dict(finite_bootstrap_metrics(result["runs"]))
    output_dir = result.get("output_dir")
    if not isinstance(output_dir, str):
        return {}
    return dict(load_test_bootstrap_c_indexes(Path(output_dir) / "metadata.json"))


@lru_cache(maxsize=None)
def load_test_bootstrap_c_indexes(
    metadata_path: Path,
) -> tuple[tuple[int, float], ...]:
    """Load finite test C-index values by bootstrap run from model metadata."""
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        LOGGER.warning("could not load benchmark inference metadata %s: %s", metadata_path, error)
        return ()
    if not isinstance(metadata, Mapping):
        return ()
    test_summary = metadata.get("test")
    if not isinstance(test_summary, Mapping):
        return ()
    return finite_bootstrap_metrics(test_summary.get("runs"))


def finite_bootstrap_metrics(runs: object, metric: str = "harrell_c_index") -> tuple[tuple[int, float], ...]:
    """Return finite values keyed by run ID for paired metric comparisons."""
    if not isinstance(runs, list):
        return ()
    values: dict[int, float] = {}
    for run in runs:
        if not isinstance(run, Mapping):
            continue
        bootstrap_run = run.get("bootstrap_run")
        value = run.get(metric)
        if not isinstance(bootstrap_run, int) or not isinstance(value, int | float):
            continue
        numeric = float(value)
        if math.isfinite(numeric):
            values[bootstrap_run] = numeric
    return tuple(sorted(values.items()))


def paired_metric_differences(
    model_runs: Mapping[int, float],
    baseline_runs: Mapping[int, float],
) -> tuple[float, ...]:
    matching_runs = sorted(model_runs.keys() & baseline_runs.keys())
    return tuple(model_runs[run] - baseline_runs[run] for run in matching_runs)


def difference_inference_fields(
    comparison: Literal["chance", "baseline"],
    differences: tuple[float, ...],
) -> dict[str, object]:
    prefix = f"harrell_c_index_{comparison}_difference"
    mean = None
    lower = None
    if differences:
        mean = float(np.mean(differences))
        lower = float(np.percentile(differences, ONE_SIDED_CI95_LOWER_PERCENTILE))
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_one_sided_ci95_lower": lower,
        f"{prefix}_valid_runs": len(differences),
    }


def positive_lower_bound(
    result: Mapping[str, object],
    comparison: Literal["chance", "baseline"],
) -> bool:
    value = result.get(
        f"harrell_c_index_{comparison}_difference_one_sided_ci95_lower"
    )
    return isinstance(value, int | float) and math.isfinite(float(value)) and float(value) > 0.0
