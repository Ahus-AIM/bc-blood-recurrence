from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from src.model.cindex_benchmark import emphasize_metric_cell, metric_values, write_benchmark_artifacts
from src.model.complete_case_brier import METRIC
from src.script.single_bloodtest_benchmark_reporting import (
    CLINICAL_BASELINE_MODEL_KEY,
    add_rank_emphasis,
    finite_bootstrap_metrics,
    paired_metric_differences,
)
from src.script.single_bloodtest_washout import washout_table

CAPTION = (
    "Complete-case Brier score at 180 days by endpoint site and pre-boundary blood-test washout. "
    "Lower is better; two-sided 95\\% bootstrap percentile intervals condition on fixed fitted models. "
    "In each column, bold indicates the lowest mean score and underline the second-lowest distinct mean score. "
    "A blood-test model is italicized when the one-sided 95\\% upper bootstrap bound of its paired score "
    "difference from the clinical baseline is below zero. The clinical baseline is never italicized. "
    "Draws match the C-index evaluation, including draws without target events. "
    "Early-censored outcomes are omitted; these complete-case scores may be biased by censoring."
)


def with_clinical_reference(results: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    references = {
        (result["site_label"], result["horizon_label"], result["washout_days"]): result
        for result in results
        if result["model_key"] == CLINICAL_BASELINE_MODEL_KEY
    }
    reported = []
    for result in results:
        reference = references.get((result["site_label"], result["horizon_label"], result["washout_days"]))
        updated = dict(result)
        updated["clinical_reference_status"] = "unavailable" if reference is None else reference["status"]
        updated["clinical_reference_complete_case_brier_score"] = None if reference is None else reference.get(METRIC)
        differences = ()
        if reference is not None and reference["status"] == "complete" and result["status"] == "complete":
            differences = paired_metric_differences(
                dict(finite_bootstrap_metrics(result["runs"], METRIC)),
                dict(finite_bootstrap_metrics(reference["runs"], METRIC)),
            )
        updated["paired_reference_run_count"] = len(differences)
        upper_bound = None if not differences else float(np.percentile(differences, 95))
        updated["table_italic"] = (
            result["model_key"] != CLINICAL_BASELINE_MODEL_KEY
            and metric_values(result, METRIC) is not None
            and upper_bound is not None
            and upper_bound < 0
        )
        for name, value in (
            ("mean", None if not differences else float(np.mean(differences))),
            ("ci95_lower", None if not differences else float(np.percentile(differences, 2.5))),
            ("ci95_upper", None if not differences else float(np.percentile(differences, 97.5))),
            ("one_sided_ci95_upper", upper_bound),
        ):
            updated[f"complete_case_brier_difference_from_clinical_{name}"] = value
        reported.append(updated)
    add_rank_emphasis(
        reported, metric=METRIC, lower_is_better=True,
        column_fields=("site_label", "horizon_label", "washout_days"),
    )
    return reported


def write_brier_outputs(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: Sequence[Mapping[str, object]],
    destination: Path,
    config_dir: Path,
) -> dict[str, str]:
    """Write Brier tables and metadata, returning their generated paths."""
    reported = with_clinical_reference(results)
    metadata = {
        "metric": "complete-case Brier score",
        "censoring_limitation": "Complete-case Brier scores may be biased by censoring.",
        "ipcw": False,
        "sampling": "same patient-then-one-landmark draws as C-index, before its zero-event exclusion",
        "washout_order": "filter pool, recalculate patient selection probabilities, then reuse C-index draws",
        "prediction_policy": "fixed full-pool predictions; no refitting during washouts or bootstrap",
        "selected_blood_tests": selected_tests,
        "results": reported,
    }
    return write_benchmark_artifacts(
        destination,
        {
            "output_csv_path": "complete_case_brier_washout.csv",
            "output_latex_path": "complete_case_brier_washout.tex",
            "output_metadata_path": "metadata.json",
        },
        [{key: value for key, value in result.items() if key != "runs"} for result in reported],
        metadata,
        brier_table(benchmark, selected_tests, reported, config_dir),
        allow_nan=False,
    )


def brier_table(
    benchmark: Mapping[str, object],
    selected_tests: tuple[str, ...],
    results: Sequence[Mapping[str, object]],
    config_dir: Path,
) -> str:
    configured = {**benchmark, "table_caption": CAPTION, "table_label": "tab:complete_case_brier_washout"}
    labeled = [{**result, "horizon_label": f"{result['horizon_label']} {result['washout_days']}d"} for result in results]
    return washout_table(configured, selected_tests, labeled, config_dir, format_cell=brier_cell)


def brier_cell(result: Mapping[str, object] | None) -> str:
    if result is None or result["status"] != "complete":
        return "--"
    values = metric_values(result, METRIC)
    if values is None:
        return "--"
    cell = f"{values[0]:.3f} [{values[1]:.3f}, {values[2]:.3f}]"
    return emphasize_metric_cell(cell, result)
