from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd

from src.model.survival_evaluation import EVALUATION_SPLITS
from src.reporting import latex_escape, recurrence_endpoint_label, report_output_path

CellFormatter = Callable[[Mapping[str, object] | None], str]

CINDEX_SUMMARY_KEYS = (
    "harrell_c_index",
    "harrell_c_index_ci95_lower",
    "harrell_c_index_ci95_upper",
    "harrell_c_index_valid_runs",
    "bootstrap_runs",
    "bootstrap_sample_count",
    "event_count_mean",
    "patient_count_mean",
    "row_count_mean",
)


def cindex_summary_fields(summary: Mapping[str, object]) -> dict[str, object]:
    """Project an evaluation summary onto the shared benchmark result schema."""
    return {key: summary[key] for key in CINDEX_SUMMARY_KEYS}


def result_records_for_fields(
    fields: Mapping[str, object],
    split_summaries: Mapping[str, Mapping[str, object]],
) -> list[dict[str, object]]:
    """Build successful or split-level non-estimable result records."""
    return [result_record_for_fields(fields, split, summary) for split, summary in split_summaries.items()]


def result_record_for_fields(
    fields: Mapping[str, object],
    split: str,
    summary: Mapping[str, object],
) -> dict[str, object]:
    status = "complete" if math.isfinite(float(summary["harrell_c_index"])) else "not_estimable"
    return {
        **fields,
        "split": split,
        "status": status,
        "not_estimable_reason": None if status == "complete" else "no comparable patient pairs",
        **cindex_summary_fields(summary),
    }


def unavailable_results_for_fields(
    fields: Mapping[str, object],
    status: Literal["not_estimable", "failed"],
    details: Mapping[str, object],
) -> list[dict[str, object]]:
    """Build empty metric records for a run-level failure or non-estimability."""
    return [
        {
            **fields,
            "split": split,
            "status": status,
            **details,
            **dict.fromkeys(CINDEX_SUMMARY_KEYS),
        }
        for split in EVALUATION_SPLITS.values()
    ]


@dataclass(frozen=True)
class CIndexTableRow:
    """Describe one C-index table row.

    ``key`` locates the row's results and ``label`` is displayed in LaTeX.
    ``separator_before`` adds a double rule before the row.
    """

    key: str
    label: str
    separator_before: bool = False


@dataclass(frozen=True)
class CIndexTableLabel:
    """One site or horizon key; endpoint names are translated only for display."""

    label: str


def latex_cindex_table(
    benchmark: Mapping[str, object],
    sites: Sequence[CIndexTableLabel],
    horizons: Sequence[CIndexTableLabel],
    rows: Sequence[CIndexTableRow],
    results: Mapping[tuple[str, str, str], Mapping[str, object]],
    *,
    format_cell: CellFormatter | None = None,
) -> str:
    """Render benchmark axes with C-index cells or a supplied metric formatter."""
    render_cell = metric_cell if format_cell is None else format_cell
    lines = table_header(benchmark, sites, horizons)
    for row in rows:
        if row.separator_before:
            lines.extend((r"\hline", r"\hline"))
        cells = [
            render_cell(results.get((row.key, site.label, horizon.label))) for site in sites for horizon in horizons
        ]
        lines.append(f"{latex_escape(row.label)} & " + " & ".join(cells) + r" \\")
    lines.extend((r"\hline", r"\end{tabular}%", "}", r"\end{table*}"))
    return "\n".join(lines) + "\n"


def write_benchmark_artifacts(
    output_dir: Path,
    benchmark: Mapping[str, object],
    results: Sequence[Mapping[str, object]],
    metadata: Mapping[str, object],
    latex: str,
    *,
    allow_nan: bool = True,
) -> dict[str, str]:
    """Write the common CSV, LaTeX, and metadata benchmark artifacts."""
    paths = {
        "csv": report_output_path(output_dir, benchmark.get("output_csv_path", "cindex_benchmark.csv")),
        "latex": report_output_path(
            output_dir,
            benchmark.get("output_latex_path", "cindex_benchmark.tex"),
        ),
        "metadata": report_output_path(output_dir, benchmark.get("output_metadata_path", "metadata.json")),
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(paths["csv"], index=False)
    paths["latex"].write_text(latex, encoding="utf-8")
    paths["metadata"].write_text(
        json.dumps(dict(metadata), indent=2, sort_keys=True, default=str, allow_nan=allow_nan),
        encoding="utf-8",
    )
    return {key: str(path) for key, path in paths.items()}


def table_header(
    benchmark: Mapping[str, object],
    sites: Sequence[CIndexTableLabel],
    horizons: Sequence[CIndexTableLabel],
) -> list[str]:
    metric_column_count = len(sites) * len(horizons)
    return [
        r"\begin{table*}[h]",
        r"\centering",
        rf"\caption{{{benchmark['table_caption']}}}",
        rf"\label{{{benchmark['table_label']}}}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{{latex_columns(len(sites), len(horizons))}}}",
        r"\hline",
        latex_site_header(sites, len(horizons)),
        rf"\cline{{2-{metric_column_count + 1}}}",
        latex_horizon_header(sites, horizons),
        r"\hline",
    ]


def metric_cell(
    summary: Mapping[str, object] | None,
) -> str:
    if summary is None or ("status" in summary and summary.get("status") != "complete"):
        return "--"
    values = metric_values(summary)
    if values is None:
        return "--"
    c_index, lower, upper = values
    cell = f"{100 * c_index:.1f} [{100 * lower:.1f}, {100 * upper:.1f}]"
    return emphasize_metric_cell(cell, summary)


def emphasize_metric_cell(cell: str, summary: Mapping[str, object]) -> str:
    """Apply shared rank and inference markup to an already formatted metric."""
    if summary.get("table_italic") is True:
        cell = rf"\textit{{{cell}}}"
    if summary.get("table_underline") is True:
        cell = rf"\underline{{{cell}}}"
    if summary.get("table_bold") is True:
        cell = rf"\textbf{{{cell}}}"
    return cell


def metric_values(
    summary: Mapping[str, object], metric: str = "harrell_c_index",
) -> tuple[float, float, float] | None:
    values = (
        summary.get(metric),
        summary.get(f"{metric}_ci95_lower"),
        summary.get(f"{metric}_ci95_upper"),
    )
    if not all(isinstance(value, int | float) for value in values):
        return None
    numeric = tuple(float(value) for value in values)
    return numeric if all(math.isfinite(value) for value in numeric) else None


def latex_site_header(
    sites: Sequence[CIndexTableLabel],
    horizon_count: int,
) -> str:
    groups = [
        rf"\multicolumn{{{horizon_count}}}{{c|}}{{{latex_escape(recurrence_endpoint_label(site.label))}}}"
        for site in sites
    ]
    return "& " + "\n& ".join(groups) + r" \\"


def latex_horizon_header(
    sites: Sequence[CIndexTableLabel],
    horizons: Sequence[CIndexTableLabel],
) -> str:
    labels = [latex_escape(horizon.label) for _site in sites for horizon in horizons]
    return "& " + " & ".join(labels) + r"\\"


def latex_columns(site_count: int, horizon_count: int) -> str:
    return "l" + "|".join("c" * horizon_count for _site in range(site_count))
