from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from src.reporting import report_output_path
from src.script.single_bloodtest_benchmark_reporting import latex_table, load_saved_summary


def regenerate_table(
    benchmark: Mapping[str, object],
    output_dir: Path,
    config_dir: Path,
) -> Path:
    """Write only the LaTeX table using a saved aggregate summary."""
    metadata_path = report_output_path(
        output_dir,
        benchmark.get("output_metadata_path", "metadata.json"),
    )
    selected_tests, results = load_saved_summary(metadata_path)
    table_path = report_output_path(
        output_dir,
        benchmark.get(
            "output_latex_path",
            "cindex_benchmark_individual_blood_tests.tex",
        ),
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(
        latex_table(benchmark, selected_tests, results, config_dir),
        encoding="utf-8",
    )
    return table_path
