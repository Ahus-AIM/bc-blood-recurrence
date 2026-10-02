from __future__ import annotations

import json
from pathlib import Path

from src.regeneration.benchmark_table import regenerate_table
from src.script.single_bloodtest_benchmark_reporting import (
    CLINICAL_BASELINE_MODEL_KEY,
)


def test_regenerate_table_writes_only_latex_from_saved_results(tmp_path: Path) -> None:
    benchmark = {
        "table_caption": "Caption",
        "table_label": "tab:test",
        "output_metadata_path": "metadata.json",
        "output_latex_path": "table.tex",
        "include_all_blood_tests": False,
        "sites": ({"label": "All"},),
        "horizons": ({"label": "1m"},),
    }
    baseline_dir = write_model_metadata(tmp_path, "baseline", [(0, 0.55), (1, 0.56)])
    model_dir = write_model_metadata(tmp_path, "model", [(0, 0.58), (1, 0.59)])
    metadata = {
        "selected_blood_tests": ["A|||u"],
        "results": [
            result(CLINICAL_BASELINE_MODEL_KEY, 0.56, baseline_dir),
            result("A|||u", 0.59, model_dir),
        ],
    }
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    original_metadata = metadata_path.read_bytes()

    table_path = regenerate_table(benchmark, tmp_path, tmp_path)

    latex = table_path.read_text(encoding="utf-8")
    assert r"Clinical baseline & \underline{56.0 [51.0, 61.0]}" in latex
    assert r"A & \textbf{\textit{59.0 [54.0, 64.0]}}" in latex
    assert metadata_path.read_bytes() == original_metadata
    assert not (tmp_path / "cindex_benchmark.csv").exists()


def result(model_key: str, c_index: float, output_dir: Path) -> dict[str, object]:
    return {
        "model_key": model_key,
        "site_label": "All",
        "horizon_label": "1m",
        "split": "test",
        "status": "complete",
        "output_dir": str(output_dir),
        "harrell_c_index": c_index,
        "harrell_c_index_ci95_lower": c_index - 0.05,
        "harrell_c_index_ci95_upper": c_index + 0.05,
    }


def write_model_metadata(
    output_root: Path,
    name: str,
    runs: list[tuple[int, float]],
) -> Path:
    output_dir = output_root / name
    output_dir.mkdir()
    metadata = {
        "test": {
            "runs": [
                {"bootstrap_run": bootstrap_run, "harrell_c_index": c_index}
                for bootstrap_run, c_index in runs
            ]
        }
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return output_dir
