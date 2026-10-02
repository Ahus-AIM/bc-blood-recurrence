import json
from pathlib import Path

import pandas as pd
import pytest

from src.model.cindex_benchmark import metric_cell
from src.script.single_bloodtest_benchmark_reporting import (
    ALL_MODEL_KEY,
    CLINICAL_BASELINE_MODEL_KEY,
    latex_table,
    rendered_latex_table,
    with_test_inference,
    write_summary_outputs,
)


def test_latex_table_uses_dynamic_columns_and_percent_ci(tmp_path: Path) -> None:
    benchmark = {
        "table_caption": "Caption",
        "table_label": "tab:test",
        "include_all_blood_tests": True,
        "all_blood_tests_label": "All",
        "sites": ({"label": "All"}, {"label": "Liver"}),
        "horizons": ({"label": "1m"}, {"label": "1y"}),
    }
    results = [
        result("B-Hb|||g/dL", "All", "1m", 0.856, 0.778, 0.942),
        result(ALL_MODEL_KEY, "Liver", "1y", 0.5, 0.4, 0.6),
    ]

    latex = rendered_latex_table(benchmark, ("B-Hb|||g/dL",), results, tmp_path)

    assert r"\begin{tabular}{lcc|cc}" in latex
    assert r"\multicolumn{2}{c|}{Any}" in latex
    assert "& 1m & 1y & 1m & 1y" in latex
    assert "B-Hb & 85.6 [77.8, 94.2] & -- & -- & --" in latex
    assert "g/dL" not in latex
    assert "All blood tests & -- & -- & -- & 50.0 [40.0, 60.0]" in latex


def test_latex_table_uses_rank_and_combined_inference_emphasis(
    tmp_path: Path,
) -> None:
    benchmark = {
        "table_caption": "Caption",
        "table_label": "tab:test",
        "include_all_blood_tests": True,
        "sites": ({"label": "All"},),
        "horizons": ({"label": "1m"},),
    }
    results = [
        result(CLINICAL_BASELINE_MODEL_KEY, "All", "1m", 0.56, 0.50, 0.62, tmp_path),
        result("A|||u", "All", "1m", 0.59, 0.53, 0.65, tmp_path),
        result("B|||u", "All", "1m", 0.54, 0.51, 0.57, tmp_path),
        result(ALL_MODEL_KEY, "All", "1m", 0.55, 0.49, 0.61, tmp_path),
    ]
    write_test_runs(tmp_path, CLINICAL_BASELINE_MODEL_KEY, [(0, 0.55), (1, 0.56), (2, 0.57)])
    write_test_runs(tmp_path, "A|||u", [(0, 0.58), (1, 0.59), (2, 0.60)])
    write_test_runs(tmp_path, "B|||u", [(0, 0.53), (1, 0.54), (2, 0.55)])
    write_test_runs(tmp_path, ALL_MODEL_KEY, [(0, 0.49), (1, 0.55), (2, 0.61)])

    latex = latex_table(benchmark, ("A|||u", "B|||u"), results, tmp_path)

    assert r"Clinical baseline & \underline{56.0 [50.0, 62.0]}" in latex
    assert r"A & \textbf{\textit{59.0 [53.0, 65.0]}}" in latex
    assert r"B & 54.0 [51.0, 57.0]" in latex
    assert "All blood tests & 55.0 [49.0, 61.0]" in latex


def test_test_inference_pairs_only_matching_finite_bootstrap_runs(
    tmp_path: Path,
) -> None:
    results = [
        result(CLINICAL_BASELINE_MODEL_KEY, "All", "1m", 0.55, 0.45, 0.65, tmp_path),
        result("A|||u", "All", "1m", 0.60, 0.50, 0.70, tmp_path),
    ]
    write_test_runs(
        tmp_path,
        CLINICAL_BASELINE_MODEL_KEY,
        [(0, 0.50), (1, None), (2, 0.55)],
    )
    write_test_runs(tmp_path, "A|||u", [(0, 0.60), (1, 0.70), (3, 0.80)])

    reported = with_test_inference(results)
    baseline, model = reported

    assert baseline["harrell_c_index_chance_difference_valid_runs"] == 2
    assert baseline[
        "harrell_c_index_chance_difference_one_sided_ci95_lower"
    ] == pytest.approx(0.0025)
    assert baseline["table_bold"] is False
    assert baseline["table_underline"] is True
    assert baseline["table_italic"] is False
    assert model["harrell_c_index_baseline_difference_valid_runs"] == 1
    assert model[
        "harrell_c_index_baseline_difference_one_sided_ci95_lower"
    ] == pytest.approx(0.1)
    assert model["table_bold"] is True
    assert model["table_underline"] is False
    assert model["table_italic"] is True


def test_test_inference_rejects_missing_run_metadata(
    tmp_path: Path,
) -> None:
    results = [
        result(CLINICAL_BASELINE_MODEL_KEY, "All", "1m", 0.60, 0.55, 0.65, tmp_path),
        result("A|||u", "All", "1m", 0.70, 0.65, 0.75, tmp_path),
    ]

    with pytest.raises(ValueError, match="no finite bootstrap C-index values"):
        with_test_inference(results)


def test_test_inference_rejects_no_matching_baseline_runs(tmp_path: Path) -> None:
    results = [
        result(CLINICAL_BASELINE_MODEL_KEY, "All", "1m", 0.60, 0.55, 0.65, tmp_path),
        result("A|||u", "All", "1m", 0.70, 0.65, 0.75, tmp_path),
    ]
    write_test_runs(tmp_path, CLINICAL_BASELINE_MODEL_KEY, [(0, 0.60)])
    write_test_runs(tmp_path, "A|||u", [(1, 0.70)])

    with pytest.raises(ValueError, match="no matching finite clinical-baseline bootstrap runs"):
        with_test_inference(results)


def test_rank_emphasis_uses_full_precision_and_distinct_ties(tmp_path: Path) -> None:
    results = [
        result(CLINICAL_BASELINE_MODEL_KEY, "All", "1m", 0.60004, 0.55, 0.65, tmp_path),
        result("A|||u", "All", "1m", 0.60005, 0.55, 0.65, tmp_path),
        result("B|||u", "All", "1m", 0.59, 0.54, 0.64, tmp_path),
        result(ALL_MODEL_KEY, "All", "1m", 0.60005, 0.55, 0.65, tmp_path),
    ]
    write_test_runs(tmp_path, CLINICAL_BASELINE_MODEL_KEY, [(0, 0.55)])
    write_test_runs(tmp_path, "A|||u", [(0, 0.60)])
    write_test_runs(tmp_path, "B|||u", [(0, 0.59)])
    write_test_runs(tmp_path, ALL_MODEL_KEY, [(0, 0.61)])

    baseline, first, third, tied_first = with_test_inference(results)

    assert first["table_bold"] is True
    assert tied_first["table_bold"] is True
    assert baseline["table_underline"] is True
    assert third["table_bold"] is False
    assert third["table_underline"] is False


def test_failed_test_result_needs_no_inference_evidence(tmp_path: Path) -> None:
    failed = {
        **result("A|||u", "All", "1m", 0.70, 0.65, 0.75, tmp_path),
        "status": "failed",
    }

    [reported] = with_test_inference([failed])

    assert reported["table_bold"] is False
    assert reported["table_underline"] is False
    assert reported["table_italic"] is False


def test_summary_outputs_store_test_inference_but_not_other_splits(
    tmp_path: Path,
) -> None:
    benchmark = {
        "table_caption": "Caption",
        "table_label": "tab:test",
        "include_all_blood_tests": False,
        "sites": ({"label": "All"},),
        "horizons": ({"label": "1m"},),
    }
    baseline = result(
        CLINICAL_BASELINE_MODEL_KEY,
        "All",
        "1m",
        0.56,
        0.50,
        0.62,
        tmp_path,
    )
    model = result("A|||u", "All", "1m", 0.59, 0.53, 0.65, tmp_path)
    validation = {**model, "split": "validation"}
    write_test_runs(tmp_path, CLINICAL_BASELINE_MODEL_KEY, [(0, 0.55), (1, 0.57)])
    write_test_runs(tmp_path, "A|||u", [(0, 0.58), (1, 0.60)])

    write_summary_outputs(
        benchmark,
        tmp_path,
        ("A|||u",),
        [baseline, model, validation],
        tmp_path,
    )

    csv_rows = pd.read_csv(tmp_path / "cindex_benchmark.csv")
    test_model = csv_rows[
        (csv_rows["model_key"] == "A|||u") & (csv_rows["split"] == "test")
    ].iloc[0]
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))
    validation_metadata = next(
        result for result in metadata["results"] if result["split"] == "validation"
    )

    assert test_model["harrell_c_index_baseline_difference_valid_runs"] == 2
    assert test_model["harrell_c_index_baseline_difference_one_sided_ci95_lower"] == pytest.approx(0.03)
    assert bool(test_model["table_bold"])
    assert not bool(test_model["table_underline"])
    assert bool(test_model["table_italic"])
    assert "table_bold" not in validation_metadata
    assert "table_underline" not in validation_metadata


def test_metric_cell_uses_dash_for_missing_metric() -> None:
    assert metric_cell({"harrell_c_index": float("nan")}) == "--"


def test_metric_cell_combines_underline_and_italics() -> None:
    assert metric_cell(
        {
            "harrell_c_index": 0.6,
            "harrell_c_index_ci95_lower": 0.55,
            "harrell_c_index_ci95_upper": 0.65,
            "table_underline": True,
            "table_italic": True,
        }
    ) == r"\underline{\textit{60.0 [55.0, 65.0]}}"


def result(
    model_key: str,
    site_label: str,
    horizon_label: str,
    c_index: float,
    lower: float,
    upper: float,
    output_root: Path | None = None,
) -> dict[str, object]:
    output_dir = None if output_root is None else output_root / model_key
    return {
        "model_key": model_key,
        "site_label": site_label,
        "horizon_label": horizon_label,
        "split": "test",
        "status": "complete",
        "output_dir": None if output_dir is None else str(output_dir),
        "harrell_c_index": c_index,
        "harrell_c_index_ci95_lower": lower,
        "harrell_c_index_ci95_upper": upper,
    }


def write_test_runs(
    output_root: Path,
    model_key: str,
    runs: list[tuple[int, float | None]],
) -> None:
    output_dir = output_root / model_key
    output_dir.mkdir(parents=True)
    metadata = {
        "test": {
            "runs": [
                {"bootstrap_run": bootstrap_run, "harrell_c_index": c_index}
                for bootstrap_run, c_index in runs
            ]
        }
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )
