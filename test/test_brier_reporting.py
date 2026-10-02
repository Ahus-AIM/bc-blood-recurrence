from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.model.complete_case_brier import METRIC
from src.script.brier_reporting import brier_cell, with_clinical_reference, write_brier_outputs
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY, CLINICAL_BASELINE_MODEL_KEY

UPPER_BOUND = "complete_case_brier_difference_from_clinical_one_sided_ci95_upper"


def result(
    model_key: str,
    score: float,
    runs: tuple[tuple[int, float | None], ...] = (),
    *,
    washout: int = 0,
) -> dict[str, object]:
    return {
        "model_key": model_key,
        "site_label": "Liver",
        "horizon_label": "6m",
        "washout_days": washout,
        "split": "test",
        "status": "complete",
        METRIC: score,
        f"{METRIC}_ci95_lower": score - 0.01,
        f"{METRIC}_ci95_upper": score + 0.01,
        "runs": [{"bootstrap_run": run, METRIC: value} for run, value in runs],
    }


def test_brier_ranks_use_full_precision_distinct_ties_and_separate_columns() -> None:
    rows = [
        result(CLINICAL_BASELINE_MODEL_KEY, 0.20005),
        result("A|||u", 0.20004),
        result("B|||u", 0.3),
        result(ALL_MODEL_KEY, 0.20004),
        result(CLINICAL_BASELINE_MODEL_KEY, 0.1, washout=30),
        result("A|||u", 0.2, washout=30),
        {**result("A|||u", 0.4), "site_label": "Lung"},
        {**result("A|||u", 0.5), "horizon_label": "other"},
    ]
    reported = with_clinical_reference(rows)
    assert [row["table_bold"] for row in reported] == [False, True, False, True, True, False, True, True]
    assert [row["table_underline"] for row in reported] == [True, False, False, False, False, True, False, False]
    assert all(not row["table_italic"] for row in reported)
    assert all("table_bold" not in row for row in rows)


@pytest.mark.parametrize(
    "differences,italic",
    [([-0.1] * 19 + [0.1], True), ([0.0] * 20, False), ([-0.1] * 18 + [0.1] * 2, False)],
)
def test_brier_italics_use_strict_one_sided_95_percent_bound(differences: list[float], italic: bool) -> None:
    baseline_runs = tuple((run, 0.3) for run in range(len(differences)))
    model_runs = tuple((run, 0.3 + difference) for run, difference in enumerate(differences))
    baseline, model = with_clinical_reference([
        result(CLINICAL_BASELINE_MODEL_KEY, 0.3, baseline_runs),
        result("A|||u", 0.3 + float(np.mean(differences)), model_runs),
    ])
    assert model[UPPER_BOUND] == pytest.approx(np.percentile(differences, 95))
    assert model["table_italic"] is italic
    assert baseline["table_italic"] is False
    if italic:
        assert model["complete_case_brier_difference_from_clinical_ci95_upper"] > 0


def test_brier_inference_uses_only_matching_finite_run_ids() -> None:
    _, model = with_clinical_reference([
        result(CLINICAL_BASELINE_MODEL_KEY, 0.3, ((0, 0.4), (1, None), (2, 0.1), (4, 0.3), (5, float("inf")))),
        result("A|||u", 0.2, ((4, None), (2, float("nan")), (0, 0.2), (3, 0.0), (5, 0.1))),
    ])
    assert model["paired_reference_run_count"] == 1
    assert model[UPPER_BOUND] == pytest.approx(-0.2)
    assert model["table_italic"] is True


@pytest.mark.parametrize("reference_status", ["missing", "failed", "disjoint"])
def test_brier_without_paired_reference_keeps_score_without_italics(reference_status: str) -> None:
    rows = [result("A|||u", 0.2, ((0, 0.2),))]
    if reference_status != "missing":
        baseline = result(CLINICAL_BASELINE_MODEL_KEY, 0.3, ((1, 0.3),))
        if reference_status == "failed":
            baseline["status"] = "failed"
        rows.insert(0, baseline)
    model = with_clinical_reference(rows)[-1]
    assert model["paired_reference_run_count"] == 0
    assert model[UPPER_BOUND] is None
    assert model["table_italic"] is False
    assert "0.200" in brier_cell(model)


@pytest.mark.parametrize("status", ["failed", "invalid_predictions", "not_estimable", "skipped"])
def test_unavailable_brier_results_receive_no_highlighting(status: str) -> None:
    rows = [
        result(CLINICAL_BASELINE_MODEL_KEY, 0.3, ((0, 0.3),)),
        {**result("A|||u", 0.1, ((0, 0.1),)), "status": status},
    ]
    baseline, unavailable = with_clinical_reference(rows)
    assert baseline["table_bold"] is True
    assert all(unavailable[key] is False for key in ("table_bold", "table_underline", "table_italic"))
    assert brier_cell(unavailable) == "--"


def test_brier_outputs_include_combined_markup_and_inference_metadata(tmp_path: Path) -> None:
    benchmark = {
        "sites": [{"label": "Liver"}],
        "horizons": [{"label": "6m", "days": 180}],
        "washout": {"days": [0]},
        "include_all_blood_tests": False,
    }
    rows = [
        result(CLINICAL_BASELINE_MODEL_KEY, 0.3, ((0, 0.3),)),
        result("A|||u", 0.2, ((0, 0.2),)),
        result("B|||u", 0.25, ((0, 0.25),)),
    ]
    write_brier_outputs(benchmark, ("A|||u", "B|||u"), rows, tmp_path, tmp_path)
    latex = (tmp_path / "complete_case_brier_washout.tex").read_text()
    assert r"A & \textbf{\textit{0.200 [0.190, 0.210]}}" in latex
    assert r"B & \underline{\textit{0.250 [0.240, 0.260]}}" in latex
    assert "Clinical baseline & 0.300 [0.290, 0.310]" in latex
    assert "one-sided 95\\% upper bootstrap bound" in latex
    assert "No IPCW is used" not in latex
    assert "may be biased by censoring" in latex
    metadata = json.loads((tmp_path / "metadata.json").read_text())
    assert metadata["ipcw"] is False
    model = metadata["results"][1]
    assert model[UPPER_BOUND] == pytest.approx(-0.1)
    assert model["table_bold"] is True
    assert model["table_italic"] is True
    exported = pd.read_csv(tmp_path / "complete_case_brier_washout.csv")
    assert exported.loc[1, UPPER_BOUND] == pytest.approx(-0.1)
    assert bool(exported.loc[1, "table_italic"])
