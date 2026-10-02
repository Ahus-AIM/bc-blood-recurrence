from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.model.brier_diagnostics import numeric_summary, risk_set_diagnostics, score_diagnostics, split_diagnostics
from src.model.cause_specific_probability import weighted_breslow
from src.script.brier_diagnostics import BrierDiagnostics
from src.script.brier_model_artifacts import predict_splits, prepare_brier_cell
from src.script.single_bloodtest_benchmark_reporting import CLINICAL_BASELINE_MODEL_KEY
from src.script.single_bloodtest_brier import run_brier_benchmark
from src.script.single_bloodtest_washout import washout_landmarks
from test.test_brier_benchmark import saved_models


@pytest.mark.parametrize("risk_scale", [1.0, 1e150])
def test_risk_set_concentration_aggregates_patient_landmarks_and_keeps_ties(risk_scale: float) -> None:
    durations = np.array([0, 10, 10, 180])
    patient_ids = np.array(["A", "A", "B", "C"])
    weights = np.array([0.5, 0.5, 1.0, 1.0])
    risks = np.array([2.0, 2.0, 1.0, 1.0]) * risk_scale
    baseline = weighted_breslow(durations, np.array([True, True, False, False]), risks, weights)
    actual = risk_set_diagnostics(durations, patient_ids, weights, risks, baseline, "target")
    np.testing.assert_array_equal(actual["at_risk_landmarks"], [4, 3])
    np.testing.assert_array_equal(actual["at_risk_patients"], [3, 3])
    np.testing.assert_allclose(actual["largest_patient_risk_share"], [0.5, 1 / 3])
    np.testing.assert_allclose(actual["effective_risk_patients"], [8 / 3, 3])
    np.testing.assert_allclose(actual["event_weight"], [0.5, 0.5])


def test_score_diagnostics_distinguish_common_offsets_spread_and_extrapolation() -> None:
    scores = {"train": np.array([1000.0, 1001.0]), "val": np.array([999.0, 1002.0]), "test": np.array([2000.0])}
    summary = score_diagnostics(scores, np.array([3.0, 1.0]))
    assert summary["training_log_risk_center"] == 1000.25
    assert summary["splits"]["train"]["centered_log_risk"]["min"] == -0.25
    assert summary["splits"]["train"]["above_float64_exp_range_count"] == 0
    assert summary["splits"]["val"]["below_training_range_count"] == 1
    assert summary["splits"]["val"]["above_training_range_count"] == 1
    assert summary["splits"]["test"]["above_float64_exp_range_count"] == 1


def test_nonfinite_and_empty_summaries_remain_strict_json() -> None:
    summary = numeric_summary(np.array([np.nan, np.inf, -np.inf, 0, 1]))
    assert summary["count"] == 5
    assert summary["nonfinite_count"] == 3
    assert summary["p50"] == 0.5
    assert numeric_summary(np.array([]))["max"] is None
    json.dumps(summary, allow_nan=False)
    extreme = numeric_summary(np.array([-1e308, 1e308]))
    assert extreme["p50"] == 0.0
    json.dumps(extreme, allow_nan=False)


def test_nonfinite_training_center_is_reported_without_masking_the_numerical_failure(tmp_path: Path) -> None:
    model = SimpleNamespace(objective="survival_cox", link_="identity", predict=lambda rows: rows[:, 0])
    with np.errstate(over="ignore"):
        with pytest.raises(ValueError, match="Training log-risk center must be finite"):
            with BrierDiagnostics(tmp_path) as report:
                predict_splits(model, {"train": np.array([[1e308], [1e308]])}, np.ones(2), report=report)
    saved = json.loads((tmp_path / "diagnostics.json").read_text())
    assert saved["status"] == "failed"
    assert saved["target_scores"]["training_log_risk_center"] is None


def test_split_coverage_uses_patient_weights_and_matched_known_valid_predictions() -> None:
    frame = pd.DataFrame({
        "patient_id": ["A", "A", "B", "C", "D"],
        "patient_weight": [0.625, 0.625, 1.25, 1.25, 1.25],
        "cause": [1, 0, 2, 0, 1],
        "duration_days": [0, 20, 10, 180, 10],
        "outcome_180d": [1, np.nan, 0, 0, 1],
        "target_probability_180d": [0.2, 0.1, 0.3, 0.4, np.nan],
        "competing_probability_180d": [0.1, 0.1, 0.2, 0.1, np.nan],
        "event_free_probability_180d": [0.7, 0.8, 0.5, 0.5, np.nan],
        "max_combined_hazard_jump": [0.2, 0.15, 0.4, 0.5, 2.0],
    })
    summary = split_diagnostics(frame)
    assert summary["day_zero_target_event_count"] == 1
    assert summary["known_outcome_count"] == 4
    assert summary["early_censored_count"] == 1
    assert summary["invalid_probability_count"] == 1
    assert summary["scored_landmark_count"] == 3
    assert summary["patient_weighted_known_outcome_fraction"] == 0.875
    assert summary["complete_case_observed_target_fraction"] == pytest.approx(0.2)
    assert summary["complete_case_mean_target_probability"] == pytest.approx(0.32)
    assert summary["max_combined_hazard_jump"]["max"] == 2.0
    assert split_diagnostics(frame.iloc[:0])["patient_weighted_known_outcome_fraction"] is None


def test_score_report_survives_failed_exponentiation_and_clears_stale_output(tmp_path: Path) -> None:
    (tmp_path / "risk_set_diagnostics.csv").write_text("old output")
    model = SimpleNamespace(objective="survival_cox", link_="identity", predict=lambda rows: rows[:, 0])
    with pytest.raises(ValueError, match="Centered model relative risks"):
        with BrierDiagnostics(tmp_path) as report:
            predict_splits(
                model, {"train": np.array([[0.0]]), "test": np.array([[1000.0]])}, np.ones(1), report=report,
            )
    saved = json.loads((tmp_path / "diagnostics.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "exponentiating_target_scores"
    assert saved["target_scores"]["splits"]["test"]["centered_log_risk"]["max"] == 1000
    assert not (tmp_path / "risk_set_diagnostics.csv").exists()


def test_saved_cell_writes_reproducible_diagnostics_and_matching_hazard_summaries(tmp_path: Path) -> None:
    dataset, sources, _ = saved_models(tmp_path)
    destination = tmp_path / "diagnostics"
    predictions, _ = prepare_brier_cell(Path(sources[0]["output_dir"]), destination, BenchmarkLandmarks(dataset), {})
    saved = json.loads((destination / "diagnostics.json").read_text())
    assert saved["status"] == "complete"
    assert set(saved["splits"]) == {"train", "val", "test"}
    for split, summary in saved["splits"].items():
        frame = predictions.loc[predictions["split"] == split]
        assert summary["max_combined_hazard_jump"]["max"] == frame["max_combined_hazard_jump"].max()
        assert summary["known_outcome_count"] == frame["outcome_180d"].notna().sum()
    assert len(pd.read_csv(destination / "risk_set_diagnostics.csv")) > 0
    snapshot = (destination / "diagnostics.json").read_bytes()
    prepare_brier_cell(Path(sources[0]["output_dir"]), destination, BenchmarkLandmarks(dataset), {})
    assert (destination / "diagnostics.json").read_bytes() == snapshot


def test_failed_model_validation_replaces_previous_diagnostics(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    pools = washout_landmarks(dataset, (0, 30, 60, 90, 180))
    run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    sources[0]["status"] = "failed"
    run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    destination = tmp_path / "brier" / "Liver" / "6m" / CLINICAL_BASELINE_MODEL_KEY
    saved = json.loads((destination / "diagnostics.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "validating_saved_model"
    assert "target_scores" not in saved
    assert not (destination / "risk_set_diagnostics.csv").exists()


def test_washout_failure_preserves_finished_prediction_diagnostics(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    pools = washout_landmarks(dataset, (0, 30, 60, 90, 180))
    with patch("src.script.single_bloodtest_brier.evaluate_brier_washout", side_effect=ValueError("draw mismatch")):
        run_brier_benchmark(benchmark, dataset.selected_test_keys, sources, tmp_path, tmp_path, pools)
    destination = tmp_path / "brier" / "Liver" / "6m" / CLINICAL_BASELINE_MODEL_KEY
    saved = json.loads((destination / "diagnostics.json").read_text())
    assert saved["status"] == "failed"
    assert saved["stage"] == "evaluating_washouts"
    assert "target_scores" in saved
    assert saved["failure_reason"] == "draw mismatch"
