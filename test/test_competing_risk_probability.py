from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data.competing_risk_outcomes import competing_risk_outcomes, competing_training_table
from src.data.sample_weighting import patient_uniform_weights
from src.model.cause_specific_probability import (
    BreslowBaseline,
    centered_relative_risks,
    cumulative_incidence,
    log_relative_risks,
    prediction_scale,
    weighted_breslow,
)
from src.model.complete_case_brier import METRIC, score_draw, summarize_brier_runs
from src.model.ebm_fitting import fit_ebm_model
from test.test_ebm_fitting import survival_table


def baseline(times: list[float], increments: list[float]) -> BreslowBaseline:
    return BreslowBaseline(np.asarray(times, dtype=float), np.asarray(increments), np.array([]), np.array([]))


def test_installed_cox_predict_returns_log_risk_and_is_exponentiated_once() -> None:
    table = survival_table()
    model, _ = fit_ebm_model(table, table.features, {"outer_bags": 1, "validation_size": 0, "max_rounds": 5})
    assert model.link_ == "identity"
    assert prediction_scale(model) == "log_risk"
    log_risk = model.intercept_ + model.eval_terms(table.features).sum(axis=1)
    np.testing.assert_allclose(model.predict(table.features), log_risk)
    scores = log_relative_risks(model, table.features)
    np.testing.assert_allclose(scores, log_risk)
    risks, center = centered_relative_risks({"train": scores}, table.sample_weights)
    np.testing.assert_allclose(risks["train"], np.exp(log_risk - center))


def test_relative_risk_output_is_not_exponentiated_twice() -> None:
    model = SimpleNamespace(objective="survival_cox", link_="log", predict=lambda _features: np.array([0.5, 3]))
    np.testing.assert_allclose(log_relative_risks(model, np.zeros((2, 1))), np.log([0.5, 3]))


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_unusable_log_risk_fails_explicitly(score: float) -> None:
    model = SimpleNamespace(objective="survival_cox", link_="identity", predict=lambda _features: np.array([score]))
    with pytest.raises(ValueError, match="log-relative risks"):
        log_relative_risks(model, np.zeros((1, 1)))


def test_weighted_breslow_uses_both_weighted_terms_and_inclusive_tied_risk_sets() -> None:
    ids = np.array(["A", "A", "B", "C", "C", "C"])
    weights = patient_uniform_weights(ids)
    np.testing.assert_allclose(weights, [1, 1, 2, 2 / 3, 2 / 3, 2 / 3])
    durations = np.array([0, 10, 10, 10, 20, 180])
    events = np.array([1, 1, 0, 1, 1, 0], dtype=bool)
    risks = np.array([2, 1, 3, 2, 1, 4])
    actual = weighted_breslow(durations, events, risks, weights)
    np.testing.assert_equal(actual.times, [0, 10, 20])
    np.testing.assert_allclose(actual.event_weights, [1, 5 / 3, 2 / 3])
    np.testing.assert_allclose(actual.risk_totals, [41 / 3, 35 / 3, 10 / 3])
    np.testing.assert_allclose(actual.increments, [3 / 41, 1 / 7, 1 / 5])
    reordered = np.array([4, 1, 0, 5, 2, 3])
    permuted = weighted_breslow(durations[reordered], events[reordered], risks[reordered], weights[reordered] * 13)
    np.testing.assert_allclose(actual.increments, permuted.increments)


def test_product_integral_uses_left_survival_combines_ties_and_includes_day_180() -> None:
    target = baseline([0, 10, 180, 181], [0.1, 0.2, 0.1, 0.9])
    competing = baseline([10], [0.3])
    result = cumulative_incidence(target, competing, np.array([1.0, 2.0]), np.array([1.0, 0.5]))
    np.testing.assert_allclose(result.target, [0.325, 0.592])
    np.testing.assert_allclose(result.competing, [0.27, 0.12])
    np.testing.assert_allclose(result.survival, [0.405, 0.288])
    np.testing.assert_allclose(result.max_combined_hazard_jump, [0.5, 0.55])
    assert result.invalid_hazards == ()


def test_unity_relative_risks_match_weighted_aalen_johansen_hand_calculation() -> None:
    durations = np.array([5, 5, 10, 20])
    causes = np.array([1, 2, 1, 0])
    weights = np.array([0.5, 0.5, 1.0, 1.0])
    risks = np.ones(4)
    target = weighted_breslow(durations, causes == 1, risks, weights)
    competing = weighted_breslow(durations, causes == 2, risks, weights)
    result = cumulative_incidence(target, competing, np.ones(1), np.ones(1))
    np.testing.assert_allclose(result.target, [0.5])
    np.testing.assert_allclose(result.competing, [1 / 6])
    np.testing.assert_allclose(result.survival, [1 / 3])


def test_invalid_patient_hazards_are_reported_and_never_clipped() -> None:
    result = cumulative_incidence(baseline([10], [0.3]), baseline([10], [0.3]), np.array([1.0, 3.0]), np.ones(2))
    np.testing.assert_allclose(result.target[0], 0.3)
    assert np.isnan(result.target[1])
    assert np.isnan(result.competing[1])
    assert np.isnan(result.survival[1])
    assert len(result.invalid_hazards) == 1
    assert result.invalid_hazards[0].row_index == 1
    assert result.invalid_hazards[0].time_days == 10
    assert result.invalid_hazards[0].target_increment == pytest.approx(0.9)
    np.testing.assert_allclose(result.max_combined_hazard_jump, [0.6, 1.2])


@pytest.mark.parametrize("increment", [-0.1, np.inf, np.nan])
def test_invalid_baseline_increments_fail(increment: float) -> None:
    with pytest.raises(ValueError, match="Invalid baseline hazard increments"):
        cumulative_incidence(baseline([0], [increment]), baseline([], []), np.ones(1), np.ones(1))


def test_no_event_baselines_give_zero_incidence() -> None:
    fitted = weighted_breslow(np.array([5, 180]), np.array([False, False]), np.ones(2), np.ones(2))
    result = cumulative_incidence(fitted, fitted, np.ones(2), np.ones(2))
    np.testing.assert_equal(result.target, [0, 0])
    np.testing.assert_equal(result.survival, [1, 1])


def test_known_outcomes_keep_competitors_and_horizon_censoring_but_not_early_censoring() -> None:
    original = survival_table()
    durations = np.array([0, 180, 30, 179, 180, 180])
    reasons = np.array(
        [
            "configured_recurrence",
            "configured_recurrence",
            "death",
            "last_observation",
            "last_observation",
            "censoring_horizon",
        ]
    )
    table = replace(
        original, durations=durations, events=reasons == "configured_recurrence", target_end_reasons=reasons
    )
    actual = competing_risk_outcomes(table)
    np.testing.assert_equal(actual.causes, [1, 1, 2, 0, 0, 0])
    np.testing.assert_equal(actual.binary_outcomes, [1, 1, 0, np.nan, 0, 0])


def test_competing_training_flips_event_cause_and_balances_patients() -> None:
    original = survival_table()
    reasons = np.resize(np.array(["configured_recurrence", "death", "other_cancer", "non_endpoint_recurrence"]), 80)
    table = replace(original, events=reasons == "configured_recurrence", target_end_reasons=reasons)
    actual = competing_training_table(table)
    np.testing.assert_equal(actual.events, reasons != "configured_recurrence")
    assert np.all(actual.signed_durations[actual.events] > 0)
    assert np.all(actual.signed_durations[~actual.events] < 0)
    totals = pd.Series(actual.sample_weights).groupby(actual.patient_ids).sum()
    np.testing.assert_allclose(totals, totals.iloc[0])


def test_complete_case_brier_retains_no_event_draws_and_preserves_multiplicity() -> None:
    predictions = pd.DataFrame(
        {
            "patient_id": ["A", "A", "B", "C"],
            "cause": [2, 2, 0, 0],
            "outcome_180d": [0, 0, 0, np.nan],
            "target_probability_180d": [0.2, 0.2, 0.4, 0.8],
            "squared_error": [0.04, 0.04, 0.16, np.nan],
        }
    )
    actual = score_draw(predictions, 7)
    assert actual[METRIC] == pytest.approx(0.08)
    assert actual["target_event_count"] == 0
    assert actual["known_outcome_count"] == 3
    assert actual["early_censored_count"] == 1
    summary = summarize_brier_runs([actual], 0)
    assert summary["status"] == "complete"
    assert summary["zero_event_bootstrap_runs_discarded"] == 0
    predictions.loc[0, "target_probability_180d"] = np.nan
    invalid = score_draw(predictions, 7)
    assert invalid[METRIC] is None
    assert summarize_brier_runs([invalid], 1)["status"] == "invalid_predictions"
    assert score_draw(predictions.iloc[-1:], 8)[METRIC] is None
