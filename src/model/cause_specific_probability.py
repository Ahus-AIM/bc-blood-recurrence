from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from interpret.glassbox import ExplainableBoostingRegressor

from src.data.competing_risk_outcomes import HORIZON_DAYS


@dataclass(frozen=True)
class BreslowBaseline:
    """Training event times, hazard increments, weighted events and weighted risk sets.

    Both Breslow terms use the same patient-balanced landmark weights. Risk
    sets include every row observed through the event time, including ties.
    """

    times: np.ndarray
    increments: np.ndarray
    event_weights: np.ndarray
    risk_totals: np.ndarray


@dataclass(frozen=True)
class InvalidHazard:
    """A patient-row hazard jump that cannot define a probability transition."""

    row_index: int
    time_days: float
    target_increment: float
    competing_increment: float


@dataclass(frozen=True)
class CumulativeIncidence:
    """Day-180 target/competing probabilities, survival and invalid-jump diagnostics.

    A row with an invalid jump has NaN probabilities; increments are never
    clipped or silently replaced by another survival construction.
    ``max_combined_hazard_jump`` retains each row's largest target-plus-competing
    jump through the horizon, including invalid jumps.
    """

    target: np.ndarray
    competing: np.ndarray
    survival: np.ndarray
    invalid_hazards: tuple[InvalidHazard, ...]
    max_combined_hazard_jump: np.ndarray


def prediction_scale(model: ExplainableBoostingRegressor) -> Literal["log_risk", "relative_risk"]:
    """Inspect the fitted Cox EBM link instead of assuming predict() exponentiates."""
    if model.objective != "survival_cox":
        raise ValueError("Absolute-risk conversion requires a survival_cox EBM.")
    if model.link_ == "identity":
        return "log_risk"
    if model.link_ == "log":
        return "relative_risk"
    raise ValueError(f"Unsupported fitted Cox link: {model.link_!r}.")


def log_relative_risks(model: ExplainableBoostingRegressor, features: np.ndarray) -> np.ndarray:
    """Return finite log risks, retaining large offsets until training-based centering."""
    scores = np.asarray(model.predict(features), dtype=float)
    if prediction_scale(model) == "relative_risk":
        require_positive_vector(scores, "Model relative risks")
        scores = np.log(scores)
    if scores.ndim != 1 or np.any(~np.isfinite(scores)):
        raise ValueError("Model log-relative risks must be finite and one-dimensional.")
    return scores


def centered_relative_risks(
    log_risks: Mapping[str, np.ndarray],
    training_weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    """Subtract the patient-weighted training mean from every split before exponentiation."""
    training_scores = log_risks["train"]
    require_positive_vector(training_weights, "Training weights")
    if training_scores.size == 0 or training_scores.shape != training_weights.shape:
        raise ValueError("Risk centering requires nonempty, matching training scores and weights.")
    center = float(np.average(training_scores, weights=training_weights))
    if not np.isfinite(center):
        raise ValueError("Training log-risk center must be finite.")
    risks: dict[str, np.ndarray] = {}
    for split, scores in log_risks.items():
        with np.errstate(over="ignore", under="ignore", invalid="ignore"):
            risks[split] = np.exp(scores - center)
        require_positive_vector(risks[split], "Centered model relative risks")
    return risks, center


def require_positive_vector(values: np.ndarray, label: str) -> None:
    if values.ndim != 1 or np.any(~np.isfinite(values)) or np.any(values <= 0):
        raise ValueError(f"{label} must be finite, positive and one-dimensional.")


def weighted_breslow(
    durations: np.ndarray,
    events: np.ndarray,
    risks: np.ndarray,
    weights: np.ndarray,
) -> BreslowBaseline:
    """Estimate weighted event counts divided by weighted relative-risk totals."""
    durations = np.asarray(durations, dtype=float)
    events = np.asarray(events, dtype=bool)
    risks = np.asarray(risks, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if durations.ndim != 1 or any(values.shape != durations.shape for values in (events, risks, weights)):
        raise ValueError("Breslow inputs must be matching one-dimensional arrays.")
    if len(durations) == 0 or np.any(~np.isfinite(durations)) or np.any(durations < 0):
        raise ValueError("Breslow fitting requires finite nonnegative training durations.")
    require_positive_vector(risks, "Training relative risks")
    require_positive_vector(weights, "Training weights")
    times, inverse = np.unique(durations[events], return_inverse=True)
    numerator = np.bincount(inverse, weights=weights[events], minlength=len(times))
    order = np.argsort(durations, kind="stable")
    with np.errstate(over="ignore"):
        risk_sets = np.cumsum((weights * risks)[order][::-1])[::-1]
    denominator = risk_sets[np.searchsorted(durations[order], times, side="left")]
    require_positive_vector(denominator, "Breslow risk-set totals")
    increments = numerator / denominator
    if np.any(~np.isfinite(increments)) or np.any(increments <= 0):
        raise ValueError("Invalid fitted Breslow hazard increments.")
    return BreslowBaseline(times, increments, numerator, denominator)


def cumulative_incidence(
    target_baseline: BreslowBaseline,
    competing_baseline: BreslowBaseline,
    target_risks: np.ndarray,
    competing_risks: np.ndarray,
) -> CumulativeIncidence:
    """Apply F_k += S(t-) dA_k and S *= 1-dA_target-dA_competing through day 180."""
    require_positive_vector(target_risks, "Target relative risks")
    require_positive_vector(competing_risks, "Competing relative risks")
    if target_risks.shape != competing_risks.shape:
        raise ValueError("Target and competing relative-risk rows must match.")
    for baseline in (target_baseline, competing_baseline):
        validate_baseline(baseline)
    times = np.union1d(target_baseline.times, competing_baseline.times)
    target_steps = aligned_increments(target_baseline, times)
    competing_steps = aligned_increments(competing_baseline, times)
    target = np.zeros(len(target_risks))
    competing = np.zeros(len(target_risks))
    survival = np.ones(len(target_risks))
    invalid_rows = np.zeros(len(target_risks), dtype=bool)
    diagnostics: list[InvalidHazard] = []
    max_combined_jump = np.zeros(len(target_risks))
    for time, target_step, competing_step in zip(times, target_steps, competing_steps, strict=True):
        if time > HORIZON_DAYS:
            break
        with np.errstate(over="ignore", invalid="ignore"):
            target_jump = target_step * target_risks
            competing_jump = competing_step * competing_risks
            total_jump = target_jump + competing_jump
        max_combined_jump = np.maximum(max_combined_jump, total_jump)
        invalid = ~np.isfinite(total_jump) | (target_jump < 0) | (competing_jump < 0) | (total_jump > 1)
        diagnostics.extend(
            InvalidHazard(int(row), float(time), float(target_jump[row]), float(competing_jump[row]))
            for row in np.flatnonzero(invalid)
        )
        invalid_rows |= invalid
        valid = ~invalid_rows
        target[valid] += survival[valid] * target_jump[valid]
        competing[valid] += survival[valid] * competing_jump[valid]
        survival[valid] *= 1 - total_jump[valid]
    for probability in (target, competing, survival):
        probability[invalid_rows] = np.nan
        valid_values = probability[~invalid_rows]
        if np.any((valid_values < 0) | (valid_values > 1 + 1e-12)):
            raise ValueError("Product-integral probability outside [0, 1].")
    if not np.allclose((target + competing + survival)[~invalid_rows], 1.0):
        raise ValueError("Product-integral probabilities do not sum to one.")
    return CumulativeIncidence(target, competing, survival, tuple(diagnostics), max_combined_jump)


def validate_baseline(baseline: BreslowBaseline) -> None:
    if baseline.times.ndim != 1 or baseline.times.shape != baseline.increments.shape:
        raise ValueError("Baseline times and hazard increments must be matching vectors.")
    if np.any(~np.isfinite(baseline.times)) or np.any(baseline.times < 0) or np.any(np.diff(baseline.times) <= 0):
        raise ValueError("Baseline event times must be finite, nonnegative and strictly increasing.")
    if np.any(~np.isfinite(baseline.increments)) or np.any(baseline.increments < 0):
        raise ValueError("Invalid baseline hazard increments: expected finite nonnegative jumps.")


def aligned_increments(baseline: BreslowBaseline, times: np.ndarray) -> np.ndarray:
    values = np.zeros(len(times))
    values[np.searchsorted(times, baseline.times)] = baseline.increments
    return values
