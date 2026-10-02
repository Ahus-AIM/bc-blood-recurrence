from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from src.model.cause_specific_probability import BreslowBaseline


def numeric_summary(values: np.ndarray) -> dict[str, int | float | None]:
    """Summarize finite values while retaining counts of every nonfinite entry."""
    finite = values[np.isfinite(values)]
    result: dict[str, int | float | None] = {
        "count": int(values.size),
        "nonfinite_count": int(values.size - finite.size),
    }
    names = ("min", "p01", "p50", "p99", "max")
    quantiles = [None] * len(names)
    if finite.size:
        scale = float(np.max(np.abs(finite)))
        quantiles = np.quantile(finite / scale, [0, 0.01, 0.5, 0.99, 1]) * scale if scale else np.zeros(5)
    result.update({name: None if value is None else float(value) for name, value in zip(names, quantiles, strict=True)})
    return result


def score_diagnostics(log_risks: Mapping[str, np.ndarray], weights: np.ndarray) -> dict[str, object]:
    """Describe score offsets, spread and extrapolation before exponentiation can fail."""
    train = log_risks["train"]
    with np.errstate(over="ignore", invalid="ignore"):
        center = float(np.average(train, weights=weights))
    minimum, maximum = float(train.min()), float(train.max())
    splits = {}
    for split, scores in log_risks.items():
        with np.errstate(over="ignore", invalid="ignore"):
            centered = scores - center
        splits[split] = {
            "raw_log_risk": numeric_summary(scores),
            "centered_log_risk": numeric_summary(centered),
            "below_training_range_count": int(np.count_nonzero(scores < minimum)),
            "above_training_range_count": int(np.count_nonzero(scores > maximum)),
            "above_float64_exp_range_count": int(np.count_nonzero(centered > np.log(np.finfo(float).max))),
            "below_float64_normal_exp_range_count": int(np.count_nonzero(centered < np.log(np.finfo(float).tiny))),
        }
    return {"training_log_risk_center": center if np.isfinite(center) else None, "splits": splits}


def risk_set_diagnostics(
    durations: np.ndarray,
    patient_ids: np.ndarray,
    weights: np.ndarray,
    risks: np.ndarray,
    baseline: BreslowBaseline,
    cause: str,
) -> pd.DataFrame:
    """Aggregate each patient's risk contribution using an inclusive descending risk-set sweep.

    Effective risk patients is 1 / sum(patient risk shares squared), a
    concentration measure rather than an inferential sample size.
    """
    columns = [
        "cause", "time_days", "at_risk_landmarks", "at_risk_patients", "event_weight",
        "risk_total", "baseline_increment", "largest_patient_risk_share", "effective_risk_patients",
    ]
    patients, patient_index = np.unique(patient_ids, return_inverse=True)
    order = np.argsort(durations, kind="stable")
    sorted_durations = durations[order]
    patient_risks = np.zeros(len(patients))
    present = np.zeros(len(patients), dtype=bool)
    contributions = weights * risks
    previous = len(order)
    rows = []
    for index in range(len(baseline.times) - 1, -1, -1):
        time = baseline.times[index]
        start = int(np.searchsorted(sorted_durations, time, side="left"))
        entering = order[start:previous]
        np.add.at(patient_risks, patient_index[entering], contributions[entering])
        present[patient_index[entering]] = True
        shares = patient_risks / baseline.risk_totals[index]
        rows.append({
            "cause": cause,
            "time_days": float(time),
            "at_risk_landmarks": len(order) - start,
            "at_risk_patients": int(present.sum()),
            "event_weight": float(baseline.event_weights[index]),
            "risk_total": float(baseline.risk_totals[index]),
            "baseline_increment": float(baseline.increments[index]),
            "largest_patient_risk_share": float(shares.max()),
            "effective_risk_patients": float(1 / np.dot(shares, shares)),
        })
        previous = start
    return pd.DataFrame(rows[::-1], columns=columns)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    return float(np.average(values, weights=weights)) if values.size else None


def risk_set_summary(frame: pd.DataFrame) -> dict[str, object]:
    summaries = {}
    for cause in ("target", "competing"):
        rows = frame.loc[frame["cause"] == cause]
        summaries[cause] = {
            "event_time_count": len(rows),
            "no_training_events": rows.empty,
            "at_risk_patients": numeric_summary(rows["at_risk_patients"].to_numpy(dtype=float)),
            "largest_patient_risk_share": numeric_summary(rows["largest_patient_risk_share"].to_numpy(dtype=float)),
            "effective_risk_patients": numeric_summary(rows["effective_risk_patients"].to_numpy(dtype=float)),
            "single_patient_majority_time_count": int((rows["largest_patient_risk_share"] > 0.5).sum()),
            "day_zero_event_weight": float(rows.loc[rows["time_days"] == 0, "event_weight"].sum()),
            "total_event_weight": float(rows["event_weight"].sum()),
        }
    return summaries


def split_diagnostics(frame: pd.DataFrame) -> dict[str, object]:
    """Report full-split coverage and probabilities; these are not bootstrap estimates."""
    weights = frame["patient_weight"].to_numpy()
    outcome = frame["outcome_180d"].to_numpy()
    known = np.isfinite(outcome)
    columns = ["target_probability_180d", "competing_probability_180d", "event_free_probability_180d"]
    probabilities = frame[columns].to_numpy()
    finite = np.isfinite(probabilities).all(axis=1)
    out_of_bounds = ((probabilities < 0) | (probabilities > 1)).any(axis=1)
    valid = finite & ~out_of_bounds
    scored = known & valid
    target_events = frame["cause"].to_numpy() == 1
    return {
        "landmark_count": len(frame),
        "patient_count": int(frame["patient_id"].nunique()),
        "target_event_count": int(target_events.sum()),
        "competing_event_count": int((frame["cause"] == 2).sum()),
        "day_zero_target_event_count": int((target_events & (frame["duration_days"] == 0)).sum()),
        "known_outcome_count": int(known.sum()),
        "early_censored_count": int((~known).sum()),
        "patient_weighted_known_outcome_fraction": weighted_mean(known, weights),
        "invalid_probability_count": int((~valid).sum()),
        "scored_landmark_count": int(scored.sum()),
        "complete_case_observed_target_fraction": weighted_mean(outcome[scored], weights[scored]),
        "complete_case_mean_target_probability": weighted_mean(probabilities[scored, 0], weights[scored]),
        "probabilities": {column: numeric_summary(frame[column].to_numpy()) for column in columns},
        "probability_sum_absolute_error": numeric_summary(np.abs(probabilities[finite].sum(axis=1) - 1)),
        "max_combined_hazard_jump": numeric_summary(frame["max_combined_hazard_jump"].to_numpy()),
        "out_of_bounds_probability_count": int(out_of_bounds.sum()),
    }
