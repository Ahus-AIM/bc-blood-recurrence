from __future__ import annotations

import numpy as np
import pandas as pd

METRIC = "complete_case_brier_score"


def score_draw(predictions: pd.DataFrame, bootstrap_run: int) -> dict[str, object]:
    """Average known-outcome errors; sampled rows already implement patient-uniform weighting."""
    known = predictions["outcome_180d"].notna()
    valid_probability = np.isfinite(predictions["target_probability_180d"]) & predictions[
        "target_probability_180d"
    ].between(0, 1)
    invalid = known & ~valid_probability
    score = None
    if known.any() and not invalid.any():
        score = float(predictions.loc[known, "squared_error"].mean())
    return {
        "bootstrap_run": bootstrap_run,
        METRIC: score,
        "row_count": len(predictions),
        "patient_count": int(predictions["patient_id"].nunique()),
        "target_event_count": int((predictions["cause"] == 1).sum()),
        "competing_event_count": int((predictions["cause"] == 2).sum()),
        "known_outcome_count": int(known.sum()),
        "early_censored_count": int((~known).sum()),
        "invalid_known_prediction_count": int(invalid.sum()),
    }


def summarize_brier_runs(runs: list[dict[str, object]], invalid_predictions: int) -> dict[str, object]:
    values = np.asarray([run[METRIC] for run in runs if run[METRIC] is not None], dtype=float)
    status = "invalid_predictions" if invalid_predictions else "complete" if len(values) else "not_estimable"
    estimable = status == "complete"
    return {
        "status": status,
        METRIC: float(values.mean()) if estimable else None,
        f"{METRIC}_ci95_lower": float(np.percentile(values, 2.5)) if estimable else None,
        f"{METRIC}_ci95_upper": float(np.percentile(values, 97.5)) if estimable else None,
        "bootstrap_runs_attempted": len(runs),
        "bootstrap_runs": len(runs),
        "finite_metric_runs": len(values),
        "zero_event_bootstrap_runs_discarded": 0,
        "no_known_outcome_runs": sum(run["known_outcome_count"] == 0 for run in runs),
        "invalid_prediction_count": invalid_predictions,
        "runs": runs,
    }
