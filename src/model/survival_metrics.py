from __future__ import annotations

import numpy as np


def harrell_c_index_excluding_same_patient(
    durations: np.ndarray,
    events: np.ndarray,
    risk_scores: np.ndarray,
    patient_ids: np.ndarray,
) -> float:
    """Compute Harrell's C-index while excluding same-patient pairs."""
    concordant = 0.0
    comparable = 0
    order = np.argsort(durations, kind="mergesort")
    sorted_durations = durations[order]
    sorted_events = events[order]
    sorted_scores = risk_scores[order]
    sorted_patients = patient_ids[order]
    for left in range(len(sorted_durations)):
        if not sorted_events[left]:
            continue
        later = sorted_durations > sorted_durations[left]
        same_time_censored = (sorted_durations == sorted_durations[left]) & ~sorted_events
        different_patient = sorted_patients != sorted_patients[left]
        comparable_mask = (later | same_time_censored) & different_patient
        pair_count = int(np.sum(comparable_mask))
        if pair_count == 0:
            continue
        score_diff = sorted_scores[left] - sorted_scores[comparable_mask]
        concordant += float(np.sum(score_diff > 0.0))
        concordant += 0.5 * float(np.sum(score_diff == 0.0))
        comparable += pair_count
    if comparable == 0:
        return float("nan")
    return concordant / comparable
