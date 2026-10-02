from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from src.data import PatientLandmarkTable
from src.data.sample_weighting import patient_uniform_weights
from src.data.survival_targets import signed_survival_durations

HORIZON_DAYS = 180
COMPETING_REASONS = ("death", "other_cancer", "non_endpoint_recurrence")
CENSORING_REASONS = ("last_observation", "censoring_horizon")


@dataclass(frozen=True)
class CompetingRiskOutcomes:
    """Landmark follow-up and mutually exclusive causes: 0=censor, 1=target, 2=competing.

    ``binary_outcomes`` is the observed target indicator at day 180; early
    ordinary censoring remains NaN. Times and ties come from the existing
    endpoint projection, including zero-day and simultaneous-site events.
    """

    durations: np.ndarray
    causes: np.ndarray
    binary_outcomes: np.ndarray


def competing_risk_outcomes(table: PatientLandmarkTable) -> CompetingRiskOutcomes:
    """Reclassify resolved censoring reasons without changing endpoint precedence."""
    durations = np.asarray(table.durations, dtype=float)
    if np.any(~np.isfinite(durations)) or np.any((durations < 0) | (durations > HORIZON_DAYS)):
        raise ValueError("Competing-risk durations must be between 0 and 180 days.")
    reasons = table.target_end_reasons
    if reasons is None or reasons.shape != durations.shape:
        raise ValueError("Competing-risk outcomes require a resolved reason for every landmark.")
    if not np.isin(reasons, (*COMPETING_REASONS, *CENSORING_REASONS, "configured_recurrence")).all():
        raise ValueError("Unknown landmark outcome reason.")
    if not np.array_equal(table.events, reasons == "configured_recurrence"):
        raise ValueError("Target event flags disagree with resolved outcome reasons.")
    causes = np.zeros(len(durations), dtype=np.int8)
    causes[table.events] = 1
    causes[np.isin(reasons, COMPETING_REASONS)] = 2
    binary = np.full(len(durations), np.nan)
    binary[(causes == 2) | (durations == HORIZON_DAYS)] = 0.0
    binary[causes == 1] = 1.0
    return CompetingRiskOutcomes(durations, causes, binary)


def competing_training_table(table: PatientLandmarkTable) -> PatientLandmarkTable:
    """Treat pooled competitors as events and target recurrence as censoring."""
    outcomes = competing_risk_outcomes(table)
    events = outcomes.causes == 2
    return replace(
        table,
        events=events,
        signed_durations=signed_survival_durations(outcomes.durations, events),
        sample_weights=patient_uniform_weights(table.patient_ids),
        metadata={**table.metadata, "event_cause": "pooled_competing_event"},
    )
