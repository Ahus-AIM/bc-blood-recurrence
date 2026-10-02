from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter

from src.data.other_cancer_outcomes import recurrence_precedes_censoring
from src.data.recurrence_site_common import (
    RecurrenceSiteRule,
    patients_at_risk,
    site_label,
    validate_min_patients_at_risk,
)
from src.data.types import RightCensoringReason
from src.time_units import DAYS_PER_YEAR

AJ_EVENT_OF_INTEREST = 1
AJ_COMPETING_EVENT = 2
AJ_CENSORED = 0
AJ_JITTER_LEVEL_DAYS = 0.0001
AJ_EVENT_OFFSET_DAYS = 0.25
AJ_CENSOR_OFFSET_DAYS = 0.5
AJ_JITTER_LOOKAHEAD_DAYS = 0.75
INCIDENCE_COLUMNS = (
    "site",
    "years_since_diagnosis",
    "cumulative_incidence_percent",
    "confidence_interval_lower_percent",
    "confidence_interval_upper_percent",
    "patients_at_risk",
    "target_events",
    "competing_events",
    "censorings",
)
OutcomeReason = Literal["recurrence"] | RightCensoringReason


@dataclass(frozen=True)
class CompetingRiskPatient:
    """One patient-level observation for competing-risk estimation.

    `patient_id` identifies the final-cohort member. `outcome_day` is measured
    from BC diagnosis. `outcome_reason` distinguishes recurrence from the
    resolved competing or censoring outcome, while `recurrence_sites` contains
    every site registered on the first-recurrence day.
    """

    patient_id: str
    outcome_day: float
    outcome_reason: OutcomeReason
    recurrence_sites: tuple[str, ...]


def build_recurrence_site_cumulative_incidence(
    endpoint_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    *,
    max_years_since_diagnosis: int,
    min_days_since_diagnosis: int,
    min_patients_at_risk: int = 1,
    confidence_level: float = 0.95,
    random_seed: int = 42,
    overall_label: str | None,
    other_label: str | None,
) -> pd.DataFrame:
    """Build Panel D Aalen-Johansen estimates from patient-level outcomes."""
    validate_min_patients_at_risk(min_patients_at_risk)
    validate_confidence_level(confidence_level)
    patients = competing_risk_patients(
        endpoint_df,
        site_rules,
        min_days_since_diagnosis,
        other_label,
    )
    durations = np.asarray([patient.outcome_day for patient in patients], dtype=float)
    rows: list[dict[str, object]] = []
    for label in incidence_labels(site_rules, other_label, overall_label):
        statuses = competing_risk_statuses(
            patients,
            label,
            overall_label=overall_label,
        )
        rows.extend(
            aalen_johansen_rows(
                label,
                durations,
                statuses,
                max_years_since_diagnosis,
                min_patients_at_risk,
                confidence_level,
                random_seed,
            )
        )
    return pd.DataFrame(rows, columns=INCIDENCE_COLUMNS)


def incidence_labels(
    site_rules: tuple[RecurrenceSiteRule, ...],
    other_label: str | None,
    overall_label: str | None,
) -> tuple[str, ...]:
    labels = [rule.label for rule in site_rules]
    if other_label is not None:
        labels.append(other_label)
    if overall_label is not None:
        labels.append(overall_label)
    return tuple(labels)


def competing_risk_patients(
    endpoint_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    min_days_since_diagnosis: int,
    other_label: str | None,
) -> tuple[CompetingRiskPatient, ...]:
    """Reduce endpoint rows to one resolved outcome per eligible patient."""
    return tuple(
        competing_risk_patient(
            str(patient_id),
            patient_df,
            site_rules,
            min_days_since_diagnosis,
            other_label,
        )
        for patient_id, patient_df in endpoint_df.groupby("patient_id", sort=False)
    )


def competing_risk_patient(
    patient_id: str,
    patient_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    min_days_since_diagnosis: int,
    other_label: str | None,
) -> CompetingRiskPatient:
    recurrence_df = eligible_recurrence_rows(
        patient_df,
        min_days_since_diagnosis,
    )
    censoring_day, censoring_reason = patient_censoring(patient_df)
    recurrence_day = first_recurrence_day(recurrence_df)
    if recurrence_precedes_censoring(
        recurrence_day,
        censoring_day,
        censoring_reason,
    ):
        return recurrence_patient(
            patient_id,
            recurrence_df,
            cast(int, recurrence_day),
            site_rules,
            other_label,
        )
    if censoring_day is None or censoring_reason is None:
        raise ValueError(f"Patient {patient_id} has no recurrence or censoring time.")
    return CompetingRiskPatient(
        patient_id,
        float(max(0, censoring_day)),
        censoring_reason,
        (),
    )


def eligible_recurrence_rows(
    patient_df: pd.DataFrame,
    min_days_since_diagnosis: int,
) -> pd.DataFrame:
    return patient_df.loc[
        patient_df["days_since_diagnosis"].notna() & (patient_df["days_since_diagnosis"] >= min_days_since_diagnosis)
    ]


def recurrence_patient(
    patient_id: str,
    recurrence_df: pd.DataFrame,
    recurrence_day: int,
    site_rules: tuple[RecurrenceSiteRule, ...],
    other_label: str | None,
) -> CompetingRiskPatient:
    first_day_rows = recurrence_df[recurrence_df["days_since_diagnosis"] == recurrence_day]
    sites = tuple(
        dict.fromkeys(
            label
            for code in first_day_rows["code"]
            if (label := site_label(str(code), site_rules, other_label)) is not None
        )
    )
    return CompetingRiskPatient(
        patient_id,
        float(max(0, recurrence_day)),
        "recurrence",
        sites,
    )


def first_recurrence_day(recurrence_df: pd.DataFrame) -> int | None:
    if recurrence_df.empty:
        return None
    return int(recurrence_df["days_since_diagnosis"].min())


def patient_censoring(
    patient_df: pd.DataFrame,
) -> tuple[int | None, RightCensoringReason | None]:
    row = patient_df.iloc[0]
    if "right_censoring_day" in patient_df:
        day = optional_int(row["right_censoring_day"])
        return day, right_censoring_reason(row["right_censoring_reason"])
    death_day = optional_int(row["death_day"])
    last_observation_day = optional_int(row["last_observation_day"])
    if death_day is not None and (last_observation_day is None or death_day <= last_observation_day):
        return death_day, "death"
    return last_observation_day, "last_observation"


def optional_int(value: object) -> int | None:
    return None if pd.isna(value) else int(value)


def right_censoring_reason(value: object) -> RightCensoringReason | None:
    if value is None or pd.isna(value):
        return None
    reason = str(value)
    valid_reasons = {
        "death",
        "last_observation",
        "non_endpoint_recurrence",
        "other_cancer",
    }
    if reason not in valid_reasons:
        raise ValueError(f"Unknown right-censoring reason: {reason}")
    return cast(RightCensoringReason, reason)


def competing_risk_statuses(
    patients: tuple[CompetingRiskPatient, ...],
    site: str,
    *,
    overall_label: str | None,
) -> np.ndarray:
    return np.asarray(
        [competing_risk_status(patient, site, overall_label) for patient in patients],
        dtype=int,
    )


def competing_risk_status(
    patient: CompetingRiskPatient,
    site: str,
    overall_label: str | None,
) -> int:
    if patient.outcome_reason == "recurrence":
        if site == overall_label or site in patient.recurrence_sites:
            return AJ_EVENT_OF_INTEREST
        return AJ_COMPETING_EVENT
    if patient.outcome_reason == "last_observation":
        return AJ_CENSORED
    return AJ_COMPETING_EVENT


def aalen_johansen_rows(
    label: str,
    durations: np.ndarray,
    statuses: np.ndarray,
    max_years_since_diagnosis: int,
    min_patients_at_risk: int,
    confidence_level: float,
    random_seed: int,
) -> list[dict[str, object]]:
    max_day = float(max_years_since_diagnosis * DAYS_PER_YEAR)
    final_day = supported_final_day(durations, max_day, min_patients_at_risk)
    if final_day is None:
        return []
    days = outcome_days(durations, final_day)
    estimates = fit_aalen_johansen(
        durations,
        statuses,
        days,
        confidence_level,
        random_seed,
    )
    return incidence_rows(label, days, durations, statuses, estimates)


def supported_final_day(
    durations: np.ndarray,
    max_day: float,
    min_patients_at_risk: int,
) -> float | None:
    finite = np.sort(durations[np.isfinite(durations)])
    if len(finite) < min_patients_at_risk:
        return None
    if patients_at_risk(finite, max_day) >= min_patients_at_risk:
        return max_day
    return float(finite[-min_patients_at_risk])


def outcome_days(durations: np.ndarray, final_day: float) -> tuple[float, ...]:
    observed_days = {float(day) for day in durations if np.isfinite(day) and 0.0 <= day <= final_day}
    return tuple(sorted({0.0, final_day, *observed_days}))


def fit_aalen_johansen(
    durations: np.ndarray,
    statuses: np.ndarray,
    days: tuple[float, ...],
    confidence_level: float,
    random_seed: int,
) -> pd.DataFrame:
    """Fit after outcome resolution and return values on original integer days."""
    fitter = AalenJohansenFitter(
        jitter_level=AJ_JITTER_LEVEL_DAYS,
        seed=random_seed,
        alpha=1.0 - confidence_level,
    )
    random_state = np.random.get_state()
    try:
        fit_with_resolved_ties(
            fitter,
            lifelines_durations(durations, statuses),
            statuses,
        )
    finally:
        np.random.set_state(random_state)
    values = pd.concat(
        [fitter.cumulative_density_, fitter.confidence_interval_],
        axis=1,
    )
    queries = [day + AJ_JITTER_LOOKAHEAD_DAYS for day in days]
    sampled = values.reindex(values.index.union(queries)).sort_index().ffill().loc[queries]
    sampled.index = range(len(sampled))
    return normalized_confidence_intervals(sampled)


def lifelines_durations(
    durations: np.ndarray,
    statuses: np.ndarray,
) -> np.ndarray:
    """Place same-day censoring after events before lifelines jitters event ties."""
    offsets = np.where(
        statuses == AJ_CENSORED,
        AJ_CENSOR_OFFSET_DAYS,
        AJ_EVENT_OFFSET_DAYS,
    )
    return durations + offsets


def fit_with_resolved_ties(
    fitter: AalenJohansenFitter,
    durations: np.ndarray,
    statuses: np.ndarray,
) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Tied event times were detected.*",
            category=Warning,
        )
        fitter.fit(
            durations,
            statuses,
            event_of_interest=AJ_EVENT_OF_INTEREST,
            ci_labels=["confidence_interval_lower", "confidence_interval_upper"],
        )


def normalized_confidence_intervals(estimates: pd.DataFrame) -> pd.DataFrame:
    estimate_column = estimates.columns[0]
    zero = estimates[estimate_column] <= 0.0
    one = estimates[estimate_column] >= 1.0
    interval_columns = [
        "confidence_interval_lower",
        "confidence_interval_upper",
    ]
    estimates.loc[zero, interval_columns] = 0.0
    estimates.loc[one, interval_columns] = 1.0
    return estimates


def incidence_rows(
    label: str,
    days: tuple[float, ...],
    durations: np.ndarray,
    statuses: np.ndarray,
    estimates: pd.DataFrame,
) -> list[dict[str, object]]:
    point_estimates = estimates.iloc[:, 0].to_numpy(dtype=float)
    lower = estimates["confidence_interval_lower"].to_numpy(dtype=float)
    upper = estimates["confidence_interval_upper"].to_numpy(dtype=float)
    return [
        incidence_row(label, day, durations, statuses, estimate, low, high)
        for day, estimate, low, high in zip(
            days,
            point_estimates,
            lower,
            upper,
            strict=True,
        )
    ]


def incidence_row(
    label: str,
    day: float,
    durations: np.ndarray,
    statuses: np.ndarray,
    estimate: float,
    confidence_lower: float,
    confidence_upper: float,
) -> dict[str, object]:
    on_day = durations == day
    return {
        "site": label,
        "years_since_diagnosis": day / DAYS_PER_YEAR,
        "cumulative_incidence_percent": 100.0 * estimate,
        "confidence_interval_lower_percent": 100.0 * confidence_lower,
        "confidence_interval_upper_percent": 100.0 * confidence_upper,
        "patients_at_risk": patients_at_risk(durations, day),
        "target_events": int(np.sum(on_day & (statuses == AJ_EVENT_OF_INTEREST))),
        "competing_events": int(np.sum(on_day & (statuses == AJ_COMPETING_EVENT))),
        "censorings": int(np.sum(on_day & (statuses == AJ_CENSORED))),
    }


def validate_confidence_level(confidence_level: float) -> None:
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("`confidence_level` must be between 0 and 1.")
