from __future__ import annotations

import re

import numpy as np
import pandas as pd
from lifelines import KaplanMeierFitter

from src.data.dataset_config import PatientDatasetConfig
from src.data.other_cancer_outcomes import (
    ResolvedRightCensoring,
    recurrence_precedes_censoring,
    resolve_right_censoring,
)
from src.data.recurrence_site_common import (
    RecurrenceSiteRule,
    code_matches_site_rule as code_matches_site_rule,
    patients_at_risk,
    site_label,
    validate_min_patients_at_risk,
)
from src.data.recurrence_site_cumulative_incidence import (
    build_recurrence_site_cumulative_incidence as build_recurrence_site_cumulative_incidence,
)
from src.data.types import PatientBloodTestItem, PatientEndpointEvent
from src.time_units import DAYS_PER_YEAR

POST_RECURRENCE_SURVIVAL_VERSION = 2


def endpoint_row(
    patient: PatientBloodTestItem,
    event: PatientEndpointEvent | None,
) -> dict[str, object]:
    return {
        "patient_id": patient.patient_id,
        "diagnosis_year": patient.diagnosis_year,
        "days_since_diagnosis": np.nan if event is None else event.days_since_diagnosis,
        "code": "" if event is None else event.code,
        "description": "" if event is None else event.description,
        "death_day": patient.death_day,
        "last_observation_day": patient.source_last_observation_day,
    }


def build_first_recurrence_site_kaplan_meier(
    endpoint_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    *,
    max_years_since_origin: int,
    min_days_since_diagnosis: int,
    min_patients_at_risk: int = 1,
    time_origin: str,
    other_label: str | None,
    overall_label: str | None = None,
) -> pd.DataFrame:
    validate_min_patients_at_risk(min_patients_at_risk)
    first_df = first_recurrence_rows(endpoint_df, site_rules, min_days_since_diagnosis, other_label)
    rows = []
    if overall_label is not None:
        overall_df = first_recurrence_rows_any(endpoint_df, min_days_since_diagnosis)
        rows.extend(
            kaplan_meier_rows(
                overall_label,
                overall_df,
                time_origin,
                max_years_since_origin,
                min_patients_at_risk,
            )
        )
    for site, site_df in first_df.groupby("site", sort=False):
        rows.extend(
            kaplan_meier_rows(
                str(site),
                site_df,
                time_origin,
                max_years_since_origin,
                min_patients_at_risk,
            )
        )
    return pd.DataFrame(rows)


def load_item_endpoint_site_rows(
    items: tuple[PatientBloodTestItem, ...],
    config: PatientDatasetConfig,
) -> pd.DataFrame:
    """Return eligible first-recurrence rows with source dates and recurrence censoring."""
    other_cancer_regex = (
        re.compile(config.other_cancer_code_filter_regex) if config.exclude_and_censor_other_cancer_patients else None
    )
    rows: list[dict[str, object]] = []
    for item in items:
        # First recurrence starts post-recurrence survival, even for a different target site.
        right_censoring = resolve_right_censoring(
            item.death_day,
            item.source_last_observation_day,
            item.clinical_events,
            other_cancer_regex,
        )
        recurrence_day = item.input_recurrence_day
        recurrence_observed = recurrence_precedes_censoring(
            recurrence_day,
            right_censoring.day,
            right_censoring.reason,
        )
        events = tuple(
            event
            for event in item.distant_recurrence_events
            if recurrence_observed
            and event.days_since_diagnosis == recurrence_day
        )
        rows.extend(item_endpoint_row(item, event, right_censoring) for event in events or (None,))
    return pd.DataFrame(rows)


def item_endpoint_row(
    item: PatientBloodTestItem,
    event: PatientEndpointEvent | None,
    right_censoring: ResolvedRightCensoring,
) -> dict[str, object]:
    return {
        **endpoint_row(item, event),
        "right_censoring_day": right_censoring.day,
        "right_censoring_reason": right_censoring.reason,
    }


def kaplan_meier_rows(
    label: str,
    site_df: pd.DataFrame,
    time_origin: str,
    max_years_since_origin: int,
    min_patients_at_risk: int,
) -> list[dict[str, object]]:
    durations, events = survival_durations(site_df, time_origin)
    points = kaplan_meier_step_points(
        durations,
        events,
        max_years_since_origin,
        min_patients_at_risk,
    )
    return [
        {
            "site": label,
            "years_since_origin": year,
            "survival_percent": percent,
        }
        for year, percent in points
    ]


def first_site_events(
    endpoint_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    min_days_since_diagnosis: int,
    other_label: str | None,
) -> dict[str, np.ndarray]:
    first_df = first_recurrence_rows(
        endpoint_df,
        site_rules,
        min_days_since_diagnosis,
        other_label,
    )
    labels = [rule.label for rule in site_rules]
    if other_label is not None:
        labels.append(other_label)
    return {
        label: first_event_days(
            first_df.loc[first_df["site"] == label],
            min_days_since_diagnosis,
        )
        for label in labels
    }


def first_event_days(endpoint_df: pd.DataFrame, min_days_since_diagnosis: int) -> np.ndarray:
    event_df = endpoint_df.loc[endpoint_df["days_since_diagnosis"].notna()].copy()
    event_df = event_df[event_df["days_since_diagnosis"] >= min_days_since_diagnosis]
    if event_df.empty:
        return np.asarray([], dtype=float)
    first_days = event_df.groupby("patient_id")["days_since_diagnosis"].min()
    return first_days.to_numpy(dtype=float)


def first_recurrence_rows(
    endpoint_df: pd.DataFrame,
    site_rules: tuple[RecurrenceSiteRule, ...],
    min_days_since_diagnosis: int,
    other_label: str | None,
) -> pd.DataFrame:
    event_df = endpoint_df.loc[endpoint_df["days_since_diagnosis"].notna()].copy()
    event_df = event_df[event_df["days_since_diagnosis"] >= min_days_since_diagnosis]
    if event_df.empty:
        return pd.DataFrame(columns=[*endpoint_df.columns, "site"])
    event_df["site"] = event_df["code"].map(lambda code: site_label(str(code), site_rules, other_label))
    event_df = event_df[event_df["site"].notna()]
    return event_df.sort_values(["patient_id", "days_since_diagnosis"]).drop_duplicates(
        ["patient_id", "site"],
        keep="first",
    )


def first_recurrence_rows_any(
    endpoint_df: pd.DataFrame,
    min_days_since_diagnosis: int,
) -> pd.DataFrame:
    event_df = endpoint_df.loc[endpoint_df["days_since_diagnosis"].notna()].copy()
    event_df = event_df[event_df["days_since_diagnosis"] >= min_days_since_diagnosis]
    if event_df.empty:
        return pd.DataFrame(columns=endpoint_df.columns)
    return event_df.sort_values(["patient_id", "days_since_diagnosis"]).drop_duplicates("patient_id", keep="first")


def survival_durations(site_df: pd.DataFrame, time_origin: str) -> tuple[np.ndarray, np.ndarray]:
    """Follow recurrence to recorded death, otherwise the last source observation.

    Recurrence-model censoring does not truncate post-recurrence survival.
    """
    if time_origin != "recurrence":
        raise ValueError("Only `recurrence` time origin is currently supported.")
    start = site_df["days_since_diagnosis"].to_numpy(dtype=float)
    death = site_df["death_day"].to_numpy(dtype=float)
    last = site_df["last_observation_day"].to_numpy(dtype=float)
    event = ~np.isnan(death)
    end = np.where(event, death, last)
    durations = np.maximum(0.0, end - start)
    return durations, event


def kaplan_meier_step_points(
    durations: np.ndarray,
    events: np.ndarray,
    max_years_since_origin: int,
    min_patients_at_risk: int,
) -> tuple[tuple[float, float], ...]:
    """Adapt lifelines estimates to death-time steps with figure-only risk-set suppression."""
    max_days = max_years_since_origin * DAYS_PER_YEAR
    finite = np.isfinite(durations)
    finite_durations = durations[finite]
    if len(finite_durations) < min_patients_at_risk:
        return ()
    points: list[tuple[float, float]] = [(0.0, 100.0)]
    if len(finite_durations) == 0:
        return tuple(points)
    fitter = KaplanMeierFitter().fit(finite_durations, event_observed=events[finite])
    event_table = fitter.event_table
    death_days = event_table.index[
        (event_table["observed"] > 0)
        & (event_table.index <= max_days)
        & (event_table["at_risk"] >= min_patients_at_risk)
    ]
    survival = fitter.survival_function_.iloc[:, 0].loc[death_days]
    points.extend((float(day) / DAYS_PER_YEAR, 100.0 * float(value)) for day, value in survival.items())
    final_year = float(max_years_since_origin)
    if points[-1][0] < final_year and patients_at_risk(durations, max_days) >= min_patients_at_risk:
        points.append((final_year, points[-1][1]))
    return tuple(points)
