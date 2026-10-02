from __future__ import annotations

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.recurrence_site_incidence import RecurrenceSiteRule
from src.data.types import PatientBloodTestItem, PatientEndpointEvent, PatientLandmarkTable
from src.model.temporal_risk import (
    TemporalRiskSiteConfig,
    assign_bootstrap_mean_quantiles,
    build_temporal_risk_figure,
    build_temporal_risk_landmark_frame,
    summarize_temporal_risk,
    temporal_risk_site_order,
)


def test_bootstrap_quantiles_average_per_run_empirical_cdfs() -> None:
    landmark_df = pd.DataFrame({"risk_score": [5.0, 10.0, 20.0]})
    bootstrap_risk_scores = pd.DataFrame(
        {
            "bootstrap_run": [0, 0, 1, 1],
            "risk_score": [0.0, 10.0, 10.0, 20.0],
        }
    )

    quantile_df = assign_bootstrap_mean_quantiles(landmark_df, bootstrap_risk_scores)

    assert quantile_df["risk_quantile"].tolist() == pytest.approx([0.25, 0.75, 1.0])
    assert quantile_df["risk_percentile"].tolist() == pytest.approx([25.0, 75.0, 100.0])


def test_temporal_risk_site_matching_accepts_dotted_icd_codes() -> None:
    site_config = temporal_site_config()
    table = landmark_table(
        patient_ids=("P0",),
        landmark_days=(40,),
        risk_scores=(0.8,),
    )
    items = (item("P0", PatientEndpointEvent(100, "C78.7")),)

    landmark_df = build_temporal_risk_landmark_frame(
        table,
        np.asarray([0.8], dtype=float),
        items,
        site_config,
        input_recurrence_code_filter_regex=r"C78\d*",
        censoring_horizon_days=90,
        month_days=30,
    )

    assert set(landmark_df["site"]) == {"Any", "Liver"}


def test_temporal_risk_can_include_recurrence_day() -> None:
    table = landmark_table(patient_ids=("P0",), landmark_days=(100,), risk_scores=(0.8,))
    frame = build_temporal_risk_landmark_frame(
        table, np.asarray([0.8]), (item("P0", PatientEndpointEvent(100, "C787")),), temporal_site_config(),
        input_recurrence_code_filter_regex=r"C78\d*", censoring_horizon_days=30, month_days=30,
        include_recurrence_day=True,
    )
    assert set(frame["site"]) == {"Any", "Liver"}
    assert set(frame["days_before_recurrence"]) == {0}


def test_month_bins_start_at_zero_and_keep_the_inclusive_horizon() -> None:
    days_before = (0, 1, 29, 30, 59, 60, 89, 90, 91)
    scores = (0.8,) * len(days_before)
    table = landmark_table(
        patient_ids=("P0",) * len(days_before),
        landmark_days=tuple(100 - day for day in days_before),
        risk_scores=scores,
    )
    frame = build_temporal_risk_landmark_frame(
        table, np.asarray(scores), (item("P0", PatientEndpointEvent(100, "C787")),), temporal_site_config(),
        input_recurrence_code_filter_regex=r"C78\d*", censoring_horizon_days=90, month_days=30,
        include_recurrence_day=True,
    )
    overall = frame.loc[frame["site"].eq("Any")]
    assert overall["days_before_recurrence"].tolist() == [0, 1, 29, 30, 59, 60, 89, 90]
    assert overall["months_before_recurrence"].tolist() == [0, 0, 0, 1, 1, 2, 2, 3]


def test_temporal_risk_uses_first_site_horizon_and_patient_month_averages() -> None:
    site_config = temporal_site_config()
    table = landmark_table(
        patient_ids=("P0", "P0", "P0", "P1", "P2", "P3", "P4", "P4", "P5"),
        landmark_days=(35, 40, 10, 50, 60, 50, 35, 40, 10),
        risk_scores=(0.8, 0.6, 0.2, 0.1, 0.3, 0.4, 0.5, 0.7, 0.9),
    )
    items = (
        item("P0", PatientEndpointEvent(100, "C787")),
        item("P1", PatientEndpointEvent(100, "C780")),
        item("P2", PatientEndpointEvent(100, "C789")),
        item("P3", PatientEndpointEvent(80, "C780"), PatientEndpointEvent(120, "C787")),
        item("P4", PatientEndpointEvent(100, "C787")),
        item("P5", PatientEndpointEvent(200, "C787")),
    )

    landmark_df = build_temporal_risk_landmark_frame(
        table,
        np.asarray([0.8, 0.6, 0.2, 0.1, 0.3, 0.4, 0.5, 0.7, 0.9], dtype=float),
        items,
        site_config,
        input_recurrence_code_filter_regex=r"C78\d*",
        censoring_horizon_days=90,
        month_days=30,
    )
    stats_df = summarize_temporal_risk(
        landmark_df.assign(risk_quantile=landmark_df["risk_score"]),
        site_order=temporal_risk_site_order(site_config),
    )

    assert set(landmark_df["site"]) == {"Any", "Liver", "Lung", "Other"}
    assert landmark_df[landmark_df["patient_id"] == "P5"].empty
    assert landmark_df[(landmark_df["patient_id"] == "P3") & (landmark_df["site"] == "Liver")].empty

    liver_month_2 = stats_df[
        (stats_df["site"] == "Liver")
        & (stats_df["months_before_recurrence"] == 2)
    ].iloc[0]
    assert liver_month_2["patient_count"] == 2
    assert liver_month_2["landmark_count"] == 4
    assert liver_month_2["mean_risk_quantile"] == pytest.approx(0.65)
    assert liver_month_2["median_risk_quantile"] == pytest.approx(0.65)
    assert liver_month_2["q25_risk_quantile"] == pytest.approx(0.625)
    assert liver_month_2["q75_risk_quantile"] == pytest.approx(0.675)
    assert liver_month_2["std_risk_quantile"] == pytest.approx(math.sqrt(0.005))


def test_temporal_risk_figure_builds_all_configured_site_panels() -> None:
    site_config = temporal_site_config()
    stats_df = pd.DataFrame(
        [
            {
                "site": "Any",
                "months_before_recurrence": 1,
                "patient_count": 1,
                "landmark_count": 1,
                "mean_risk_quantile": 0.5,
                "std_risk_quantile": 0.0,
                "q25_risk_quantile": 0.5,
                "median_risk_quantile": 0.5,
                "q75_risk_quantile": 0.5,
                "mean_risk_percentile": 50.0,
                "std_risk_percentile": 0.0,
                "q25_risk_percentile": 50.0,
                "median_risk_percentile": 50.0,
                "q75_risk_percentile": 50.0,
            }
        ]
    )

    figure = build_temporal_risk_figure(
        stats_df,
        site_order=temporal_risk_site_order(site_config),
        max_month=3,
        figure_width=5.0,
        panel_height=1.2,
        font_size=7.0,
        title_font_size=8.0,
    )

    assert len(figure.axes) == 4
    plt.close(figure)


def temporal_site_config() -> TemporalRiskSiteConfig:
    return TemporalRiskSiteConfig(
        site_rules=(
            RecurrenceSiteRule("Liver", r"C787\d*"),
            RecurrenceSiteRule("Lung", r"C780\d*"),
        ),
        include_overall=True,
        overall_label="Any",
        other_label="Other",
    )


def item(
    patient_id: str,
    *events: PatientEndpointEvent,
) -> PatientBloodTestItem:
    return PatientBloodTestItem(
        patient_id=patient_id,
        split="test",
        diagnosis_year=2020,
        age_at_diagnosis=float("nan"),
        blood_days_since_diagnosis=torch.empty(0, dtype=torch.long),
        blood_test_type_ids=torch.empty(0, dtype=torch.long),
        blood_values=torch.empty(0, dtype=torch.float32),
        input_recurrence_day=events[0].days_since_diagnosis if events else None,
        endpoint_day=events[0].days_since_diagnosis if events else None,
        death_day=None,
        right_censoring_day=365,
        last_observation_day=365,
        source_last_observation_day=365,
        distant_recurrence_events=events,
        valid_landmark_days=(),
    )


def landmark_table(
    *,
    patient_ids: tuple[str, ...],
    landmark_days: tuple[int, ...],
    risk_scores: tuple[float, ...],
) -> PatientLandmarkTable:
    row_count = len(patient_ids)
    durations = np.ones(row_count, dtype=np.int32)
    return PatientLandmarkTable(
        features=np.asarray(risk_scores, dtype=np.float32).reshape(row_count, 1),
        signed_durations=durations.astype(np.float64),
        durations=durations,
        events=np.ones(row_count, dtype=bool),
        sample_weights=None,
        feature_names=("risk",),
        patient_ids=np.asarray(patient_ids, dtype=object),
        landmark_days=np.asarray(landmark_days, dtype=np.int32),
        prior_non_endpoint_recurrence=np.zeros(row_count, dtype=bool),
        metadata={},
    )
