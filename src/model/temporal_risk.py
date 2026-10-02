from __future__ import annotations

import math
import re
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator

from src.data.icd_codes import matches_icd_code
from src.data.other_cancer_outcomes import recurrence_precedes_censoring
from src.data.recurrence_site_common import RecurrenceSiteRule, site_label
from src.data.types import PatientBloodTestItem, PatientEndpointEvent, PatientLandmarkTable

TEMPORAL_RISK_COLUMNS = (
    "site",
    "patient_id",
    "landmark_day",
    "first_recurrence_day",
    "days_before_recurrence",
    "months_before_recurrence",
    "risk_score",
)
TEMPORAL_RISK_STAT_COLUMNS = (
    "site",
    "months_before_recurrence",
    "patient_count",
    "landmark_count",
    "mean_risk_quantile",
    "std_risk_quantile",
    "q25_risk_quantile",
    "median_risk_quantile",
    "q75_risk_quantile",
    "mean_risk_percentile",
    "std_risk_percentile",
    "q25_risk_percentile",
    "median_risk_percentile",
    "q75_risk_percentile",
)
SITE_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#999999",
)


@dataclass(frozen=True)
class TemporalRiskSiteConfig:
    """Recurrence-site rules for assigning first-recurrence patients to plot panels.

    `site_rules` define explicit anatomical sites. `include_overall` adds an
    overall panel for any configured input recurrence. `other_label` collects
    patients whose first configured input recurrence matches none of the
    explicit site rules.
    """

    site_rules: tuple[RecurrenceSiteRule, ...]
    include_overall: bool
    overall_label: str
    other_label: str | None


def build_temporal_risk_landmark_frame(
    table: PatientLandmarkTable,
    risk_scores: np.ndarray,
    items: tuple[PatientBloodTestItem, ...],
    site_config: TemporalRiskSiteConfig,
    *,
    input_recurrence_code_filter_regex: str,
    censoring_horizon_days: int,
    month_days: int,
    include_recurrence_day: bool = False,
) -> pd.DataFrame:
    validate_temporal_risk_inputs(table, risk_scores, censoring_horizon_days, month_days)
    recurrences = first_recurrence_sites_by_patient(
        items,
        site_config,
        input_recurrence_code_filter_regex,
    )
    rows: list[dict[str, object]] = []
    for patient_id, landmark_day, risk_score in zip(
        table.patient_ids,
        table.landmark_days,
        risk_scores,
        strict=True,
    ):
        recurrence = recurrences.get(str(patient_id))
        if recurrence is None:
            continue
        days_before_recurrence = int(recurrence["first_recurrence_day"]) - int(landmark_day)
        minimum_days = 0 if include_recurrence_day else 1
        if days_before_recurrence < minimum_days or days_before_recurrence > censoring_horizon_days:
            continue
        for site in recurrence["sites"]:
            rows.append(
                {
                    "site": site,
                    "patient_id": str(patient_id),
                    "landmark_day": int(landmark_day),
                    "first_recurrence_day": int(recurrence["first_recurrence_day"]),
                    "days_before_recurrence": days_before_recurrence,
                    "months_before_recurrence": days_before_recurrence // month_days,
                    "risk_score": float(risk_score),
                }
            )
    return pd.DataFrame(rows, columns=TEMPORAL_RISK_COLUMNS)


def validate_temporal_risk_inputs(
    table: PatientLandmarkTable,
    risk_scores: np.ndarray,
    censoring_horizon_days: int,
    month_days: int,
) -> None:
    if len(risk_scores) != len(table.landmark_days):
        raise ValueError("Risk-score count must match the landmark table row count.")
    if censoring_horizon_days < 1:
        raise ValueError("`dataset.censoring_horizon_days` must be at least 1.")
    if month_days < 1:
        raise ValueError("`temporal_risk.month_days` must be at least 1.")


def first_recurrence_sites_by_patient(
    items: tuple[PatientBloodTestItem, ...],
    site_config: TemporalRiskSiteConfig,
    input_recurrence_code_filter_regex: str,
) -> dict[str, dict[str, object]]:
    input_regex = re.compile(input_recurrence_code_filter_regex)
    recurrences: dict[str, dict[str, object]] = {}
    for item in items:
        event = first_input_recurrence_event(item, input_regex)
        if event is None:
            continue
        sites = site_labels_for_first_recurrence(event, site_config)
        if not sites:
            continue
        recurrences[item.patient_id] = {
            "first_recurrence_day": event.days_since_diagnosis,
            "sites": sites,
        }
    return recurrences


def first_input_recurrence_event(
    item: PatientBloodTestItem,
    input_regex: re.Pattern[str],
) -> PatientEndpointEvent | None:
    recurrence_day = item.input_recurrence_day
    if not recurrence_precedes_censoring(
        recurrence_day,
        item.right_censoring_day,
        item.right_censoring_reason,
    ):
        return None
    return next(
        (
            event
            for event in item.distant_recurrence_events
            if event.days_since_diagnosis == recurrence_day and matches_icd_code(event.code, input_regex)
        ),
        None,
    )


def site_labels_for_first_recurrence(
    event: PatientEndpointEvent,
    site_config: TemporalRiskSiteConfig,
) -> tuple[str, ...]:
    labels: list[str] = []
    if site_config.include_overall:
        labels.append(site_config.overall_label)
    site_label = explicit_site_label(event.code, site_config.site_rules)
    if site_label is None:
        site_label = site_config.other_label
    if site_label is not None:
        labels.append(site_label)
    return tuple(dict.fromkeys(labels))


def explicit_site_label(
    code: str,
    site_rules: tuple[RecurrenceSiteRule, ...],
) -> str | None:
    return site_label(code, site_rules, None)


def assign_bootstrap_mean_quantiles(
    landmark_df: pd.DataFrame,
    bootstrap_risk_scores: pd.DataFrame,
) -> pd.DataFrame:
    if landmark_df.empty:
        return landmark_df.assign(risk_quantile=pd.Series(dtype=float), risk_percentile=pd.Series(dtype=float))
    output_df = landmark_df.copy()
    quantiles = mean_bootstrap_quantiles(
        output_df["risk_score"].to_numpy(dtype=float),
        bootstrap_risk_scores,
    )
    output_df["risk_quantile"] = quantiles
    output_df["risk_percentile"] = quantiles * 100.0
    return output_df


def mean_bootstrap_quantiles(
    risk_scores: np.ndarray,
    bootstrap_risk_scores: pd.DataFrame,
) -> np.ndarray:
    validate_bootstrap_risk_scores(bootstrap_risk_scores)
    quantiles = np.full(len(risk_scores), np.nan, dtype=float)
    finite_values = np.isfinite(risk_scores)
    if not bool(finite_values.any()):
        return quantiles
    quantile_sum = np.zeros(int(finite_values.sum()), dtype=float)
    valid_run_count = 0
    for _run, run_df in bootstrap_risk_scores.groupby("bootstrap_run", sort=True):
        reference_scores = finite_sorted_scores(run_df["risk_score"])
        if len(reference_scores) == 0:
            continue
        values = risk_scores[finite_values]
        quantile_sum += np.searchsorted(reference_scores, values, side="right") / float(len(reference_scores))
        valid_run_count += 1
    if valid_run_count == 0:
        raise ValueError("No bootstrap runs contain finite test risk scores.")
    quantiles[finite_values] = quantile_sum / valid_run_count
    return quantiles


def validate_bootstrap_risk_scores(bootstrap_risk_scores: pd.DataFrame) -> None:
    required_columns = {"bootstrap_run", "risk_score"}
    missing = required_columns.difference(bootstrap_risk_scores.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Bootstrap risk-score table is missing columns: {joined}.")
    if bootstrap_risk_scores.empty:
        raise ValueError("Bootstrap risk-score table is empty.")


def finite_sorted_scores(scores: pd.Series) -> np.ndarray:
    values = scores.to_numpy(dtype=float)
    return np.sort(values[np.isfinite(values)])


def summarize_temporal_risk(
    landmark_df: pd.DataFrame,
    *,
    site_order: tuple[str, ...],
) -> pd.DataFrame:
    if landmark_df.empty:
        return pd.DataFrame(columns=TEMPORAL_RISK_STAT_COLUMNS)
    valid_df = landmark_df.dropna(subset=["risk_quantile"]).copy()
    if valid_df.empty:
        return pd.DataFrame(columns=TEMPORAL_RISK_STAT_COLUMNS)
    patient_month_df = patient_month_temporal_risk(valid_df)
    rows = [
        temporal_risk_stat_row(site, month, group)
        for (site, month), group in patient_month_df.groupby(
            ["site", "months_before_recurrence"],
            sort=False,
        )
    ]
    stats_df = pd.DataFrame(rows, columns=TEMPORAL_RISK_STAT_COLUMNS)
    return sort_temporal_risk_stats(stats_df, site_order)


def patient_month_temporal_risk(landmark_df: pd.DataFrame) -> pd.DataFrame:
    grouped = landmark_df.groupby(
        ["site", "patient_id", "months_before_recurrence"],
        sort=False,
    )
    return grouped.agg(
        risk_quantile=("risk_quantile", "mean"),
        landmark_count=("risk_quantile", "size"),
    ).reset_index()


def temporal_risk_stat_row(
    site: str,
    month: int,
    group: pd.DataFrame,
) -> dict[str, object]:
    values = group["risk_quantile"].to_numpy(dtype=float)
    stats = risk_quantile_stats(values)
    return {
        "site": str(site),
        "months_before_recurrence": int(month),
        "patient_count": int(len(group)),
        "landmark_count": int(group["landmark_count"].sum()),
        **stats,
        **{key.replace("quantile", "percentile"): value * 100.0 for key, value in stats.items()},
    }


def risk_quantile_stats(values: np.ndarray) -> dict[str, float]:
    return {
        "mean_risk_quantile": float(np.mean(values)),
        "std_risk_quantile": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        "q25_risk_quantile": float(np.quantile(values, 0.25)),
        "median_risk_quantile": float(np.quantile(values, 0.50)),
        "q75_risk_quantile": float(np.quantile(values, 0.75)),
    }


def sort_temporal_risk_stats(
    stats_df: pd.DataFrame,
    site_order: tuple[str, ...],
) -> pd.DataFrame:
    site_rank = {site: index for index, site in enumerate(site_order)}
    ranked_df = stats_df.assign(
        _site_rank=stats_df["site"].map(lambda site: site_rank.get(str(site), len(site_rank))),
    )
    return (
        ranked_df.sort_values(["_site_rank", "months_before_recurrence"], kind="mergesort")
        .drop(columns="_site_rank")
        .reset_index(drop=True)
    )


def temporal_risk_site_order(site_config: TemporalRiskSiteConfig) -> tuple[str, ...]:
    labels: list[str] = []
    if site_config.include_overall:
        labels.append(site_config.overall_label)
    labels.extend(rule.label for rule in site_config.site_rules)
    if site_config.other_label is not None:
        labels.append(site_config.other_label)
    return tuple(dict.fromkeys(labels))


def build_temporal_risk_figure(
    stats_df: pd.DataFrame,
    *,
    site_order: tuple[str, ...],
    max_month: int,
    figure_width: float,
    panel_height: float,
    font_size: float,
    title_font_size: float,
) -> Figure:
    columns = 2 if len(site_order) > 1 else 1
    rows = max(1, math.ceil(len(site_order) / columns))
    figure_height = max(panel_height, rows * panel_height)
    plt.rcParams.update({"font.size": font_size, "axes.titlesize": title_font_size})
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_width, figure_height),
        constrained_layout=True,
        squeeze=False,
    )
    flat_axes = tuple(axes.ravel())
    for index, site in enumerate(site_order):
        color = SITE_COLORS[index % len(SITE_COLORS)]
        site_df = stats_df[stats_df["site"] == site]
        draw_temporal_risk_panel(flat_axes[index], site_df, site, max_month, color)
    for axis in flat_axes[len(site_order) :]:
        axis.set_axis_off()
    return figure


def draw_temporal_risk_panel(
    axis: Axes,
    site_df: pd.DataFrame,
    site: str,
    max_month: int,
    color: str,
) -> None:
    axis.set_title(site)
    axis.set_xlabel("Months before first recurrence")
    axis.set_ylabel("Test-set risk percentile")
    axis.set_xlim(float(max_month), 0.0)
    axis.set_ylim(0.0, 100.0)
    axis.set_yticks((0, 25, 50, 75, 100))
    axis.xaxis.set_major_locator(FixedLocator(month_ticks(max_month)))
    axis.xaxis.set_major_formatter(FixedFormatter(tuple(str(tick) for tick in month_ticks(max_month))))
    axis.grid(color="0.9", linewidth=0.6)
    axis.set_axisbelow(True)
    if site_df.empty:
        draw_empty_temporal_risk_panel(axis)
        return
    x = site_df["months_before_recurrence"].to_numpy(dtype=float)
    mean = site_df["mean_risk_percentile"].to_numpy(dtype=float)
    std = site_df["std_risk_percentile"].to_numpy(dtype=float)
    q25 = site_df["q25_risk_percentile"].to_numpy(dtype=float)
    median = site_df["median_risk_percentile"].to_numpy(dtype=float)
    q75 = site_df["q75_risk_percentile"].to_numpy(dtype=float)
    axis.fill_between(x, clipped_percent(mean - std), clipped_percent(mean + std), color=color, alpha=0.12)
    axis.fill_between(x, q25, q75, color=color, alpha=0.24)
    axis.plot(x, median, color=color, linewidth=1.7, label="Median")
    axis.plot(x, mean, color=color, linewidth=1.2, linestyle="--", label="Mean")
    axis.legend(frameon=False, loc="upper left", fontsize=6.0)


def month_ticks(max_month: int) -> tuple[int, ...]:
    if max_month <= 12:
        step = 3
    elif max_month <= 36:
        step = 6
    else:
        step = 12
    ticks = list(range(0, max_month + 1, step))
    if ticks[-1] != max_month:
        ticks.append(max_month)
    return tuple(ticks)


def clipped_percent(values: np.ndarray) -> np.ndarray:
    return np.clip(values, 0.0, 100.0)


def draw_empty_temporal_risk_panel(axis: Axes) -> None:
    axis.text(
        0.5,
        0.5,
        "No eligible patient-months",
        ha="center",
        va="center",
        transform=axis.transAxes,
        color="0.45",
    )
