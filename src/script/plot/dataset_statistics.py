from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FixedFormatter, FixedLocator, MultipleLocator

from src.config import load_yml, resolve_path
from src.figure_style import figure_typography
from src.data.dataset import PatientBloodTestDataset
from src.data.cohort_age_exchange import age_display_summary, find_age_summary
from src.data.dataset_config import PatientDatasetConfig
from src.data.cohort_figure import CohortFigureData, suppress_risk_counts
from src.data.dataset_statistics import prepared_recurrence_site_rows
from src.data.dataset_statistics_export import (
    dataset_statistics_updates_from_payload,
    load_dataset_statistics_payload,
    recurrence_incidence_from_payload,
    recurrence_survival_from_payload,
    validate_payload_dataset_config,
    write_dataset_statistics_payload,
)
from src.data.recurrence_site_cumulative_incidence import (
    build_recurrence_site_cumulative_incidence,
)
from src.data.recurrence_site_incidence import (
    RecurrenceSiteRule,
    build_first_recurrence_site_kaplan_meier,
)
from src.script.plot.blood_test_imputation_chart import (
    draw_bloodtest_count_bar_chart,
    draw_empty_panel,
)
from src.script.plot.patient_flow_diagram import display_text, draw_patient_flow_diagram
from src.script.plot.cohort_composition_panels import (
    draw_annual_curve_table,
    draw_cohort_composition,
)

from src.script.plot.dataset_statistics_config import (
    BloodTestCountPlotConfig as BloodTestCountPlotConfig,
    DatasetStatisticsFigureConfig as DatasetStatisticsFigureConfig,
    DatasetStatisticsPlotConfig as DatasetStatisticsPlotConfig,
    DatasetStatisticsSourceConfig as DatasetStatisticsSourceConfig,
    DiagnosisYearBarChartConfig as DiagnosisYearBarChartConfig,
    FirstRecurrenceSiteSurvivalPlotConfig as FirstRecurrenceSiteSurvivalPlotConfig,
    PatientFlowDiagramConfig as PatientFlowDiagramConfig,
    RecurrenceSiteIncidencePlotConfig as RecurrenceSiteIncidencePlotConfig,
    load_dataset_section,
)
from src.script.plot.overview_layout import (
    cohort_headers, create_cohort_axes, fill_right_column, fit_cohort_footer, fit_recurrence_tick_spacing,
    load_overview_design, pack_cohort_panels, stack_survival_panel,
)
from src.script.plot.overview_style import (
    CURVE_WIDTH,
    DIAGNOSIS_YEAR_STACK_COLORS,
    OVERALL_CURVE_WIDTH,
    OVERALL_SITE_COLOR,
    publication_rc,
    recurrence_site_colors,
    save_publication_figure,
    style_plot,
    wrap_to_width,
)

CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "dataset_statistics.yml"
LOGGER = logging.getLogger(__name__)
DIAGNOSIS_YEAR_YTICK_STEP = 100
DIAGNOSIS_YEAR_Y_MAX = 600
DIAGNOSIS_YEAR_X_MIN = 2004
DIAGNOSIS_YEAR_FIRST_TICK = 2005
DIAGNOSIS_YEAR_XTICK_STEP = 5
DIAGNOSIS_YEAR_BAR_WIDTH = 0.8
DIAGNOSIS_YEAR_EDGE_PAD = 1.0 - DIAGNOSIS_YEAR_BAR_WIDTH


def render_saved_dataset_figure(run_dir: Path, *, plot_config_path: Path = CONFIG_PATH) -> tuple[Path, ...]:
    """Redraw one saved cohort with current settings and return the figure paths."""
    raw_config = load_yml(plot_config_path)
    raw_config["dataset"]["statistics_output_path"] = str((run_dir / "dataset_statistics_values.yml").resolve())
    raw_config["dataset"]["dataset_config_path"] = str((run_dir / "training_config.yml").resolve())
    raw_config["figure"]["output_path"] = str((run_dir / "dataset_statistics.png").resolve())
    source = raw_config["dataset"]
    payload_path = resolve_path(str(source["statistics_output_path"]), plot_config_path.parent)
    payload = load_dataset_statistics_payload(payload_path)
    if "cohort_figure" not in payload:
        raise ValueError("This saved statistics file predates the six-panel layout. Regenerate the benchmark's "
                         "dataset artifacts to export population composition and annual curve tables.")
    dataset_config_path = resolve_path(str(source["dataset_config_path"]), plot_config_path.parent)
    validate_statistics_source(payload, load_dataset_section(dataset_config_path), dataset_config_path.parent)
    config = plot_config_from_payload(raw_config, payload, config_dir=plot_config_path.parent)
    figure = build_dataset_statistics_figure(
        config,
        recurrence_incidence_df=recurrence_incidence_from_payload(payload),
        recurrence_survival_df=recurrence_survival_from_payload(payload),
    )
    paths = save_publication_figure(figure, config.figure.output_path, config.figure.dpi)
    plt.close(figure)
    payload["metadata"]["manual_age_summary"] = asdict(config.manual_age)
    payload["metadata"]["age_return_sources"] = [str(path) for path in config.age_return_sources]
    write_dataset_statistics_payload(payload, payload_path)
    print(f"wrote {config.figure.output_path}", flush=True)
    return paths


def plot_config_from_payload(
    raw_config: dict[str, Any],
    payload: dict[str, Any],
    *,
    config_dir: Path,
) -> DatasetStatisticsPlotConfig:
    """Apply saved values in memory, preserving both source files and attribution."""
    updated = deepcopy(raw_config)
    for (section, key), value in dataset_statistics_updates_from_payload(payload).items():
        updated[section][key] = value
    configured_sources = updated["panel_c"].get("show_imputation_sources", False)
    updated["panel_c"]["show_imputation_sources"] = payload["metadata"].get(
        "show_imputation_sources", configured_sources
    )
    updated["figure"]["manual_age_config_path"] = None
    config = DatasetStatisticsPlotConfig.from_mapping(updated, config_dir=config_dir)
    if "cohort_figure" in payload:
        config = replace(config, cohort_figure=CohortFigureData.from_mapping(payload["cohort_figure"]))
    cohort_hash = payload["metadata"].get("cohort_hash")
    returned = None if cohort_hash is None else find_age_summary(cohort_hash)
    if cohort_hash is None:
        LOGGER.warning("Saved statistics have no cohort hash; ages remain unavailable. Regenerate the cohort artifacts.")
    return replace(config, manual_age=age_display_summary(None if returned is None else returned.value),
                   age_return_sources=() if returned is None else returned.paths)


def validate_statistics_source(payload: dict[str, Any], dataset_config: dict[str, Any], config_dir: Path) -> None:
    """Compare saved snapshots, resolving paths for run-local dataset configs."""
    if payload["metadata"].get("dataset_config_format") == "resolved":
        dataset_config = asdict(PatientDatasetConfig.from_mapping(dataset_config, config_dir=config_dir))
    validate_payload_dataset_config(payload, dataset_config)


def load_config(config_path: Path) -> DatasetStatisticsPlotConfig:
    raw_config = load_yml(config_path)
    raw_config["figure"]["manual_age_config_path"] = None
    return DatasetStatisticsPlotConfig.from_mapping(raw_config, config_dir=config_path.parent)


def build_dataset_statistics_figure(
    config: DatasetStatisticsPlotConfig,
    *,
    recurrence_incidence_df: pd.DataFrame,
    recurrence_survival_df: pd.DataFrame,
) -> Figure:
    fonts = figure_typography("dataset_statistics", config.figure.figure_width)
    with plt.rc_context(publication_rc(fonts.axis_label, fonts.panel_title) | fonts.rc()):
        return draw_dataset_statistics_figure(config, recurrence_incidence_df, recurrence_survival_df)


def draw_dataset_statistics_figure(
    config: DatasetStatisticsPlotConfig,
    incidence: pd.DataFrame,
    survival: pd.DataFrame,
) -> Figure:
    data = config.cohort_figure
    if data is None:
        raise ValueError("The six-panel figure requires run-local cohort composition.")
    risk_table = suppress_risk_counts(data.survival, config.first_recurrence_site_survival.min_patients_at_risk)
    figure = plt.figure(figsize=(config.figure.figure_width, config.figure.figure_height))
    design = load_overview_design(config.figure.layout)
    axes = create_cohort_axes(figure)
    draw_patient_flow_diagram(axes[0], config.patient_flow, fit_height=True, footnotes=config.patient_flow.footnotes)
    numbered = {label: f"{index}  {label}" for index, label in enumerate(config.patient_flow.exclusion_box_labels, 1)}
    calendar = replace(config.diagnosis_years, series_labels=tuple(
        numbered.get(label, label) for label in config.diagnosis_years.series_labels))
    draw_diagnosis_year_bar_chart(axes[1], calendar)
    draw_cohort_composition(axes[2], data, config.manual_age, table_style=design.table_style)
    draw_bloodtest_count_bar_chart(axes[3], config.blood_test_counts)
    draw_recurrence_site_incidence(axes[4], config.recurrence_site_incidence, incidence)
    draw_first_recurrence_site_survival(axes[5], config.first_recurrence_site_survival, survival)
    pack_cohort_panels(figure, axes, design)
    fit_recurrence_tick_spacing((axes[4], axes[5]))
    for axis, table, is_survival, maximum in (
        (axes[4], data.incidence, False, config.recurrence_site_incidence.max_years_since_diagnosis),
        (axes[5], risk_table, True, config.first_recurrence_site_survival.max_years_since_origin),
    ):
        axis.set_axis_on()
        axis.set_xlim(0, maximum)
        if is_survival:
            axis.set_ylim(0, 105)
            axis.set_yticks(tuple(range(0, 101, 10)))
        else:
            apply_integer_percent_ticks(axis, axis.get_ylim()[1])
        style_plot(axis, vertical_grid=True)
        apply_year_ticks(axis, maximum, 1)
        draw_annual_curve_table(axis, table, survival=is_survival)
        figure_typography("dataset_statistics", figure.get_figwidth()).apply_axis(axis)
    if design.stacked_curves:
        stack_survival_panel(axes[4], axes[5], design.panel_gap_mm)
    cohort_headers(figure, axes, ("Cohort selection", config.diagnosis_years.title, "Study cohort composition",
                                  config.blood_test_counts.title, config.recurrence_site_incidence.title,
                                  config.first_recurrence_site_survival.title))
    if not (design.aligned_columns or design.stacked_curves):
        fill_right_column(figure, axes, design.panel_gap_mm)
    fit_cohort_footer(figure)
    return figure


def build_recurrence_incidence_table(
    config: DatasetStatisticsPlotConfig,
    *,
    endpoint_rows: pd.DataFrame,
) -> pd.DataFrame:
    panel_config = config.recurrence_site_incidence
    return build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        panel_config.site_rules,
        max_years_since_diagnosis=panel_config.max_years_since_diagnosis,
        min_days_since_diagnosis=panel_config.min_days_since_diagnosis,
        min_patients_at_risk=panel_config.min_patients_at_risk,
        confidence_level=panel_config.confidence_level,
        random_seed=panel_config.random_seed,
        overall_label=panel_config.overall_label if panel_config.show_overall else None,
        other_label=panel_config.other_label,
    )


def build_recurrence_survival_table(
    config: DatasetStatisticsPlotConfig,
    *,
    endpoint_rows: pd.DataFrame,
) -> pd.DataFrame:
    """Suppress panel f curve points whose risk count would display as a bound."""
    panel_config = config.first_recurrence_site_survival
    return build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        panel_config.site_rules,
        max_years_since_origin=panel_config.max_years_since_origin,
        min_days_since_diagnosis=config.recurrence_site_incidence.min_days_since_diagnosis,
        # Counts <= the figure threshold are suppressed; the estimator's minimum is inclusive.
        min_patients_at_risk=panel_config.min_patients_at_risk + 1,
        time_origin=panel_config.time_origin,
        other_label=panel_config.other_label,
        overall_label=panel_config.overall_label if panel_config.show_overall else None,
    )


def prepared_endpoint_rows(source: DatasetStatisticsSourceConfig) -> pd.DataFrame:
    """Build shared Panel D/E rows from the configured prepared patient dataset."""
    if source.dataset_config_path is None:
        raise ValueError("`dataset.dataset_config_path` is required to derive Panels D and E from prepared patients.")
    dataset_config = PatientDatasetConfig.from_mapping(
        load_dataset_section(source.dataset_config_path),
        config_dir=source.dataset_config_path.parent,
    )
    return prepared_recurrence_site_rows(PatientBloodTestDataset(dataset_config))


def draw_diagnosis_year_bar_chart(
    axis: Axes,
    config: DiagnosisYearBarChartConfig,
) -> None:
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    legend_order = diagnosis_year_legend_order(config.series_labels)
    bottoms_by_series, totals = diagnosis_year_stack_bottoms(
        config.series_counts,
        legend_order,
    )
    for index, (label, counts) in enumerate(zip(config.series_labels, config.series_counts, strict=True)):
        axis.bar(
            config.diagnosis_years,
            counts,
            bottom=bottoms_by_series[index],
            color=DIAGNOSIS_YEAR_STACK_COLORS[index % len(DIAGNOSIS_YEAR_STACK_COLORS)],
            edgecolor="0.25",
            linewidth=0.35,
            label=display_text(label),
            width=DIAGNOSIS_YEAR_BAR_WIDTH,
            align="edge",
        )
    axis.set_xlabel(config.x_label)
    axis.set_ylabel(config.y_label)
    axis.set_ylim(0.0, max(DIAGNOSIS_YEAR_Y_MAX, max(totals) * 1.08))
    axis.yaxis.set_major_locator(MultipleLocator(DIAGNOSIS_YEAR_YTICK_STEP))
    style_plot(axis)
    axis.set_xlim(
        min(config.diagnosis_years) - DIAGNOSIS_YEAR_EDGE_PAD,
        max(config.diagnosis_years) + 1.0,
    )
    format_year_ticks(axis, config.diagnosis_years)
    handles, labels = axis.get_legend_handles_labels()
    axis.legend(
        tuple(handles[index] for index in legend_order),
        tuple(diagnosis_year_legend_label(axis, labels[index]) for index in legend_order),
        frameon=False,
        fontsize=fonts.legend,
        loc="lower left",
        bbox_to_anchor=(0.0, 1.02),
        borderaxespad=0.0,
        ncols=2,
        columnspacing=1.5,
        labelspacing=0.85,
        handlelength=1.2,
        handletextpad=0.35,
        borderpad=0.0,
    )


def diagnosis_year_legend_label(axis: Axes, label: str) -> str:
    width_points = axis.get_position().width * axis.figure.get_figwidth() * 72 / 2 - 19
    return wrap_to_width(axis, label, width_points,
                         figure_typography("dataset_statistics", axis.figure.get_figwidth()).legend)


def diagnosis_year_legend_order(labels: tuple[str, ...]) -> tuple[int, ...]:
    return tuple(index for index, label in enumerate(labels) if label != "Included") + tuple(
        index for index, label in enumerate(labels) if label == "Included"
    )


def diagnosis_year_stack_bottoms(
    series_counts: tuple[tuple[int, ...], ...],
    legend_order: tuple[int, ...],
) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Stack series so their top-to-bottom order matches the legend."""
    year_count = len(series_counts[0])
    bottoms_by_series = [[0] * year_count for _ in series_counts]
    totals = [0] * year_count
    for series_index in reversed(legend_order):
        bottoms_by_series[series_index] = totals.copy()
        totals = [
            bottom + count
            for bottom, count in zip(
                totals,
                series_counts[series_index],
                strict=True,
            )
        ]
    return tuple(tuple(values) for values in bottoms_by_series), tuple(totals)


def draw_recurrence_site_incidence(
    axis: Axes,
    config: RecurrenceSiteIncidencePlotConfig,
    incidence_df: pd.DataFrame,
) -> None:
    axis.set_xlabel(config.x_label)
    axis.set_ylabel(config.y_label)
    if incidence_df.empty:
        draw_empty_panel(axis, "No Panel D rows in dataset-statistics payload")
        return
    overall_label = config.overall_label if config.show_overall else None
    color_by_site = site_color_mapping(config.site_rules, config.other_label, overall_label)
    incidence_by_site = {str(site_label): site_df for site_label, site_df in incidence_df.groupby("site", sort=False)}
    for label in ordered_site_labels(config.site_rules, config.other_label, overall_label):
        site_df = incidence_by_site.get(label)
        if site_df is None:
            continue
        if label != overall_label and not site_has_incidence(site_df):
            continue
        draw_incidence_line(
            axis,
            site_df,
            label,
            color=color_by_site[label],
            is_overall=label == config.overall_label,
        )
    y_max = incidence_y_max(incidence_df["cumulative_incidence_percent"])
    axis.set_xlim(0.0, float(config.max_years_since_diagnosis))
    axis.set_ylim(0.0, y_max)
    apply_year_ticks(axis, config.max_years_since_diagnosis, config.year_step)
    apply_integer_percent_ticks(axis, y_max)
    style_plot(axis)


def draw_incidence_line(axis: Axes, site_df: pd.DataFrame, label: str, *, color: str, is_overall: bool) -> None:
    axis.step(
        site_df["years_since_diagnosis"],
        site_df["cumulative_incidence_percent"],
        where="post",
        color=color,
        linewidth=OVERALL_CURVE_WIDTH if is_overall else CURVE_WIDTH,
        label=label,
    )


def site_has_incidence(site_df: pd.DataFrame) -> bool:
    return bool(site_df["cumulative_incidence_percent"].max() > 0.0)


def draw_first_recurrence_site_survival(
    axis: Axes,
    config: FirstRecurrenceSiteSurvivalPlotConfig,
    survival_df: pd.DataFrame,
) -> None:
    axis.set_xlabel(config.x_label)
    axis.set_ylabel(config.y_label)
    if survival_df.empty:
        draw_empty_panel(axis, "No Panel E rows in dataset-statistics payload")
        return
    overall_label = config.overall_label if config.show_overall else None
    color_by_site = site_color_mapping(config.site_rules, config.other_label, overall_label)
    survival_by_site = {str(site_label): site_df for site_label, site_df in survival_df.groupby("site", sort=False)}
    for label in ordered_site_labels(config.site_rules, config.other_label, overall_label):
        site_df = survival_by_site.get(label)
        if site_df is None:
            continue
        axis.step(
            site_df["years_since_origin"],
            site_df["survival_percent"],
            where="post",
            color=color_by_site[label],
            linewidth=OVERALL_CURVE_WIDTH if label == overall_label else CURVE_WIDTH,
            label=label,
        )
    axis.set_xlim(0.0, float(config.max_years_since_origin))
    axis.set_ylim(0.0, 105.0)
    apply_year_ticks(axis, config.max_years_since_origin, config.year_step)
    axis.set_yticks(tuple(range(0, 101, 10)))
    style_plot(axis)


def year_ticks(max_year: int, configured_step: int) -> tuple[int, ...]:
    step = max(1, configured_step)
    return tuple(range(0, max_year + 1, step))


def apply_year_ticks(axis: Axes, max_year: int, configured_step: int) -> None:
    ticks = year_ticks(max_year, configured_step)
    axis.xaxis.set_major_locator(FixedLocator(ticks))
    axis.xaxis.set_major_formatter(FixedFormatter(tuple(str(tick) for tick in ticks)))
    axis.tick_params(axis="x", labelsize=figure_typography("dataset_statistics", axis.figure.get_figwidth()).tick_label, pad=1.5)


def apply_integer_percent_ticks(axis: Axes, y_max: float) -> None:
    ticks = tuple(range(0, int(y_max) + 1))
    axis.yaxis.set_major_locator(FixedLocator(ticks))
    axis.yaxis.set_major_formatter(FixedFormatter(tuple(str(tick) for tick in ticks)))


def ordered_site_labels(
    site_rules: tuple[RecurrenceSiteRule, ...],
    other_label: str | None,
    overall_label: str | None = None,
) -> tuple[str, ...]:
    labels = [rule.label for rule in site_rules]
    if other_label is not None:
        labels.append(other_label)
    if overall_label is not None:
        labels.append(overall_label)
    return tuple(sorted(labels, key=str.casefold))


def site_color_mapping(
    site_rules: tuple[RecurrenceSiteRule, ...],
    other_label: str | None,
    overall_label: str | None = None,
) -> dict[str, str]:
    labels = ordered_site_labels(site_rules, other_label)
    color_by_site = recurrence_site_colors(labels)
    if overall_label is not None:
        color_by_site[overall_label] = OVERALL_SITE_COLOR
    return color_by_site


def format_year_ticks(axis: Axes, years: tuple[int, ...]) -> None:
    tick_years = tuple(
        range(
            DIAGNOSIS_YEAR_FIRST_TICK,
            max(years) + 1,
            DIAGNOSIS_YEAR_XTICK_STEP,
        )
    )
    tick_positions = tuple(year + DIAGNOSIS_YEAR_BAR_WIDTH / 2.0 for year in tick_years)
    axis.xaxis.set_major_locator(FixedLocator(tick_positions))
    axis.xaxis.set_major_formatter(FixedFormatter(tuple(str(year) for year in tick_years)))
    axis.tick_params(axis="x", labelrotation=0)


def incidence_y_max(values: pd.Series) -> float:
    max_value = float(max(values))
    if max_value <= 0.0:
        return 1.0
    return min(100.0, max_value * 1.15)
