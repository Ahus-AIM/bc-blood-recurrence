from __future__ import annotations

from dataclasses import replace
from textwrap import fill

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Patch
from matplotlib.text import Text
from matplotlib.ticker import MaxNLocator, StrMethodFormatter
from matplotlib.transforms import blended_transform_factory

from src.data.blood_test_display import format_blood_test_unit
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig
from src.data.recurrence_blood_test_display import window_display_label
from src.data.recurrence_blood_test_plot_config import RecurrenceBloodTestPlotConfig
from src.data.reference_intervals import reference_interval_label as reference_label
from src.model.normal_ranges import NormalRange
from src.model.normal_range_plotting import normal_range_color, NORMAL_RANGE_SPAN_ALPHA, NORMAL_RANGE_LEGEND_ALPHA
from src.figure_style import Typography, figure_typography, publication_style, wrap_text
from src.reporting import p_value_text, percentage_text
from src.script.plot.overview_style import BORDER_COLOR, GRID_COLOR, ROW_COLOR, SECONDARY_COLOR, TEXT_COLOR
from src.script.plot.recurrence_reference_legend import inset_reference_legend

FIGURE_LEFT = 0.46
FIGURE_RIGHT = 0.32
FIGURE_TOP = 1.12
PLOT_WIDTH_FRACTION = 0.76
PLOT_HEIGHT_FRACTION = 0.42
TABLE_HEIGHT_FRACTION = 0.35
PERCENT_COLOR = publication_style().colors["outside_reference"]
PERCENT_FILL = publication_style().colors["outside_reference_fill"]
OUTSIDE_REFERENCE_LABEL = "Outside reference range (%)"
BOUNDARY_STYLES = ("--", ":", "-.")


def window_tick_labels(config: RecurrenceBloodTestConfig) -> list[str]:
    """Show month/week ranges from farther to nearer recurrence, keeping strict bounds."""
    labels = []
    for window in config.windows:
        label = window_display_label(window.label)
        lower, separator, upper = label.partition("–")
        if len(label) > 14 and separator and any(character.isalpha() for character in lower):
            label = f"{lower}–\n{upper}"  # Keep each number with its unit.
        else:
            label = fill(label, width=14, break_long_words=False, break_on_hyphens=False)
        labels.append(label)
    labels[0] += "\n(baseline)"
    return labels


def text_width_points(text: str, font: FontProperties, renderer: RendererAgg) -> float:
    return max(renderer.get_text_width_height_descent(line, font, ismath=False)[0] for line in text.splitlines())


def fitted_window_labels(config: RecurrenceBloodTestConfig, fonts: Typography, panel_width: float) -> list[str]:
    """Wrap range boundaries at the available printed width, keeping numbers and units together."""
    renderer = RendererAgg(1, 1, 72)
    font = FontProperties(family=fonts.family, size=fonts.tick_label)
    available = panel_width * PLOT_WIDTH_FRACTION * 72 / len(config.windows) - 0.2 * fonts.tick_label
    return [label.replace("–", "–\n") if text_width_points(label, font, renderer) > available else label
            for label in window_tick_labels(config)]


def trajectory_typography(config: RecurrenceBloodTestConfig, width: float) -> Typography:
    """Apply the trajectory-only reduction to every shared text role."""
    return figure_typography("recurrence_blood_test_trajectories", width * config.plot.font_scale)


def content_panel_width(selected: pd.DataFrame, config: RecurrenceBloodTestConfig, columns: int) -> float:
    """Measure the width needed for wrapped labels and values, including bold text."""
    width = config.plot.panel_width
    fonts = trajectory_typography(config, FIGURE_LEFT + columns * width + FIGURE_RIGHT)
    renderer = RendererAgg(1, 1, 72)
    tick_font = FontProperties(family=fonts.family, size=fonts.tick_label)
    available = width * PLOT_WIDTH_FRACTION * 72 / len(config.windows) - 0.5 * fonts.table
    widest = max(text_width_points(label, tick_font, renderer)
                 for label in fitted_window_labels(config, fonts, width)) + 0.2 * fonts.tick_label
    for _, rows in selected.groupby(["site", "test_key"], sort=False):
        significant = rows["p_value_bh"].to_numpy(dtype=float) < config.fdr_alpha
        for index, (_, cells) in enumerate(numerical_rows(rows, config.decimals)):
            for column, cell in enumerate(cells):
                bold = index in (2, 3, 5) and significant[column]
                font = FontProperties(family=fonts.family, size=fonts.table, weight="bold" if bold else "normal")
                display = fitted_table_cell(cell, available, font, renderer)
                widest = max(widest, text_width_points(display, font, renderer) + 0.5 * fonts.table)
    return len(config.windows) * widest / 72 / PLOT_WIDTH_FRACTION


def fitted_panel_layout(
    selected: pd.DataFrame, config: RecurrenceBloodTestConfig,
    *, columns: int | None = None,
) -> tuple[int, RecurrenceBloodTestPlotConfig]:
    """Fit text within the chosen columns; by default keep one row per site."""
    if columns is None:
        counts = selected.groupby("site")["test_key"].nunique()
        columns = max(1, int(counts.max())) if not counts.empty else 1
    if columns < 1:
        raise ValueError("Trajectory layouts require at least one column.")
    for _ in range(8):
        required = content_panel_width(selected, config, columns)
        if required <= config.plot.panel_width:
            return columns, config.plot
        scale = config.plot.font_scale * config.plot.panel_width / required * 0.98
        config = replace(config, plot=replace(config.plot, font_scale=scale))
    raise ValueError("Trajectory text cannot fit at its manuscript width; reduce displayed precision.")


def fitted_table_cell(cell: str, available: float, font: FontProperties, renderer: RendererAgg) -> str:
    """Split interval/count pairs over two lines when needed; preserve every digit."""
    if text_width_points(cell, font, renderer) <= available:
        return cell
    return cell.replace(", ", ",\n").replace(" / ", " /\n")


def content_row_height(selected: pd.DataFrame, config: RecurrenceBloodTestConfig, width: float) -> float:
    """Reserve table height for wrapped text at the fitted font size."""
    fonts = trajectory_typography(config, width)
    available = config.plot.panel_width * PLOT_WIDTH_FRACTION * 72 / len(config.windows) - 0.5 * fonts.table
    heights = [sum(height for _, _, height in numeric_table_layout(rows, config, fonts, available))
               for _, rows in selected.groupby(["site", "test_key"], sort=False)]
    return max(config.plot.row_height, max(heights, default=0) / 72 / TABLE_HEIGHT_FRACTION)


def site_row_height(config: RecurrenceBloodTestConfig) -> float:
    """Remove only the unused graph height from the fitted site row."""
    removed_fraction = PLOT_HEIGHT_FRACTION * (1 - config.plot.graph_height_scale)
    return config.plot.row_height * (1 - removed_fraction)


def numeric_table_layout(
    rows: pd.DataFrame, config: RecurrenceBloodTestConfig, fonts: Typography, available: float,
) -> list[tuple[str, list[str], float]]:
    """Return display cells and row heights in points, including bold significant cells."""
    renderer = RendererAgg(1, 1, 72)
    significant = rows["p_value_bh"].to_numpy(dtype=float) < config.fdr_alpha
    label_font = FontProperties(family=fonts.family, size=fonts.table)
    label_width = 0.18 * config.plot.panel_width * 72
    layout = []
    for index, (label, cells) in enumerate(numerical_rows(rows, config.decimals)):
        displays = []
        for column, cell in enumerate(cells):
            bold = index in (2, 3, 5) and significant[column]
            font = FontProperties(family=fonts.family, size=fonts.table, weight="bold" if bold else "normal")
            displays.append(fitted_table_cell(cell, available, font, renderer))
        label = wrap_text(label, label_width, label_font, renderer)
        lines = max(len(cell.splitlines()) for cell in (label, *displays))
        layout.append((label, displays, fonts.table * (lines * 1.2 + 0.25)))
    return layout


def draw_test_panel(
    figure: Figure,
    rows: pd.DataFrame,
    config: RecurrenceBloodTestConfig,
    ranges: tuple[NormalRange, ...],
    color: str,
    column: int,
    top: float,
    y_limits: tuple[float, float],
    *, heading_size: float, panel_number: int | None = None,
) -> None:
    settings = config.plot
    fonts = trajectory_typography(config, figure.get_figwidth())
    width, height = figure.get_size_inches()
    left = FIGURE_LEFT + column * settings.panel_width
    plot_left, plot_width = left + 0.17 * settings.panel_width, PLOT_WIDTH_FRACTION * settings.panel_width
    plot_top = top - 0.12 * settings.row_height
    plot_height = PLOT_HEIGHT_FRACTION * settings.row_height * settings.graph_height_scale
    axis = figure.add_axes(
        (plot_left / width, (plot_top - plot_height) / height, plot_width / width, plot_height / height)
    )
    style_axis(axis, fonts)
    axis.set_ylim(y_limits)
    axis.set_xlim(-0.5, len(rows) - 0.5)
    draw_reference_bands(axis, ranges, fonts.legend)
    percent_axis = draw_percentage_area(axis, rows["outside_reference_percent"].to_numpy(dtype=float), fonts)
    medians = rows["median"].to_numpy(dtype=float)
    errors = np.array([medians - rows["q1"].to_numpy(dtype=float), rows["q3"].to_numpy(dtype=float) - medians])
    axis.errorbar(
        np.arange(len(rows)),
        medians,
        yerr=errors,
        fmt="o-",
        color=color,
        markersize=4,
        markerfacecolor="white",
        markeredgewidth=1.3,
        linewidth=1.4,
        elinewidth=1.5,
        capsize=3,
        zorder=5,
    )
    axis.set_xticks(np.arange(len(rows)), fitted_window_labels(config, fonts, config.plot.panel_width))
    axis.set_ylabel(format_blood_test_unit(str(rows.iloc[0]["unit"])), fontsize=fonts.axis_label)
    title = f"{column + 1 if panel_number is None else panel_number}. {rows.iloc[0]['blood_test']}"
    title = wrap_text(title, plot_width * figure.dpi,
                      FontProperties(family=fonts.family, size=heading_size, weight="bold"),
                      figure.canvas.get_renderer())
    heading = figure.text(
        plot_left / width,
        (top - 0.01 * settings.row_height) / height,
        title,
        fontsize=heading_size,
        weight="bold",
        va="top",
        color=TEXT_COLOR,
        gid=f"test-heading-{len(figure.texts)}",
    )
    fit_reference_heading(figure, axis, heading)
    inset_reference_legend(axis, percent_axis, heading, ranges)
    draw_numeric_table(
        figure,
        rows,
        config,
        plot_left / width,
        (top - site_row_height(config) + 0.03 * settings.row_height) / height,
        plot_width / width,
        TABLE_HEIGHT_FRACTION * settings.row_height / height,
    )


def fit_reference_heading(figure: Figure, axis: Axes, heading: Text) -> None:
    """Reserve space for the full reference legend beneath the panel heading."""
    legend = axis.get_legend()
    if legend is None:
        return
    renderer = RendererAgg(1, 1, figure.dpi)
    padding = 4 * figure.dpi / 72
    overlap = legend.get_window_extent(renderer).y1 + padding - heading.get_window_extent(renderer).y0
    if overlap > 0:
        position = axis.get_position()
        # Lower the graph's top edge; its bottom and the table remain aligned.
        axis.set_position((position.x0, position.y0, position.width, position.height - overlap / figure.bbox.height))


def style_axis(axis: Axes, fonts: Typography) -> None:
    fonts.apply_axis(axis)
    for edge in ("top", "right"):
        axis.spines[edge].set_visible(False)
    for edge in ("left", "bottom"):
        axis.spines[edge].set_color(BORDER_COLOR)
    axis.tick_params(color=BORDER_COLOR, labelsize=fonts.tick_label)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True, min_n_ticks=2))
    axis.yaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
    axis.grid(axis="y", color=GRID_COLOR, linewidth=0.5)
    axis.set_axisbelow(True)


def draw_reference_bands(axis: Axes, ranges: tuple[NormalRange, ...], font_size: float) -> None:
    if not ranges:
        axis.text(
            0,
            1.03,
            "Reference range unavailable",
            transform=axis.transAxes,
            fontsize=font_size,
            color=SECONDARY_COLOR,
        )
        return
    handles = []
    lower_limit, upper_limit = axis.get_ylim()
    for index, interval in enumerate(ranges):
        color = normal_range_color(interval)
        style = BOUNDARY_STYLES[index % len(BOUNDARY_STYLES)]
        lower = lower_limit if interval.lower is None else interval.lower
        upper = upper_limit if interval.upper is None else interval.upper
        axis.axhspan(lower, upper, color=color, alpha=NORMAL_RANGE_SPAN_ALPHA, zorder=0)
        for boundary in (interval.lower, interval.upper):
            if boundary is not None:
                axis.axhline(boundary, color=color, linestyle=style, linewidth=0.9, zorder=2)
        handles.append(
            Patch(facecolor=color, alpha=NORMAL_RANGE_LEGEND_ALPHA, edgecolor=color, linestyle=style, label=reference_label(interval))
        )
    axis.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0, 1.01),
        frameon=False,
        borderaxespad=0,
        fontsize=font_size,
        handlelength=2,
        labelspacing=0.25,
    )
    axis.set_ylim(lower_limit, upper_limit)


def draw_percentage_area(axis: Axes, percentages: np.ndarray, fonts: Typography) -> Axes:
    """Draw categorical percentage steps on a fixed 0–100% right-hand scale."""
    percent_axis = axis.twinx()
    fonts.apply_axis(percent_axis)
    percent_axis.set_ylim(0, 100)
    percent_axis.set_yticks((0, 50, 100))
    percent_axis.set_ylabel(fill(OUTSIDE_REFERENCE_LABEL, width=18), fontsize=fonts.axis_label, color=PERCENT_COLOR)
    percent_axis.tick_params(axis="y", labelsize=fonts.tick_label, colors=PERCENT_COLOR)
    percent_axis.spines["top"].set_visible(False)
    percent_axis.spines["left"].set_visible(False)
    percent_axis.spines["bottom"].set_visible(False)
    percent_axis.spines["right"].set_color(PERCENT_COLOR)
    if np.isfinite(percentages).any():
        transform = blended_transform_factory(axis.transData, axis.transAxes)
        edges = np.arange(len(percentages) + 1) - 0.5
        axis.stairs(
            percentages / 100,
            edges,
            baseline=0,
            fill=True,
            color=PERCENT_FILL,
            alpha=0.23,
            linewidth=0,
            transform=transform,
            zorder=1,
        )
        axis.stairs(
            percentages / 100, edges, baseline=None, color=PERCENT_COLOR, linewidth=0.8, transform=transform, zorder=3
        )
    return percent_axis


def number(value: float, decimals: int, *, signed: bool = False) -> str:
    if not np.isfinite(value):
        return "—"
    return f"{value:+.{decimals}f}" if signed else f"{value:.{decimals}f}"


def p_number(value: float, *, baseline: bool = False) -> str:
    if baseline:
        return "—"
    return p_value_text(value)


def numerical_rows(rows: pd.DataFrame, decimals: int) -> list[tuple[str, list[str]]]:
    cells: list[list[str]] = [[] for _ in range(8)]
    for row in rows.to_dict("records"):
        baseline = row["test_status"] == "baseline"
        outside_text = percentage_text(float(row["outside_reference_percent"]))
        values = (
            number(row["median"], decimals),
            interval_number(row["q1"], row["q3"], decimals),
            "—" if baseline else number(row["median_change"], decimals, signed=True),
            "—" if baseline else interval_number(row["change_q1"], row["change_q3"], decimals),
            p_number(row["p_value"], baseline=baseline),
            p_number(row["p_value_bh"], baseline=baseline),
            f"{row['n_patients']} / {'—' if baseline else row['n_pairs']}",
            outside_text,
        )
        for column, value in zip(cells, values, strict=True):
            column.append(value)
    return list(
        zip(
            (
                "Median",
                "IQR [Q1, Q3]",
                "Paired Δ median",
                "Paired Δ IQR",
                "p value",
                "Adjusted p",
                "Tested / paired",
                OUTSIDE_REFERENCE_LABEL,
            ),
            cells,
            strict=True,
        )
    )


def interval_number(q1: float, q3: float, decimals: int) -> str:
    return f"[{number(q1, decimals)}, {number(q3, decimals)}]" if np.isfinite([q1, q3]).all() else "—"


def draw_numeric_table(
    figure: Figure,
    rows: pd.DataFrame,
    config: RecurrenceBloodTestConfig,
    left: float,
    bottom: float,
    width: float,
    height: float,
) -> None:
    fonts = trajectory_typography(config, figure.get_figwidth())
    available = width * figure.get_figwidth() * 72 / len(rows) - 0.5 * fonts.table
    layout = numeric_table_layout(rows, config, fonts, available)
    axis = figure.add_axes((left, bottom, width, height))
    axis.set_xlim(-0.5, len(rows) - 0.5)
    axis.set_ylim(sum(row_height for _, _, row_height in layout), 0)
    axis.set_axis_off()
    significant = rows["p_value_bh"].to_numpy(dtype=float) < config.fdr_alpha
    top = 0
    for index, (label, cells, row_height) in enumerate(layout):
        y_position = top + row_height / 2
        top += row_height
        if index in (0, 2, 7):
            axis.axhspan(y_position - row_height / 2, y_position + row_height / 2,
                         color="#FBF4E4" if index == 7 else ROW_COLOR)
        color = PERCENT_COLOR if index == 7 else TEXT_COLOR
        axis.text(
            -0.025,
            y_position,
            label,
            transform=axis.get_yaxis_transform(),
            ha="right",
            va="center",
            fontsize=fonts.table,
            color=color,
        )
        for column, cell in enumerate(cells):
            bold = index in (2, 3, 5) and significant[column]
            axis.text(
                column,
                y_position,
                cell,
                ha="center",
                va="center",
                color=color,
                fontsize=fonts.table,
                weight="bold" if bold else "normal",
            )
