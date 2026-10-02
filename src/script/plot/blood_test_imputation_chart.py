from __future__ import annotations

from typing import Protocol

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from src.figure_style import figure_typography
from src.reporting import count_percentage_text
from src.script.plot.overview_style import (
    ROW_COLOR,
    SECONDARY_COLOR,
    style_plot,
)

BLOODTEST_BAR_HEIGHT = 0.60
BLOODTEST_STACK_LABELS = ("Measured", "LOCF", "MICE")
# Label and bar boundaries followed by three equally wide number columns.
COLUMN_EDGES = (0.0, 0.295, 0.535, 0.69, 0.845, 1.0)


class BloodTestStackPlotConfig(Protocol):
    """Blood-test labels, colors, and aligned measured/LOCF/MICE counts."""

    x_label: str
    source_x_label: str
    y_label: str
    show_imputation_sources: bool
    bar_color: str
    locf_color: str
    mice_color: str
    blood_test_names: tuple[str, ...]
    blood_test_counts: tuple[int, ...]
    blood_test_locf_counts: tuple[int, ...]
    blood_test_mice_counts: tuple[int, ...]


def draw_measured_count_bars(axis: Axes, config: BloodTestStackPlotConfig) -> None:
    """Preserve the measured-only path when fitted source attribution is absent."""
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    if not config.blood_test_counts:
        draw_empty_panel(axis, "Panel C counts pending")
        return
    axis.set_axis_off()
    plot = axis.inset_axes((COLUMN_EDGES[1], 0.09, 1 - COLUMN_EDGES[1], 0.80))
    counts = config.blood_test_counts
    plot.barh(range(len(counts)), counts, color=config.bar_color, height=BLOODTEST_BAR_HEIGHT)
    plot.set_yticks(range(len(counts)), config.blood_test_names, fontsize=fonts.table)
    plot.invert_yaxis()
    plot.set_xlim(0, max(counts) * 1.22 if any(counts) else 1)
    plot.set_xlabel(config.x_label, fontsize=fonts.axis_label)
    style_plot(plot)
    for row, count in enumerate(counts):
        plot.text(count, row, f" {count:,}", va="center", fontsize=fonts.table)


def bloodtest_bar_values(config: BloodTestStackPlotConfig) -> tuple[tuple[str, ...], tuple[tuple[int, ...], ...]]:
    return config.blood_test_names, (
        config.blood_test_counts,
        config.blood_test_locf_counts,
        config.blood_test_mice_counts,
    )


def draw_empty_panel(axis: Axes, message: str) -> None:
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    axis.set_axis_off()
    axis.text(
        0.5,
        0.5,
        message,
        ha="center",
        va="center",
        transform=axis.transAxes,
        color=SECONDARY_COLOR,
        fontsize=fonts.table,
    )


def draw_bloodtest_count_bar_chart(
    axis: Axes, config: BloodTestStackPlotConfig, *, width: float = 84, height: float = 91,
) -> None:
    """Render the approved narrow assay panel with count/percentage columns."""
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    if not config.show_imputation_sources:
        draw_measured_count_bars(axis, config)
        return
    names, counts = bloodtest_bar_values(config)
    axis.set_axis_off()
    if not names:
        return
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    colors = (config.bar_color, config.locf_color, config.mice_color)
    column_width = (width - 35) / 3
    ends = tuple(35 + (index + 1) * column_width for index in range(3))
    for label, color, x in zip(BLOODTEST_STACK_LABELS, colors, ends, strict=True):
        axis.add_patch(Rectangle((x - column_width + 1, -.6), 1.2, 1.2, color=color, clip_on=False))
        axis.text(x, 0, label, ha="right", va="center", fontsize=fonts.table, weight="bold")
        axis.text(x, 3, "n (%)", ha="right", va="center", fontsize=fonts.table)
    axis.text(28.5, 0, "0–100%", ha="center", va="center", fontsize=fonts.table)
    axis.text(0, 3, "Blood test", va="center", fontsize=fonts.table)
    step = min(3.54, (height - 7) / len(names))
    for index, (name, row) in enumerate(zip(names, zip(*counts, strict=True), strict=True)):
        y = 6 + step * index
        if index % 2 == 0:
            axis.add_patch(Rectangle((0, y - step / 2), width, step, facecolor=ROW_COLOR, edgecolor="none"))
        axis.text(0, y, name, va="center", fontsize=fonts.table)
        total, left = sum(row), 24.0
        for count, color, x in zip(row, colors, ends, strict=True):
            bar_width = 0 if total == 0 else 9 * count / total
            axis.add_patch(Rectangle((left, y - .65), bar_width, 1.3, facecolor=color, edgecolor="none"))
            left += bar_width
            axis.text(x, y, count_percentage_text(count, total).replace("%", ""),
                      ha="right", va="center", fontsize=fonts.table)
