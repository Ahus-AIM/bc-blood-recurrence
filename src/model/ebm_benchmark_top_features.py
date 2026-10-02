from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from interpret.api.templates import FeatureValueExplanation
from matplotlib.font_manager import FontProperties
from matplotlib.transforms import Bbox

from src.figure_style import figure_typography, recurrence_site_colors, wrap_text
from src.figure_headings import ebm_row_gap_fraction, position_figure_title
from src.model.ebm_global_importance import PNG_DPI, draw_global_importance_axis
from src.model.ebm_visualizations import (
    HAZARD_RATIO_GRID_SUBPLOT_HEIGHT,
    HAZARD_RATIO_GRID_SUBPLOT_WIDTH,
    HazardRatioGridStyle,
    draw_hazard_ratio_grid_term,
    univariate_terms_by_importance,
)
from src.model.normal_range_plotting import fit_numeric_reference_labels

TOP_FEATURE_COUNT = 3
GRID_COLUMN_COUNT = TOP_FEATURE_COUNT + 1
ROW_LABEL_MARGIN = 0.1
FIGURE_TITLE_MARGIN = 0.96
FIGURE_BOTTOM_MARGIN = 0.01
FIGURE_RIGHT_MARGIN = 0.99
SAVE_PADDING_INCHES = 0.01


@dataclass(frozen=True)
class EbmTopFeatureRow:
    """One labeled recurrence-site row in the benchmark comparison figure.

    ``label`` identifies the site. ``explanation`` and ``style`` are absent
    when its model is unavailable; otherwise they hold its global explanation
    and saved plot settings.
    """

    label: str
    explanation: FeatureValueExplanation | None
    style: HazardRatioGridStyle | None


def write_ebm_top_feature_grid(
    rows: Sequence[EbmTopFeatureRow],
    png_path: Path,
    svg_path: Path,
    title: str,
) -> None:
    """Write three importance-ranked term effects and their importance per row."""
    figure = create_ebm_top_feature_grid(rows, title)
    try:
        figure.canvas.draw()
        bounds = grid_export_bounds(figure).padded(SAVE_PADDING_INCHES)
        with plt.rc_context(figure_typography("ebm_top_features", bounds.width).rc()):
            for path in (png_path, svg_path):
                path.parent.mkdir(parents=True, exist_ok=True)
                figure.savefig(path, bbox_inches=bounds)
    finally:
        plt.close(figure)


def create_ebm_top_feature_grid(rows: Sequence[EbmTopFeatureRow], title: str) -> plt.Figure:
    """Build the original four-column layout with manuscript-scaled typography."""
    if not rows:
        raise ValueError("At least one recurrence-site row is required.")
    sorted_rows = tuple(sorted(rows, key=lambda row: row.label.casefold()))
    y_limits = next((row.style.y_limits for row in sorted_rows if row.style is not None), (0.3, 4.0))
    importance_limits = selected_importance_x_limits(sorted_rows)
    size = figure_size(len(sorted_rows))
    fonts = figure_typography("ebm_top_features", size[0])
    with plt.rc_context(fonts.rc()):
        figure, axes = plt.subplots(len(sorted_rows), GRID_COLUMN_COUNT, figsize=size, dpi=PNG_DPI, squeeze=False)
        for row_index, row in enumerate(sorted_rows):
            draw_row(axes[row_index], row, y_limits, importance_limits)
        figure.suptitle(title, fontsize=fonts.figure_title, weight="bold", gid="figure-title")
        colors = recurrence_site_colors(row.label for row in sorted_rows)
        for row in sorted_rows:
            figure.text(ROW_LABEL_MARGIN - 0.015, 0.5, row.label, ha="right", va="center",
                        fontsize=fonts.site_label, fontweight="bold", color=colors[row.label], gid="site-label")
        fit_grid_typography(figure)
    return figure


def grid_export_bounds(figure: plt.Figure) -> Bbox:
    """Keep the empty grid's canvas when unavailable models leave no plot to crop."""
    if not any(axis.get_visible() for axis in figure.axes):
        return figure.bbox_inches
    return figure.get_tightbbox(figure.canvas.get_renderer())


def fit_grid_typography(figure: plt.Figure) -> None:
    """Measure the tight export, then converge on fonts at its actual paper scale."""
    width = figure.get_figwidth()
    titles = [axis.get_title() for axis in figure.axes]
    for _ in range(12):
        fonts = figure_typography("ebm_top_features", width)
        for index, (axis, title) in enumerate(zip(figure.axes, titles, strict=True)):
            fonts.apply_axis(axis)
            axis.title.set_fontsize(fonts.site_label)
            axis.title.set_gid(f"ebm-panel-title-{index}")
            axis.title.set_text(wrap_text(title, axis.bbox.width,
                                          FontProperties(family=fonts.family, size=fonts.site_label, weight="bold"),
                                          figure.canvas.get_renderer()))
            fit_numeric_reference_labels(axis)
        for text in figure.texts:
            text.set_fontsize(fonts.site_label if text.get_gid() == "site-label" else fonts.figure_title)
        figure.tight_layout(
            rect=(ROW_LABEL_MARGIN, FIGURE_BOTTOM_MARGIN, FIGURE_RIGHT_MARGIN, FIGURE_TITLE_MARGIN),
            pad=0.5, h_pad=0.45 * ebm_row_gap_fraction(), w_pad=0.2,
        )
        row_labels = [text for text in figure.texts if text.get_gid() == "site-label"]
        for axis, label in zip(figure.axes[::GRID_COLUMN_COUNT], row_labels, strict=True):
            bounds = axis.get_position()
            label.set_y((bounds.y0 + bounds.y1) / 2)
        figure.canvas.draw()
        bounds = grid_export_bounds(figure)
        for text in figure.texts:
            if text.get_gid() != "site-label":
                text.set_x((bounds.x0 + bounds.x1) / (2 * figure.get_figwidth()))
        measured = bounds.width + 2 * SAVE_PADDING_INCHES
        if abs(measured / width - 1) < 0.0001:
            position_figure_title(figure, "ebm_top_features", export_width_inches=measured)
            return
        width = measured
    raise ValueError("EBM export width did not stabilize while fitting publication typography.")


def draw_row(
    axes: Sequence[plt.Axes],
    row: EbmTopFeatureRow,
    y_limits: tuple[float, float],
    importance_limits: tuple[float, float],
) -> None:
    """Draw one available model row and leave unavailable cells blank."""
    if row.explanation is None or row.style is None:
        for axis in axes:
            axis.set_visible(False)
        return
    style = replace(row.style, y_limits=y_limits, color=recurrence_site_colors((row.label,))[row.label])
    terms = tuple(univariate_terms_by_importance(row.explanation)[:TOP_FEATURE_COUNT])
    if not terms:
        for axis in axes:
            axis.set_visible(False)
        return
    for column, (term_index, term_name) in enumerate(terms):
        draw_hazard_ratio_grid_term(
            axes[column],
            row.explanation.data(term_index),
            term_name,
            style,
            show_ylabel=column == 0,
        )
    for axis in axes[len(terms) : TOP_FEATURE_COUNT]:
        axis.set_visible(False)
    draw_global_importance_axis(
        axes[-1],
        row.explanation.data(),
        ordered_term_indices=[term_index for term_index, _term_name in terms],
        test_name_translation=style.test_name_translation,
        x_limits=importance_limits,
        title="Global Importance",
        compact=True,
        color=style.color,
    )


def selected_importance_x_limits(
    rows: Sequence[EbmTopFeatureRow],
) -> tuple[float, float]:
    """Return one zero-based importance scale for every available row."""
    selected_scores: list[float] = []
    for row in rows:
        if row.explanation is None:
            continue
        scores = np.asarray(row.explanation.data()["scores"], dtype=float)
        selected_scores.extend(
            float(scores[term_index])
            for term_index, _term_name in univariate_terms_by_importance(row.explanation)[
                :TOP_FEATURE_COUNT
            ]
            if np.isfinite(scores[term_index])
        )
    maximum = max(selected_scores, default=0.0)
    return (0.0, maximum * 1.05 if maximum > 0.0 else 1.0)


def figure_size(row_count: int) -> tuple[float, float]:
    return (
        GRID_COLUMN_COUNT * HAZARD_RATIO_GRID_SUBPLOT_WIDTH + 0.8,
        row_count * HAZARD_RATIO_GRID_SUBPLOT_HEIGHT + 0.6,
    )
