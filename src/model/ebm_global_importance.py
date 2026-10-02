from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import MaxNLocator

from src.data.blood_test_display import feature_display_title
from src.figure_style import publication_style, wrap_text
from src.figure_headings import heading_typography, position_figure_title
from src.model.ebm_explanation_data import sequence_value as sequence_value
from src.model.ebm_plotting_defaults import IMPORTANCE_TITLE

PNG_DPI = 160
FIGURE_SIZE = (7.5, 5.2)
GLOBAL_IMPORTANCE_ROW_HEIGHT = 0.24
GLOBAL_IMPORTANCE_EXTRA_HEIGHT = 1.4
GLOBAL_IMPORTANCE_MAX_WIDTH = 14.0
GLOBAL_IMPORTANCE_LABEL_SIZE = 9.0
GLOBAL_IMPORTANCE_MIN_LABEL_SIZE = 5.0
COMPACT_BAR_HEIGHT = 0.30
COMPACT_BAR_Y_OFFSET = -0.21
COMPACT_LABEL_X_POSITION = 0.015
COMPACT_LABEL_Y_OFFSET = 0.03


def write_global_importance_plot(
    overall_data: Mapping[str, object],
    output_path: Path,
    *,
    test_name_translation: Mapping[str, str] | None = None,
    title: str = IMPORTANCE_TITLE,
    color: str | None = None,
) -> None:
    """Write a ranked table with inline bars from the model's exact term scores."""
    translations = {} if test_name_translation is None else test_name_translation
    names = [
        feature_display_title(str(name), translations)
        for name in sequence_value(overall_data["names"])
    ]
    figure_size = global_importance_figure_size(names)
    style = publication_style()
    headings = heading_typography("global_importance", *figure_size)
    with plt.rc_context(style.reference_sizes.rc()):
        figure, axis = plt.subplots(figsize=figure_size, dpi=PNG_DPI)
        try:
            figure.subplots_adjust(left=0.035, right=0.965, top=0.87, bottom=0.09)
            draw_global_importance_table(axis, names, sequence_value(overall_data["scores"]), color=color)
            figure.text(0.5, 0.96, title, fontsize=headings.figure_title,
                        fontweight="bold", va="top", ha="center", gid="figure-title")
            figure.text(0.035, 0.914, f"{len(names)} features · Ranked by global importance",
                        fontsize=headings.site_label, color=style.colors["secondary"], va="top")
            figure.add_artist(plt.Line2D([0.035, 0.965], [0.89, 0.89], transform=figure.transFigure,
                                        color=style.colors["overall"] if color is None else color, linewidth=1.2))
            figure.text(0.035, 0.032, "Score: mean absolute contribution to log-relative hazard",
                        fontsize=headings.site_label * 8.5 / 9, color=style.colors["secondary"])
            position_figure_title(figure, "global_importance")
            figure.savefig(output_path)
        finally:
            plt.close(figure)


def draw_global_importance_table(
    axis: plt.Axes,
    names: Sequence[str],
    scores: Sequence[object],
    *,
    color: str | None = None,
    x_max: float | None = None,
    score_heading: str = "Score",
) -> None:
    """Draw exact scores on a zero-based bar scale, with stable descending ranks.

    Only displayed numbers are rounded. Long feature names wrap within their
    column, with taller rows to keep text clear of the bars and other labels.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) != len(names):
        raise ValueError("Importance names and scores must have the same length.")
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Global importance scores must be finite and nonnegative.")
    maximum = float(values.max()) if len(values) else 0.0
    if x_max is None:
        x_max = float(MaxNLocator(nbins=4).tick_values(0, maximum)[-1]) if maximum > 0 else 1.0
    if not np.isfinite(x_max) or x_max <= 0 or x_max < maximum:
        raise ValueError("The importance scale must be positive and include every score.")
    style = publication_style()
    bar_color = style.colors["overall"] if color is None else color
    font_size = heading_typography("global_importance", *axis.get_figure().get_size_inches()).site_label
    font = FontProperties(family=style.reference_sizes.family, size=font_size)
    renderer = axis.get_figure().canvas.get_renderer()
    order = np.argsort(-values, kind="stable")
    labels = [wrap_text(names[index], axis.bbox.width * 0.42, font, renderer) for index in order]
    heights = np.asarray([max(1.0, 0.7 * (label.count("\n") + 1)) for label in labels])
    positions = np.cumsum(heights) - heights / 2
    axis.set_xlim(0, 1)
    axis.set_ylim(max(1.0, float(heights.sum())), -0.85)
    axis.axis("off")
    for rank, (index, label, height, position) in enumerate(zip(order, labels, heights, positions, strict=True)):
        if rank % 2 == 0:
            axis.axhspan(position - height / 2, position + height / 2,
                          color=style.colors["row"], linewidth=0, zorder=0)
        axis.text(0.032, position, str(rank + 1), ha="right", va="center",
                  fontsize=font_size * 8 / 9, color=style.colors["secondary"])
        axis.text(0.06, position, label, va="center", fontsize=font_size,
                  gid=f"importance-feature-{rank}")
        bars = axis.barh(position, values[index] / x_max * 0.34, left=0.51, height=0.35,
                         color=bar_color, linewidth=0)
        bars[0].set_gid(f"importance-bar-{rank}")
        score = values[index]
        displayed = f"{score:.3g}" if 0 < score < 0.001 else f"{score:.3f}"
        axis.text(0.98, position, displayed, ha="right", va="center", fontsize=font_size,
                  gid=f"importance-score-{rank}")
    for x, label, alignment in ((0.06, "Feature", "left"), (0.51, f"0–{x_max:g}", "left"),
                                (0.98, score_heading, "right")):
        axis.text(x, -0.52, label, fontsize=font_size, fontweight="bold", ha=alignment,
                  color=style.colors["secondary"])
    axis.axhline(-0.18, color=style.colors["border"], linewidth=0.65)


def draw_global_importance_axis(
    axis: plt.Axes,
    overall_data: Mapping[str, object],
    *,
    ordered_term_indices: Sequence[int] | None = None,
    test_name_translation: Mapping[str, str] | None = None,
    x_limits: tuple[float, float] | None = None,
    title: str = IMPORTANCE_TITLE,
    compact: bool = False,
    color: str | None = None,
) -> None:
    """Draw term importance, placing compact labels above thinner bars."""
    translations = {} if test_name_translation is None else test_name_translation
    names = [
        feature_display_title(str(name), translations)
        for name in sequence_value(overall_data["names"])
    ]
    scores = np.asarray(overall_data["scores"], dtype=float)
    descending_order = (
        np.argsort(scores)[::-1]
        if ordered_term_indices is None
        else np.asarray(ordered_term_indices, dtype=int)
    )
    plot_order = descending_order[::-1]
    positions = np.arange(len(plot_order))
    label_size = 6.0 if compact else global_importance_label_size(len(names))
    axis.barh(
        positions + COMPACT_BAR_Y_OFFSET if compact else positions,
        scores[plot_order],
        height=COMPACT_BAR_HEIGHT if compact else 0.72,
        color="#4daf4a" if color is None else color,
        edgecolor="#2b7a2b" if color is None else color,
    )
    axis.set_yticks(positions + COMPACT_LABEL_Y_OFFSET if compact else positions)
    axis.set_yticklabels([names[index] for index in plot_order], fontsize=label_size)
    axis.set_xlabel("Mean Absolute Score")
    axis.set_title(title)
    axis.grid(axis="x", linestyle=":", color="0.78")
    if x_limits is not None:
        axis.set_xlim(x_limits)
    if compact:
        axis.title.set_fontsize(8.0)
        axis.xaxis.label.set_fontsize(7.0)
        axis.tick_params(axis="x", labelsize=6.0)
        axis.set_ylim(-0.5, max(1, len(plot_order)) - 0.5)
        axis.tick_params(axis="y", length=0, pad=0)
        for label in axis.get_yticklabels():
            label.set_horizontalalignment("left")
            label.set_verticalalignment("bottom")
            label.set_x(COMPACT_LABEL_X_POSITION)


def global_importance_figure_size(names: Sequence[str]) -> tuple[float, float]:
    term_count = max(1, len(names))
    max_label_length = max((len(name) for name in names), default=0)
    width = max(FIGURE_SIZE[0], min(GLOBAL_IMPORTANCE_MAX_WIDTH, 6.0 + max_label_length * 0.055))
    height = max(FIGURE_SIZE[1], GLOBAL_IMPORTANCE_EXTRA_HEIGHT + term_count * GLOBAL_IMPORTANCE_ROW_HEIGHT)
    return width, height


def global_importance_label_size(term_count: int) -> float:
    if term_count <= 25:
        return GLOBAL_IMPORTANCE_LABEL_SIZE
    dense_plot_reduction = (term_count - 25) * 0.06
    return max(GLOBAL_IMPORTANCE_MIN_LABEL_SIZE, GLOBAL_IMPORTANCE_LABEL_SIZE - dense_plot_reduction)
