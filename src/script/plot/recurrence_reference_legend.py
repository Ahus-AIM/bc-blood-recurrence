"""Fit uniformly sized reference legends in continuous trajectory-axis headroom."""

from __future__ import annotations

import numpy as np
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.legend import Legend
from matplotlib.patches import StepPatch
from matplotlib.text import Text
from matplotlib.ticker import FixedLocator
from matplotlib.transforms import Affine2D, blended_transform_factory

from src.figure_style import wrap_text
from src.model.normal_range_plotting import NORMAL_RANGE_SPAN_ALPHA, normal_range_color
from src.model.normal_ranges import NormalRange

HEADROOM_POINTS = 30.0
LEGEND_PADDING_POINTS = 6.0
LEGACY_LAYOUT_GID = "trajectory-continuous-legend"
LAYOUT_GID = "trajectory-continuous-legend-uniform"


def wrapped_reference_legend(axis: Axes, original: Legend, renderer: RendererAgg) -> Legend:
    """Reduce the number of columns, never the font size, and retain reading order by row."""
    figure = axis.get_figure()
    size = original.get_texts()[0].get_fontsize()
    handles = original.legend_handles
    maximum_width = axis.bbox.width - 2 * LEGEND_PADDING_POINTS * figure.dpi / 72
    font = original.get_texts()[0].get_fontproperties()
    labels = [wrap_text(label.get_text(), maximum_width - 2.8 * size * figure.dpi / 72, font, renderer)
              for label in original.get_texts()]
    for handle in handles:
        handle.set_alpha(NORMAL_RANGE_SPAN_ALPHA)
        handle.set_edgecolor("none")
    original.remove()
    for columns in range(len(labels), 0, -1):
        order = [index for column in range(columns) for index in range(column, len(labels), columns)]
        legend = axis.legend(handles=[handles[index] for index in order], labels=[labels[index] for index in order],
                             loc="upper left", ncols=columns, frameon=False, borderaxespad=0, borderpad=0,
                             fontsize=size, handlelength=1.35, handletextpad=0.45, columnspacing=1,
                             labelspacing=0.25)
        if legend.get_window_extent(renderer).width <= maximum_width:
            return legend
        legend.remove()
    raise ValueError("Trajectory reference labels cannot fit at the shared font size.")


def inset_reference_legend(axis: Axes, percent_axis: Axes, heading: Text,
                           ranges: tuple[NormalRange, ...]) -> None:
    """Extend the y scale through clear legend space while keeping both scales aligned."""
    legend = axis.get_legend()
    if legend is None:
        return
    figure = axis.get_figure()
    renderer = RendererAgg(1, 1, figure.dpi)
    legend = wrapped_reference_legend(axis, legend, renderer)
    position = axis.get_position()
    height_points = position.height * figure.get_figheight() * 72
    headroom = max(HEADROOM_POINTS, legend.get_window_extent(renderer).height * 72 / figure.dpi
                   + 2 * LEGEND_PADDING_POINTS)
    heading_bottom = heading.get_window_extent(renderer).y0 / figure.bbox.height
    available = (heading_bottom - position.y1) * figure.get_figheight() * 72 - 4
    # Reuse the space vacated by the outside legend; keep a clear title gap.
    growth = min(headroom, max(0, available))
    new_height = height_points + growth
    data_fraction = (new_height - headroom) / new_height
    if data_fraction <= 0:
        raise ValueError("Trajectory panel is too short for its reference legend.")
    lower, upper = axis.get_ylim()
    old_ticks = axis.get_yticks()
    axis.set_position((position.x0, position.y0, position.width, new_height / (figure.get_figheight() * 72)))
    axis.set_ylim(lower, lower + (upper - lower) / data_fraction)
    if len(old_ticks) > 1:
        step = old_ticks[1] - old_ticks[0]
        axis.yaxis.set_major_locator(FixedLocator(np.arange(old_ticks[0], axis.get_ylim()[1] + step, step)))
    percent_axis.set_ylim(0, 100 / data_fraction)
    # Percentage steps are normalized on the left axes, independently of its units.
    transform = blended_transform_factory(axis.transData, Affine2D().scale(1, data_fraction) + axis.transAxes)
    for patch in axis.patches:
        if isinstance(patch, StepPatch):
            patch.set_transform(transform)
    legend.set_bbox_to_anchor((LEGEND_PADDING_POINTS / (position.width * figure.get_figwidth() * 72),
                               1 - LEGEND_PADDING_POINTS / new_height))
    legend.set_gid(LAYOUT_GID)
    for label in legend.get_texts():
        label.set_bbox({"facecolor": "white", "edgecolor": "none", "pad": 0.5})
    for interval in ranges:
        if interval.upper is None:
            # The shading retains the original data window; the arrow denotes continuation.
            axis.annotate("", xy=(0.98, upper + (axis.get_ylim()[1] - upper) * 0.72), xytext=(0.98, upper),
                          xycoords=blended_transform_factory(axis.transAxes, axis.transData),
                          arrowprops={"arrowstyle": "->", "color": normal_range_color(interval), "lw": 0.8})
