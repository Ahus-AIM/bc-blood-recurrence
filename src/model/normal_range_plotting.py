from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.colors import to_hex, to_rgb
from matplotlib.legend import Legend
from matplotlib.patches import Patch

from src.data.reference_intervals import reference_interval_label
from src.model.normal_ranges import NormalRange
from src.figure_style import reference_group_color, wrap_text

NORMAL_RANGE_LEGEND_FONT_SIZE = 7.0
NUMERIC_REFERENCE_FONT_SIZE = 7.4
NORMAL_RANGE_SPAN_ALPHA = 0.14
NORMAL_RANGE_LEGEND_ALPHA = 0.55
PUBLICATION_REFERENCE_ALPHA = 0.32
NUMERIC_REFERENCE_SWATCH_ALPHA = 0.70
NORMAL_RANGE_INDICATOR_LOWER_Y = 0.04
NORMAL_RANGE_INDICATOR_UPPER_Y = 0.16
NORMAL_RANGE_INDICATOR_LINE_WIDTH = 3.0
NUMERIC_REFERENCE_GID = "numeric-reference-intervals"
ADULT_EQUIVALENT_AGE_GROUPS = frozenset(
    {
        "adult",
        "allages",
        ">18years",
        ">=18years",
        ">1month",
        ">=1month",
        "adultfastingserum",
        "adultg1",
    }
)
AGE_INTERVAL_PATTERN = re.compile(
    r"^(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s+years?$"
)
AGE_BOUND_PATTERN = re.compile(r"^([<>≤≥])\s*(\d+(?:\.\d+)?)\s+years?$")


def add_univariate_normal_ranges(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    show_legend: bool = True,
    show_boundaries: bool = True,
    show_multiple_as_horizontal_bars: bool = False,
    span_alpha: float = NORMAL_RANGE_SPAN_ALPHA,
) -> None:
    """Draw x-axis references, optionally replacing multiple visible spans with bars."""
    if not ranges:
        return
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    use_horizontal_bars = (
        show_multiple_as_horizontal_bars
        and visible_range_count(axis, ranges, orientation="x") > 1
    )
    if use_horizontal_bars:
        add_horizontal_range_indicators(axis, ranges)
    else:
        add_normal_ranges(axis, ranges, orientation="x", show_boundaries=show_boundaries, span_alpha=span_alpha)
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    if show_legend:
        add_deduplicated_legend(axis)


def add_interaction_normal_ranges(
    axis: plt.Axes,
    left_label: str,
    right_label: str,
    ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None,
) -> None:
    if not ranges_by_feature:
        return
    x_limits = axis.get_xlim()
    y_limits = axis.get_ylim()
    add_normal_ranges(axis, ranges_by_feature.get(left_label, ()), orientation="x")
    add_normal_ranges(axis, ranges_by_feature.get(right_label, ()), orientation="y")
    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    add_deduplicated_legend(axis)


def add_numeric_reference_labels(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    font_size: float = NUMERIC_REFERENCE_FONT_SIZE,
) -> None:
    """Label full reference bounds inside the upper-left corner of each panel.

    Borderless legends retain shared sex/age colors. Numeric limits stay local
    to their feature and include sex when more than one sex is displayed.
    """
    if not ranges:
        return
    include_sex = len({interval.sex for interval in ranges}) > 1
    unique = {reference_interval_label(interval, include_sex=include_sex): interval for interval in ranges}
    handles = [Patch(facecolor=numeric_reference_swatch_color(axis, interval), edgecolor="none")
               for interval in unique.values()]
    legend = Legend(axis, handles, list(unique), loc="upper left",
                    frameon=False, borderaxespad=0.7, borderpad=0, fontsize=font_size,
                    handlelength=1.2, handletextpad=0.45, labelspacing=0.2)
    legend.set_gid(NUMERIC_REFERENCE_GID)
    axis.add_artist(legend)
    fit_numeric_reference_labels(axis, font_size)


def reference_swatch_color(color: str, alpha: float) -> str:
    """Composite on white so a legend swatch cannot darken over a shaded reference band."""
    return to_hex(tuple(alpha * component + (1 - alpha) for component in to_rgb(color)))


def numeric_reference_swatch_color(axis: plt.Axes, interval: NormalRange) -> str:
    """Strengthen area-legend colors while retaining fully colored interval-bar swatches."""
    label = normal_range_label(interval)
    alpha = PUBLICATION_REFERENCE_ALPHA
    for patch in axis.patches:
        if patch.get_label() == label:
            alpha = patch.get_facecolor()[3]
            break
    for line in axis.lines:
        if line.get_label() == label:
            alpha = 1.0 if line.get_alpha() is None else line.get_alpha()
            break
    return reference_swatch_color(normal_range_color(interval), max(alpha, NUMERIC_REFERENCE_SWATCH_ALPHA))


def fit_numeric_reference_labels(axis: plt.Axes, font_size: float | None = None) -> None:
    """Wrap in-panel reference labels after font or axes-size changes."""
    figure = axis.get_figure()
    renderer = figure.canvas.get_renderer()
    for artist in axis.artists:
        if not isinstance(artist, Legend) or artist.get_gid() != NUMERIC_REFERENCE_GID:
            continue
        for text in artist.get_texts():
            if font_size is not None:
                text.set_fontsize(font_size)
            size = text.get_fontsize()
            handle_width = 3.5 * size * figure.dpi / 72
            text.set_text(wrap_text(text.get_text().replace("\n", " "), axis.bbox.width - handle_width,
                                   text.get_fontproperties(), renderer))


def add_normal_ranges(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    orientation: str,
    show_boundaries: bool = True,
    span_alpha: float = NORMAL_RANGE_SPAN_ALPHA,
) -> None:
    add_range_spans(axis, ranges, orientation=orientation, alpha=span_alpha)
    if show_boundaries:
        add_range_boundaries(axis, ranges, orientation=orientation)


def add_range_spans(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    orientation: str,
    alpha: float = NORMAL_RANGE_SPAN_ALPHA,
) -> None:
    for normal_range in ranges:
        bounds = visible_range_bounds(axis, normal_range, orientation)
        if bounds is None:
            continue
        label = normal_range_label(normal_range)
        color = normal_range_color(normal_range)
        if orientation == "x":
            axis.axvspan(
                bounds[0],
                bounds[1],
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                label=label,
                zorder=0,
            )
        else:
            axis.axhspan(
                bounds[0],
                bounds[1],
                facecolor=color,
                edgecolor="none",
                alpha=alpha,
                label=label,
                zorder=0,
            )


def add_range_boundaries(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    orientation: str,
) -> None:
    for normal_range in ranges:
        color = normal_range_color(normal_range)
        for bound in normal_range_boundaries(normal_range):
            if bound is None:
                continue
            draw_range_boundary(axis, bound, orientation, color, None)


def add_horizontal_range_indicators(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
) -> None:
    """Draw staggered solid segments so overlapping x-range spans remain identifiable."""
    visible_ranges = [
        (normal_range, bounds)
        for normal_range in ranges
        if (bounds := visible_range_bounds(axis, normal_range, "x")) is not None
    ]
    levels = horizontal_indicator_levels(len(visible_ranges))
    for (normal_range, bounds), level in zip(visible_ranges, levels, strict=True):
        axis.plot(
            bounds,
            (level, level),
            color=normal_range_color(normal_range),
            linewidth=NORMAL_RANGE_INDICATOR_LINE_WIDTH,
            solid_capstyle="butt",
            transform=axis.get_xaxis_transform(),
            label=normal_range_label(normal_range),
            zorder=2.5,
        )


def horizontal_indicator_levels(count: int) -> np.ndarray:
    if count < 1:
        return np.asarray([], dtype=float)
    if count == 1:
        return np.asarray([NORMAL_RANGE_INDICATOR_LOWER_Y], dtype=float)
    return np.linspace(
        NORMAL_RANGE_INDICATOR_LOWER_Y,
        NORMAL_RANGE_INDICATOR_UPPER_Y,
        count,
    )


def visible_range_count(
    axis: plt.Axes,
    ranges: tuple[NormalRange, ...],
    *,
    orientation: str,
) -> int:
    return sum(
        visible_range_bounds(axis, normal_range, orientation) is not None
        for normal_range in ranges
    )


def draw_range_boundary(
    axis: plt.Axes,
    bound: float,
    orientation: str,
    color: str,
    label: str | None,
) -> None:
    if axis_is_log_scaled(axis, orientation) and bound <= 0.0:
        return
    if orientation == "x":
        axis.axvline(bound, color=color, linestyle="--", linewidth=0.85, label=label)
    else:
        axis.axhline(bound, color=color, linestyle="--", linewidth=0.85, label=label)


def axis_is_log_scaled(axis: plt.Axes, orientation: str) -> bool:
    if orientation == "x":
        return axis.get_xscale() == "log"
    return axis.get_yscale() == "log"


def normal_range_label(normal_range: NormalRange) -> str:
    sex_label = normal_range.sex.capitalize()
    age_group = normalized_age_group_label(normal_range.age_group)
    return f"{sex_label} {age_group} reference" if age_group else f"{sex_label} reference"


def normalized_age_group_label(age_group: str) -> str:
    cleaned = " ".join(age_group.strip().split())
    compact = cleaned.lower().replace(" ", "")
    if compact in ADULT_EQUIVALENT_AGE_GROUPS:
        return "adult"
    return cleaned.replace(">=", "≥").replace("<=", "≤")


def normal_range_color(normal_range: NormalRange) -> str:
    label = normal_range_label(normal_range)
    return reference_group_color(label)


def normal_range_boundaries(normal_range: NormalRange) -> tuple[float | None, float | None]:
    return normal_range.lower, normal_range.upper


def visible_range_bounds(
    axis: plt.Axes,
    normal_range: NormalRange,
    orientation: str,
) -> tuple[float, float] | None:
    lower_limit, upper_limit = axis.get_xlim() if orientation == "x" else axis.get_ylim()
    lower = lower_limit if normal_range.lower is None else normal_range.lower
    upper = upper_limit if normal_range.upper is None else normal_range.upper
    lower = max(lower, min(lower_limit, upper_limit))
    upper = min(upper, max(lower_limit, upper_limit))
    if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
        return None
    return float(lower), float(upper)


def add_deduplicated_legend(axis: plt.Axes) -> None:
    handles, labels = deduplicated_axis_legend_items(axis)
    if handles:
        axis.legend(handles, labels, loc="best", fontsize=NORMAL_RANGE_LEGEND_FONT_SIZE)


def deduplicated_axis_legend_items(axis: plt.Axes) -> tuple[tuple[object, ...], tuple[str, ...]]:
    handles, labels = axis.get_legend_handles_labels()
    return deduplicated_legend_items(handles, labels)


def deduplicated_figure_legend_items(axes: Iterable[plt.Axes]) -> tuple[tuple[object, ...], tuple[str, ...]]:
    handles: list[object] = []
    labels: list[str] = []
    for axis in axes:
        axis_handles, axis_labels = axis.get_legend_handles_labels()
        handles.extend(axis_handles)
        labels.extend(axis_labels)
    unique_handles, unique_labels = deduplicated_legend_items(handles, labels)
    ordered_items = sorted(
        zip(unique_handles, unique_labels, strict=True),
        key=lambda item: reference_legend_sort_key(item[1]),
    )
    return (
        tuple(handle for handle, _label in ordered_items),
        tuple(label for _handle, label in ordered_items),
    )


def deduplicated_legend_items(
    handles: Iterable[object],
    labels: Iterable[str],
) -> tuple[tuple[object, ...], tuple[str, ...]]:
    unique: dict[str, object] = {}
    for handle, label in zip(handles, labels, strict=True):
        if label and label not in unique:
            unique[label] = legend_handle(label, handle)
    return tuple(unique.values()), tuple(unique.keys())


def legend_handle(label: str, handle: object) -> object:
    if not label.endswith(" reference") or isinstance(handle, Line2D):
        return handle
    color = reference_group_color(label)
    return Patch(facecolor=color, edgecolor=color, alpha=NORMAL_RANGE_LEGEND_ALPHA)


def reference_legend_sort_key(label: str) -> tuple[int, float, float, str]:
    """Place the adult reference first, then references by numeric age bounds."""
    age_group = reference_age_group(label)
    if age_group == "adult":
        return 0, 0.0, 0.0, label
    interval_match = AGE_INTERVAL_PATTERN.fullmatch(age_group)
    if interval_match is not None:
        lower, upper = (float(value) for value in interval_match.groups())
        return 1, lower, upper, label
    bound_match = AGE_BOUND_PATTERN.fullmatch(age_group)
    if bound_match is not None:
        operator, value_text = bound_match.groups()
        value = float(value_text)
        if operator in {"<", "≤"}:
            return 1, 0.0, value, label
        return 1, value, float("inf"), label
    return 2, float("inf"), float("inf"), label


def reference_age_group(label: str) -> str:
    if not label.endswith(" reference"):
        return ""
    label_without_suffix = label.removesuffix(" reference")
    _sex, separator, age_group = label_without_suffix.partition(" ")
    return age_group if separator else ""


def add_grid_cell_legend(
    axis: plt.Axes,
    handles: tuple[object, ...],
    labels: tuple[str, ...],
) -> None:
    if not handles:
        return
    axis.legend(
        handles,
        labels,
        loc="center left",
        bbox_to_anchor=(0.0, 0.5),
        ncol=1,
        frameon=False,
        fontsize=NORMAL_RANGE_LEGEND_FONT_SIZE,
    )
