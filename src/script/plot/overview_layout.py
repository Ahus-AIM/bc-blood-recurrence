"""Shared six-panel geometry and physical spacing for cohort figures."""
from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass
from typing import Any

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.text import Text
from matplotlib.transforms import Bbox

import yaml

from src.figure_style import figure_typography

LAYOUT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config/dataset_statistics_designs.yml"
LEGEND_GAP_POINTS = 5.0
DESIGN_CONFIG_PATH = LAYOUT_CONFIG_PATH.with_name("dataset_statistics_layouts.yml")


@dataclass(frozen=True)
class OverviewDesign:
    """Table emphasis, column alignment and physical gaps for a six-panel layout.

    Stacked curves use the full width; otherwise e/f share one row. All lengths
    are millimetres at the configured export width. Titles describe preview choices.
    """

    title: str
    description: str
    table_style: str
    aligned_columns: bool
    stacked_curves: bool
    panel_gap_mm: float
    curve_height_mm: float
    assay_height_mm: float


def load_overview_design(name: str = "compact") -> OverviewDesign:
    raw = yaml.safe_load(DESIGN_CONFIG_PATH.read_text(encoding="utf-8"))
    if name not in raw["presets"]:
        raise ValueError(f"Unknown dataset-statistics layout: {name}")
    return OverviewDesign(**(raw["defaults"] | raw["presets"][name]))


def load_cohort_layout() -> dict[str, Any]:
    """Read presentation settings only; manuscript counts and ages remain preview inputs."""
    raw = yaml.safe_load(LAYOUT_CONFIG_PATH.read_text(encoding="utf-8"))
    return {**raw["cohort_box_layouts"], "subcategory_indent_mm": raw["table_refinements"]["subcategory_indent_mm"]}


def create_cohort_axes(figure: Figure) -> tuple[Axes, ...]:
    """Seed six panel axes; production packs their rendered content before exporting."""
    figure.set_layout_engine(None)
    settings = load_cohort_layout()
    layout = settings["option_one"]
    legend_height = 4 * settings["calendar_legend_row_height_mm"]
    margins = settings["curve_tables"]["left_margins_mm"]
    widths = settings["curve_tables"]["panel_widths_mm"]
    right = settings["curve_tables"]["right_margin_mm"]
    curve_top = layout["curves_top_mm"] + 5
    curve_height = layout["curves_bottom_mm"] - layout["curves_top_mm"] - 12
    rectangles = ((3, 9, 102, 110), (120, 8 + legend_height, 75, settings["calendar_height_mm"] - 11 - legend_height),
                  (3, layout["table_header_y_mm"], 102, 52), (113, layout["assay_top_mm"] + 4, 84, 91),
                  (3 + margins[0], curve_top, widths[0] - margins[0] - right, curve_height),
                  (107 + margins[1], curve_top, widths[1] - margins[1] - right, curve_height))
    return tuple(
        figure.add_axes((x / settings["figure_width_mm"], 1 - (y + height) / settings["figure_height_mm"],
                         width / settings["figure_width_mm"], height / settings["figure_height_mm"]), label=f"panel-{letter}")
        for letter, (x, y, width, height) in zip("abcdef", rectangles, strict=True)
    )


def cohort_headers(figure: Figure, axes: tuple[Axes, ...], titles: tuple[str, ...]) -> None:
    """Align headings by column and match B's legend inset to D's content."""
    fonts = figure_typography("dataset_statistics", figure.get_figwidth())
    width, height = figure.get_size_inches() * 25.4
    positions = [(3, 3), (113, 3)]
    for index, axis in enumerate(axes[2:], 2):
        x = positions[index % 2][0]
        if index == 5 and axis.get_position().x0 == axes[4].get_position().x0:
            x = positions[0][0]
        positions.append((x, (1 - axis.get_position().y1) * height - 5))
    panel_headers: list[tuple[Text, Text]] = []
    for letter, title, (x, y) in zip("abcdef", titles, positions, strict=True):
        label = figure.text(x / width, 1 - y / height, letter, weight="bold", fontsize=fonts.panel_title, va="center")
        heading = figure.text((x + 4) / width, 1 - y / height, title, weight="bold", fontsize=fonts.panel_title, va="center", gid="panel-title")
        panel_headers.append((label, heading))
    renderer = figure.canvas.get_renderer()
    assay_top = max((artist.get_window_extent(renderer).y1
                     for artist in (*axes[3].texts, *axes[3].patches, *axes[3].lines) if artist.get_visible()),
                    default=axes[3].bbox.y1)
    title_gap = min(header.get_window_extent(renderer).y0 for header in panel_headers[3]) - assay_top
    place_diagnosis_year_legend(axes[1], panel_headers[1], title_gap)


def pack_cohort_panels(figure: Figure, axes: tuple[Axes, ...], design: OverviewDesign) -> None:
    """Place tables below rendered content, then fit curves below both upper columns."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    mm = figure.dpi / 25.4
    flow, calendar, composition, assays, incidence, survival = axes
    flow_bottom = min(patch.get_window_extent(renderer).y0 for patch in flow.patches
                      if patch.get_gid() == "study-cohort")
    flow_bottom = min((flow_bottom, *(text.get_window_extent(renderer).y0 for text in flow.texts
                                     if text.get_gid() == "flow-note")))
    calendar_bottom = calendar.xaxis.label.get_window_extent(renderer).y0
    if design.aligned_columns:
        shift = calendar_bottom - flow_bottom
        position = calendar.get_position()
        calendar.set_position((position.x0, position.y0 - shift / figure.bbox.height,
                               position.width, position.height + shift / figure.bbox.height))
        calendar_bottom = flow_bottom
    gap = (design.panel_gap_mm + 5) * mm
    set_panel_top(composition, (flow_bottom - gap) / figure.bbox.height)
    set_panel_top(assays, (calendar_bottom - gap) / figure.bbox.height,
                  height=design.assay_height_mm / (figure.get_figheight() * 25.4))
    figure.canvas.draw()
    upper_bottom = min(text.get_window_extent(renderer).y0 for axis in (composition, assays) for text in axis.texts)
    curve_top = (upper_bottom - gap) / figure.bbox.height
    for axis in (incidence, survival):
        set_panel_top(axis, curve_top, height=design.curve_height_mm / (figure.get_figheight() * 25.4))
        if design.stacked_curves:
            position = axis.get_position()
            left = incidence.get_position().x0
            axis.set_position((left, position.y0, .985 - left, position.height))


def set_panel_top(axis: Axes, top: float, *, height: float | None = None) -> None:
    position = axis.get_position()
    size = position.height if height is None else height
    axis.set_position((position.x0, top - size, position.width, size))


def fill_right_column(figure: Figure, axes: tuple[Axes, ...], gap_mm: float) -> None:
    """Place d above fixed panel f and extend b downward, using equal visible gaps."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    calendar, assays, survival = axes[1], axes[3], axes[5]
    headings = [text for text in figure.texts if text.get_gid() == "panel-title"]
    assay_headers = [headings[3], *(text for text in figure.texts if text.get_text() == "d")]
    survival_headers = [headings[5], *(text for text in figure.texts if text.get_text() == "f")]
    assay_bounds = Bbox.union([
        artist.get_window_extent(renderer)
        for artist in (*assays.texts, *assays.patches, *assays.lines, *assay_headers)
        if artist.get_visible()
    ])
    upper_bottom = calendar.get_tightbbox(renderer).y0
    lower_top = max(survival.get_tightbbox(renderer).y1,
                    *(text.get_window_extent(renderer).y1 for text in survival_headers))
    gap = gap_mm * figure.dpi / 25.4
    shift_pixels = lower_top + gap - assay_bounds.y0
    shift = shift_pixels / figure.bbox.height
    position = assays.get_position()
    assays.set_position((position.x0, position.y0 + shift, position.width, position.height))
    for text in assay_headers:
        x, y = text.get_position()
        text.set_position((x, y + shift))
    calendar_shift = (assay_bounds.y1 + shift_pixels + gap - upper_bottom) / figure.bbox.height
    position = calendar.get_position()
    calendar.set_position((position.x0, position.y0 + calendar_shift,
                           position.width, position.height - calendar_shift))


def stack_survival_panel(incidence: Axes, survival: Axes, gap_mm: float) -> None:
    """Place f after e's annual table and time-axis label have been drawn."""
    figure = incidence.figure
    figure.canvas.draw()
    bottom = incidence.get_tightbbox(figure.canvas.get_renderer()).y0
    top = (bottom - (gap_mm + 5) * figure.dpi / 25.4) / figure.bbox.height
    set_panel_top(survival, top)


def fit_cohort_footer(figure: Figure) -> None:
    """Fit the canvas to the annual tables with a 3-mm bottom margin."""
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bottom = min(axis.get_tightbbox(renderer).y0 for axis in figure.axes)
    required = (3 / 25.4 * figure.dpi - bottom) / figure.dpi
    if abs(required) > .01:
        reserve_footer(figure, required)


def fit_recurrence_tick_spacing(axes: tuple[Axes, Axes]) -> None:
    """Keep integer incidence labels legible by extending both plots down together."""
    figure = axes[0].figure
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    incidence = axes[0]
    lower, upper = incidence.get_ylim()
    labels = [label for value, label in zip(incidence.get_yticks(), incidence.get_yticklabels(), strict=True)
              if lower <= value <= upper]
    if len(labels) < 2:
        return
    label_height = max(label.get_window_extent(renderer).height for label in labels)
    needed = (len(labels) - 1) * (label_height + 1.5) / figure.bbox.height
    height = max(needed, axes[0].get_position().height)
    for axis in axes:
        bounds = axis.get_position()
        axis.set_position((bounds.x0, bounds.y1 - height, bounds.width, height))


def place_diagnosis_year_legend(axis: Axes, headers: tuple[Text, Text], title_gap: float) -> None:
    """Match D's title gap in display pixels; keep B's legend clear of its plot."""
    legend = axis.get_legend()
    if legend is None:
        return
    figure = axis.figure
    renderer = figure.canvas.get_renderer()
    gap = LEGEND_GAP_POINTS * figure.dpi / 72
    header_bottom = min(header.get_window_extent(renderer).y0 for header in headers)
    position = axis.get_position()
    legend.set_loc("upper left")
    legend.set_bbox_to_anchor(
        (position.x0, (header_bottom - title_gap) / figure.bbox.height),
        transform=figure.transFigure,
    )
    legend_bottom = legend.get_window_extent(renderer).y0
    plot_top = min(position.y1, (legend_bottom - gap) / figure.bbox.height)
    axis.set_position((position.x0, position.y0, position.width, plot_top - position.y0))


def reserve_footer(figure: Figure, extra_height: float) -> None:
    """Resize the bottom margin while preserving physical panel and legend positions."""
    old_height = figure.get_figheight()
    new_height = old_height + extra_height
    scale = old_height / new_height
    offset = extra_height / new_height
    legends = [*figure.legends, *(axis.get_legend() for axis in figure.axes if axis.get_legend() is not None)]
    legend_bounds = [legend.get_bbox_to_anchor().transformed(figure.transFigure.inverted()) for legend in legends]
    figure.set_size_inches(figure.get_figwidth(), new_height)
    for axis in figure.axes:
        bounds = axis.get_position()
        axis.set_position((bounds.x0, offset + bounds.y0 * scale, bounds.width, bounds.height * scale))
    for text in figure.texts:
        x, y = text.get_position()
        text.set_position((x, offset + y * scale))
    for legend, bounds in zip(legends, legend_bounds, strict=True):
        translated = Bbox.from_bounds(bounds.x0, offset + bounds.y0 * scale, bounds.width, bounds.height * scale)
        legend.set_bbox_to_anchor(translated, transform=figure.transFigure)
