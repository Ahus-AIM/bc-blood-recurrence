"""Shared publication typography, colors, and physical spacing (points)."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib as mpl
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from src.figure_style import VECTOR_FONT_RC, publication_style, recurrence_site_colors as recurrence_site_colors, wrap_text

FONT_FAMILY = publication_style().reference_sizes.family
TEXT_COLOR = publication_style().colors["text"]
SECONDARY_COLOR = publication_style().colors["secondary"]
BORDER_COLOR = publication_style().colors["border"]
GRID_COLOR = publication_style().colors["grid"]
ROW_COLOR = publication_style().colors["row"]
RETAINED_COLOR = publication_style().colors["retained"]
EXCLUDED_COLOR = publication_style().colors["excluded"]
INCLUDED_COLOR = publication_style().colors["included"]
SMALL_FONT_SIZE = 5.5
FLOW_FONT_SIZE = 5.7
BORDER_WIDTH = 0.55
GRID_WIDTH = 0.4
CURVE_WIDTH = 1.0
OVERALL_CURVE_WIDTH = 1.4
BOX_PADDING = 2.5
FOOTNOTE_GAP_POINTS = 4.0
FLOW_GAP = 5.0
LINE_SPACING = 1.20
BLOODTEST_STACK_COLORS = ("#d62728", "#0072B2", "#E69F00")
OVERALL_SITE_COLOR = publication_style().colors["overall"]
DIAGNOSIS_YEAR_STACK_COLORS = (
    "#009E73",
    "#0072B2",
    "#D55E00",
    "#E69F00",
    "#CC79A7",
    "#56B4E9",
    "#999999",
    "#6A3D9A",
    "#8C510A",
)


def publication_rc(font_size: float, title_font_size: float) -> dict[str, object]:
    return {
        "font.family": FONT_FAMILY,
        "font.size": font_size,
        "text.color": TEXT_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.labelsize": font_size,
        "axes.titlesize": title_font_size,
        "axes.linewidth": BORDER_WIDTH,
        "axes.edgecolor": BORDER_COLOR,
        "xtick.labelsize": SMALL_FONT_SIZE,
        "ytick.labelsize": SMALL_FONT_SIZE,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.major.width": BORDER_WIDTH,
        "ytick.major.width": BORDER_WIDTH,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        **VECTOR_FONT_RC,
        "pgf.rcfonts": False,
    }


def style_plot(axis: Axes, *, vertical_grid: bool = False) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="both" if vertical_grid else "y", color=GRID_COLOR, linewidth=GRID_WIDTH)
    axis.set_axisbelow(True)


def wrap_to_width(axis: Axes, text: str, width_points: float, font_size: float) -> str:
    """Wrap using the actual font metrics, including long slash-separated labels."""
    renderer = axis.figure.canvas.get_renderer()
    font = FontProperties(family=FONT_FAMILY, size=font_size)
    available = width_points * axis.figure.dpi / 72.0
    return wrap_text(text.replace("/", "/ "), available, font, renderer)


def text_width_points(axis: Axes, text: str, font_size: float) -> float:
    renderer = axis.figure.canvas.get_renderer()
    font = FontProperties(family=FONT_FAMILY, size=font_size)
    width, _, _ = renderer.get_text_width_height_descent(text, font, False)
    return width * 72 / axis.figure.dpi


def wrap_numbered_note(axis: Axes, note: str, width: float, font_size: float) -> tuple[str, float]:
    """Wrap a numbered note, reserving its marker width for a hanging indent."""
    match = re.match(r"^[⁰¹²³⁴⁵⁶⁷⁸⁹]+\s+", note)
    prefix = match.group() if match else ""
    indent = text_width_points(axis, prefix, font_size)
    body = wrap_to_width(axis, note[len(prefix):], width - indent, font_size)
    return prefix + body, indent


def save_publication_figure(figure: Figure, output_path: Path, dpi: int) -> tuple[Path, ...]:
    """Export the same canvas as PNG, embedded-font PDF, and selectable-text SVG."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    paths = tuple(output_path.with_suffix(suffix) for suffix in (".png", ".pdf", ".svg"))
    with mpl.rc_context(VECTOR_FONT_RC):
        for path in paths:
            figure.savefig(path, dpi=dpi, facecolor="white")
        if output_path not in paths:
            figure.savefig(output_path, dpi=dpi)
    return paths
