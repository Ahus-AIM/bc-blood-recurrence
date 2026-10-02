"""Heading sizes and spacing at the final manuscript scale, including height limits."""

from __future__ import annotations

from dataclasses import fields
from fnmatch import fnmatchcase
from pathlib import Path

from matplotlib.backend_bases import RendererBase
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.text import Text
from matplotlib.textpath import TextToPath

from src.config import load_yml
from src.figure_style import Typography, publication_style

CONFIG_PATH = Path(__file__).parent / "config/figure_headings.yml"


def manuscript_scale(stem: str, width_points: float, height_points: float) -> float:
    """Return PDF points per SVG point for the configured includegraphics limits."""
    config = load_yml(CONFIG_PATH)
    style = publication_style()
    for pattern, placement in config["placements"].items():
        if fnmatchcase(stem, pattern):
            width_scale = style.text_width_tex_pt * 72 / 72.27 * placement["width"] / width_points
            if placement["height"] is None:
                return width_scale
            height_scale = config["manuscript"]["text_height_tex_pt"] * 72 / 72.27 * placement["height"] / height_points
            return min(width_scale, height_scale)
    raise ValueError(f"No manuscript placement configured for {stem}.")


def heading_typography(stem: str, width_inches: float, height_inches: float) -> Typography:
    """Match Figure 3's printed title and site-label sizes on any configured canvas."""
    style = publication_style()
    reference_scale = style.printed_width_inches(style.reference_figure) / style.reference_export_width_inches
    factor = style.scale * reference_scale / manuscript_scale(stem, width_inches * 72, height_inches * 72)
    sizes = {
        field.name: getattr(style.reference_sizes, field.name) * factor
        for field in fields(Typography) if field.name != "family"
    }
    return Typography(style.reference_sizes.family, **sizes)


def title_gap_points(stem: str, width_points: float, height_points: float) -> float:
    """Convert the shared printed gap to the figure's own coordinate scale."""
    gap = load_yml(CONFIG_PATH)["manuscript"]["title_gap_pdf_pt"]
    return gap / manuscript_scale(stem, width_points, height_points)


def ebm_row_gap_fraction() -> float:
    """Fraction of the previous inter-row padding retained in EBM effect grids."""
    return load_yml(CONFIG_PATH)["ebm_rows"]["gap_fraction"]


def glyph_vertical_metrics(text: str, font: FontProperties) -> tuple[float, float]:
    """Return visible ascent and descent in points, excluding font-box padding."""
    _, height, descent = TextToPath().get_text_width_height_descent(text, font, ismath=False)
    return height - descent, descent


def visible_text_bounds(text: Text, renderer: RendererBase) -> tuple[float, float]:
    """Return bottom and top in display pixels, using glyph edges for upright text."""
    bounds = text.get_window_extent(renderer)
    if text.get_rotation() or "$" in text.get_text():
        return bounds.y0, bounds.y1
    font = text.get_fontproperties()
    lines = text.get_text().split("\n")
    _, minimum_height, minimum_descent = renderer.get_text_width_height_descent("lp", font, False)
    _, first_height, first_descent = renderer.get_text_width_height_descent(lines[0], font, False)
    _, _, last_descent = renderer.get_text_width_height_descent(lines[-1], font, False)
    first_ascent = max(first_height, minimum_height) - max(first_descent, minimum_descent)
    ascent, _ = glyph_vertical_metrics(lines[0], font)
    _, descent = glyph_vertical_metrics(lines[-1], font)
    pixels_per_point = text.get_figure().dpi / 72
    return (
        bounds.y0 + max(last_descent, minimum_descent) - descent * pixels_per_point,
        bounds.y1 - first_ascent + ascent * pixels_per_point,
    )


def position_figure_title(figure: Figure, stem: str, *, export_width_inches: float | None = None) -> None:
    """Place the overall title above the nearest visible content after layout is final."""
    title = next(text for text in figure.texts if text.get_gid() == "figure-title")
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    axes = [axis for axis in figure.axes if axis.get_visible()]
    texts = [*figure.texts, *(text for axis in axes for text in axis.findobj(Text)),
             *(text for legend in figure.legends for text in legend.get_texts())]
    tops = [visible_text_bounds(text, renderer)[1] for text in texts
            if text is not title and text.get_visible() and text.get_text().strip()]
    tops.extend(axis.bbox.y1 for axis in axes if axis.axison)
    if not tops:
        return
    width = figure.get_figwidth() if export_width_inches is None else export_width_inches
    gap = title_gap_points(stem, width * 72, figure.get_figheight() * 72) * figure.dpi / 72
    bottom, _ = visible_text_bounds(title, renderer)
    title.set_y(title.get_position()[1] + (max(tops) + gap - bottom) / figure.bbox.height)
