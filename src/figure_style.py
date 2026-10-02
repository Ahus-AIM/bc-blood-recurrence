"""Shared colors and typography measured at the figures' manuscript widths."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from functools import lru_cache
from math import isfinite
from pathlib import Path
from typing import Literal

from matplotlib.axes import Axes
from matplotlib.backend_bases import RendererBase
from matplotlib.colors import is_color_like
from matplotlib.font_manager import FontProperties

from src.config import load_yml

FigureKind = Literal["dataset_statistics", "recurrence_blood_test_trajectories", "ebm_top_features"]
FIGURE_KINDS: tuple[FigureKind, ...] = (
    "dataset_statistics", "recurrence_blood_test_trajectories", "ebm_top_features",
)
CONFIG_PATH = Path(__file__).parent / "config/publication_style.yml"
VECTOR_FONT_RC: dict[str, str | int] = {"pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none"}


@dataclass(frozen=True)
class Typography:
    """Font family and point sizes by text role at one exported canvas width.

    Titles and site labels are bold; table values keep their statistical emphasis.
    Sizes account for manuscript scaling, so wider exports use larger source text.
    """

    family: str
    figure_title: float
    site_label: float
    panel_title: float
    axis_label: float
    tick_label: float
    legend: float
    table: float

    def rc(self) -> dict[str, object]:
        return {
            "font.family": self.family,
            "font.size": self.table,
            "text.color": publication_style().colors["text"],
            "axes.labelcolor": publication_style().colors["text"],
            "xtick.color": publication_style().colors["text"],
            "ytick.color": publication_style().colors["text"],
            "axes.titlesize": self.panel_title,
            "axes.titleweight": "bold",
            "axes.labelsize": self.axis_label,
            "xtick.labelsize": self.tick_label,
            "ytick.labelsize": self.tick_label,
            "legend.fontsize": self.legend,
            **VECTOR_FONT_RC,
        }

    def apply_axis(self, axis: Axes) -> None:
        """Override local plot defaults, including minor ticks and offset text."""
        axis.title.set_fontsize(self.panel_title)
        axis.title.set_fontweight("bold")
        for label in (axis.xaxis.label, axis.yaxis.label):
            label.set_fontsize(self.axis_label)
        axis.tick_params(axis="both", which="both", labelsize=self.tick_label)
        for label in (axis.xaxis.get_offset_text(), axis.yaxis.get_offset_text()):
            label.set_fontsize(self.tick_label)


@dataclass(frozen=True)
class PublicationStyle:
    """Manuscript dimensions, original EBM typography, and semantic color maps.

    ``text_width_tex_pt`` uses TeX points; reference font sizes use Matplotlib
    points. ``reference_export_width_inches`` includes the original EBM crop.
    ``scale`` multiplies that reference's printed text size for every figure.
    """

    text_width_tex_pt: float
    width_fractions: Mapping[str, float]
    reference_figure: str
    reference_export_width_inches: float
    scale: float
    reference_sizes: Typography
    colors: Mapping[str, str]
    sites: Mapping[str, str]
    reference_groups: Mapping[str, str]

    def printed_width_inches(self, kind: str) -> float:
        return self.text_width_tex_pt / 72.27 * self.width_fractions[kind]

    def typography(self, kind: FigureKind, export_width_inches: float) -> Typography:
        positive_number(export_width_inches, "Export width")
        reference_scale = self.printed_width_inches(self.reference_figure) / self.reference_export_width_inches
        export_scale = export_width_inches / self.printed_width_inches(kind)
        factor = self.scale * reference_scale * export_scale
        sizes = {
            field.name: getattr(self.reference_sizes, field.name) * factor
            for field in fields(Typography) if field.name != "family"
        }
        return Typography(self.reference_sizes.family, **sizes)


def positive_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be a number.")
    number = float(value)
    if not isfinite(number) or number <= 0:
        raise ValueError(f"{name} must be finite and positive.")
    return number


def color_mapping(values: Mapping[str, object]) -> dict[str, str]:
    result = {name: str(color) for name, color in values.items()}
    if not result or not all(is_color_like(color) for color in result.values()):
        raise ValueError("Publication palettes must be nonempty and contain valid colors.")
    return result


def mapping_value(values: Mapping[str, object], key: str) -> Mapping[str, object]:
    value = values[key]
    if not isinstance(value, Mapping) or not all(isinstance(name, str) for name in value):
        raise ValueError(f"Publication {key} must be a mapping with string keys.")
    return value


@lru_cache(maxsize=1)
def publication_style() -> PublicationStyle:
    """Read the shared config once per rendering process."""
    raw = load_yml(CONFIG_PATH)
    manuscript, typography = raw["manuscript"], raw["typography"]
    configured_widths = mapping_value(manuscript, "width_fractions")
    configured_sizes = mapping_value(typography, "reference_sizes")
    fractions = {
        kind: positive_number(configured_widths[kind], kind) for kind in FIGURE_KINDS
    }
    if max(fractions.values()) > 1:
        raise ValueError("Manuscript width fractions must not exceed one.")
    sizes = {
        field.name: positive_number(configured_sizes[field.name], field.name)
        for field in fields(Typography) if field.name != "family"
    }
    reference = str(typography["reference_figure"])
    if reference not in fractions:
        raise ValueError("Typography reference must name a configured figure.")
    return PublicationStyle(
        positive_number(manuscript["text_width_tex_pt"], "Manuscript text width"), fractions, reference,
        positive_number(typography["reference_export_width_inches"], "Reference export width"),
        positive_number(typography["scale"], "Typography scale"),
        Typography(str(typography["family"]), **sizes),
        color_mapping(raw["colors"]), color_mapping(raw["sites"]), color_mapping(raw["reference_groups"]),
    )


def figure_typography(kind: FigureKind, export_width_inches: float) -> Typography:
    return publication_style().typography(kind, export_width_inches)


def stable_color(label: str, palette: Mapping[str, str]) -> str:
    """Assign extra groups deterministically without depending on visible groups."""
    colors = tuple(palette.values())
    index = sum((index + 1) * ord(character) for index, character in enumerate(label.casefold()))
    return colors[index % len(colors)]


def recurrence_site_colors(labels: Iterable[str]) -> dict[str, str]:
    """Resolve display labels and configured recurrence endpoint names to shared colors."""
    style = publication_style()
    named = {name.casefold(): color for name, color in style.sites.items()}
    named.update({label: style.colors["overall"] for label in ("any", "all")})
    colors: dict[str, str] = {}
    for label in labels:
        site = label.replace("_", " ").strip().casefold().removesuffix(" recurrence")
        if site == "distant":
            site = "any"
        colors[label] = named.get(site, stable_color(label, style.sites))
    return colors


def reference_group_color(label: str) -> str:
    palette = publication_style().reference_groups
    return palette.get(label, stable_color(label, palette))


def wrap_text(text: str, width_pixels: float, font: FontProperties, renderer: RendererBase) -> str:
    """Wrap at word boundaries using the renderer's actual glyph widths."""
    lines: list[str] = []
    for paragraph in text.splitlines():
        line = ""
        for word in paragraph.split():
            candidate = f"{line} {word}".strip()
            width, _, _ = renderer.get_text_width_height_descent(candidate, font, ismath=False)
            if width > width_pixels and line:
                lines.append(line)
                line = word
            else:
                line = candidate
        if line:
            lines.append(line)
    return "\n".join(lines)
