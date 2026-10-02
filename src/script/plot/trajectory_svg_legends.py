"""Original SVG legend helpers, separated from the omitted preview application."""
from __future__ import annotations

from typing import Literal
import xml.etree.ElementTree as ET

from src.script.plot.ebm_reference_svg import bounds, element, label_width, reference_entries, text
from src.script.style_paper_figures import SVG, css

Variant = Literal["current", "stacked", "row"]

def local_legend(axis: ET.Element) -> ET.Element | None:
    return next((group for group in axis.findall(SVG + "g")
                 if group.get("id", "").startswith("legend_")), None)

def reference_alpha(axis: ET.Element, colour: str) -> float:
    """Read each band opacity from the saved plot instead of assuming an EBM setting."""
    for node in axis.iter(SVG + "path"):
        style = css(node)
        if "clip-path" in node.attrib and style.get("fill", "").strip() == colour:
            return float(style["opacity"])
    raise ValueError(f"No shaded reference band found for {colour}.")

def inset_legend(axis: ET.Element, variant: Variant) -> None:
    legend = local_legend(axis)
    if legend is None:
        return
    entries = reference_entries(legend)
    font_size = float(css(next(legend.iter(SVG + "text")))["font-size"].strip().removesuffix("px"))
    left, top, right, _ = bounds(axis)
    alphas = [reference_alpha(axis, colour) for _, colour in entries]
    axis.remove(legend)
    if variant == "row":
        total_width = sum(label_width(label, font_size) + 2.8 * font_size for label, _ in entries)
        font_size *= min(1, (right - left - 12) / total_width)
    group = element(axis, "g", id=legend.attrib["id"], data_legend_preview=variant)
    x, y = left + 6, top + 6 + font_size
    for (label, colour), alpha in zip(entries, alphas, strict=True):
        # White backing prevents a swatch from becoming darker when it lies over a shaded band.
        element(group, "rect", x=x, y=y - 0.7 * font_size, width=1.35 * font_size,
                height=0.7 * font_size, fill="white")
        element(group, "rect", x=x, y=y - 0.7 * font_size, width=1.35 * font_size,
                height=0.7 * font_size, fill=colour, fill_opacity=alpha)
        text(group, x + 1.8 * font_size, y, label, font_size)
        if variant == "row":
            x += label_width(label, font_size) + 2.8 * font_size
        else:
            y += 1.22 * font_size
