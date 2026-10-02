"""Shared SVG primitives for EBM reference-legend styling and layout previews."""

from __future__ import annotations

import re
from typing import Literal
import xml.etree.ElementTree as ET

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from src.model.normal_range_plotting import NUMERIC_REFERENCE_FONT_SIZE
from src.script.style_paper_figures import SVG, css

Variant = Literal["boxed", "minimal", "direct"]
FONT_SIZE = NUMERIC_REFERENCE_FONT_SIZE
TEXT_COLOUR = "#202830"


def element(parent: ET.Element, tag: str, **attributes: object) -> ET.Element:
    return ET.SubElement(parent, SVG + tag, {key.replace("_", "-"): str(value)
                                           for key, value in attributes.items()})


def text(parent: ET.Element, x: float, y: float, value: str, size: float = FONT_SIZE,
         *, colour: str = TEXT_COLOUR, weight: str = "normal", halo: bool = False) -> None:
    if halo:
        backing = element(parent, "text", x=x, y=y, fill="white", stroke="white", stroke_width=1.5,
                          font_size=size, font_family="DejaVu Sans", font_weight=weight)
        backing.text = value
    node = element(parent, "text", x=x, y=y, fill=colour, font_size=size,
                   font_family="DejaVu Sans", font_weight=weight)
    node.set("style", f"font-size: {size}px; font-family: 'DejaVu Sans'; font-weight: {weight}; fill: {colour}")
    node.text = value


def bounds(axis: ET.Element) -> tuple[float, float, float, float]:
    path = axis.find(f"{SVG}g/{SVG}path")
    values = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", path.attrib["d"])]
    return min(values[::2]), min(values[1::2]), max(values[::2]), max(values[1::2])


def label_width(value: str, size: float = FONT_SIZE) -> float:
    return TextToPath().get_text_width_height_descent(
        value, FontProperties(family="DejaVu Sans", size=size), False,
    )[0]


def wrap_label(value: str, width: float) -> list[str]:
    lines = [""]
    for word in value.split():
        candidate = f"{lines[-1]} {word}".strip()
        if lines[-1] and label_width(candidate) > width:
            lines.append(word)
        else:
            lines[-1] = candidate
    return lines


def reference_entries(legend: ET.Element) -> list[tuple[str, str]]:
    colours = [node.attrib["fill"] if "fill" in node.attrib else css(node)["fill"].strip()
               for node in legend.iter() if node.tag in (SVG + "path", SVG + "rect")]
    labels = [" ".join("".join(node.itertext()) for node in group.iter(SVG + "text"))
              for group in legend if group.find(SVG + "text") is not None]
    return list(zip(labels, colours, strict=True))


def corner_legend(axis: ET.Element, entries: list[tuple[str, str]], variant: Variant) -> None:
    left, top, right, bottom = bounds(axis)
    lines = [(wrap_label(label, right - left - 32), colour) for label, colour in entries]
    width = max(label_width(line) for labels, _ in lines for line in labels) + 22
    height = sum(len(labels) for labels, _ in lines) * 10 + 9
    x, y = left + 5, top + 5
    if variant == "boxed" and len(entries) == 1:
        x, y = right - width - 5, bottom - height - 5
    group = element(axis, "g", id="numeric-reference-intervals")
    if variant == "boxed":
        element(group, "rect", x=x, y=y, width=width, height=height, rx=2,
                fill="white", fill_opacity=0.96, stroke="#b8c2c9", stroke_width=0.55)
    baseline = y + 11
    for index, (labels, colour) in enumerate(lines):
        entry = element(group, "g", id=f"reference-label-{index}")
        element(entry, "rect", x=x + 5, y=baseline - 5.5, width=7, height=5.5, fill=colour)
        for line in labels:
            text(entry, x + 16, baseline, line)
            baseline += 10
