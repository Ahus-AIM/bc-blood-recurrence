"""Match saved manuscript reference areas and legend swatches to the shared palette."""

from __future__ import annotations

from html import escape
import re
import xml.etree.ElementTree as ET

from src.config import load_yml, resolve_path
from src.figure_headings import CONFIG_PATH
from src.figure_style import publication_style
from src.model.normal_range_plotting import (
    NORMAL_RANGE_SPAN_ALPHA,
    NUMERIC_REFERENCE_GID,
    NUMERIC_REFERENCE_SWATCH_ALPHA,
    PUBLICATION_REFERENCE_ALPHA,
    reference_swatch_color,
)
from src.script.style_paper_figures import SVG, css

SOURCE_COLOR = "data-reference-color"
SHAPE_TAG = re.compile(r"<(?:path|rect)\b[^>]*>", re.DOTALL)


def presentation(node: ET.Element, key: str, default: str = "") -> str:
    return css(node).get(key, node.get(key, default)).strip()


def set_presentation(node: ET.Element, key: str, value: str) -> None:
    """Respect CSS precedence while retaining the original formatting of other properties."""
    style = node.get("style", "")
    pattern = rf"(^|;)(\s*{re.escape(key)}\s*:\s*)[^;]+"
    if re.search(pattern, style):
        node.set("style", re.sub(pattern, lambda match: match[1] + match[2] + value, style))
    else:
        node.set(key, value)


def style_axis_references(axis: ET.Element, palette: set[str], span_alpha: float) -> None:
    solid_bar_colors = set()
    for node in axis.iter():
        if node.tag not in (SVG + "path", SVG + "rect") or "clip-path" not in node.attrib:
            continue
        color = presentation(node, "fill")
        if color in palette:
            set_presentation(node, "opacity", str(span_alpha))
        stroke = presentation(node, "stroke")
        if color == "none" and stroke in palette and float(presentation(node, "stroke-width", "1")) >= 2:
            solid_bar_colors.add(stroke)
    for legend in axis.findall(SVG + "g"):
        identifier = legend.get("id", "")
        if identifier != NUMERIC_REFERENCE_GID and not identifier.startswith("legend_"):
            continue
        for node in legend.iter():
            if node.tag not in (SVG + "path", SVG + "rect"):
                continue
            color = node.get(SOURCE_COLOR, presentation(node, "fill"))
            if color not in palette:
                continue
            alpha = 1.0 if color in solid_bar_colors else span_alpha
            if identifier == NUMERIC_REFERENCE_GID:
                alpha = max(alpha, NUMERIC_REFERENCE_SWATCH_ALPHA)
            node.set(SOURCE_COLOR, color)
            set_presentation(node, "fill", reference_swatch_color(color, alpha))
            for key in ("opacity", "fill-opacity"):
                if presentation(node, key):
                    set_presentation(node, key, "1")


def patch_shape_tag(tag: str, before: dict[str, str], after: dict[str, str]) -> str:
    for key, value in after.items():
        if before.get(key) == value:
            continue
        attribute = f'{key}="{escape(value, quote=True)}"'
        pattern = rf'(?<![\w:-]){re.escape(key)}="[^"]*"'
        if re.search(pattern, tag):
            tag = re.sub(pattern, lambda _match: attribute, tag)
        else:
            tag = re.sub(r"\s*/?>$", lambda match: " " + attribute + match[0], tag)
    return tag


def style_reference_colors(source: str, *, span_alpha: float = PUBLICATION_REFERENCE_ALPHA) -> str:
    """Edit presentation attributes only; preserve all geometry, text and surrounding XML."""
    root = ET.fromstring(source)
    shapes = [node for node in root.iter() if node.tag in (SVG + "path", SVG + "rect")]
    before = [dict(node.attrib) for node in shapes]
    palette = set(publication_style().reference_groups.values())
    for axis in root.iter(SVG + "g"):
        if axis.get("id", "").startswith("axes_"):
            style_axis_references(axis, palette, span_alpha)
    replacements = iter(zip(before, shapes, strict=True))

    def replace_shape(match: re.Match[str]) -> str:
        previous, node = next(replacements)
        return patch_shape_tag(match[0], previous, node.attrib)

    result = SHAPE_TAG.sub(replace_shape, source)
    assert next(replacements, None) is None
    return result


def main() -> None:
    config = load_yml(CONFIG_PATH)
    directory = resolve_path(config["manuscript"]["figure_dir"], CONFIG_PATH.parent)
    paths = [directory / "ebm_top_features.svg", *sorted(directory.glob("terms_hazard_ratio_grid*.svg")),
             *sorted(directory.glob("recurrence_blood_test_trajectories*.svg"))]
    updates = {}
    for path in paths:
        alpha = (NORMAL_RANGE_SPAN_ALPHA if path.stem.startswith("recurrence_blood_test_trajectories")
                 else PUBLICATION_REFERENCE_ALPHA)
        source = path.read_text(encoding="utf-8")
        result = style_reference_colors(source, span_alpha=alpha)
        assert style_reference_colors(result, span_alpha=alpha) == result
        if result != source:
            updates[path] = result
    for path, result in updates.items():
        path.write_text(result, encoding="utf-8")
        print(f"Matched reference shading: {path.name}")


if __name__ == "__main__":
    main()
