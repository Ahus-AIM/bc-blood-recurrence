"""Update saved trajectory legends without loading or recomputing clinical data.

Run with ``uv run python -m src.script.style_paper_trajectory_legends``.
"""

from __future__ import annotations

from copy import deepcopy
import re
import xml.etree.ElementTree as ET

from src.config import load_yml, resolve_path
from src.figure_headings import CONFIG_PATH
from src.model.normal_range_plotting import NORMAL_RANGE_SPAN_ALPHA, reference_swatch_color
from src.script.plot.ebm_reference_svg import bounds, element, label_width, reference_entries, text
from src.script.plot.trajectory_svg_legends import inset_legend, local_legend
from src.script.plot.recurrence_reference_legend import HEADROOM_POINTS, LEGACY_LAYOUT_GID, LAYOUT_GID, LEGEND_PADDING_POINTS
from src.script.style_paper_figures import SVG, css, text_vertical_bounds
from src.script.style_paper_reference_colors import SOURCE_COLOR, style_reference_colors


def extend_spine(axis: ET.Element, x: float, top: float, bottom: float, new_top: float) -> None:
    """Extend only the vertical spine; data paths and their clipping remain untouched."""
    for group in axis:
        if not group.get("id", "").startswith("patch_"):
            continue
        path = group.find(SVG + "path")
        if path is None or "clip-path" in path.attrib:
            continue
        values = [float(value) for value in re.findall(r"-?\d+(?:\.\d+)?", path.get("d", ""))]
        if len(values) == 4 and all(abs(a - b) < 1e-5 for a, b in zip(values, (x, bottom, x, top), strict=True)):
            path.set("d", f"M {x} {bottom} L {x} {new_top}")


def continue_ticks(axis: ET.Element, new_top: float) -> None:
    """Extrapolate the existing linear tick spacing, keeping all original tick positions."""
    group = next(child for child in axis if any(node.get("id", "").startswith("ytick_") for node in child))
    for child in list(group):
        if child.get("id", "").startswith(f"headroom-{axis.attrib['id']}-"):
            group.remove(child)
    ticks = []
    for tick in group:
        if not tick.get("id", "").startswith("ytick_"):
            continue
        label = next(tick.iter(SVG + "text"))
        marker = next(node for node in tick.iter(SVG + "use") if "y" in node.attrib)
        ticks.append((float(label.text.replace("−", "-")), float(marker.attrib["y"]), tick))
    ticks.sort()
    if len(ticks) < 2:
        raise ValueError("At least two saved y ticks are needed to continue the scale.")
    value, y, template = ticks[-1]
    value_step = value - ticks[-2][0]
    pixel_step = ticks[-2][1] - y
    index = 1
    while y - index * pixel_step >= new_top + 2:
        tick = deepcopy(template)
        tick.set("transform", f"translate(0 {-index * pixel_step})")
        for node in tick.iter():
            if "id" in node.attrib:
                node.set("id", f"headroom-{axis.attrib['id']}-{index}-{node.attrib['id']}")
            node.attrib.pop("clip-path", None)
            if node.tag == SVG + "text":
                node.text = f"{value + index * value_step:g}".replace("-", "−")
        group.append(tick)
        index += 1


def legend_backing(legend: ET.Element) -> None:
    """Keep grid lines clear of text; this backing occupies only the new empty headroom."""
    for label in list(legend.iter(SVG + "text")):
        size = float(css(label)["font-size"].removesuffix("px"))
        top, bottom = text_vertical_bounds(label)
        backing = element(legend, "rect", x=float(label.attrib["x"]) - 1, y=top - 1,
                          width=label_width(label.text, size) + 2, height=bottom - top + 2, fill="white")
        legend.remove(backing)
        legend.insert(list(legend).index(label), backing)


def continuation_arrow(axis: ET.Element, colour: str, x: float, top: float) -> None:
    """Mark an upper-unbounded interval without shading over the inset legend."""
    group = element(axis, "g", id=f"reference-continuation-{axis.attrib['id']}")
    end = top - HEADROOM_POINTS * 0.72
    element(group, "path", d=f"M {x} {top} L {x} {end} M {x - 2.5} {end + 4} L {x} {end} L {x + 2.5} {end + 4}",
            fill="none", stroke=colour, stroke_width=0.8)


def update_panel(root: ET.Element, axis: ET.Element) -> None:
    legend = local_legend(axis)
    entries = reference_entries(legend)
    left, top, right, bottom = bounds(axis)
    new_top = top - HEADROOM_POINTS
    index = int(axis.attrib["id"].removeprefix("axes_"))
    twin = root.find(f".//{SVG}g[@id='axes_{index + 1}']")
    inset_legend(axis, "row")
    legend = local_legend(axis)
    legend.set("transform", f"translate(0 {-HEADROOM_POINTS})")
    legend.set("data-reference-layout", LEGACY_LAYOUT_GID)
    legend_backing(legend)
    # Extend the white axes background and both ordinary spines, with no header divider.
    background = axis.find(f"{SVG}g/{SVG}path")
    background.set("d", f"M {left} {bottom} L {right} {bottom} L {right} {new_top} L {left} {new_top} z")
    extend_spine(axis, left, top, bottom, new_top)
    extend_spine(twin, right, top, bottom, new_top)
    continue_ticks(axis, new_top)
    for label, colour in entries:
        if label.startswith(("≥", ">")):
            continuation_arrow(axis, colour, right - 0.02 * (right - left), top)
    axis.remove(legend)
    axis.append(legend)
    for node in legend.iter(SVG + "text"):
        text_top, text_bottom = text_vertical_bounds(node)
        assert new_top < text_top - HEADROOM_POINTS < text_bottom - HEADROOM_POINTS < top - 6


def uniform_legend(axis: ET.Element, font_size: float) -> tuple[ET.Element, float]:
    """Pack entries into rows at the figure's common font and swatch size."""
    original = local_legend(axis)
    labels = [node.text for node in original.iter(SVG + "text")]
    colors = [node.attrib[SOURCE_COLOR] for node in original.iter() if SOURCE_COLOR in node.attrib]
    if not colors:
        colors = [node.attrib["fill"] for node in original.iter(SVG + "rect") if node.get("fill") != "white"]
    left, top, right, _ = bounds(axis)
    rows: list[list[tuple[str, str]]] = [[]]
    used = 0.0
    for label, color in zip(labels, colors, strict=True):
        width = label_width(label, font_size) + 2.8 * font_size
        if rows[-1] and used + width > right - left - 2 * LEGEND_PADDING_POINTS:
            rows.append([])
            used = 0.0
        rows[-1].append((label, color))
        used += width
    headroom = max(HEADROOM_POINTS, 2 * LEGEND_PADDING_POINTS + 1.25 * font_size * len(rows))
    new_top = top + HEADROOM_POINTS - headroom
    axis.remove(original)
    legend = element(axis, "g", id=original.attrib["id"], data_reference_layout=LAYOUT_GID,
                     data_reference_headroom=headroom)
    baseline = new_top + LEGEND_PADDING_POINTS + font_size
    for row in rows:
        x = left + LEGEND_PADDING_POINTS
        for label, color in row:
            element(legend, "rect", x=x, y=baseline - 0.7 * font_size, width=1.35 * font_size,
                    height=0.7 * font_size, fill=reference_swatch_color(color, NORMAL_RANGE_SPAN_ALPHA),
                    data_reference_color=color)
            text(legend, x + 1.8 * font_size, baseline, label, font_size)
            x += label_width(label, font_size) + 2.8 * font_size
        baseline += 1.25 * font_size
    legend_backing(legend)
    assert [node.text for node in legend.iter(SVG + "text")] == labels
    for node in legend.iter(SVG + "text"):
        text_top, text_bottom = text_vertical_bounds(node)
        assert new_top < text_top < text_bottom < top + HEADROOM_POINTS - LEGEND_PADDING_POINTS
        assert float(node.attrib["x"]) + label_width(node.text, font_size) < right - LEGEND_PADDING_POINTS
    return legend, new_top


def apply_uniform_size(root: ET.Element, axis: ET.Element, font_size: float) -> None:
    left, top, right, bottom = bounds(axis)
    _, new_top = uniform_legend(axis, font_size)
    background = axis.find(f"{SVG}g/{SVG}path")
    background.set("d", f"M {left} {bottom} L {right} {bottom} L {right} {new_top} L {left} {new_top} z")
    index = int(axis.attrib["id"].removeprefix("axes_"))
    twin = root.find(f".//{SVG}g[@id='axes_{index + 1}']")
    extend_spine(axis, left, top, bottom, new_top)
    extend_spine(twin, right, top, bottom, new_top)
    continue_ticks(axis, new_top)


def clinical_content(root: ET.Element) -> list[tuple[str, tuple[tuple[str, str], ...], str | None]]:
    """Snapshot every clipped data shape and every numerical table cell."""
    result = []
    for axis in root.iter(SVG + "g"):
        identifier = axis.get("id", "")
        if not identifier.startswith("axes_"):
            continue
        table = int(identifier.removeprefix("axes_")) % 3 == 0
        for node in axis.iter():
            if "clip-path" in node.attrib or table and node.tag == SVG + "text":
                result.append((node.tag, tuple(sorted(node.attrib.items())), node.text))
    return result


def style_trajectory_legends(source: str) -> str:
    root = ET.fromstring(source)
    before = clinical_content(root)
    axes = [axis for axis in root.iter(SVG + "g") if axis.get("id", "").startswith("axes_")
            and local_legend(axis) is not None]
    if not axes:
        return style_reference_colors(source, span_alpha=NORMAL_RANGE_SPAN_ALPHA)
    font_size = max(float(css(node)["font-size"].strip().removesuffix("px"))
                    for axis in axes for node in local_legend(axis).iter(SVG + "text"))
    changed = False
    for axis in axes:
        legend = local_legend(axis)
        layout = legend.get("data-reference-layout")
        if layout != LAYOUT_GID:
            if layout != LEGACY_LAYOUT_GID:
                update_panel(root, axis)
            apply_uniform_size(root, axis, font_size)
            changed = True
    if not changed:
        return style_reference_colors(source, span_alpha=NORMAL_RANGE_SPAN_ALPHA)
    assert clinical_content(root) == before
    ET.register_namespace("", SVG[1:-1])
    ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")
    return style_reference_colors(ET.tostring(root, encoding="unicode"), span_alpha=NORMAL_RANGE_SPAN_ALPHA)


def main() -> None:
    config = load_yml(CONFIG_PATH)
    directory = resolve_path(config["manuscript"]["figure_dir"], CONFIG_PATH.parent)
    updates = {path: style_trajectory_legends(path.read_text(encoding="utf-8"))
               for path in sorted(directory.glob("recurrence_blood_test_trajectories*.svg"))}
    for path, result in updates.items():
        assert style_trajectory_legends(result) == result
        if result != path.read_text(encoding="utf-8"):
            path.write_text(result, encoding="utf-8")
            print(f"Updated trajectory legend and y-axis: {path.name}")


if __name__ == "__main__":
    main()
