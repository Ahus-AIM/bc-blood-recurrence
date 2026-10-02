"""Tighten supplied EBM grids by translating complete rows, preserving their data."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import re
import xml.etree.ElementTree as ET

from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath
from matplotlib.transforms import Affine2D

from src.config import load_yml, resolve_path
from src.figure_headings import CONFIG_PATH, manuscript_scale
from src.script.style_paper_figures import SVG, css, style_svg

ORIGINAL_HEIGHT = "data-row-layout-original-height"
ROW_SHIFT = "data-row-layout-shift"
GROUP_OPENING = re.compile(r'<g\b[^>]*\bid="(?:axes_\d+|site-label)"[^>]*(?<!/)>')


@dataclass
class Row:
    """One subplot row: its SVG axes groups and visible upper/lower content edges."""

    axes: list[ET.Element]
    top: float
    bottom: float


def text_y_bounds(node: ET.Element) -> tuple[float, float]:
    """Include rotated ticks and axis labels when measuring a row's occupied space."""
    style = css(node)
    font = FontProperties(family="DejaVu Sans", size=float(style["font-size"].strip().removesuffix("px")),
                          weight=style.get("font-weight", "normal").strip())
    width, height, descent = TextToPath().get_text_width_height_descent("".join(node.itertext()), font, False)
    anchor = {"start": 0, "middle": 0.5, "end": 1}[style.get("text-anchor", "start").strip()]
    x, y = float(node.get("x", "0")) - width * anchor, float(node.get("y", "0"))
    transform = Affine2D()
    for operation, arguments in reversed(re.findall(r"(\w+)\(([^)]+)\)", node.get("transform", ""))):
        values = [float(value) for value in re.split(r"[ ,]+", arguments)]
        if operation == "translate":
            transform.translate(*values)
        elif operation == "rotate" and len(values) == 3:
            transform.rotate_deg_around(values[1], values[2], values[0])
        elif operation == "rotate":
            transform.rotate_deg(values[0])
        else:
            raise ValueError(f"Unsupported text transform: {operation}")
    ys = transform.transform([(x, y - height + descent), (x + width, y - height + descent),
                              (x, y + descent), (x + width, y + descent)])[:, 1]
    return float(min(ys)), float(max(ys))


def figure_rows(root: ET.Element) -> list[Row]:
    """Group axes by their plot tops, retaining all labels when measuring row edges."""
    rows: dict[float, Row] = {}
    for axis in root.find(f"{SVG}g").findall(f"{SVG}g"):
        if not axis.get("id", "").startswith("axes_"):
            continue
        background = axis.find(f"{SVG}g/{SVG}path")
        if background is None:
            continue
        coordinates = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", background.attrib["d"])]
        top, bottom = min(coordinates[1::2]), max(coordinates[1::2])
        row = rows.setdefault(round(top, 3), Row([], top, bottom))
        row.axes.append(axis)
        for node in axis.iter(f"{SVG}text"):
            text_top, text_bottom = text_y_bounds(node)
            row.top, row.bottom = min(row.top, text_top), max(row.bottom, text_bottom)
    return [row for _, row in sorted(rows.items())]


def row_offsets(rows: Sequence[Row], scale: float) -> list[float]:
    """Close a fraction of each gap, leaving already tight rows in place."""
    config = load_yml(CONFIG_PATH)["ebm_rows"]
    minimum = config["minimum_gap_pdf_pt"] / scale
    offsets = [0.0]
    for previous, current in zip(rows, rows[1:]):
        gap = current.top - previous.bottom
        reduction = max(0.0, min(gap * (1 - config["gap_fraction"]), gap - minimum))
        offsets.append(offsets[-1] - reduction)
    return offsets


def opening_tag(node: ET.Element) -> str:
    """Serialize only an opening tag so paths, metadata and text remain byte-for-byte intact."""
    return ET.tostring(ET.Element(node.tag.removeprefix(SVG), node.attrib), encoding="unicode").removesuffix(" />") + ">"


def restore_row_layout(source: str) -> str:
    """Undo only our previous translations and crop before recomputing the configured gaps."""
    opening = re.search(r"<svg\b[^>]*>", source)
    root = ET.fromstring(source)
    if ORIGINAL_HEIGHT not in root.attrib:
        return source
    height = root.attrib.pop(ORIGINAL_HEIGHT)
    viewbox = root.attrib["viewBox"].split()
    root.set("height", height + "pt")
    root.set("viewBox", " ".join([*viewbox[:3], height]))
    # Retain namespace declarations from the original opening rather than reserializing the document.
    replacement = re.sub(r' height="[^"]+"', f' height="{height}pt"', opening[0])
    replacement = re.sub(r' viewBox="[^"]+"', f' viewBox="{root.attrib["viewBox"]}"', replacement)
    replacement = re.sub(rf' {ORIGINAL_HEIGHT}="[^"]+"', "", replacement)
    source = source[:opening.start()] + replacement + source[opening.end():]

    def restore_group(match: re.Match[str]) -> str:
        group = ET.fromstring(match[0] + "</g>")
        if ROW_SHIFT not in group.attrib:
            return match[0]
        shift = group.attrib.pop(ROW_SHIFT)
        if group.attrib.pop("transform") != f"translate(0 {shift})":
            raise ValueError("The stored row translation was changed externally.")
        return opening_tag(group)

    return GROUP_OPENING.sub(restore_group, source)


def compact_svg(source: str, stem: str) -> str:
    """Translate rows together with their clipping paths and crop the freed bottom space."""
    source = style_svg(restore_row_layout(source), stem)
    root = ET.fromstring(source)
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    rows = figure_rows(root)
    if not rows:
        raise ValueError(f"No EBM rows found in {stem}.")
    offsets = row_offsets(rows, manuscript_scale(stem, width, height))
    shifts = {axis.attrib["id"]: offset for row, offset in zip(rows, offsets, strict=True) for axis in row.axes}
    site_offsets = iter(offsets)

    def shift_group(match: re.Match[str]) -> str:
        group = ET.fromstring(match[0] + "</g>")
        identifier = group.attrib["id"]
        if identifier == "site-label":
            offset = next(site_offsets)
        elif identifier in shifts:
            offset = shifts[identifier]
        else:
            return match[0]
        if not offset:
            return match[0]
        if "transform" in group.attrib:
            raise ValueError(f"Unexpected transform on row group {identifier}.")
        shift = f"{offset:.6f}"
        group.set("transform", f"translate(0 {shift})")
        group.set(ROW_SHIFT, shift)
        return opening_tag(group)

    source = GROUP_OPENING.sub(shift_group, source)
    opening = re.search(r"<svg\b[^>]*>", source)
    new_height = f"{height + offsets[-1]:.6f}"
    replacement = re.sub(r' height="[^"]+"', f' height="{new_height}pt"', opening[0])
    original_width = root.attrib["viewBox"].split()[2]
    replacement = re.sub(r' viewBox="[^"]+"', f' viewBox="0 0 {original_width} {new_height}"', replacement)
    replacement = replacement[:-1] + f' {ORIGINAL_HEIGHT}="{height:.6f}">'
    compacted = source[:opening.start()] + replacement + source[opening.end():]
    return style_svg(compacted, stem)


def main() -> None:
    config = load_yml(CONFIG_PATH)
    directory = resolve_path(config["manuscript"]["figure_dir"], CONFIG_PATH.parent)
    paths = [directory / "ebm_top_features.svg", *sorted(directory.glob("terms_hazard_ratio_grid*.svg"))]
    updates = {path: compact_svg(path.read_text(encoding="utf-8"), path.stem) for path in paths}
    for path, result in updates.items():
        if result != path.read_text(encoding="utf-8"):
            path.write_text(result, encoding="utf-8")
            print(f"Compacted rows: {path.name}")


if __name__ == "__main__":
    main()
