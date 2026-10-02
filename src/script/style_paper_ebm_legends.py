"""Apply borderless upper-left reference legends to the supplied manuscript SVGs."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from src.config import load_yml, resolve_path
from src.figure_headings import CONFIG_PATH
from src.model.normal_range_plotting import NUMERIC_REFERENCE_GID
from src.script.plot.ebm_reference_svg import bounds, corner_legend, reference_entries
from src.script.style_paper_figures import SVG, TEXT_GROUP, position, style_svg
from src.script.style_paper_reference_colors import style_reference_colors

LAYOUT_ATTRIBUTE = "data-reference-layout"
LAYOUT_VALUE = "borderless-upper-left"


def reference_group_spans(source: str) -> list[tuple[int, int]]:
    """Locate complete reference groups without reserializing scientific paths."""
    spans = []
    depth = 0
    start: int | None = None
    for match in re.finditer(r"<g\b[^>]*>|</g>", source):
        if match[0] == "</g>":
            if start is not None:
                depth -= 1
                if depth == 0:
                    spans.append((start, match.end()))
                    start = None
        elif start is not None:
            depth += 1
        elif f'id="{NUMERIC_REFERENCE_GID}"' in match[0]:
            start, depth = match.start(), 1
    if start is not None:
        raise ValueError("Unclosed numeric reference group.")
    return spans


def serialise_group(group: ET.Element) -> str:
    group.tail = None
    for node in group.iter():
        node.tag = node.tag.removeprefix(SVG)
    return ET.tostring(group, encoding="unicode")


def style_reference_legends(source: str, stem: str) -> str:
    """Replace reference labels and close their former heading gap; retain all plot data."""
    root = ET.fromstring(source)
    replacements: list[str] = []
    headings: dict[str, str] = {}
    for axis in root.iter(SVG + "g"):
        if not axis.get("id", "").startswith("axes_"):
            continue
        legend = axis.find(f"{SVG}g[@id='{NUMERIC_REFERENCE_GID}']")
        if legend is None:
            continue
        if legend.get(LAYOUT_ATTRIBUTE) == LAYOUT_VALUE:
            replacements.append("")
            continue
        entries = reference_entries(legend)
        axis.remove(legend)
        corner_legend(axis, entries, "minimal")
        replacement = axis[-1]
        replacement.set(LAYOUT_ATTRIBUTE, LAYOUT_VALUE)
        replacements.append(serialise_group(replacement))
        for heading in axis.findall(SVG + "g"):
            nodes = heading.findall(SVG + "text")
            if not nodes:
                continue
            shift = bounds(axis)[1] - 6 - max(position(node)[1] for node in nodes)
            for node in nodes:
                x, y = position(node)
                node.attrib.pop("transform", None)
                node.set("x", f"{x:.6f}")
                node.set("y", f"{y + shift:.6f}")
            headings[heading.attrib["id"]] = serialise_group(heading)
    spans = reference_group_spans(source)
    for (start, end), replacement in reversed(list(zip(spans, replacements, strict=True))):
        if replacement:
            source = source[:start] + replacement + source[end:]
    source = TEXT_GROUP.sub(lambda match: headings.get(match["id"], match[0]), source)
    return style_reference_colors(re.sub(r"(?m)^[ \t]+(?=\n)", "", style_svg(source, stem)))


def main() -> None:
    config = load_yml(CONFIG_PATH)
    directory = resolve_path(config["manuscript"]["figure_dir"], CONFIG_PATH.parent)
    paths = [directory / "ebm_top_features.svg", *sorted(directory.glob("terms_hazard_ratio_grid*.svg"))]
    updates = {path: style_reference_legends(path.read_text(encoding="utf-8"), path.stem) for path in paths}
    for path, result in updates.items():
        if result != path.read_text(encoding="utf-8"):
            path.write_text(result, encoding="utf-8")
            print(f"Updated reference legends: {path.name}")


if __name__ == "__main__":
    main()
