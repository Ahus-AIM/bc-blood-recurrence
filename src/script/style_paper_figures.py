"""Restyle supplied vector figure headings without regenerating clinical results."""

from __future__ import annotations

from fnmatch import fnmatchcase
import re
from typing import Literal
import xml.etree.ElementTree as ET

from matplotlib.backends.backend_agg import RendererAgg
from matplotlib.font_manager import FontProperties
from matplotlib.textpath import TextToPath

from src.config import load_yml, resolve_path
from src.figure_headings import CONFIG_PATH, glyph_vertical_metrics, heading_typography, title_gap_points
from src.figure_style import publication_style, wrap_text

SVG = "{http://www.w3.org/2000/svg}"
Role = Literal["figure_title", "test_heading", "site_label", "body"]
TEXT_GROUP = re.compile(
    r'(?P<opening><g\b[^>]*\bid="(?P<id>[^"]+)"[^>]*>)'
    r'\s*(?:<text\b[^>]*>.*?</text>\s*)+</g>', re.DOTALL,
)


def text_nodes(group: ET.Element) -> list[ET.Element]:
    return group.findall(f"{SVG}text")


def label(group: ET.Element) -> str:
    return " ".join(" ".join(node.itertext()).strip() for node in text_nodes(group))


def css(node: ET.Element) -> dict[str, str]:
    return dict(part.strip().split(":", 1) for part in node.get("style", "").split(";") if ":" in part)


def position(node: ET.Element) -> tuple[float, float]:
    """Read the unrotated text coordinates used for Matplotlib headings."""
    transform = node.get("transform", "")
    translated = re.fullmatch(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\)", transform)
    if translated:
        return float(translated[1]), float(translated[2])
    if transform and not re.fullmatch(r"rotate\(-?0(?:[ ,]+[-\d.]+){2}\)", transform):
        raise ValueError(f"Unsupported heading transform: {transform}")
    return float(node.get("x", "0")), float(node.get("y", "0"))


def group_center(group: ET.Element) -> float:
    bounds = []
    for node in text_nodes(group):
        style = css(node)
        font = FontProperties(family="DejaVu Sans", size=float(style["font-size"].strip().removesuffix("px")),
                              weight=style.get("font-weight", "normal").strip())
        width, _, _ = TextToPath().get_text_width_height_descent(node.text, font, ismath=False)
        anchor = style.get("text-anchor", "start").strip()
        x, _ = position(node)
        left = x - width * {"start": 0, "middle": 0.5, "end": 1}[anchor]
        bounds.append((left, left + width))
    return (min(left for left, _ in bounds) + max(right for _, right in bounds)) / 2


def axes_width(axis: ET.Element) -> float:
    path = axis.find(f"{SVG}g/{SVG}path")
    coordinates = [float(value) for value in re.findall(r"[-+]?\d*\.?\d+", path.attrib["d"])]
    xs = coordinates[::2]
    return max(xs) - min(xs)


def heading_groups(root: ET.Element, stem: str) -> dict[str, tuple[ET.Element, Role, float]]:
    """Identify semantic heading groups, excluding legends, ticks and statistical values."""
    width = float(root.attrib["viewBox"].split()[2])
    figure = root.find(f"{SVG}g")
    groups: dict[str, tuple[ET.Element, Role, float]] = {}
    trajectory_groups = []
    for group in root.iter(f"{SVG}g"):
        if not text_nodes(group) or len(text_nodes(group)) != len(group):
            continue
        identifier = group.attrib["id"]
        text = label(group)
        if text.startswith(("Top EBM feature effects", "EBM feature importance", "EBM feature effects")) or (
            re.search(r"\bBlood[- ]test trajectories\b", text)
        ):
            groups[identifier] = (group, "figure_title", width * 0.96)
        elif identifier.startswith("ebm-panel-title-"):
            groups[identifier] = (group, "test_heading", 0)
        elif stem.startswith("global_importance"):
            groups[identifier] = (group, "body", 0)
        elif stem == "recurrence_blood_test_trajectories" and re.fullmatch(r".+ \((?:N = )?[\d,]+ patients?\)", text):
            groups[identifier] = (group, "site_label", 0)
        elif stem.startswith("recurrence_blood_test_trajectories") and re.match(r"\d+\.\s", text):
            trajectory_groups.append(group)
    for axis in figure.findall(f"{SVG}g"):
        if not axis.get("id", "").startswith("axes_"):
            continue
        for group in axis.findall(f"{SVG}g"):
            identifier = group.get("id", "")
            if identifier.startswith("ebm-panel-title-") or (
                stem.startswith("terms_hazard_ratio_grid") and text_nodes(group)
            ):
                groups[identifier] = (group, "test_heading", axes_width(axis))
    if trajectory_groups:
        xs = sorted({position(text_nodes(group)[0])[0] for group in trajectory_groups})
        spacing = min((b - a for a, b in zip(xs, xs[1:])), default=width)
        for group in trajectory_groups:
            groups[group.attrib["id"]] = (group, "test_heading", spacing * 0.76)
    if sum(role == "figure_title" for _, role, _ in groups.values()) != 1:
        raise ValueError(f"Expected exactly one overall title in {stem}.")
    return groups


def styled_heading(group: ET.Element, role: Role, size: float, width: float, figure_width: float,
                   *, left_aligned: bool, number: int | None = None) -> str:
    font = FontProperties(family=publication_style().reference_sizes.family, size=size, weight="bold")
    text = label(group).replace("Blood-test", "Blood test")
    if number is not None:
        text = f"{number}. {re.sub(r'^\d+\.\s+', '', text)}"
    if role == "figure_title" and "Blood test trajectories" in text:
        text = text.replace(" · N = ", " · ")
    lines = wrap_text(text, width, font, RendererAgg(1, 1, 72)).splitlines()
    bottom = max(position(node)[1] for node in text_nodes(group))
    if role == "figure_title":
        x = figure_width / 2
        ascent = max(TextToPath().get_text_width_height_descent(line, font, False)[1] for line in lines)
        bottom = max(bottom, ascent + 0.72 + (len(lines) - 1) * size * 1.12)
    else:
        x = position(text_nodes(group)[0])[0] if left_aligned else group_center(group)
    anchor = "start" if left_aligned and role != "figure_title" else "middle"
    style = (f"font-weight: 700; font-size: {size:.6f}px; font-family: '{font.get_family()[0]}'; "
             f"text-anchor: {anchor}; fill: {publication_style().colors['text']}")
    rendered = []
    for index, line in enumerate(lines):
        y = bottom - (len(lines) - 1 - index) * size * 1.12
        node = ET.Element("text", {"style": style, "x": f"{x:.6f}", "y": f"{y:.6f}"})
        node.text = line
        rendered.append(ET.tostring(node, encoding="unicode"))
    return "\n    ".join(rendered)


def styled_body(group: ET.Element, size: float) -> str:
    rendered = []
    for original in text_nodes(group):
        text = original.text
        factor = 8 / 9 if text.isdigit() else 8.5 / 9 if text.startswith("Score:") else 1
        style = css(original)
        style["font-size"] = f"{size * factor:.6f}px"
        node = ET.Element("text", dict(original.attrib))
        node.set("style", "; ".join(f"{key}: {value.strip()}" for key, value in style.items()))
        node.text = text
        rendered.append(ET.tostring(node, encoding="unicode"))
    return "\n    ".join(rendered)


def styled_site_label(group: ET.Element, size: float) -> str:
    """Style a centred site label with a plain patient count, preserving its colour."""
    original, = text_nodes(group)
    text = original.text.replace("(N = ", "(")
    transform = re.fullmatch(r"translate\(([-\d.]+)[ ,]+([-\d.]+)\) rotate\(-90\)", original.get("transform", ""))
    if transform is None:
        raise ValueError(f"Unsupported site-label transform: {original.get('transform')}")
    x, y = float(transform[1]), float(transform[2])
    style = css(original)
    font = FontProperties(family=publication_style().reference_sizes.family,
                          size=float(style["font-size"].strip().removesuffix("px")), weight="bold")
    width, _, _ = TextToPath().get_text_width_height_descent(original.text, font, False)
    anchor = {"start": 0, "middle": 0.5, "end": 1}[style.get("text-anchor", "start").strip()]
    y += width * (anchor - 0.5)
    ascent, descent = glyph_vertical_metrics(original.text, font)
    font.set_size(size)
    new_ascent, new_descent = glyph_vertical_metrics(text, font)
    x += (new_ascent - new_descent - ascent + descent) / 2
    style["font-size"] = f"{size:.6f}px"
    style["text-anchor"] = "middle"
    node = ET.Element("text", dict(original.attrib))
    node.set("style", "; ".join(f"{key}: {value.strip()}" for key, value in style.items()))
    node.set("transform", f"translate({x:.6f} {y:.6f}) rotate(-90)")
    node.text = text
    return ET.tostring(node, encoding="unicode")


def text_vertical_bounds(node: ET.Element) -> tuple[float, float]:
    """Measure an upright SVG text node's visible top and bottom in SVG points."""
    style = css(node)
    font = FontProperties(family=publication_style().reference_sizes.family,
                          size=float(style["font-size"].strip().removesuffix("px")),
                          weight=style.get("font-weight", "normal").strip())
    ascent, descent = glyph_vertical_metrics("".join(node.itertext()), font)
    _, baseline = position(node)
    return baseline - ascent, baseline + descent


def centre_trajectory_site_labels(source: str, stem: str) -> str:
    """Centre vertical site labels over each complete heading/plot/table row."""
    if stem != "recurrence_blood_test_trajectories":
        return source
    root = ET.fromstring(source)
    groups = heading_groups(root, stem)
    sites = [group for group, role, _ in groups.values() if role == "site_label"]
    first_headings = [group for group, role, _ in groups.values()
                      if role == "test_heading" and label(group).startswith("1. ")]
    tops = sorted(min(text_vertical_bounds(node)[0] for node in text_nodes(group)) for group in first_headings)
    excluded = {node for group, _, _ in groups.values() for node in text_nodes(group)}
    body_bounds = []
    for node in root.iter(f"{SVG}text"):
        rotation = re.search(r"rotate\(([-\d.]+)", node.get("transform", ""))
        if node in excluded or (rotation and float(rotation[1]) != 0):
            continue
        body_bounds.append(text_vertical_bounds(node))
    replacements = {}
    for group, top, next_top in zip(sites, tops, [*tops[1:], float("inf")], strict=True):
        bottom = max(lower for upper, lower in body_bounds if top <= upper < next_top)
        original, = text_nodes(group)
        node = ET.Element("text", dict(original.attrib))
        node.text = original.text
        transform = re.sub(r"(translate\([-\d.]+[ ,]+)[-\d.]+", rf"\g<1>{(top + bottom) / 2:.6f}",
                           original.attrib["transform"])
        node.set("transform", transform)
        replacements[group.attrib["id"]] = ET.tostring(node, encoding="unicode")
    return TEXT_GROUP.sub(lambda match: match["opening"] + "\n    " + replacements[match["id"]] + "\n   </g>"
                          if match["id"] in replacements else match[0], source)


def space_figure_title(source: str, stem: str) -> str:
    """Move only the title to give every manuscript figure the same visible gap."""
    root = ET.fromstring(source)
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    title = next(group for group, role, _ in heading_groups(root, stem).values() if role == "figure_title")
    nodes = text_nodes(title)
    # Rotated tick/axis/site labels are below or beside the panels' first content row.
    content_tops = []
    for node in root.iter(f"{SVG}text"):
        rotation = re.search(r"rotate\(([-\d.]+)", node.get("transform", ""))
        if node in nodes or (rotation and float(rotation[1]) != 0):
            continue
        content_tops.append(text_vertical_bounds(node)[0])
    content_top = min(content_tops)
    bottom = max(text_vertical_bounds(node)[1] for node in nodes)
    shift = content_top - title_gap_points(stem, width, height) - bottom
    if min(text_vertical_bounds(node)[0] for node in nodes) + shift < 0:
        raise ValueError(f"Insufficient room above the content for the shared title gap in {stem}.")
    rendered = []
    for original in nodes:
        node = ET.Element("text", dict(original.attrib))
        node.set("y", f"{position(original)[1] + shift:.6f}")
        node.text = original.text
        rendered.append(ET.tostring(node, encoding="unicode"))
    content = "\n    ".join(rendered)
    return TEXT_GROUP.sub(lambda match: match["opening"] + "\n    " + content + "\n   </g>"
                          if match["id"] == title.attrib["id"] else match[0], source)


def style_svg(source: str, stem: str) -> str:
    """Rewrite only text groups; all paths, clinical values and canvas dimensions survive."""
    root = ET.fromstring(source)
    _, _, width, height = map(float, root.attrib["viewBox"].split())
    fonts = heading_typography(stem, width / 72, height / 72)
    groups = heading_groups(root, stem)
    numbered_panels = [identifier for identifier, (_, role, _) in groups.items()
                       if stem.startswith("terms_hazard_ratio_grid") and role == "test_heading"]
    panel_numbers = {identifier: number for number, identifier in enumerate(numbered_panels, start=1)}
    applied = set()

    def replace_group(match: re.Match[str]) -> str:
        identifier = match["id"]
        if identifier not in groups:
            return match[0]
        group, role, available = groups[identifier]
        size = fonts.figure_title if role == "figure_title" else fonts.site_label
        if role == "body":
            content = styled_body(group, size)
        elif role == "site_label":
            content = styled_site_label(group, size)
        else:
            content = styled_heading(
                group, role, size, available, width, left_aligned=stem.startswith("recurrence_blood_test_trajectories"),
                number=panel_numbers.get(identifier),
            )
        applied.add(identifier)
        return match["opening"] + "\n    " + content + "\n   </g>"

    styled = TEXT_GROUP.sub(replace_group, source)
    if applied != set(groups):
        raise ValueError(f"Could not safely rewrite heading groups: {set(groups) - applied}")
    return centre_trajectory_site_labels(space_figure_title(styled, stem), stem)


def main() -> None:
    config = load_yml(CONFIG_PATH)
    directory = resolve_path(config["manuscript"]["figure_dir"], CONFIG_PATH.parent)
    updates = {}
    for path in sorted(directory.glob("*.svg")):
        if any(fnmatchcase(path.stem, pattern) for pattern in config["placements"]):
            source = path.read_text(encoding="utf-8")
            styled = style_svg(source, path.stem)
            if source != styled:
                updates[path] = styled
    for path, styled in updates.items():
        path.write_text(styled, encoding="utf-8")
        print(f"Styled headings: {path.name}")


if __name__ == "__main__":
    main()
