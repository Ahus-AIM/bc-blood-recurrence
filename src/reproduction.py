"""Strict artifact comparisons that retain plotted values, geometry and table formatting."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "{http://www.w3.org/2000/svg}"
XLINK = "{http://www.w3.org/1999/xlink}"


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def normalized_svg(content: str) -> bytes:
    """Ignore metadata, XML indentation and generated IDs; retain all rendering attributes and text."""
    root = ET.fromstring(content)
    for parent in root.iter():
        for child in list(parent):
            if child.tag == SVG + "metadata":
                parent.remove(child)
            elif parent.tag not in (SVG + "text", SVG + "tspan") and child.tail and not child.tail.strip():
                child.tail = None
    identifiers = {node.attrib["id"]: f"element_{index}" for index, node in enumerate(root.iter()) if "id" in node.attrib}
    for node in root.iter():
        for key, value in list(node.attrib.items()):
            if key == "id":
                value = identifiers[value]
            elif key in ("href", XLINK + "href") and value.startswith("#"):
                value = "#" + identifiers.get(value[1:], value[1:])
            else:
                value = re.sub(r"url\(#([^)]*)\)", lambda match: f"url(#{identifiers.get(match[1], match[1])})", value)
            node.set(key, value)
        # Indentation outside text elements does not affect rendering.
        if node.tag not in (SVG + "text", SVG + "tspan", SVG + "style") and node.text and not node.text.strip():
            node.text = None
    return ET.canonicalize(ET.tostring(root, encoding="unicode")).encode()


def normalized_tabular(content: str) -> bytes:
    """Compare the complete tabular body, including labels and significance markup, ignoring whitespace."""
    tables = re.findall(r"\\begin\{tabular\}.*?\\end\{tabular\}", content, flags=re.DOTALL)
    if len(tables) != 1:
        raise ValueError(f"Expected one tabular environment, found {len(tables)}.")
    return re.sub(r"\s+", "", tables[0]).encode()


def artifact_hash(path: Path) -> str:
    if path.suffix == ".svg":
        return sha256(normalized_svg(path.read_text()))
    if path.suffix == ".tex":
        return sha256(normalized_tabular(path.read_text()))
    return sha256(path.read_bytes())


def compare_files(reference: Path, actual: Path) -> dict[str, object]:
    if not actual.is_file():
        return {"status": "missing_output"}
    expected_hash, actual_hash = artifact_hash(reference), artifact_hash(actual)
    return {
        "status": "match" if expected_hash == actual_hash else "different",
        "expected_sha256": expected_hash,
        "actual_sha256": actual_hash,
        "bytes_equal": reference.read_bytes() == actual.read_bytes(),
    }
