"""Check the published SVG numbers without requiring patient-level source data.

These checks establish internal consistency, not correctness of clinical inputs.
P-value intervals retain display-rounding uncertainty instead of treating the
printed values as exact inputs to the multiple-testing adjustment.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, nan
from pathlib import Path
import re
from xml.etree import ElementTree as ET

import pytest

from test.recurrence_statistics_oracles import bh_reference

FIGURES = Path(__file__).resolve().parents[1] / "reference/paper/figure"
SVG = "{http://www.w3.org/2000/svg}"
SITES = ("any", "bone", "brain", "liver", "lung")
SUPERSCRIPTS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


@dataclass(frozen=True)
class PrintedPanel:
    """A site's named test and its eight rows of five displayed window cells."""

    site: str
    test: str
    rows: tuple[tuple[str, ...], ...]


def text_cells(group: ET.Element) -> list[str]:
    cells = []
    for node in group:
        if node.find(SVG + "text") is not None:
            text = " ".join("".join(t.itertext()) for t in node.iter(SVG + "text"))
            cells.append(" ".join(text.split()))
    return cells


def printed_panels(path: Path, *, site: str | None = None) -> list[PrintedPanel]:
    root = ET.parse(path)
    figure = next(node for node in root.iter(SVG + "g") if node.get("id") == "figure_1")
    identities = []
    for text in text_cells(figure):
        if match := re.match(r"^(Any|Bone|Brain|Liver|Lung) \((?:N = )?\d+ patients?\)", text):
            site = match[1].lower()
        if match := re.match(r"^\d+\. (.+)$", text):
            assert site is not None
            identities.append((site, match[1]))
    tables = []
    for node in figure:
        if not node.get("id", "").startswith("axes_"):
            continue
        cells = text_cells(node)
        if not cells or cells[0] != "Median":
            continue
        assert len(cells) == 48
        assert cells[24] == "p value" and cells[30] == "Adjusted p"
        tables.append(tuple(tuple(cells[i + 1:i + 6]) for i in range(0, 48, 6)))
    return [PrintedPanel(site, test, rows) for (site, test), rows in zip(identities, tables, strict=True)]


def rounding_interval(text: str) -> tuple[float, float]:
    if text in ("—", "N/A"):
        return nan, nan
    if text == "<0.001":
        return 0.0, 0.001
    if " × 10" in text:
        coefficient, exponent = text.split(" × 10")
        scale = 10 ** int(exponent.translate(SUPERSCRIPTS))
        value, half_step = float(coefficient) * scale, 0.05 * scale
    else:
        value, half_step = float(text), 0.0005
    return max(0.0, value - half_step), min(1.0, value + half_step)


@pytest.fixture(scope="module")
def site_panels() -> list[PrintedPanel]:
    if not FIGURES.is_dir():
        pytest.skip("Manuscript reference figures are supplied only for clinical validation.")
    return [panel for site in SITES for panel in printed_panels(
        FIGURES / f"recurrence_blood_test_trajectories_{site}.svg", site=site,
    )]


def test_printed_adjusted_p_values_are_compatible_with_global_bh(site_panels: list[PrintedPanel]) -> None:
    comparisons = [(panel, window) for panel in site_panels for window in range(1, 5)
                   if isfinite(rounding_interval(panel.rows[4][window])[0])]
    intervals = [rounding_interval(panel.rows[4][window]) for panel, window in comparisons]
    low = bh_reference([value[0] for value in intervals])
    high = bh_reference([value[1] for value in intervals])
    for (panel, window), lower, upper in zip(comparisons, low, high, strict=True):
        printed_low, printed_high = rounding_interval(panel.rows[5][window])
        assert lower <= printed_high + 1e-14 and upper >= printed_low - 1e-14, (panel.site, panel.test, window)


def test_overview_cells_match_full_site_figures(site_panels: list[PrintedPanel]) -> None:
    lookup = {(panel.site, panel.test): panel.rows for panel in site_panels}
    overview = printed_panels(FIGURES / "recurrence_blood_test_trajectories.svg")
    assert len(overview) == 25
    for panel in overview:
        assert panel.rows == lookup[panel.site, panel.test], (panel.site, panel.test)


def test_printed_counts_and_quartiles_are_consistent(site_panels: list[PrintedPanel]) -> None:
    assert len(site_panels) == 120
    for panel in site_panels:
        baseline_n = int(panel.rows[6][0].split(" / ")[0])
        for window in range(5):
            tested, paired = panel.rows[6][window].split(" / ")
            if paired != "—":
                assert 0 <= int(paired) <= min(int(tested), baseline_n)
            for value_row, interval_row in ((0, 1), (2, 3)):
                value, interval = panel.rows[value_row][window], panel.rows[interval_row][window]
                if value != "—":
                    lower, upper = map(float, interval.strip("[]").split(", "))
                    assert lower <= float(value) <= upper, (panel.site, panel.test, window)
