from __future__ import annotations

from dataclasses import dataclass
from collections import Counter
from itertools import combinations

import matplotlib.pyplot as plt
import pytest
from matplotlib.patches import FancyArrowPatch, Rectangle

from src.script.plot.patient_flow_diagram import display_text, draw_patient_flow_diagram
from src.script.plot.overview_style import INCLUDED_COLOR, SECONDARY_COLOR, TEXT_COLOR
from matplotlib.colors import to_hex


@dataclass(frozen=True)
class FlowConfig:
    """Source labels and counts for retained, excluded, and split test cohorts."""

    inclusion_box_labels: tuple[str, ...]
    inclusion_box_counts: tuple[str, ...]
    exclusion_box_labels: tuple[str, ...]
    exclusion_box_counts: tuple[str, ...]
    split_box_labels: tuple[str, ...]
    split_box_counts: tuple[str, ...]


def test_patient_flow_text_is_separate_and_contained() -> None:
    figure, axis = plt.subplots(figsize=(4, 8), dpi=180)
    draw_patient_flow_diagram(axis, dense_flow_config())
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    texts = [text for text in axis.texts if text.get_gid()]
    text_bounds = [text.get_window_extent(renderer) for text in texts]
    boxes = [patch.get_window_extent(renderer) for patch in axis.patches if isinstance(patch, Rectangle)]
    assert not any(a.overlaps(b) for a, b in combinations(text_bounds, 2))
    assert all(any(box.contains(*text.min) and box.contains(*text.max) for box in boxes) for text in text_bounds)
    assert all(axis.bbox.contains(*box.min) and axis.bbox.contains(*box.max) for box in boxes)
    plt.close(figure)


def test_patient_flow_preserves_counts_dates_and_text_hierarchy() -> None:
    figure, axis = plt.subplots(figsize=(4, 8))
    config = dense_flow_config()
    draw_patient_flow_diagram(axis, config)
    patients = [text for text in axis.texts if text.get_gid() == "cohort-patients"]
    blood = [text for text in axis.texts if text.get_gid() == "cohort-blood-tests"]
    source_counts = (*config.inclusion_box_counts, *config.exclusion_box_counts, config.inclusion_box_counts[-1])
    assert Counter(text.get_text() for text in patients) == Counter(
        "n = " + next(line.split("=")[1].strip() for line in source.splitlines() if line.startswith("N ="))
        for source in source_counts
    )
    assert Counter(text.get_text() for text in blood) == Counter(
        source.split("blood tests = ")[1] + " blood tests" for source in source_counts
    )
    assert all(text.get_weight() == "bold" and text.get_color() == TEXT_COLOR for text in patients)
    assert all(text.get_weight() == "normal" and text.get_color() == SECONDARY_COLOR for text in blood)
    assert "Study cohort" in [text.get_text() for text in axis.texts]
    assert "Train" not in [text.get_text() for text in axis.texts]
    assert "1.12.1998 - 08.06.2026" in [text.get_text() for text in axis.texts]
    plt.close(figure)


def test_patient_flow_has_one_cohort_and_straight_midpoint_branches() -> None:
    figure, axis = plt.subplots(figsize=(4, 8))
    config = dense_flow_config()
    draw_patient_flow_diagram(axis, config)
    retained = [patch for patch in axis.patches if patch.get_gid() == "retained"]
    excluded = [patch for patch in axis.patches if patch.get_gid() == "excluded"]
    cohort = [patch for patch in axis.patches if patch.get_gid() == "study-cohort"]
    assert len(retained) == len(config.inclusion_box_labels)
    assert len(excluded) == len(config.exclusion_box_labels)
    assert len(cohort) == 1 and to_hex(cohort[0].get_facecolor()) == INCLUDED_COLOR
    assert len({box.get_width() for box in (*retained, *excluded, *cohort)}) == 1
    branches = [patch for patch in axis.patches if patch.get_gid() == "exclusion-connector"]
    for index, branch in enumerate(branches):
        incoming_bottom = retained[index].get_y()
        outgoing_top = retained[index + 1].get_y() + retained[index + 1].get_height()
        shaft = branch.get_path().vertices[:3]
        assert all(y == pytest.approx((incoming_bottom + outgoing_top) / 2) for y in shaft[:, 1])
        assert shaft[0, 1] == pytest.approx(excluded[index].get_y() + excluded[index].get_height() / 2)
    assert all(isinstance(branch, FancyArrowPatch) for branch in branches)
    plt.close(figure)


def test_patient_flow_displays_comparison_signs() -> None:
    assert display_text(
        "x <= y and x >= z; Other cancer < 31 days after BC diagnosis; "
        "No blood tests >= 31 days after BC diagnosis; Patients with blood tests ≥ day 31"
    ) == (
        "x ≤ y and x ≥ z; Other cancer ≤ 30 days after BC diagnosis; "
        "No blood tests > 30 days after BC diagnosis; Patients with blood tests > day 30"
    )


def dense_flow_config() -> FlowConfig:
    return FlowConfig(
        inclusion_box_labels=(
            "Females diagnosed with BC at Ahus\n\n1.12.1998 - 08.06.2026",
            "Initial BC diagnosis >= 2004",
            "No distant recurrence before BC diagnosis",
            "No distant recurrence ≤ 30 days after BC diagnosis",
            "Blood test on-/one day before a BC checkup",
            "Blood tests > 30 days after BC diagnosis",
            "Blood tests before recurrence/death/right-censoring",
            "Days with > 12 different blood tests",
        ),
        inclusion_box_counts=(
            "N = 10,317\nblood tests = 543,210",
            "N = 10,090\nblood tests = 531,442",
            "N = 9,541\nblood tests = 502,118",
            "N = 8,845\nblood tests = 410,774",
            "N = 5,156\nblood tests = 286,201",
            "N = 5,109\nblood tests = 264,649",
            "N = 5,109\nblood tests = 264,649",
            "N = 5,109\nblood tests = 264,649",
        ),
        exclusion_box_labels=(
            "Initial BC diagnosis before 2004",
            "Distant recurrence before BC diagnosis",
            "Distant recurrence ≤ 30 days after BC diagnosis",
            "Blood test unrelated to BC checkup",
            "No blood tests > 30 days after BC diagnosis",
            "No blood tests before recurrence/death/right-censoring",
            "No days with > 12 different blood tests",
        ),
        exclusion_box_counts=(
            "N = 227\nblood tests = 11,768",
            "N = 549\nblood tests = 29,324",
            "N = 696\nblood tests = 91,344",
            "N = 3,389\nblood tests = 124,573",
            "N = 47\nblood tests = 21,552",
            "N = 47\nblood tests = 21,552",
            "N = 0\nblood tests = 0",
        ),
        split_box_labels=("Train", "Validation", "Test"),
        split_box_counts=(
            "2004 - 2016\nN = 2,543\nblood tests = 140,219",
            "2017 - 2021\nN = 1,315\nblood tests = 71,438",
            "2022 - 2026\nN = 1,251\nblood tests = 52,992",
        ),
    )
