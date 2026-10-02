from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping, Protocol, Sequence

from matplotlib.axes import Axes
from matplotlib.patches import ArrowStyle, FancyArrowPatch, Rectangle

from src.figure_style import figure_typography
from src.script.plot.overview_layout import load_cohort_layout
from src.script.plot.overview_style import (
    BORDER_COLOR,
    BORDER_WIDTH,
    BOX_PADDING,
    EXCLUDED_COLOR,
    FLOW_FONT_SIZE,
    FOOTNOTE_GAP_POINTS,
    INCLUDED_COLOR,
    LINE_SPACING,
    RETAINED_COLOR,
    SECONDARY_COLOR,
    TEXT_COLOR,
    wrap_numbered_note,
    wrap_to_width,
)

class PatientFlowDiagramConfigProtocol(Protocol):
    """Aligned source labels and counts for retained and excluded cohorts."""

    inclusion_box_labels: tuple[str, ...]
    inclusion_box_counts: tuple[str, ...]
    exclusion_box_labels: tuple[str, ...]
    exclusion_box_counts: tuple[str, ...]


@dataclass(frozen=True)
class BoxContent:
    """Wrapped criterion/date text, unchanged patient count, and measurement count.

    `criterion` includes any date lines supplied with the counts;
    `patients` and `blood_tests` only change the source count's display syntax.
    `font_size` controls both wrapping and physical row height.
    """

    criterion: str
    patients: str
    blood_tests: str
    font_size: float = FLOW_FONT_SIZE

    @property
    def height(self) -> float:
        lines = len(self.criterion.splitlines()) + bool(self.patients) + bool(self.blood_tests)
        return lines * self.font_size * LINE_SPACING + 2 * BOX_PADDING


def box_content(axis: Axes, label: str, counts: str, width: float, *, font_size: float = FLOW_FONT_SIZE) -> BoxContent:
    criterion = [display_text(label)]
    patients = ""
    blood_tests = ""
    for line in display_text(counts).splitlines():
        if re.match(r"[Nn]\s*=", line):
            patients = "n = " + line.split("=", 1)[1].strip()
        elif line.lower().startswith("blood tests ="):
            blood_tests = line.split("=", 1)[1].strip() + " blood tests"
        else:
            criterion.append(line)
    available = axis.get_position().width * axis.figure.get_figwidth() * 72 * width - 2 * BOX_PADDING
    return BoxContent(wrap_to_width(axis, "\n".join(criterion), available, font_size), patients, blood_tests, font_size)


def draw_patient_flow_diagram(
    axis: Axes, config: PatientFlowDiagramConfigProtocol, *,
    layout: Mapping[str, Any] | None = None, wide_cohort: bool = False, fit_height: bool = False,
    footnotes: Sequence[str] = (),
) -> None:
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    settings = load_cohort_layout() if layout is None else layout
    axis.set_axis_off()
    axis.set_xlim(0, 1)
    width = 43 / 102
    column_gap = settings["flow_column_gap_mm"] / 102
    left = (1 - width - column_gap) / 2
    right = (1 + width + column_gap) / 2
    retained = [box_content(axis, label, count, width, font_size=fonts.table)
                for label, count in zip(config.inclusion_box_labels, config.inclusion_box_counts, strict=True)]
    excluded = [box_content(axis, label, count, width - 3.6 / 102, font_size=fonts.table)
                for label, count in zip(config.exclusion_box_labels, config.exclusion_box_counts, strict=True)]
    cohort = box_content(axis, "Study cohort", config.inclusion_box_counts[-1], width, font_size=fonts.table)
    note_width = axis.get_position().width * axis.figure.get_figwidth() * 72
    notes = [wrap_numbered_note(axis, note, note_width, fonts.table) for note in footnotes]
    note_height = sum(len(note.splitlines()) * fonts.table * LINE_SPACING + BOX_PADDING for note, _ in notes)
    if notes:
        note_height += FOOTNOTE_GAP_POINTS + BORDER_WIDTH / 2 - BOX_PADDING
    head = (settings["retained_arrow_head_length_mm"], settings["retained_arrow_head_width_mm"])
    gap = 5.1
    heights = [max(content.height, excluded[index - 1].height if index else 0)
               for index, content in enumerate(retained)]
    # Larger exclusion boxes must also fit the gap between successive branch centres.
    for index in range(1, len(heights) - 1):
        heights[index] = max(heights[index], (excluded[index - 1].height + excluded[index].height) / 2)
    # Leave room below the final box for its full border stroke.
    total = 16 + sum(heights) + len(heights) * gap + cohort.height + BORDER_WIDTH + note_height
    if fit_height:
        position = axis.get_position()
        height = total / (axis.figure.get_figheight() * 72)
        axis.set_position((position.x0, position.y1 - height, position.width, height))
    available = axis.get_position().height * axis.figure.get_figheight() * 72
    axis.set_ylim(0, max(total, available))
    top = max(total, available) - 16
    for x, text in ((left, "Retained"), (right, "Excluded at this step")):
        axis.text(x, top + 10, text, ha="center", va="center", fontsize=fonts.table, weight="bold")
    for index, (content, height) in enumerate(zip(retained, heights, strict=True)):
        if index:
            branch = top + gap / 2
            arrow(axis, (left, top + gap), (left, top), head_mm=head)
            arrow(axis, (left, branch), (right - width / 2, branch))
            draw_exclusion(axis, right, branch, width, excluded[index - 1], index)
        draw_text_box(axis, left, top - height / 2, width, height, content, RETAINED_COLOR, role="retained")
        top -= height + gap
    arrow(axis, (left, top + gap), (left, top), head_mm=head)
    draw_text_box(axis, .5 if wide_cohort else left, top - cohort.height / 2,
                  2 * width + column_gap if wide_cohort else width, cohort.height, cohort, INCLUDED_COLOR,
                  role="study-cohort")
    note_top = top - cohort.height - BORDER_WIDTH / 2 - FOOTNOTE_GAP_POINTS
    for note, indent in notes:
        for index, line in enumerate(note.splitlines()):
            axis.text(indent / note_width if index else 0, note_top, line, va="top",
                      fontsize=fonts.table, color=TEXT_COLOR, gid="flow-note")
            note_top -= fonts.table * LINE_SPACING
        note_top -= BOX_PADDING


def draw_exclusion(axis: Axes, center: float, y: float, width: float, content: BoxContent, number: int) -> None:
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    count = len(axis.texts)
    draw_text_box(axis, center, y, width, content.height, content, EXCLUDED_COLOR, role="excluded")
    for text in axis.texts[count:]:
        x, text_y = text.get_position()
        text.set_position((x + 1.8 / 102, text_y))
    axis.text(center - width / 2 + 1.8 / 102, y, str(number), ha="left", va="center", fontsize=fonts.table,
              weight="bold", gid="exclusion-rule-number")


def arrow(axis: Axes, start: tuple[float, float], end: tuple[float, float], *,
          head_mm: tuple[float, float] | None = None) -> None:
    scale_mm = 5 * 25.4 / 72
    style = (ArrowStyle.CurveB(head_length=head_mm[0] / scale_mm, head_width=head_mm[1] / (2 * scale_mm))
             if head_mm is not None else ArrowStyle.CurveB())
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=5, linewidth=.5,
                                  color=BORDER_COLOR, shrinkA=0, shrinkB=0, zorder=1,
                                  gid="retained-connector" if head_mm is not None else "exclusion-connector"))


def draw_text_box(
    axis: Axes,
    x: float,
    y: float,
    width: float,
    height: float,
    content: BoxContent,
    color: str,
    *, role: str,
) -> None:
    box = Rectangle(
        (x - width / 2, y - height / 2),
        width,
        height,
        linewidth=BORDER_WIDTH,
        edgecolor=BORDER_COLOR,
        facecolor=color,
        zorder=2,
    )
    box.set_gid(role)
    axis.add_patch(box)
    line_height = content.font_size * LINE_SPACING
    top = y + (content.height - 2 * BOX_PADDING) / 2
    parts = (
        (content.criterion, "normal", TEXT_COLOR, "criterion"),
        (content.patients, "bold", TEXT_COLOR, "patients"),
        (content.blood_tests, "normal", SECONDARY_COLOR, "blood-tests"),
    )
    for text, weight, text_color, role in parts:
        if not text:
            continue
        for line in text.splitlines():
            artist = axis.text(
                x,
                top,
                line,
                fontsize=content.font_size,
                weight=weight,
                color=text_color,
                ha="center",
                va="top",
                zorder=3,
            )
            artist.set_gid(f"cohort-{role}")
            top -= line_height


def display_text(text: str) -> str:
    """Format saved integer-day cutoffs using ≤ for exclusions and > for eligibility."""
    text = text.replace("<=", "≤").replace(">=", "≥").replace("\\n", "\n")
    text = re.sub(r"≥\s*(\d+)(?=\s+days\b)", lambda match: f"> {int(match[1]) - 1}", text)
    text = re.sub(r"≥\s*day\s+(\d+)\b", lambda match: f"> day {int(match[1]) - 1}", text)
    return re.sub(r"<\s*(\d+)(?=\s+days\b)", lambda match: f"≤ {int(match[1]) - 1}", text)
