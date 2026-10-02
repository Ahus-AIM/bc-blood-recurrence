"""Shared composition and annual tables for training figures and design previews."""
from __future__ import annotations

from collections.abc import Sequence

from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from src.figure_style import figure_typography, publication_style
from src.data.cohort_figure import AnnualCurveTable, CohortFigureData
from src.data.population_statistics import PopulationStatisticsRow
from src.manual_age_summary import ManualAgeSummary
from src.script.plot.overview_layout import load_cohort_layout
from src.script.plot.overview_style import (
    BORDER_COLOR, FOOTNOTE_GAP_POINTS, LINE_SPACING, OVERALL_SITE_COLOR,
    recurrence_site_colors, text_width_points, wrap_numbered_note,
)

IMPUTATION_NOTE = "LOCF: last observation carried forward. MICE: multivariate imputation by chained equations."
COMPOSITION_NOTES = (
    "¹ Observation time was measured from BC diagnosis until another cancer, "
    "first distant recurrence, death or last observation, whichever occurred first.",
    "² Patients may contribute to more than one site category if multiple distant recurrence sites were recorded "
    "on the date of first distant recurrence.",
    IMPUTATION_NOTE,
)
STATISTICS_LABELS = {
    "Observation, years": "Observation time¹\nin years (median, (IQR))",
    "Age, years": "Age in years\n(median, (IQR))",
    "Site occurrences*": "Site occurrences²",
    "All sites*": "All sites²",
}


def draw_statistics_table(
    axis: Axes, rows: Sequence[PopulationStatisticsRow], x: float, y: float, width: float,
    step: float = 3.4, label_width: float = 29, *, subcategory_indent_mm: float = 0,
    shade_overall: bool = True, section_bands: bool = False, bold_overall: bool = True,
    emphasize_totals: bool = False, header_details: Sequence[str] | None = None,
    column_labels: Sequence[str] = ("Train", "Validation", "Test", "Overall"),
) -> float:
    """Draw rows in millimetres, allowing wrapped labels, and return the table bottom."""
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    site_colors = recurrence_site_colors((*publication_style().sites, "Any", "All"))
    column_width = (width - label_width) / 4.2
    ends = [x + label_width + column_width * factor for factor in (1, 2, 3, 4.2)]
    header_extra = 0 if header_details is None else 2.6
    labels = [STATISTICS_LABELS.get(row.label, row.label) for row in rows]
    heights = [step * (label.count("\n") + 1) for label in labels]
    if shade_overall:
        axis.add_patch(Rectangle((ends[2], y - 2), column_width * 1.2,
                                 sum(heights) + step + 1 + header_extra,
                                 facecolor="#f2f5f6", edgecolor="none", clip_on=False))
    for label, end in zip(column_labels, ends, strict=True):
        axis.text(end - .5, y, label, weight="bold", ha="right", va="center", fontsize=fonts.table)
    if header_details is not None:
        for end, detail in zip(ends, header_details, strict=True):
            axis.text(end - .5, y + header_extra, detail, ha="right", va="center", fontsize=fonts.table, color="#505860")
    axis.plot((x, x + width), (y + header_extra + step / 2,) * 2, color=BORDER_COLOR, linewidth=.45)
    row_top = y + header_extra + step / 2
    for index, (row, label, height) in enumerate(zip(rows, labels, heights, strict=True)):
        row_y = row_top + height / 2
        total = row.label in ("Patients", "Blood test values", "Site occurrences*", "All sites*")
        if section_bands and row.label in ("Blood test values", "Site occurrences*", "All sites*"):
            axis.add_patch(Rectangle((x, row_top), width, height, facecolor="#eaf0f3", edgecolor="none"))
        if index and row.label in ("Blood test values", "Observation, years", "Site occurrences*", "All sites*"):
            axis.plot((x, x + width), (row_top,) * 2, color=BORDER_COLOR, linewidth=.45)
        is_site = row.label in site_colors
        if row.label == "Site occurrences*" and any("(" in cell for cell in row.cells):
            label = "Site occurrences (unq. pts.)²"
        axis.text(x + (subcategory_indent_mm if row.indented else 0), row_y, label,
                  va="center", fontsize=fonts.table,
                  color=site_colors.get(row.label, publication_style().colors["text"]),
                  weight="bold" if total or is_site else "normal")
        for column, (cell, end) in enumerate(zip(row.cells, ends, strict=True)):
            bold = (bold_overall and column == 3) or (emphasize_totals and total)
            axis.text(end - .5, row_y, cell, va="center", ha="right", fontsize=fonts.table,
                      weight="bold" if bold else "normal")
        row_top += height
    return row_top


def draw_table_notes(axis: Axes, x: float, y: float) -> None:
    """Wrap definitions across the full table width, with hanging indents and height-based spacing."""
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    left, right = axis.get_xlim()
    width_points = (right - x) / (right - left) * axis.get_position().width * axis.figure.get_figwidth() * 72
    renderer = axis.figure.canvas.get_renderer()
    units_per_pixel = abs(axis.get_ylim()[1] - axis.get_ylim()[0]) / axis.bbox.height
    line_step = fonts.table * LINE_SPACING * axis.figure.dpi / 72 * units_per_pixel
    for note in COMPOSITION_NOTES:
        text, indent = wrap_numbered_note(axis, note, width_points, fonts.table)
        for index, line in enumerate(text.splitlines()):
            line_x = x + (right - x) * indent / width_points if index else x
            artist = axis.text(line_x, y + index * line_step, line, va="top",
                               fontsize=fonts.table, gid="composition-note")
        y = artist.get_position()[1] + artist.get_window_extent(renderer).height * units_per_pixel + .8


def draw_cohort_composition(axis: Axes, data: CohortFigureData, ages: ManualAgeSummary | None,
                            *, table_style: str = "overall") -> None:
    rows = list(data.population.rows)
    site_index = next(index for index, row in enumerate(rows) if row.label == "Site occurrences*")
    rows.insert(site_index, PopulationStatisticsRow("Age, years", tuple(ages.cells) if ages else ("—",) * 4))
    settings = load_cohort_layout()
    layout = settings["option_one"]
    axis.set_axis_off()
    axis.set_xlim(0, 102)
    axis.set_ylim(52, 0)
    draw_statistics_table(axis, rows, 0, 0, 102, layout["table_row_height_mm"], 24,
                          header_details=data.split_years, column_labels=data.population.column_labels,
                          subcategory_indent_mm=settings["subcategory_indent_mm"],
                          shade_overall=table_style == "overall", section_bands=table_style == "bands",
                          emphasize_totals=table_style == "plain")
    renderer = axis.figure.canvas.get_renderer()
    bottom = min(artist.get_window_extent(renderer).y0 for artist in (*axis.texts, *axis.patches, *axis.lines))
    note_top = bottom - FOOTNOTE_GAP_POINTS * axis.figure.dpi / 72
    _, note_y = axis.transData.inverted().transform((axis.bbox.x0, note_top))
    draw_table_notes(axis, 0, note_y)


def draw_annual_curve_table(axis: Axes, table: AnnualCurveTable, *, survival: bool, first_row_offset_mm: float | None = None,
                            label_margin_mm: float | None = None) -> None:
    """Put one numerical column beneath each annual tick, with no repeated year row."""
    if tuple(axis.get_xticks()) != table.years:
        raise ValueError("Annual table years must match every curve-axis tick.")
    fonts = figure_typography("dataset_statistics", axis.figure.get_figwidth())
    colors = recurrence_site_colors(table.sites)
    colors["Any"] = OVERALL_SITE_COLOR
    figure_height_mm = axis.figure.get_figheight() * 25.4
    plot_height_mm = axis.get_position().height * figure_height_mm
    layout = load_cohort_layout()["curve_tables"]
    offset = layout["first_row_offset_mm"] if first_row_offset_mm is None else first_row_offset_mm
    margin = layout.get("label_gap_mm", 2) if label_margin_mm is None else label_margin_mm
    first_column = max((text_width_points(axis, row[0], fonts.table) for row in table.cells if row), default=0)
    margin = max(margin, first_column / 2 * 25.4 / 72 + .8)
    label_x = -margin / (axis.get_position().width * axis.figure.get_figwidth() * 25.4)
    row_step = 3.2
    title = "At risk" if survival else "Cumulative incidence (%)"
    # Wrap the metric heading to keep it clear of the annual ticks.
    display_title = title if survival else "Cumulative\nincidence (%)"
    heading = axis.text(label_x, -2 / plot_height_mm, display_title, transform=axis.transAxes,
                        fontsize=fonts.table, va="top", ha="right", clip_on=False)
    heading.set_gid(title)
    for index, (site, cells) in enumerate(zip(table.sites, table.cells, strict=True)):
        y = -(offset + index * row_step) / plot_height_mm
        axis.text(label_x, y, site, transform=axis.transAxes, va="center", fontsize=fonts.table, color=colors[site],
                  weight="bold", ha="right", clip_on=False, gid=f"curve-site-{site}")
        for year, cell in zip(table.years, cells, strict=True):
            artist = axis.text(year, y, cell, transform=axis.get_xaxis_transform(), ha="center", va="center",
                               fontsize=fonts.table, clip_on=False)
            artist.set_gid(f"annual-{'survival' if survival else 'incidence'}-{site}-{year}")
    axis.xaxis.set_label_coords(.5, -(offset + 5 + max(0, len(table.sites) - 1) * row_step) / plot_height_mm)
