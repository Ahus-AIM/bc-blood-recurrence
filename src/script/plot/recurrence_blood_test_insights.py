from __future__ import annotations

import logging
from dataclasses import replace
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from src.figure_style import VECTOR_FONT_RC, Typography, publication_style
from src.figure_headings import heading_typography, position_figure_title
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig
from src.data.recurrence_blood_test_insights import site_patient_counts
from src.data.recurrence_blood_test_selection import (
    RANKING_DESCRIPTION,
    SELECTION_COLUMNS,
    rank_recurrence_tests,
    select_recurrence_plot_rows,
)
from src.data.reference_intervals import reference_ranges_for_test
from src.script.ebm_survival_config import safe_path_component
from src.script.plot.overview_style import TEXT_COLOR, recurrence_site_colors
from src.script.plot.recurrence_blood_test_panel import (
    FIGURE_LEFT,
    FIGURE_RIGHT,
    FIGURE_TOP,
    PERCENT_FILL,
    OUTSIDE_REFERENCE_LABEL,
    fitted_panel_layout,
    content_row_height,
    draw_test_panel,
    site_row_height,
    trajectory_typography,
)

LOGGER = logging.getLogger(__name__)
FIGURE_STEM = "recurrence_blood_test_trajectories"


def write_recurrence_blood_test_figure(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> dict[str, object]:
    if not config.plot.enabled:
        return {"enabled": False}
    if config.plot.layout == "per_site":
        return write_recurrence_site_figures(frame, config)
    overview = write_combined_recurrence_figure(frame, config)
    if config.plot.layout == "combined":
        return overview
    per_site = write_recurrence_site_figures(frame, config)
    return {
        **overview,
        "layout": "both",
        "files": [*overview["files"], *per_site["files"]],
        "per_site": per_site,
    }


def write_combined_recurrence_figure(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> dict[str, object]:
    """Export the ranked overview and its selected values independently of the site pages."""
    selected = select_recurrence_plot_rows(frame, config)
    values_path = config.output_dir / f"{FIGURE_STEM}_values.csv"
    selected.to_csv(values_path, index=False)
    files = [values_path.name]
    figure = create_recurrence_blood_test_figure(frame, config)
    files.extend(save_trajectory_figure(figure, config, FIGURE_STEM))
    selection_rows = selected[["site", "test_key", *SELECTION_COLUMNS]].drop_duplicates()
    selections = selection_rows.astype(object).where(selection_rows.notna(), None).to_dict("records")
    LOGGER.info("Exported recurrence trajectories with %d selected site/test combinations", len(selections))
    return {
        "enabled": True,
        "layout": "combined",
        "files": files,
        "selected_tests": selections,
        "site_patient_counts": site_patient_counts(frame),
        "ranking": RANKING_DESCRIPTION,
    }


def save_trajectory_figure(figure: Figure, config: RecurrenceBloodTestConfig, stem: str) -> list[str]:
    """Export every requested format and release the canvas even if saving fails."""
    files = []
    try:
        with plt.rc_context(VECTOR_FONT_RC):
            for suffix in config.plot.formats:
                path = config.output_dir / f"{stem}.{suffix}"
                figure.savefig(path, dpi=config.plot.dpi, facecolor="white")
                files.append(path.name)
    finally:
        plt.close(figure)
    return files


def write_recurrence_site_figures(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> dict[str, object]:
    """Export all tests for each configured site, including tests with no paired comparisons."""
    stems = {site: f"{FIGURE_STEM}_{safe_path_component(site.casefold())}" for site in config.sites}
    if len(set(stems.values())) != len(stems):
        raise ValueError("Recurrence-site labels must produce distinct figure filenames.")
    frames = [ordered_site_rows(frame.loc[frame["site"].eq(site)], config) for site in config.sites]
    values = pd.concat(frames, ignore_index=True)
    values_stem = f"{FIGURE_STEM}_per_site" if config.plot.layout == "both" else FIGURE_STEM
    values_path = config.output_dir / f"{values_stem}_values.csv"
    values.to_csv(values_path, index=False)
    files = [values_path.name]
    for site, rows in zip(config.sites, frames, strict=True):
        figure = create_recurrence_site_figure(rows, config)
        files.extend(save_trajectory_figure(figure, config, stems[site]))
    LOGGER.info("Exported all blood tests in %d recurrence-site figures", len(frames))
    return {
        "enabled": True,
        "layout": "per_site",
        "files": files,
        "selected_tests": values[["site", "test_key"]].drop_duplicates().to_dict("records"),
        "site_patient_counts": site_patient_counts(values),
        "ranking": f"all report tests; {RANKING_DESCRIPTION}; unrankable tests last, alphabetical by name then key",
    }


def ordered_site_rows(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> pd.DataFrame:
    """Use the overview ranking for all tests; append unrankable tests alphabetically."""
    if frame["site"].nunique() != 1:
        raise ValueError("A site figure requires report rows for exactly one recurrence group.")
    ranks = rank_recurrence_tests(frame, config).set_index("test_key")["selection_rank"]
    return (
        frame.assign(_test_order=frame["test_key"].map(ranks))
        .sort_values(
            ["_test_order", "blood_test", "test_key", "min_days_before"],
            ascending=[True, True, True, False], na_position="last", kind="stable",
        )
        .drop(columns="_test_order")
    )


def create_recurrence_site_figure(
    frame: pd.DataFrame, config: RecurrenceBloodTestConfig, *, synthetic: bool = False,
) -> Figure:
    """Plot every test for one site with all eight numeric rows; label synthetic previews explicitly."""
    ordered = ordered_site_rows(frame, config)
    groups = list(ordered.groupby("test_key", sort=False))
    columns, plot = fitted_panel_layout(ordered, config, columns=config.plot.columns)
    config = replace(config, plot=plot)
    width = FIGURE_LEFT + columns * plot.panel_width + FIGURE_RIGHT
    config = replace(config, plot=replace(plot, row_height=content_row_height(ordered, config, width)))
    row_height = site_row_height(config)
    height = FIGURE_TOP + ceil(len(groups) / columns) * row_height + (0.35 if synthetic else 0.12)
    fonts = trajectory_typography(config, width)
    site = str(ordered.iloc[0]["site"])
    count = site_patient_counts(ordered)[site]
    color = recurrence_site_colors(config.sites)[site]
    title = f"{site} recurrence · Blood test trajectories · {count:,} {'synthetic ' if synthetic else ''}patients"
    stem = f"{FIGURE_STEM}_{safe_path_component(site.casefold())}"
    headings = heading_typography(stem, width, height)
    with plt.rc_context(fonts.rc()):
        figure = plt.figure(figsize=(width, height), facecolor="white")
        top = draw_figure_header(figure, columns, FIGURE_TOP,
                                 replace(fonts, figure_title=headings.figure_title), title=title)
        headings = heading_typography(stem, *figure.get_size_inches())
        limits = plot_y_limits(ordered, config)
        for index, (test_key, rows) in enumerate(groups):
            row, column = divmod(index, columns)
            ranges = reference_ranges_for_test(config.reference_ranges, str(test_key))
            draw_test_panel(
                figure, rows, config, ranges, color, column,
                figure.get_figheight() - top - row * row_height,
                limits[(site, str(test_key))], heading_size=headings.site_label, panel_number=index + 1,
            )
        if synthetic:
            figure.text(
                0.5, 0.09 / figure.get_figheight(),
                "SYNTHETIC DATA · Layout review only · Tests ranked by earliest significance persisting to recurrence",
                ha="center", va="bottom", fontsize=fonts.table, color="#586572",
            )
        position_figure_title(figure, stem)
    return figure


def create_recurrence_blood_test_figure(
    frame: pd.DataFrame, config: RecurrenceBloodTestConfig, *, synthetic: bool = False,
) -> Figure:
    """Apply the shared font family even for callers that render without exporting."""
    with plt.rc_context(trajectory_typography(config, config.plot.panel_width).rc()):
        return draw_recurrence_blood_test_figure(frame, config, synthetic=synthetic)


def draw_recurrence_blood_test_figure(
    frame: pd.DataFrame, config: RecurrenceBloodTestConfig, *, synthetic: bool = False,
) -> Figure:
    """Select panels from the full report, retaining totals even for unrankable sites."""
    selected = select_recurrence_plot_rows(frame, config)
    totals = site_patient_counts(frame)
    columns, plot = fitted_panel_layout(selected, config)
    config = replace(config, plot=plot)
    width = FIGURE_LEFT + columns * config.plot.panel_width + FIGURE_RIGHT
    config = replace(config, plot=replace(config.plot, row_height=content_row_height(selected, config, width)))
    row_height = site_row_height(config)
    top_margin = FIGURE_TOP + (0.4 if columns == 1 else 0)
    figure = plt.figure(
        figsize=(
            width,
            top_margin + 0.12 + len(config.sites) * row_height,
        ),
        facecolor="white",
    )
    fonts = trajectory_typography(config, width)
    title = "Blood test trajectories before first distant recurrence"
    if synthetic:
        title += " · SYNTHETIC DATA · Layout review only"
    headings = heading_typography(FIGURE_STEM, *figure.get_size_inches())
    top_margin = draw_figure_header(figure, columns, top_margin,
                                    replace(fonts, figure_title=headings.figure_title), title=title)
    width, height = figure.get_size_inches()
    limits = plot_y_limits(selected, config)
    colors = recurrence_site_colors(config.sites)
    for site_index, site in enumerate(config.sites):
        color = colors[site]
        patients = "patient" if totals[site] == 1 else "patients"
        row_top = height - top_margin - site_index * row_height
        y_position = row_top - 0.5 * row_height
        site_label = figure.text(
            0.22 / width,
            y_position / height,
            f"{site} ({totals[site]:,} {patients})",
            color=color,
            rotation=90,
            fontsize=headings.site_label,
            weight="bold",
            va="center",
            ha="center",
            gid=f"site-label-{site_index}",
        )
        site_rows = selected.loc[selected["site"].eq(site)]
        if site_rows.empty:
            figure.text(
                0.95 / width,
                y_position / height,
                "No calculable paired comparisons",
                color="#64748B",
                fontsize=fonts.table,
            )
        first_axis, first_heading = len(figure.axes), len(figure.texts)
        for index, (test_key, rows) in enumerate(site_rows.groupby("test_key", sort=False)):
            ranges = reference_ranges_for_test(config.reference_ranges, str(test_key))
            draw_test_panel(figure, rows, config, ranges, color, index, row_top,
                            limits[(site, str(test_key))], heading_size=headings.site_label, panel_number=index + 1)
        if not site_rows.empty:
            renderer = figure.canvas.get_renderer()
            bounds = [axis.get_tightbbox(renderer) for axis in figure.axes[first_axis:]]
            bounds.extend(text.get_window_extent(renderer) for text in figure.texts[first_heading:])
            site_label.set_y((min(box.y0 for box in bounds) + max(box.y1 for box in bounds)) / (2 * figure.bbox.height))
    position_figure_title(figure, FIGURE_STEM)
    return figure


def draw_figure_header(
    figure: Figure, columns: int, top_margin: float, fonts: Typography,
    *, title: str = "Blood test trajectories before first distant recurrence",
) -> float:
    """Fit the title and legend at the fitted text size before placing site rows."""
    width, height = figure.get_size_inches()
    title = figure.text(
        0.5,
        1 - 0.14 / height,
        title,
        fontsize=fonts.figure_title,
        weight="bold",
        va="top",
        ha="center",
        color=TEXT_COLOR,
        gid="figure-title",
    )
    legend_handles = (
        Line2D([0], [0], color=TEXT_COLOR, marker="o", markerfacecolor="white", label="Median with IQR whiskers"),
        Patch(facecolor=publication_style().colors["secondary"], alpha=0.4, label="Female reference ranges"),
        Patch(
            facecolor=PERCENT_FILL,
            alpha=0.4,
            label=(
                f"{OUTSIDE_REFERENCE_LABEL}\n"
                "Baseline: entire tested population for that test/site\n"
                "Later windows: patients paired with baseline"
            ),
        ),
    )
    legend = figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1 - 0.42 / height),
        ncol=1 if columns == 1 else 3,
        frameon=False,
        fontsize=fonts.legend,
    )
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    title_bottom = title.get_window_extent(renderer).y0
    legend_offset = (figure.bbox.height - title_bottom) / figure.dpi + 0.08
    legend.set_bbox_to_anchor((0.5, 1 - legend_offset / height))
    required = (figure.bbox.height - legend.get_window_extent(renderer).y0) / figure.dpi + 0.14
    if required > top_margin:
        height += required - top_margin
        top_margin = required
        figure.set_size_inches(width, height)
        title.set_y(1 - 0.14 / height)
        legend.set_bbox_to_anchor((0.5, 1 - legend_offset / height))
    return top_margin


def plot_y_limits(
    selected: pd.DataFrame,
    config: RecurrenceBloodTestConfig,
) -> dict[tuple[str, str], tuple[float, float]]:
    """Pad each panel's displayed median/IQR and finite reference bounds by eight percent."""
    limits = {}
    for (site, test_key), rows in selected.groupby(["site", "test_key"], sort=False):
        bounds = rows[["q1", "median", "q3"]].to_numpy(dtype=float).ravel().tolist()
        for interval in reference_ranges_for_test(config.reference_ranges, str(test_key)):
            bounds.extend(bound for bound in (interval.lower, interval.upper) if bound is not None)
        finite = np.asarray(bounds)[np.isfinite(bounds)]
        lower, upper = (0.0, 1.0) if len(finite) == 0 else (float(finite.min()), float(finite.max()))
        padding = 0.08 * (upper - lower if upper > lower else max(abs(lower), 1.0))
        limits[(str(site), str(test_key))] = (lower - padding, upper + padding)
    return limits
