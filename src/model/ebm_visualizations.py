from __future__ import annotations

import io
import logging
import os
import re
from collections.abc import Mapping, Sequence
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axis import Axis
from matplotlib.font_manager import FontProperties
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MultipleLocator, NullFormatter
from interpret import preserve
from interpret.glassbox import ExplainableBoostingRegressor

from src.data.blood_test_display import FeatureDisplayLabel, feature_display_label
from src.data.blood_test_features import BLOOD_TEST_CHANGE_SUFFIX
from src.data.timing import timed
from src.figure_style import VECTOR_FONT_RC, publication_style, recurrence_site_colors, wrap_text
from src.figure_headings import ebm_row_gap_fraction, heading_typography, position_figure_title
from src.model.ebm_explanation_data import (
    float_array as float_array,
    optional_float_array as optional_float_array,
    sequence_value as sequence_value,
)
from src.model.ebm_explanation_tables import write_explanation_csvs
from src.model.ebm_global_importance import FIGURE_SIZE, PNG_DPI, write_global_importance_plot
from src.model.ebm_plotting_defaults import (
    DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS,
    FEATURE_EFFECTS_TITLE,
    HAZARD_RATIO_LABEL,
    IMPORTANCE_TITLE,
    recurrence_figure_title,
)
from src.model.normal_ranges import NormalRange, normal_range_metadata
from src.model.normal_range_plotting import (
    PUBLICATION_REFERENCE_ALPHA,
    add_deduplicated_legend,
    add_interaction_normal_ranges,
    add_numeric_reference_labels,
    add_univariate_normal_ranges,
    fit_numeric_reference_labels,
)

LOGGER = logging.getLogger(__name__)
INTERACTION_SIZE = (7.0, 5.8)
HAZARD_RATIO_GRID_SUBPLOT_WIDTH = 2.55
HAZARD_RATIO_GRID_SUBPLOT_HEIGHT = 2.55
HAZARD_RATIO_GRID_TITLE_INSET_INCHES = 0.10
TERM_X_AXIS_SCALES = ("linear", "log")
UNIVARIATE_Y_TICK_STEP = 0.1
MAX_UNIVARIATE_FIXED_Y_TICKS = 100
MAX_SAFE_FIXED_Y_TICKS = 1_000
UNIVARIATE_LOG_Y_TICKS = 12
GRID_FRACTIONAL_MAJOR_Y_TICKS = (0.5,)
GRID_MAJOR_Y_TICK_LENGTH = 3.5
GRID_MINOR_Y_TICK_LENGTH = 2.0
GRID_EMPHASIZED_LINE_COLOR = "0.58"
GRID_BACKGROUND_LINE_COLOR = "0.78"
GRID_LINE_WIDTH = 0.4
LOG_X_AXIS_MIN_VALUE_RATIO = 10.0
TIME_SINCE_DIAGNOSIS_LABEL_TICK_STEP = 5.0
TIME_SINCE_DIAGNOSIS_YEAR_TICK_STEP = 1.0
TIME_SINCE_DIAGNOSIS_FEATURE_NAMES = frozenset(
    {"time_since_diagnosis", "time_since_diagnosis_years"}
)
LINEAR_X_AXIS_FEATURE_NAMES = TIME_SINCE_DIAGNOSIS_FEATURE_NAMES | {
    "age_at_diagnosis"
}
NUMERIC_TICK_FORMATTER = FuncFormatter(lambda value, _position: numeric_tick_label(value))


@dataclass(frozen=True)
class HazardRatioGridStyle:
    """Shared rendering policy for compact univariate hazard-ratio panels.

    The mappings supply reference ranges and labels. ``x_axis_scale`` and
    ``y_limits`` control axes, while the final flags control term markers and
    outer-bag uncertainty rendering. Optional ``color`` gives a site-specific
    color to the effect curve, markers, and uncertainty.
    """

    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]]
    test_name_translation: Mapping[str, str]
    x_axis_scale: str
    y_limits: tuple[float, float]
    show_x_markers: bool
    show_uncertainty_area: bool
    color: str | None = None


@timed(LOGGER, "write EBM explanation visualizations")
def write_ebm_explanations(
    model: ExplainableBoostingRegressor,
    output_dir: Path,
    endpoint_name: str,
    include_missing_bins: bool = True,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None = None,
    normal_ranges_csv_path: Path | None = None,
    test_name_translation: Mapping[str, str] | None = None,
    test_name_translation_csv_path: Path | None = None,
    term_x_axis_scale: str = "linear",
    write_combined_term_hazard_ratio_plot: bool = True,
    combined_term_hazard_ratio_columns: int = 4,
    combined_term_hazard_ratio_y_limits: tuple[
        float, float
    ] = DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS,
    combined_term_hazard_ratio_show_x_markers: bool = False,
    combined_term_hazard_ratio_show_uncertainty_area: bool = True,
    fail_fast: bool = False,
) -> dict[str, object]:
    explanation = model.explain_global()
    explanation_dir = output_dir / "explanations"
    explanation_dir.mkdir(parents=True, exist_ok=True)
    artifacts = write_ebm_explanation_plots(
        model,
        explanation_dir,
        risk_score_label=f"{human_label(endpoint_name)} EBM Cox Risk-Score Contribution",
        hazard_ratio_label=HAZARD_RATIO_LABEL,
        endpoint_name=endpoint_name,
        include_missing_bins=include_missing_bins,
        term_x_axis_scale=term_x_axis_scale,
        explanation=explanation,
        normal_ranges_by_feature=normal_ranges_by_feature,
        normal_ranges_csv_path=normal_ranges_csv_path,
        test_name_translation=test_name_translation,
        test_name_translation_csv_path=test_name_translation_csv_path,
        write_combined_term_hazard_ratio_plot=write_combined_term_hazard_ratio_plot,
        combined_term_hazard_ratio_columns=combined_term_hazard_ratio_columns,
        combined_term_hazard_ratio_y_limits=combined_term_hazard_ratio_y_limits,
        combined_term_hazard_ratio_show_x_markers=combined_term_hazard_ratio_show_x_markers,
        combined_term_hazard_ratio_show_uncertainty_area=(
            combined_term_hazard_ratio_show_uncertainty_area
        ),
    )
    html_path = write_interpret_html(explanation, explanation_dir / "interpret_global.html", fail_fast=fail_fast)
    if html_path is not None:
        artifacts["interpret_html"] = str(html_path.relative_to(output_dir))
    return artifacts


def write_ebm_explanation_plots(
    model: ExplainableBoostingRegressor,
    output_dir: Path,
    risk_score_label: str,
    hazard_ratio_label: str = HAZARD_RATIO_LABEL,
    include_missing_bins: bool = True,
    explanation: Any | None = None,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None = None,
    normal_ranges_csv_path: Path | None = None,
    test_name_translation: Mapping[str, str] | None = None,
    test_name_translation_csv_path: Path | None = None,
    term_x_axis_scale: str = "linear",
    write_combined_term_hazard_ratio_plot: bool = True,
    combined_term_hazard_ratio_columns: int = 4,
    combined_term_hazard_ratio_y_limits: tuple[
        float, float
    ] = DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS,
    combined_term_hazard_ratio_show_x_markers: bool = False,
    combined_term_hazard_ratio_show_uncertainty_area: bool = True,
    endpoint_name: str | None = None,
) -> dict[str, object]:
    """Write global, per-term, and interaction figures from an InterpretML explanation."""
    if explanation is None:
        explanation = model.explain_global()
    term_x_axis_scale = normalized_term_x_axis_scale(term_x_axis_scale)
    normal_ranges = {} if normal_ranges_by_feature is None else normal_ranges_by_feature
    translations = {} if test_name_translation is None else dict(test_name_translation)
    output_dir.mkdir(parents=True, exist_ok=True)
    term_dir = output_dir / "terms"
    hazard_term_dir = output_dir / "terms_hazard_ratio"
    interaction_dir = output_dir / "interactions"
    hazard_interaction_dir = output_dir / "interactions_hazard_ratio"
    artifact_root = output_dir.parent
    term_dir.mkdir(exist_ok=True)
    hazard_term_dir.mkdir(exist_ok=True)
    interaction_dir.mkdir(exist_ok=True)
    hazard_interaction_dir.mkdir(exist_ok=True)

    term_plots: list[dict[str, object]] = []
    hazard_term_plots: list[dict[str, object]] = []
    interaction_plots: list[dict[str, object]] = []
    hazard_interaction_plots: list[dict[str, object]] = []
    artifacts: dict[str, object] = {
        "global_importance_png": str((output_dir / "global_importance.png").relative_to(artifact_root)),
        "global_importance_svg": str((output_dir / "global_importance.svg").relative_to(artifact_root)),
        "term_plots": term_plots,
        "term_hazard_ratio_plots": hazard_term_plots,
        "interaction_plots": interaction_plots,
        "interaction_hazard_ratio_plots": hazard_interaction_plots,
        "include_missing_bins": include_missing_bins,
        "term_x_axis_scale": term_x_axis_scale,
        "normal_ranges": normal_range_metadata(normal_ranges, normal_ranges_csv_path),
        "test_name_translation": test_name_translation_metadata(
            translations,
            test_name_translation_csv_path,
        ),
    }
    write_global_importance_plot(
        explanation.data(),
        output_dir / "global_importance.png",
        test_name_translation=translations,
        title=recurrence_figure_title(IMPORTANCE_TITLE, endpoint_name),
        color=None if endpoint_name is None else recurrence_site_colors((endpoint_name,))[endpoint_name],
    )
    write_global_importance_plot(
        explanation.data(),
        output_dir / "global_importance.svg",
        test_name_translation=translations,
        title=recurrence_figure_title(IMPORTANCE_TITLE, endpoint_name),
        color=None if endpoint_name is None else recurrence_site_colors((endpoint_name,))[endpoint_name],
    )

    term_count = len(explanation.data()["names"])
    for term_index in range(term_count):
        term_data = explanation.data(term_index)
        term_name = str(explanation.data()["names"][term_index])
        if term_data.get("type") == "interaction":
            path = interaction_dir / f"{term_index:03d}_{safe_filename(term_name)}.png"
            write_interaction_plot(
                term_data,
                term_name,
                risk_score_label,
                path,
                normal_ranges,
                test_name_translation=translations,
            )
            interaction_plots.append(term_artifact(term_index, term_name, path, artifact_root))
            hazard_path = hazard_interaction_dir / path.name
            write_interaction_hazard_ratio_plot(
                term_data,
                term_name,
                hazard_ratio_label,
                hazard_path,
                normal_ranges,
                test_name_translation=translations,
            )
            hazard_interaction_plots.append(term_artifact(term_index, term_name, hazard_path, artifact_root))
            continue
        path = term_dir / f"{term_index:03d}_{safe_filename(term_name)}.png"
        write_univariate_plot(
            term_data,
            term_name,
            risk_score_label,
            path,
            missing_score=missing_bin_score(model, term_index) if include_missing_bins else None,
            normal_ranges=normal_ranges.get(term_name, ()),
            test_name_translation=translations,
            x_axis_scale=term_x_axis_scale,
        )
        term_plots.append(term_artifact(term_index, term_name, path, artifact_root))
        hazard_path = hazard_term_dir / path.name
        write_univariate_hazard_ratio_plot(
            term_data,
            term_name,
            hazard_ratio_label,
            hazard_path,
            missing_score=missing_bin_score(model, term_index) if include_missing_bins else None,
            normal_ranges=normal_ranges.get(term_name, ()),
            test_name_translation=translations,
            x_axis_scale=term_x_axis_scale,
        )
        hazard_term_plots.append(term_artifact(term_index, term_name, hazard_path, artifact_root))
    if write_combined_term_hazard_ratio_plot:
        artifacts["combined_term_hazard_ratio_plot_png"] = str(
            (output_dir / "terms_hazard_ratio_grid.png").relative_to(artifact_root)
        )
        artifacts["combined_term_hazard_ratio_plot_svg"] = str(
            (output_dir / "terms_hazard_ratio_grid.svg").relative_to(artifact_root)
        )
        write_univariate_hazard_ratio_grid(
            model,
            explanation,
            output_dir / "terms_hazard_ratio_grid.png",
            output_dir / "terms_hazard_ratio_grid.svg",
            include_missing_bins=include_missing_bins,
            normal_ranges_by_feature=normal_ranges,
            test_name_translation=translations,
            x_axis_scale=term_x_axis_scale,
            columns=combined_term_hazard_ratio_columns,
            y_limits=combined_term_hazard_ratio_y_limits,
            show_x_markers=combined_term_hazard_ratio_show_x_markers,
            show_uncertainty_area=combined_term_hazard_ratio_show_uncertainty_area,
            title=recurrence_figure_title(FEATURE_EFFECTS_TITLE, endpoint_name),
            color=None if endpoint_name is None else recurrence_site_colors((endpoint_name,))[endpoint_name],
        )
    artifacts.update(write_explanation_csvs(model, explanation, output_dir, include_missing_bins))
    return artifacts


def write_interpret_html(explanation: Any, output_path: Path, *, fail_fast: bool = False) -> Path | None:
    previous_node_warnings = os.environ.get("NODE_NO_WARNINGS")
    os.environ["NODE_NO_WARNINGS"] = "1"
    try:
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            preserve(explanation, file_name=str(output_path))
    except Exception as error:
        if fail_fast:
            raise
        LOGGER.warning("could not write InterpretML HTML explanation: %s", error)
        return None
    finally:
        if previous_node_warnings is None:
            os.environ.pop("NODE_NO_WARNINGS", None)
        else:
            os.environ["NODE_NO_WARNINGS"] = previous_node_warnings
    return output_path


def write_univariate_plot(
    term_data: Mapping[str, object],
    term_name: str,
    risk_score_label: str,
    output_path: Path,
    missing_score: tuple[float, float] | None = None,
    normal_ranges: tuple[NormalRange, ...] = (),
    test_name_translation: Mapping[str, str] | None = None,
    x_axis_scale: str = "linear",
) -> None:
    scores = float_array(term_data["scores"])
    x_values, labels = univariate_x_values(sequence_value(term_data["names"]), len(scores))
    valid = np.isfinite(x_values) & np.isfinite(scores)
    display_label = feature_display_label(term_name, translation_mapping(test_name_translation))
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
    yerr = univariate_yerr(term_data, scores)
    if yerr is None:
        axis.plot(x_values[valid], scores[valid], color="#d95f02", marker="x", markersize=4, linewidth=1.1)
    else:
        axis.errorbar(
            x_values[valid],
            scores[valid],
            yerr=yerr[:, valid],
            color="#d95f02",
            ecolor="#4daf4a",
            marker="x",
            markersize=4,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2.5,
        )
    axis.set_xlabel(display_label.x_axis_label)
    axis.set_ylabel(risk_score_label)
    axis.set_title(display_label.title)
    axis.grid(linestyle=":", color="0.78")
    apply_category_labels(axis, x_values, labels)
    apply_univariate_x_axis_scale(axis, term_name, x_values, labels, valid, x_axis_scale)
    if labels is None:
        add_univariate_normal_ranges(axis, normal_ranges)
    add_missing_bin_marker(axis, missing_score)
    add_deduplicated_legend(axis)
    format_univariate_axes(axis, term_name, labels)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def write_univariate_hazard_ratio_plot(
    term_data: Mapping[str, object],
    term_name: str,
    hazard_ratio_label: str,
    output_path: Path,
    missing_score: tuple[float, float] | None = None,
    normal_ranges: tuple[NormalRange, ...] = (),
    test_name_translation: Mapping[str, str] | None = None,
    x_axis_scale: str = "linear",
) -> None:
    figure, axis = plt.subplots(figsize=FIGURE_SIZE, dpi=PNG_DPI)
    draw_univariate_hazard_ratio_axis(
        axis,
        term_data,
        term_name,
        hazard_ratio_label,
        missing_score=missing_score,
        normal_ranges=normal_ranges,
        test_name_translation=test_name_translation,
        x_axis_scale=x_axis_scale,
    )
    figure.tight_layout()
    fit_numeric_reference_labels(axis)
    figure.savefig(output_path)
    plt.close(figure)


def draw_univariate_hazard_ratio_axis(
    axis: plt.Axes,
    term_data: Mapping[str, object],
    term_name: str,
    hazard_ratio_label: str,
    missing_score: tuple[float, float] | None = None,
    normal_ranges: tuple[NormalRange, ...] = (),
    test_name_translation: Mapping[str, str] | None = None,
    x_axis_scale: str = "linear",
    y_limits: tuple[float, float] | None = None,
    compact: bool = False,
    show_ylabel: bool = True,
    show_normal_range_legend: bool = True,
    show_normal_range_boundaries: bool = True,
    show_multiple_normal_ranges_as_bars: bool = False,
    show_x_markers: bool = True,
    show_uncertainty_area: bool = False,
    show_hazard_ratio_baseline: bool = True,
    color: str | None = None,
) -> None:
    scores = float_array(term_data["scores"])
    hazard_ratios = hazard_ratio_values(scores)
    x_values, labels = univariate_x_values(sequence_value(term_data["names"]), len(scores))
    valid = np.isfinite(x_values) & np.isfinite(hazard_ratios) & (hazard_ratios > 0.0)
    yerr = univariate_hazard_ratio_yerr(term_data, scores, hazard_ratios)
    display_label = feature_display_label(term_name, translation_mapping(test_name_translation))
    marker = "x" if show_x_markers else None
    if yerr is not None:
        yerr_valid = np.all(np.isfinite(yerr), axis=0)
        valid &= yerr_valid
    if yerr is None or show_uncertainty_area:
        axis.plot(
            x_values[valid],
            hazard_ratios[valid],
            color="#7570b3" if color is None else color,
            marker=marker,
            markersize=4,
            linewidth=1.1,
        )
        if yerr is not None:
            axis.fill_between(
                x_values[valid],
                hazard_ratios[valid] - yerr[0, valid],
                hazard_ratios[valid] + yerr[1, valid],
                color="#4daf4a" if color is None else color,
                alpha=0.25,
                linewidth=0.0,
            )
    else:
        axis.errorbar(
            x_values[valid],
            hazard_ratios[valid],
            yerr=yerr[:, valid],
            color="#7570b3" if color is None else color,
            ecolor="#4daf4a" if color is None else color,
            marker=marker,
            markersize=4,
            linewidth=1.1,
            elinewidth=0.8,
            capsize=2.5,
        )
    if show_hazard_ratio_baseline:
        axis.axhline(1.0, color="0.35", linestyle="--", linewidth=0.9)
    axis.set_yscale("log")
    axis.set_xlabel(display_label.compact_x_axis_label if compact else display_label.x_axis_label)
    axis.set_ylabel(hazard_ratio_label if show_ylabel else "")
    axis.set_title(display_label.title)
    axis.grid(linestyle=":", color="0.78")
    apply_category_labels(axis, x_values, labels)
    apply_univariate_x_axis_scale(axis, term_name, x_values, labels, valid, x_axis_scale)
    if labels is None:
        add_univariate_normal_ranges(
            axis,
            normal_ranges,
            show_legend=show_normal_range_legend,
            show_boundaries=show_normal_range_boundaries,
            show_multiple_as_horizontal_bars=show_multiple_normal_ranges_as_bars,
            span_alpha=PUBLICATION_REFERENCE_ALPHA,
        )
        add_numeric_reference_labels(axis, normal_ranges)
    add_missing_hazard_ratio_marker(axis, missing_score)
    if not compact:
        add_deduplicated_legend(axis)
    if y_limits is not None:
        axis.set_ylim(y_limits)
    format_univariate_axes(axis, term_name, labels)


def draw_hazard_ratio_grid_term(
    axis: plt.Axes,
    term_data: Mapping[str, object],
    term_name: str,
    style: HazardRatioGridStyle,
    *,
    show_ylabel: bool,
    missing_score: tuple[float, float] | None = None,
) -> None:
    """Draw one term with the shared compact hazard-ratio-grid policy."""
    draw_univariate_hazard_ratio_axis(
        axis,
        term_data,
        term_name,
        HAZARD_RATIO_LABEL,
        missing_score=missing_score,
        normal_ranges=female_normal_ranges(style.normal_ranges_by_feature.get(term_name, ())),
        test_name_translation=style.test_name_translation,
        x_axis_scale=style.x_axis_scale,
        y_limits=style.y_limits,
        compact=True,
        show_ylabel=show_ylabel,
        show_normal_range_legend=False,
        show_normal_range_boundaries=False,
        show_multiple_normal_ranges_as_bars=True,
        show_x_markers=style.show_x_markers,
        show_uncertainty_area=style.show_uncertainty_area,
        show_hazard_ratio_baseline=False,
        color=style.color,
    )
    format_grid_axis(axis)
    if missing_score is not None:
        handles, labels = axis.get_legend_handles_labels()
        markers = [handle for handle, label in zip(handles, labels, strict=True) if label == "missing bin"]
        axis.legend(markers, ["missing bin"], loc="best", fontsize=6, frameon=False)


def write_univariate_hazard_ratio_grid(
    model: ExplainableBoostingRegressor,
    explanation: Any,
    png_path: Path,
    svg_path: Path,
    *,
    include_missing_bins: bool,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]],
    test_name_translation: Mapping[str, str],
    x_axis_scale: str,
    columns: int,
    y_limits: tuple[float, float],
    show_x_markers: bool,
    show_uncertainty_area: bool,
    title: str = FEATURE_EFFECTS_TITLE,
    color: str | None = None,
) -> None:
    if columns < 1:
        raise ValueError("`columns` must be at least 1.")
    terms = univariate_terms_by_importance(explanation)
    if not terms:
        return
    style = HazardRatioGridStyle(
        normal_ranges_by_feature,
        test_name_translation,
        x_axis_scale,
        y_limits,
        show_x_markers,
        show_uncertainty_area,
        color=color,
    )
    missing_scores = {
        term_index: score
        for term_index, _term_name in terms
        if include_missing_bins and (score := missing_bin_score(model, term_index)) is not None
    }
    rows = int(np.ceil(len(terms) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(
            columns * HAZARD_RATIO_GRID_SUBPLOT_WIDTH,
            rows * HAZARD_RATIO_GRID_SUBPLOT_HEIGHT,
        ),
        dpi=PNG_DPI,
        squeeze=False,
    )
    for position, (term_index, term_name) in enumerate(terms):
        row, column = divmod(position, columns)
        axis = axes[row][column]
        draw_hazard_ratio_grid_term(
            axis,
            explanation.data(term_index),
            term_name,
            style,
            show_ylabel=column == 0,
            missing_score=missing_scores.get(term_index),
        )
        axis.set_title(f"{position + 1}. {axis.get_title()}")
    for position in range(len(terms), rows * columns):
        row, column = divmod(position, columns)
        axes[row][column].axis("off")
    headings = heading_typography("terms_hazard_ratio_grid", *figure.get_size_inches())
    for axis in axes.flat:
        axis.title.set_fontsize(headings.site_label)
        axis.title.set_fontweight("bold")
        axis.title.set_fontfamily(headings.family)
        axis.title.set_color(publication_style().colors["text"])
        axis.title.set_text(wrap_text(
            axis.get_title(), axis.bbox.width,
            FontProperties(family=headings.family, size=headings.site_label, weight="bold"),
            figure.canvas.get_renderer(),
        ))
    # Reserve title space before fitting panels; measure the final gap below.
    figure.suptitle(title, y=1 - HAZARD_RATIO_GRID_TITLE_INSET_INCHES / figure.get_figheight(),
                    fontsize=headings.figure_title, fontweight="bold", fontfamily=headings.family,
                    color=publication_style().colors["text"], gid="figure-title")
    figure.tight_layout(pad=0.75, h_pad=ebm_row_gap_fraction(), w_pad=0.8)
    for axis in axes.flat:
        fit_numeric_reference_labels(axis)
    figure.tight_layout(pad=0.75, h_pad=ebm_row_gap_fraction(), w_pad=0.8)
    position_figure_title(figure, "terms_hazard_ratio_grid")
    with plt.rc_context(VECTOR_FONT_RC):
        figure.savefig(png_path)
        figure.savefig(svg_path)
    plt.close(figure)


def female_normal_ranges(ranges: tuple[NormalRange, ...]) -> tuple[NormalRange, ...]:
    return tuple(normal_range for normal_range in ranges if normal_range.sex == "female")


def univariate_terms_by_importance(explanation: Any) -> list[tuple[int, str]]:
    overall_data = explanation.data()
    names = [str(name) for name in sequence_value(overall_data["names"])]
    scores = float_array(overall_data["scores"])
    order = np.argsort(scores)[::-1]
    return [
        (int(term_index), names[int(term_index)])
        for term_index in order
        if explanation.data(int(term_index)).get("type") != "interaction"
    ]


def format_grid_axis(axis: plt.Axes) -> None:
    axis.title.set_fontsize(8.0)
    axis.xaxis.label.set_fontsize(7.0)
    axis.yaxis.label.set_fontsize(7.0)
    axis.tick_params(axis="both", labelsize=6.0)
    major_ticks = grid_major_y_ticks(axis.get_ylim())
    minor_ticks = grid_minor_y_ticks(axis.get_ylim(), major_ticks)
    axis.yaxis.set_major_locator(FixedLocator(major_ticks))
    axis.yaxis.set_minor_locator(FixedLocator(minor_ticks))
    format_numeric_axis(axis.yaxis)
    axis.tick_params(axis="y", which="major", length=GRID_MAJOR_Y_TICK_LENGTH)
    axis.tick_params(axis="y", which="minor", length=GRID_MINOR_Y_TICK_LENGTH)
    format_grid_lines(axis, major_ticks)
    for label in axis.get_xticklabels():
        label.set_rotation(45)
        label.set_ha("right")


def format_grid_lines(axis: plt.Axes, major_y_ticks: np.ndarray) -> None:
    line_style = {
        "linestyle": "-",
        "linewidth": GRID_LINE_WIDTH,
    }
    axis.grid(
        True,
        axis="x",
        which="both",
        color=GRID_EMPHASIZED_LINE_COLOR,
        **line_style,
    )
    axis.grid(
        True,
        axis="y",
        which="both",
        color=GRID_BACKGROUND_LINE_COLOR,
        **line_style,
    )
    ticks = axis.yaxis.get_major_ticks(len(major_y_ticks))
    for value, tick in zip(major_y_ticks, ticks, strict=True):
        if np.isclose(value, round(value)):
            tick.gridline.set_color(GRID_EMPHASIZED_LINE_COLOR)


def grid_major_y_ticks(limits: tuple[float, float]) -> np.ndarray:
    lower, upper = sorted(float(limit) for limit in limits)
    fractional_ticks = [
        tick for tick in GRID_FRACTIONAL_MAJOR_Y_TICKS if lower <= tick <= upper
    ]
    integer_ticks = positive_fixed_ticks(limits, 1.0)
    return np.asarray(
        sorted({lower, upper, *fractional_ticks, *integer_ticks}),
        dtype=float,
    )


def grid_minor_y_ticks(
    limits: tuple[float, float],
    major_ticks: np.ndarray,
) -> np.ndarray:
    ticks = positive_fixed_ticks(limits, UNIVARIATE_Y_TICK_STEP)
    return ticks[~np.isclose(ticks[:, np.newaxis], major_ticks).any(axis=1)]


def write_interaction_plot(
    term_data: Mapping[str, object],
    term_name: str,
    risk_score_label: str,
    output_path: Path,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None = None,
    test_name_translation: Mapping[str, str] | None = None,
) -> None:
    _write_interaction_plot(
        term_data, term_name, risk_score_label, output_path,
        normal_ranges_by_feature, test_name_translation, hazard_ratio_scale=False,
    )


def write_interaction_hazard_ratio_plot(
    term_data: Mapping[str, object],
    term_name: str,
    hazard_ratio_label: str,
    output_path: Path,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None = None,
    test_name_translation: Mapping[str, str] | None = None,
) -> None:
    _write_interaction_plot(
        term_data, term_name, hazard_ratio_label, output_path,
        normal_ranges_by_feature, test_name_translation, hazard_ratio_scale=True,
    )


def _write_interaction_plot(
    term_data: Mapping[str, object],
    term_name: str,
    colorbar_label: str,
    output_path: Path,
    normal_ranges_by_feature: Mapping[str, tuple[NormalRange, ...]] | None,
    test_name_translation: Mapping[str, str] | None,
    *,
    hazard_ratio_scale: bool,
) -> None:
    """Render the shared score heatmap with score or hazard-ratio colorbar labels."""
    scores = float_array(term_data["scores"])
    left_label, right_label = interaction_axis_labels(term_name)
    left_display = feature_display_label(left_label, translation_mapping(test_name_translation))
    right_display = feature_display_label(right_label, translation_mapping(test_name_translation))
    color_limit = symmetric_color_limit(scores)
    extent = interaction_extent(term_data, scores)
    figure, axis = plt.subplots(figsize=INTERACTION_SIZE, dpi=PNG_DPI)
    image = axis.imshow(
        scores.T,
        origin="lower",
        aspect="auto",
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        extent=extent,
    )
    axis.set_xlabel(left_display.x_axis_label)
    axis.set_ylabel(right_display.x_axis_label)
    axis.set_title(interaction_display_title(left_display, right_display))
    if extent is not None:
        add_interaction_normal_ranges(axis, left_label, right_label, normal_ranges_by_feature)
        add_deduplicated_legend(axis)
    colorbar = figure.colorbar(image, ax=axis)
    if hazard_ratio_scale:
        set_hazard_ratio_colorbar(colorbar, color_limit, colorbar_label)
    else:
        colorbar.set_label(colorbar_label)
    figure.tight_layout()
    figure.savefig(output_path)
    plt.close(figure)


def univariate_x_values(names: Sequence[object], score_count: int) -> tuple[np.ndarray, list[str] | None]:
    numeric = optional_float_array(names)
    if numeric is not None and len(numeric) == score_count + 1:
        return (numeric[:-1] + numeric[1:]) / 2.0, None
    if numeric is not None and len(numeric) == score_count:
        return numeric, None
    return np.arange(score_count, dtype=float), [str(name) for name in names[:score_count]]


def univariate_yerr(term_data: Mapping[str, object], scores: np.ndarray) -> np.ndarray | None:
    bounds = univariate_score_bounds(term_data, scores)
    if bounds is None:
        return None
    lower, upper = bounds
    return np.vstack((np.maximum(0.0, scores - lower), np.maximum(0.0, upper - scores)))


def univariate_score_bounds(
    term_data: Mapping[str, object],
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if "lower_bounds" not in term_data or "upper_bounds" not in term_data:
        return None
    lower = float_array(term_data["lower_bounds"])
    upper = float_array(term_data["upper_bounds"])
    if lower.shape != scores.shape or upper.shape != scores.shape:
        return None
    return lower, upper


def univariate_hazard_ratio_yerr(
    term_data: Mapping[str, object],
    scores: np.ndarray,
    hazard_ratios: np.ndarray,
) -> np.ndarray | None:
    bounds = univariate_score_bounds(term_data, scores)
    if bounds is None:
        return None
    lower, upper = bounds
    lower_ratios = hazard_ratio_values(lower)
    upper_ratios = hazard_ratio_values(upper)
    return np.vstack(
        (
            np.maximum(0.0, hazard_ratios - lower_ratios),
            np.maximum(0.0, upper_ratios - hazard_ratios),
        )
    )


def add_missing_bin_marker(axis: plt.Axes, missing_score: tuple[float, float] | None) -> None:
    if missing_score is None:
        return
    score, score_std = missing_score
    axis.axhline(score, color="#377eb8", linestyle="--", linewidth=1.0, label="missing bin")
    if np.isfinite(score_std):
        axis.axhspan(score - score_std, score + score_std, color="#377eb8", alpha=0.12)


def add_missing_hazard_ratio_marker(axis: plt.Axes, missing_score: tuple[float, float] | None) -> None:
    if missing_score is None:
        return
    score, score_std = missing_score
    hazard_ratio = float(hazard_ratio_values(np.asarray([score]))[0])
    if not np.isfinite(hazard_ratio) or hazard_ratio <= 0.0:
        return
    axis.axhline(hazard_ratio, color="#377eb8", linestyle="--", linewidth=1.0, label="missing bin")
    if np.isfinite(score_std):
        lower = float(hazard_ratio_values(np.asarray([score - score_std]))[0])
        upper = float(hazard_ratio_values(np.asarray([score + score_std]))[0])
        if np.isfinite(lower) and np.isfinite(upper) and lower > 0.0:
            axis.axhspan(lower, upper, color="#377eb8", alpha=0.12)


def missing_bin_score(
    model: ExplainableBoostingRegressor,
    term_index: int,
) -> tuple[float, float] | None:
    term_scores = getattr(model, "term_scores_", ())
    if term_index >= len(term_scores):
        return None
    scores = float_array(term_scores[term_index])
    if scores.ndim != 1 or len(scores) == 0:
        return None
    return float(scores[0]), missing_bin_std(model, term_index)


def missing_bin_std(model: ExplainableBoostingRegressor, term_index: int) -> float:
    standard_deviations = getattr(model, "standard_deviations_", None)
    if standard_deviations is None or term_index >= len(standard_deviations):
        return float("nan")
    stds = float_array(standard_deviations[term_index])
    if stds.ndim != 1 or len(stds) == 0:
        return float("nan")
    return float(stds[0])


def interaction_extent(term_data: Mapping[str, object], scores: np.ndarray) -> tuple[float, float, float, float] | None:
    left = optional_float_array(sequence_value(term_data["left_names"]))
    right = optional_float_array(sequence_value(term_data["right_names"]))
    if left is None or right is None:
        return None
    if len(left) == scores.shape[0] + 1 and len(right) == scores.shape[1] + 1:
        return float(left[0]), float(left[-1]), float(right[0]), float(right[-1])
    return None


def interaction_axis_labels(term_name: str) -> tuple[str, str]:
    labels = term_name.split(" & ", maxsplit=1)
    if len(labels) == 2:
        return labels[0], labels[1]
    return "Feature 1", "Feature 2"


def apply_category_labels(axis: plt.Axes, x_values: np.ndarray, labels: list[str] | None) -> None:
    if labels is None:
        return
    axis.set_xticks(x_values)
    axis.set_xticklabels(labels, rotation=45, ha="right")


def apply_univariate_x_axis_scale(
    axis: plt.Axes,
    term_name: str,
    x_values: np.ndarray,
    labels: list[str] | None,
    valid: np.ndarray,
    x_axis_scale: str,
) -> None:
    if x_axis_scale == "linear" or labels is not None:
        return
    if term_name in LINEAR_X_AXIS_FEATURE_NAMES or term_name.endswith(BLOOD_TEST_CHANGE_SUFFIX):
        return
    plotted_x_values = x_values[valid]
    if len(plotted_x_values) == 0 or bool(np.any(plotted_x_values <= 0.0)):
        LOGGER.warning("could not use log x-axis for term `%s`; plotted x values must be positive", term_name)
        return
    if plotted_value_ratio(plotted_x_values) < LOG_X_AXIS_MIN_VALUE_RATIO:
        return
    axis.set_xscale("log")


def plotted_value_ratio(values: np.ndarray) -> float:
    finite_values = values[np.isfinite(values)]
    if len(finite_values) < 2:
        return 1.0
    lower = float(np.min(finite_values))
    upper = float(np.max(finite_values))
    if lower <= 0.0:
        return 1.0
    return upper / lower


def format_univariate_axes(
    axis: plt.Axes,
    term_name: str,
    labels: list[str] | None,
) -> None:
    format_univariate_y_axis(axis)
    if labels is None:
        if term_name in TIME_SINCE_DIAGNOSIS_FEATURE_NAMES:
            format_time_since_diagnosis_x_axis(axis)
        format_numeric_axis(axis.xaxis)
        label_sparse_log_x_ticks(axis)


def label_sparse_log_x_ticks(axis: plt.Axes) -> None:
    """Label outermost visible log ticks when fewer than three values are shown."""
    if axis.get_xscale() != "log":
        return
    lower, upper = sorted(axis.get_xlim())
    major_ticks = axis.get_xticks()
    visible_major_ticks = major_ticks[(major_ticks >= lower) & (major_ticks <= upper)]
    if len(visible_major_ticks) >= 3:
        return
    if len(axis.get_xticks(minor=True)) == 0:
        # Matplotlib may suppress the log grid when its automatic stride exceeds one decade.
        axis.xaxis.set_minor_locator(LogLocator(subs="all", numticks=100))
    grid_ticks = np.union1d(major_ticks, axis.get_xticks(minor=True))
    visible_grid_ticks = grid_ticks[(grid_ticks >= lower) & (grid_ticks <= upper)]
    if len(visible_grid_ticks) == 0:
        return
    labelled_ticks = np.union1d(visible_major_ticks, visible_grid_ticks[[0, -1]])
    if len(labelled_ticks) < 3:
        # An outermost tick may already be labelled; retain an interior value too.
        log_ticks = np.log10(visible_grid_ticks)
        midpoint = (log_ticks[0] + log_ticks[-1]) / 2.0
        middle_tick = visible_grid_ticks[np.argmin(np.abs(log_ticks - midpoint))]
        labelled_ticks = np.union1d(labelled_ticks, [middle_tick])
    axis.xaxis.set_major_locator(FixedLocator(labelled_ticks))


def format_time_since_diagnosis_x_axis(axis: plt.Axes) -> None:
    """Label five-year ticks while retaining annual marks and gridlines."""
    axis.xaxis.set_major_locator(MultipleLocator(TIME_SINCE_DIAGNOSIS_LABEL_TICK_STEP))
    axis.xaxis.set_minor_locator(MultipleLocator(TIME_SINCE_DIAGNOSIS_YEAR_TICK_STEP))
    axis.grid(
        True,
        axis="x",
        which="minor",
        linestyle=":",
        color=GRID_BACKGROUND_LINE_COLOR,
    )


def format_univariate_y_axis(axis: plt.Axes) -> None:
    if axis.get_yscale() == "log":
        limits = axis.get_ylim()
        first_tick, last_tick = positive_fixed_tick_indices(limits, UNIVARIATE_Y_TICK_STEP)
        tick_count = max(0, last_tick - first_tick + 1)
        if 0 < tick_count <= MAX_UNIVARIATE_FIXED_Y_TICKS:
            y_ticks = positive_fixed_ticks(limits, UNIVARIATE_Y_TICK_STEP)
        else:
            y_ticks = sparse_log_ticks(limits, UNIVARIATE_LOG_Y_TICKS)
        axis.yaxis.set_major_locator(FixedLocator(y_ticks))
    else:
        axis.yaxis.set_major_locator(MultipleLocator(UNIVARIATE_Y_TICK_STEP))
    format_numeric_axis(axis.yaxis)


def format_numeric_axis(axis: Axis) -> None:
    axis.set_major_formatter(NUMERIC_TICK_FORMATTER)
    axis.set_minor_formatter(NullFormatter())


def positive_fixed_ticks(limits: tuple[float, float], step: float) -> np.ndarray:
    first_tick, last_tick = positive_fixed_tick_indices(limits, step)
    if last_tick < first_tick:
        return np.asarray([], dtype=float)
    tick_count = last_tick - first_tick + 1
    if tick_count > MAX_SAFE_FIXED_Y_TICKS:
        raise ValueError(f"Refusing to allocate {tick_count:,} fixed y-axis ticks.")
    return np.round(np.arange(first_tick, last_tick + 1, dtype=float) * step, 10)


def positive_fixed_tick_indices(
    limits: tuple[float, float],
    step: float,
) -> tuple[int, int]:
    """Return inclusive positive fixed-tick indices without allocating them."""
    lower, upper = sorted(float(limit) for limit in limits)
    if step <= 0.0 or not np.all(np.isfinite((lower, upper))):
        raise ValueError("Fixed tick limits must be finite and the step must be positive.")
    first_tick = max(1, int(np.ceil((lower - 1e-9) / step)))
    last_tick = int(np.floor((upper + 1e-9) / step))
    return first_tick, last_tick


def sparse_log_ticks(
    limits: tuple[float, float],
    max_tick_count: int,
) -> np.ndarray:
    """Return a bounded set of power-of-ten ticks within positive limits."""
    lower, upper = sorted(float(limit) for limit in limits)
    if lower <= 0.0 or not np.all(np.isfinite((lower, upper))):
        raise ValueError("Log tick limits must be finite and positive.")
    if max_tick_count < 1:
        raise ValueError("The maximum log tick count must be positive.")
    first_exponent = int(np.ceil(np.log10(lower)))
    last_exponent = int(np.floor(np.log10(upper)))
    if last_exponent < first_exponent:
        midpoint_exponent = (np.log10(lower) + np.log10(upper)) / 2.0
        return np.asarray([10.0**midpoint_exponent], dtype=float)
    exponents = np.arange(first_exponent, last_exponent + 1, dtype=int)
    if len(exponents) > max_tick_count:
        indices = np.linspace(0, len(exponents) - 1, num=max_tick_count, dtype=int)
        exponents = exponents[np.unique(indices)]
        if first_exponent <= 0 <= last_exponent:
            exponents[np.argmin(np.abs(exponents))] = 0
    return np.power(10.0, exponents.astype(float))


def numeric_tick_label(value: float) -> str:
    if not np.isfinite(value):
        return ""
    rounded = round(float(value), 10)
    if abs(rounded) < 1e-10:
        rounded = 0.0
    label = f"{rounded:.10f}".rstrip("0").rstrip(".")
    return label if label else "0"


def normalized_term_x_axis_scale(value: str) -> str:
    scale = str(value).strip().lower()
    if scale not in TERM_X_AXIS_SCALES:
        allowed = ", ".join(TERM_X_AXIS_SCALES)
        raise ValueError(f"`term_x_axis_scale` must be one of: {allowed}.")
    return scale


def translation_mapping(test_name_translation: Mapping[str, str] | None) -> Mapping[str, str]:
    return {} if test_name_translation is None else test_name_translation


def interaction_display_title(left_label: FeatureDisplayLabel, right_label: FeatureDisplayLabel) -> str:
    return f"{left_label.title} & {right_label.title}"


def test_name_translation_metadata(
    test_name_translation: Mapping[str, str],
    csv_path: Path | None,
) -> dict[str, object]:
    return {
        "csv_path": None if csv_path is None else str(csv_path),
        "translation_count": len(test_name_translation),
    }


def symmetric_color_limit(values: np.ndarray) -> float:
    finite_values = np.abs(values[np.isfinite(values)])
    if len(finite_values) == 0:
        return 1.0
    limit = float(np.max(finite_values))
    return limit if limit > 0.0 else 1.0


def set_hazard_ratio_colorbar(colorbar: Any, color_limit: float, hazard_ratio_label: str) -> None:
    ticks = np.linspace(-color_limit, color_limit, num=5)
    colorbar.set_ticks(ticks)
    colorbar.set_ticklabels([hazard_ratio_tick_label(tick) for tick in ticks])
    colorbar.set_label(hazard_ratio_label)


def hazard_ratio_tick_label(score: float) -> str:
    ratio = float(hazard_ratio_values(np.asarray([score]))[0])
    return "inf" if not np.isfinite(ratio) else f"{ratio:.3g}x"


def hazard_ratio_values(scores: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        return np.exp(scores)


def term_artifact(term_index: int, term_name: str, path: Path, root: Path) -> dict[str, object]:
    return {"term_index": term_index, "term_name": term_name, "path": str(path.relative_to(root))}


def human_label(value: str) -> str:
    return value.replace("_", " ").strip().title() or "Configured Endpoint"


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[\\/\s]+", "_", value.strip())
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", cleaned)
    return cleaned or "term"
