from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.collections import FillBetweenPolyCollection, LineCollection
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import pytest
from interpret.glassbox import ExplainableBoostingRegressor

from src.data.blood_test_display import feature_display_label, load_test_name_translation
from src.model import ebm_visualizations
from src.model.ebm_global_importance import (
    GLOBAL_IMPORTANCE_EXTRA_HEIGHT,
    GLOBAL_IMPORTANCE_ROW_HEIGHT,
    PNG_DPI,
    draw_global_importance_axis,
    write_global_importance_plot,
)
from src.model.ebm_visualizations import (
    write_ebm_explanation_plots,
)
from src.model.normal_range_plotting import (
    add_interaction_normal_ranges,
    add_univariate_normal_ranges,
    normal_range_color,
)
from src.model.normal_ranges import NormalRange, load_normal_ranges


def test_ebm_explanation_plots_include_main_effects_and_interactions(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    features = rng.normal(size=(90, 2))
    features[:6, 0] = np.nan
    event_signal = np.nan_to_num(features[:, 0]) * features[:, 1]
    events = event_signal > np.median(event_signal)
    durations = rng.integers(30, 365, size=len(events))
    targets = np.where(events, durations, -durations).astype(float)
    model = ExplainableBoostingRegressor(
        feature_names=["marker_a", "marker_b"],
        objective="survival_cox",
        interactions=1,
        outer_bags=2,
        validation_size=0.2,
        max_rounds=8,
        random_state=7,
    )
    model.fit(features, targets)

    artifacts = write_ebm_explanation_plots(
        model,
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=True,
    )

    assert (tmp_path / artifacts["global_importance_png"]).is_file()
    assert (tmp_path / artifacts["global_importance_svg"]).is_file()
    assert (tmp_path / artifacts["combined_term_hazard_ratio_plot_png"]).is_file()
    assert (tmp_path / artifacts["combined_term_hazard_ratio_plot_svg"]).is_file()
    assert (tmp_path / artifacts["term_scores_csv"]).is_file()
    assert (tmp_path / artifacts["interaction_scores_csv"]).is_file()
    assert len(artifacts["term_plots"]) == 2
    assert len(artifacts["term_hazard_ratio_plots"]) == 2
    assert len(artifacts["interaction_plots"]) == 1
    assert len(artifacts["interaction_hazard_ratio_plots"]) == 1
    all_plot_artifacts = [
        *artifacts["term_plots"],
        *artifacts["term_hazard_ratio_plots"],
        *artifacts["interaction_plots"],
        *artifacts["interaction_hazard_ratio_plots"],
    ]
    for artifact in all_plot_artifacts:
        path = tmp_path / str(artifact["path"])
        assert path.is_file()
        assert path.stat().st_size > 0

    term_scores = pd.read_csv(tmp_path / artifacts["term_scores_csv"])
    interaction_scores = pd.read_csv(tmp_path / artifacts["interaction_scores_csv"])
    expected_term_columns = {
        "outer_bag_score_std",
        "score_lower",
        "score_upper",
        "hazard_ratio",
        "hazard_ratio_lower",
        "hazard_ratio_upper",
        "is_missing",
    }
    assert expected_term_columns.issubset(term_scores.columns)
    assert bool(term_scores["is_missing"].any())
    assert {
        "outer_bag_score_std",
        "score_lower",
        "score_upper",
        "hazard_ratio",
        "hazard_ratio_lower",
        "hazard_ratio_upper",
    }.issubset(interaction_scores.columns)
    assert len(interaction_scores) > 0
    np.testing.assert_allclose(term_scores["hazard_ratio"], np.exp(term_scores["score"]))
    np.testing.assert_allclose(interaction_scores["hazard_ratio"], np.exp(interaction_scores["score"]))


def test_combined_hazard_ratio_grid_uses_importance_order_and_fixed_axis(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig
    normal_ranges = {
        "P-ALAT|||u/l": visible_normal_range_pair("P-ALAT", "u/l"),
        "P-Kalsium|||mmol/l": age_specific_normal_ranges("P-Kalsium", "mmol/l"),
    }
    male_colors = {
        normal_range_color(normal_range)
        for ranges in normal_ranges.values()
        for normal_range in ranges
        if normal_range.sex == "male"
    }
    female_colors = {
        normal_range_color(normal_range)
        for ranges in normal_ranges.values()
        for normal_range in ranges
        if normal_range.sex == "female"
    }
    multiple_reference_colors = {
        normal_range_color(normal_range)
        for normal_range in normal_ranges["P-Kalsium|||mmol/l"]
        if normal_range.sex == "female"
    }

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.name == "terms_hazard_ratio_grid.png":
            figure.canvas.draw()
            snapshots.append(
                {
                    "figure_size": tuple(figure.get_size_inches()),
                    "axes": [grid_axis_snapshot(axis) for axis in figure.axes],
                    **grid_legend_snapshot(figure),
                    **grid_style_snapshot(figure),
                }
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    artifacts = write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=GridTermExplanation(),
        test_name_translation={
            "P-ALAT": "ALT",
            "P-Kalsium": "Calcium",
            "P-Magnesium": "Magnesium",
            "S-CRP": "C-reactive protein",
        },
        normal_ranges_by_feature=normal_ranges,
        combined_term_hazard_ratio_columns=3,
        endpoint_name="bone_recurrence",
    )

    assert (tmp_path / artifacts["combined_term_hazard_ratio_plot_png"]).is_file()
    assert (tmp_path / artifacts["combined_term_hazard_ratio_plot_svg"]).is_file()
    assert len(snapshots) == 1
    grid_svg = (tmp_path / artifacts["combined_term_hazard_ratio_plot_svg"]).read_text()
    importance_svg = (tmp_path / artifacts["global_importance_svg"]).read_text()
    assert "EBM feature effects on relative hazard — Bone recurrence" in grid_svg
    assert "Hazard ratio" in grid_svg
    assert "EBM feature importance — Bone recurrence" in importance_svg
    assert "<text" in grid_svg and "<text" in importance_svg
    assert snapshots[0]["figure_size"][0] == pytest.approx(7.65)
    assert snapshots[0]["figure_size"][1] == pytest.approx(5.1)  # Inset legends need no extra header space.
    grid_axes = snapshots[0]["axes"]
    assert [" ".join(axis["title"].split()) for axis in grid_axes[:4]] == [
        "1. ALT",
        "2. Calcium",
        "3. Magnesium",
        "4. C-reactive protein",
    ]
    assert [axis["xlabel"] for axis in grid_axes[:4]] == ["U/L", "mmol/L", "mmol/L", "mg/L"]
    assert grid_axes[4]["visible"] is False
    assert grid_axes[4]["has_legend"] is False
    assert grid_axes[5]["visible"] is False
    for axis in grid_axes[:4]:
        np.testing.assert_allclose(axis["ylim"], (0.3, 4.0))
        all_ticks = np.sort(
            np.concatenate((axis["y_major_ticks"], axis["y_minor_ticks"]))
        )
        np.testing.assert_allclose(np.diff(all_ticks), 0.1)
        assert {0.4, 0.6, 2.1, 3.9}.issubset(set(axis["y_minor_ticks"]))
        assert axis["y_labels"] == ["0.3", "0.5", "1", "2", "3", "4"]
        assert axis["y_minor_tick_length"] < axis["y_major_tick_length"]
        assert axis["has_legend"] is False
    assert snapshots[0]["figure_legend_labels"] == []
    assert snapshots[0]["axis_legend_labels"] == []
    assert snapshots[0]["figure_legend_line_count"] == 0
    assert snapshots[0]["figure_legend_patch_count"] == 0
    assert snapshots[0]["axis_legend_line_count"] == 0
    assert snapshots[0]["axis_legend_patch_count"] == 0
    assert snapshots[0]["patch_labels"] == ["Female adult reference"]
    assert snapshots[0]["patch_edge_alphas"] == [0.0]
    assert male_colors.isdisjoint(snapshots[0]["line_colors"])
    reference_lines = [
        line
        for line in snapshots[0]["line_segments"]
        if line["color"] in female_colors
    ]
    assert len(reference_lines) == 2
    assert {line["color"] for line in reference_lines} == multiple_reference_colors
    assert all(line["orientation"] == "horizontal" for line in reference_lines)
    calcium_levels = {
        line["position"] for line in reference_lines if line["axis_index"] == 1
    }
    assert len(calcium_levels) == 2
    assert "x" not in snapshots[0]["markers"]
    assert snapshots[0]["fill_count"] == 4
    assert snapshots[0]["error_bar_count"] == 0


def test_combined_hazard_ratio_grid_labels_integers_within_custom_y_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    y_labels = []
    original_savefig = plt.Figure.savefig

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        if Path(args[0]).name == "terms_hazard_ratio_grid.png":
            figure.canvas.draw()
            y_labels.extend(
                label.get_text()
                for label in figure.axes[0].get_yticklabels()
                if label.get_text()
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=SingleTermExplanation("marker|||u/l"),
        combined_term_hazard_ratio_y_limits=(0.2, 5.0),
    )

    assert y_labels == ["0.2", "0.5", "1", "2", "3", "4", "5"]


def test_combined_hazard_ratio_grid_uses_solid_emphasized_gridlines(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        if Path(args[0]).name == "terms_hazard_ratio_grid.png":
            figure.canvas.draw()
            axis = figure.axes[0]
            major_y_values = axis.get_yticks()
            major_y_ticks = axis.yaxis.get_major_ticks(len(major_y_values))
            snapshots.append(
                {
                    "whole_y": [
                        gridline_snapshot(tick.gridline)
                        for value, tick in zip(
                            major_y_values, major_y_ticks, strict=True
                        )
                        if np.isclose(value, round(value))
                    ],
                    "other_major_y": [
                        gridline_snapshot(tick.gridline)
                        for value, tick in zip(
                            major_y_values, major_y_ticks, strict=True
                        )
                        if not np.isclose(value, round(value))
                    ],
                    "minor_y": [
                        gridline_snapshot(tick.gridline)
                        for tick in axis.yaxis.get_minor_ticks()
                    ],
                    "vertical_major": [
                        gridline_snapshot(tick.gridline)
                        for tick in axis.xaxis.get_major_ticks()
                    ],
                    "vertical_minor": [
                        gridline_snapshot(tick.gridline)
                        for tick in axis.xaxis.get_minor_ticks()
                    ],
                }
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=WideRangeTermExplanation("marker|||u/l"),
        term_x_axis_scale="log",
        combined_term_hazard_ratio_y_limits=(0.2, 5.0),
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    emphasized_lines = [
        *snapshot["whole_y"],
        *snapshot["vertical_major"],
        *snapshot["vertical_minor"],
    ]
    background_lines = [*snapshot["other_major_y"], *snapshot["minor_y"]]
    assert emphasized_lines
    assert background_lines
    assert snapshot["vertical_minor"]
    assert {line["color"] for line in emphasized_lines} == {
        snapshot["whole_y"][0]["color"]
    }
    assert {line["color"] for line in background_lines} == {
        snapshot["minor_y"][0]["color"]
    }
    assert np.mean(to_rgb(snapshot["whole_y"][0]["color"])) < np.mean(
        to_rgb(snapshot["minor_y"][0]["color"])
    )
    assert np.mean(to_rgb(snapshot["whole_y"][0]["color"])) <= 0.6
    assert np.mean(to_rgb(snapshot["minor_y"][0]["color"])) <= 0.8
    all_lines = [*emphasized_lines, *background_lines]
    assert all(line["visible"] for line in all_lines)
    assert {line["linestyle"] for line in all_lines} == {"-"}
    assert max(line["linewidth"] for line in all_lines) <= 0.5


def test_combined_grid_omits_the_standalone_hr_one_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_counts: dict[str, int] = {}
    original_savefig = plt.Figure.savefig

    def capture_baselines(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        plot_kind = None
        if output_path.parent.name == "terms_hazard_ratio":
            plot_kind = "standalone"
        elif output_path.name == "terms_hazard_ratio_grid.png":
            plot_kind = "grid"
        if plot_kind is not None:
            baseline_counts[plot_kind] = sum(
                line.get_linestyle() == "--"
                and np.allclose(np.asarray(line.get_ydata(), dtype=float), 1.0)
                for axis in figure.axes
                for line in axis.lines
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_baselines)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=SingleTermExplanation("marker|||u/l"),
    )

    assert baseline_counts == {"standalone": 1, "grid": 0}


def test_combined_hazard_ratio_grid_style_can_restore_markers_and_error_bars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.name == "terms_hazard_ratio_grid.png":
            snapshots.append(grid_style_snapshot(figure))
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=UncertainGridTermExplanation(),
        combined_term_hazard_ratio_show_x_markers=True,
        combined_term_hazard_ratio_show_uncertainty_area=False,
    )

    assert len(snapshots) == 1
    assert "x" in snapshots[0]["markers"]
    assert snapshots[0]["fill_count"] == 0
    assert snapshots[0]["error_bar_count"] == 1


def test_combined_hazard_ratio_grid_keeps_local_intervals_without_shared_legend_space(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.name == "terms_hazard_ratio_grid.png":
            snapshots.append(
                {
                    "axis_count": len(figure.axes),
                    "figure_legend_count": len(figure.legends),
                    "axis_legend_count": sum(axis.get_legend() is not None for axis in figure.axes),
                    "interval_labels": [
                        text.get_text()
                        for axis in figure.axes
                        for artist in axis.artists
                        if artist.get_gid() == "numeric-reference-intervals"
                        for text in artist.get_texts()
                    ],
                }
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=GridTermExplanation(),
        normal_ranges_by_feature={
            "P-ALAT|||u/l": (
                *reversed(common_reference_group_ranges("P-ALAT", "u/l")),
                NormalRange(
                    "P-ALAT",
                    "u/l",
                    "female",
                    "<=17 years",
                    0.5,
                    1.0,
                    ">=",
                    "<=",
                    "source",
                    "",
                ),
            ),
        },
        combined_term_hazard_ratio_columns=4,
    )

    assert len(snapshots) == 1
    snapshot = snapshots[0]
    assert snapshot["axis_count"] == 4
    assert snapshot["figure_legend_count"] == 0
    assert snapshot["axis_legend_count"] == 0
    assert len(snapshot["interval_labels"]) == 8
    assert "0.5–1 (≤17 years)" in snapshot["interval_labels"]


def test_combined_hazard_ratio_grid_plots_missing_bin_marker_when_model_includes_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig

    def capture_grid_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.name == "terms_hazard_ratio_grid.png":
            figure.canvas.draw()
            snapshots.append(
                [
                    text.get_text()
                    for axis in figure.axes
                    if (legend := axis.get_legend()) is not None
                    for text in legend.get_texts()
                ]
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_grid_savefig)

    write_ebm_explanation_plots(
        MissingBinModel(term_count=1),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=True,
        explanation=SingleTermExplanation("P-ALAT|||u/l"),
        combined_term_hazard_ratio_columns=1,
    )

    assert len(snapshots) == 1
    assert snapshots[0] == ["missing bin"]


def test_global_importance_plot_scales_for_many_covariates(tmp_path: Path) -> None:
    term_count = 100
    output_path = tmp_path / "global_importance.png"
    write_global_importance_plot(
        {
            "names": [f"covariate_{index:03d}_with_readable_name" for index in range(term_count)],
            "scores": np.linspace(0.0, 1.0, term_count),
        },
        output_path,
    )

    image = mpimg.imread(output_path)

    assert output_path.stat().st_size > 0
    expected_height = GLOBAL_IMPORTANCE_EXTRA_HEIGHT + term_count * GLOBAL_IMPORTANCE_ROW_HEIGHT
    assert image.shape[0] >= int(expected_height * PNG_DPI) - 2


@pytest.mark.parametrize("term_count", [1, 2, 3])
def test_compact_global_importance_labels_are_above_bars_inside_plot(term_count: int) -> None:
    figure, axis = plt.subplots(figsize=(3.0, 2.5), dpi=PNG_DPI)
    names = ["ALP", "C-reactive protein", "Time since diagnosis"][:term_count]
    scores = [0.265, 0.263, 0.105][:term_count]
    draw_global_importance_axis(
        axis,
        {"names": names, "scores": scores},
        compact=True,
        color="#262626",
        x_limits=(0.0, 0.5),
    )
    try:
        # Top-feature grids resize tick labels after the importance axis is drawn.
        axis.tick_params(axis="both", labelsize=11)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        labels = axis.get_yticklabels()
        bars = axis.patches
        assert [label.get_text() for label in labels] == names[::-1]
        assert [bar.get_width() for bar in bars] == scores[::-1]
        for label, bar in zip(labels, bars, strict=True):
            bounds = label.get_window_extent(renderer)
            assert bounds.y0 > bar.get_window_extent(renderer).y1
            assert axis.bbox.x0 <= bounds.x0 < bounds.x1 <= axis.bbox.x1
            assert axis.bbox.y0 <= bounds.y0 < bounds.y1 <= axis.bbox.y1
            assert all(not bounds.overlaps(other.get_window_extent(renderer)) for other in bars)
    finally:
        plt.close(figure)


def test_global_importance_plot_uses_unitless_english_bloodtest_labels(
    tmp_path: Path,
    monkeypatch,
) -> None:
    snapshots = []
    original_savefig = plt.Figure.savefig

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        figure.canvas.draw()
        snapshots.append([label.get_text() for label in figure.axes[0].texts
                          if str(label.get_gid()).startswith("importance-feature-")])
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    write_global_importance_plot(
        {
            "names": [
                "P-Magnesium|||mmol/l",
                "P-ALAT|||u/l",
                "P-Magnesium|||mmol/l & P-ALAT|||u/l",
            ],
            "scores": [1.0, 2.0, 3.0],
        },
        tmp_path / "global_importance.png",
        test_name_translation={
            "P-Magnesium": "Magnesium",
            "P-ALAT": "ALT",
        },
    )

    assert snapshots == [["Magnesium & ALT", "ALT", "Magnesium"]]


def test_translation_csv_covers_current_ebm_bloodtest_names() -> None:
    translations = load_test_name_translation(Path("src/config/blood_test_name_translations.csv"))

    assert feature_display_label("CRP|||mg/L", translations).full_label == "C-reactive protein (mg/L)"
    assert feature_display_label("P-Kalium|||mmol/L", translations).full_label == "Potassium (mmol/L)"
    assert feature_display_label("P-Natrium|||mmol/L", translations).full_label == "Sodium (mmol/L)"


def test_clinical_covariate_feature_labels_are_readable() -> None:
    translations = load_test_name_translation(Path("src/config/blood_test_name_translations.csv"))

    age_label = feature_display_label("age_at_diagnosis", translations)
    time_label = feature_display_label("time_since_diagnosis_years", translations)

    assert age_label.title == "Age at diagnosis"
    assert age_label.x_axis_label == "years"
    assert time_label.title == "Time since diagnosis"
    assert time_label.x_axis_label == "years"


def test_ahus_normal_ranges_load_by_model_feature_key() -> None:
    ranges = load_normal_ranges(Path("src/config/ahus_blood_test_normal_ranges.csv"))

    hemoglobin_ranges = ranges["B-Hemoglobin|||g/dl"]

    assert {normal_range.sex for normal_range in hemoglobin_ranges} == {"female", "male"}
    assert hemoglobin_ranges[0].source_url.startswith("https://metodebok.no/")
    assert "P-Kalsium albkorr.|||mmol/l" in ranges
    assert "P-Magnesium|||mmol/l" in ranges


def test_ebm_explanation_term_plots_can_use_log_x_axis(tmp_path: Path, monkeypatch) -> None:
    captured_scales = []
    original_set_xscale = plt.Axes.set_xscale

    def capture_xscale(axis: plt.Axes, value: str, *args: object, **kwargs: object) -> None:
        captured_scales.append(value)
        original_set_xscale(axis, value, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "set_xscale", capture_xscale)

    artifacts = write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=WideRangeTermExplanation("B-Hb|||g/dL"),
        term_x_axis_scale="log",
        write_combined_term_hazard_ratio_plot=False,
    )

    assert artifacts["term_x_axis_scale"] == "log"
    assert captured_scales == ["log", "log"]


def test_ebm_explanation_log_x_axis_skips_clinical_covariates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_scales = []
    original_set_xscale = plt.Axes.set_xscale

    def capture_xscale(axis: plt.Axes, value: str, *args: object, **kwargs: object) -> None:
        captured_scales.append((axis.get_title(), value))
        original_set_xscale(axis, value, *args, **kwargs)

    monkeypatch.setattr(plt.Axes, "set_xscale", capture_xscale)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=ClinicalCovariateExplanation(),
        term_x_axis_scale="log",
        write_combined_term_hazard_ratio_plot=False,
    )

    assert captured_scales == []


def test_ebm_hazard_ratio_grid_labels_clinical_covariates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_axes = []
    original_savefig = plt.Figure.savefig

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.name == "terms_hazard_ratio_grid.png":
            figure.canvas.draw()
            captured_axes.append(
                [
                    (" ".join(axis.get_title().split()), axis.get_xlabel())
                    for axis in figure.axes
                    if axis.get_visible() and axis.get_title()
                ]
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=ClinicalCovariateExplanation(),
        term_x_axis_scale="log",
    )

    assert captured_axes == [
        [
            ("1. Time since diagnosis", "years"),
            ("2. Age at diagnosis", "years"),
        ]
    ]


@pytest.mark.parametrize(
    "term_name",
    ["time_since_diagnosis", "time_since_diagnosis_years"],
)
def test_time_since_diagnosis_plots_label_five_years_and_mark_every_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    term_name: str,
) -> None:
    captured_axes = []
    original_savefig = plt.Figure.savefig

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.suffix == ".png":
            figure.canvas.draw()
            captured_axes.extend(
                axis
                for axis in figure.axes
                if " ".join(axis.get_title().split()) in ("Time since diagnosis", "1. Time since diagnosis")
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=TimeSinceDiagnosisExplanation(term_name),
    )

    assert len(captured_axes) == 3
    for axis in captured_axes:
        x_min, x_max = axis.get_xlim()
        major_ticks = axis.xaxis.get_major_ticks()
        minor_ticks = axis.xaxis.get_minor_ticks()
        visible_major_ticks = [
            tick for tick in major_ticks if x_min <= float(tick.get_loc()) <= x_max
        ]
        visible_minor_ticks = [
            tick for tick in minor_ticks if x_min <= float(tick.get_loc()) <= x_max
        ]
        annual_positions = sorted(
            float(tick.get_loc())
            for tick in (*visible_major_ticks, *visible_minor_ticks)
        )

        np.testing.assert_allclose(np.diff(annual_positions), 1.0)
        assert all(np.isclose(tick.get_loc() % 5.0, 0.0) for tick in visible_major_ticks)
        assert all(tick.label1.get_text() for tick in visible_major_ticks)
        assert all(not tick.label1.get_text() for tick in visible_minor_ticks)
        assert all(tick.tick1line.get_visible() for tick in visible_major_ticks)
        assert all(tick.tick1line.get_visible() for tick in visible_minor_ticks)
        assert all(tick.gridline.get_visible() for tick in visible_major_ticks)
        assert all(tick.gridline.get_visible() for tick in visible_minor_ticks)


def test_ebm_explanation_log_x_axis_keeps_narrow_lab_ranges_readable(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_axes = []
    original_savefig = plt.Figure.savefig

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        figure.canvas.draw()
        output_path = Path(args[0])
        if output_path.name.startswith("000_"):
            axis = figure.axes[0]
            captured_axes.append(
                {
                    "xscale": axis.get_xscale(),
                    "x_labels": [label.get_text() for label in axis.get_xticklabels() if label.get_text()],
                }
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=NarrowRangeTermExplanation("S-Kalium|||mmol/l"),
        term_x_axis_scale="log",
        write_combined_term_hazard_ratio_plot=False,
    )

    assert len(captured_axes) == 2
    for snapshot in captured_axes:
        assert snapshot["xscale"] == "linear"
        assert any(label for label in snapshot["x_labels"])


def test_ebm_explanation_term_plots_use_decimal_tick_labels(tmp_path: Path, monkeypatch) -> None:
    captured_axes = []
    original_savefig = plt.Figure.savefig

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        figure.canvas.draw()
        axis = figure.axes[0]
        output_path = Path(args[0])
        if axis.get_title() == "B-Hb" and output_path.name.startswith("000_"):
            captured_axes.append(axis_snapshot(axis))
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Distant Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=MultiPointTermExplanation("B-Hb|||g/dL"),
        term_x_axis_scale="log",
    )

    assert len(captured_axes) == 2
    for snapshot in captured_axes:
        y_tick_diffs = np.diff(snapshot["y_ticks"])
        np.testing.assert_allclose(y_tick_diffs, 0.1)
        assert all("^" not in label for label in snapshot["x_labels"])
        assert all("^" not in label for label in snapshot["y_labels"])
        assert all("mathdefault" not in label for label in snapshot["x_labels"])
        assert all("mathdefault" not in label for label in snapshot["y_labels"])
    assert "1" in captured_axes[1]["y_labels"]


def test_extreme_hazard_ratio_plot_does_not_build_dense_ticks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snapshots = []
    original_positive_fixed_ticks = ebm_visualizations.positive_fixed_ticks
    original_savefig = plt.Figure.savefig

    def reject_excessive_ticks(limits: tuple[float, float], step: float) -> np.ndarray:
        lower, upper = sorted(float(limit) for limit in limits)
        first_tick = max(1, int(np.ceil((lower - 1e-9) / step)))
        last_tick = int(np.floor((upper + 1e-9) / step))
        assert last_tick - first_tick + 1 <= 1_000
        return original_positive_fixed_ticks(limits, step)

    def capture_savefig(figure: plt.Figure, *args: object, **kwargs: object) -> None:
        output_path = Path(args[0])
        if output_path.parent.name == "terms_hazard_ratio":
            figure.canvas.draw()
            axis = figure.axes[0]
            snapshots.append(
                {
                    "yscale": axis.get_yscale(),
                    "y_ticks": axis.get_yticks(),
                }
            )
        original_savefig(figure, *args, **kwargs)

    monkeypatch.setattr(ebm_visualizations, "positive_fixed_ticks", reject_excessive_ticks)
    monkeypatch.setattr(plt.Figure, "savefig", capture_savefig)

    artifacts = write_ebm_explanation_plots(
        EmptyModel(),
        tmp_path / "explanations",
        "Brain Recurrence EBM Cox Risk-Score Contribution",
        include_missing_bins=False,
        explanation=ExtremeUncertainTermExplanation(),
        term_x_axis_scale="log",
        write_combined_term_hazard_ratio_plot=False,
    )

    hazard_path = tmp_path / str(artifacts["term_hazard_ratio_plots"][0]["path"])
    assert hazard_path.is_file()
    assert snapshots[0]["yscale"] == "log"
    assert len(snapshots[0]["y_ticks"]) <= ebm_visualizations.UNIVARIATE_LOG_Y_TICKS


def test_normal_range_guides_draw_univariate_reference_marks() -> None:
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-1.0, 1.0)

    add_univariate_normal_ranges(axis, normal_range_pair("A", "u"))

    assert len(axis.patches) == 2
    assert len(axis.lines) == 4
    assert {text.get_text() for text in axis.get_legend().texts} == {
        "Female adult reference",
        "Male adult reference",
    }
    plt.close(figure)


def test_normal_range_guides_collapse_adult_equivalent_labels() -> None:
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-1.0, 1.0)

    add_univariate_normal_ranges(axis, adult_equivalent_normal_ranges("A", "u"))

    assert {text.get_text() for text in axis.get_legend().texts} == {
        "Female adult reference",
        "Male adult reference",
    }
    female_colors = {
        normal_range_color(normal_range)
        for normal_range in adult_equivalent_normal_ranges("A", "u")
        if normal_range.sex == "female"
    }
    assert len(female_colors) == 1
    plt.close(figure)


def test_normal_range_guides_use_distinct_colors_by_reference_group() -> None:
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-1.0, 1.0)

    add_univariate_normal_ranges(axis, age_specific_normal_ranges("A", "u"))

    assert len({line.get_color() for line in axis.lines}) == 4
    plt.close(figure)


def test_normal_range_guide_colors_are_visually_separated() -> None:
    colors = [
        np.asarray(to_rgb(normal_range_color(normal_range)))
        for normal_range in common_reference_group_ranges("A", "u")
    ]
    distances = [
        float(np.linalg.norm(left - right))
        for left_index, left in enumerate(colors)
        for right in colors[left_index + 1 :]
    ]

    assert min(distances) > 0.25


def test_normal_range_guides_label_age_specific_reference_marks() -> None:
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-1.0, 1.0)

    add_univariate_normal_ranges(axis, age_specific_normal_ranges("A", "u"))

    assert len(axis.patches) == 4
    assert len(axis.lines) == 8
    assert {text.get_text() for text in axis.get_legend().texts} == {
        "Female 18-39 years reference",
        "Female ≥70 years reference",
        "Male 18-39 years reference",
        "Male ≥70 years reference",
    }
    plt.close(figure)


def test_normal_range_legend_uses_mathematical_comparison_symbols() -> None:
    figure, axis = plt.subplots()
    axis.set_xlim(0.0, 10.0)
    axis.set_ylim(-1.0, 1.0)
    ranges = (
        NormalRange(
            "A", "u", "female", ">=70 years", 2.0, 4.0, ">=", "<=", "source", ""
        ),
        NormalRange(
            "A", "u", "female", "<=17 years", 3.0, 5.0, ">=", "<=", "source", ""
        ),
    )

    add_univariate_normal_ranges(axis, ranges)

    assert [text.get_text() for text in axis.get_legend().texts] == [
        "Female ≥70 years reference",
        "Female ≤17 years reference",
    ]
    plt.close(figure)


def test_normal_range_guides_draw_interaction_axis_references() -> None:
    figure, axis = plt.subplots()
    axis.imshow(np.zeros((2, 2)), extent=(0.0, 10.0, 0.0, 20.0), origin="lower")

    add_interaction_normal_ranges(
        axis,
        "A|||u",
        "B|||u",
        {
            "A|||u": normal_range_pair("A", "u"),
            "B|||u": normal_range_pair("B", "u"),
        },
    )

    assert len(axis.patches) == 4
    assert len(axis.lines) == 8
    plt.close(figure)


def normal_range_pair(test_name: str, unit: str) -> tuple[NormalRange, ...]:
    return (
        NormalRange(test_name, unit, "female", "adult", 2.0, 4.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "adult", 3.0, 5.0, ">=", "<=", "source", ""),
    )


def visible_normal_range_pair(test_name: str, unit: str) -> tuple[NormalRange, ...]:
    return (
        NormalRange(test_name, unit, "female", "adult", 0.55, 1.10, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "adult", 0.75, 1.35, ">=", "<=", "source", ""),
    )


def age_specific_normal_ranges(test_name: str, unit: str) -> tuple[NormalRange, ...]:
    return (
        NormalRange(test_name, unit, "female", "18-39 years", 0.50, 1.00, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", ">=70 years", 0.65, 1.15, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "18-39 years", 0.55, 1.05, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", ">=70 years", 0.70, 1.20, ">=", "<=", "source", ""),
    )


def adult_equivalent_normal_ranges(test_name: str, unit: str) -> tuple[NormalRange, ...]:
    return (
        NormalRange(test_name, unit, "female", "adult", 2.0, 4.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", ">18 years", 2.1, 4.1, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", ">=1 month", 2.2, 4.2, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "adult fasting serum", 2.3, 4.3, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "adult G1", 2.4, 4.4, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "adult", 3.0, 5.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", ">=18 years", 3.1, 5.1, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "all ages", 3.2, 5.2, ">=", "<=", "source", ""),
    )


def common_reference_group_ranges(test_name: str, unit: str) -> tuple[NormalRange, ...]:
    return (
        NormalRange(test_name, unit, "female", "adult", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "adult", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "<40 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "<40 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", ">=40 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", ">=40 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "18-39 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "18-39 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "40-69 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "40-69 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", "18-69 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", "18-69 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "female", ">=70 years", 0.5, 1.0, ">=", "<=", "source", ""),
        NormalRange(test_name, unit, "male", ">=70 years", 0.5, 1.0, ">=", "<=", "source", ""),
    )


class EmptyModel:
    """Minimal model object for explanation rendering tests without fitted EBM state."""


class MissingBinModel:
    """Minimal model object with explicit missing-bin scores."""

    def __init__(self, term_count: int) -> None:
        self.term_scores_ = [np.asarray([0.1]) for _ in range(term_count)]
        self.standard_deviations_ = [np.asarray([0.01]) for _ in range(term_count)]


class SingleTermExplanation:
    """Minimal InterpretML-like explanation with one numeric univariate term."""

    def __init__(self, term_name: str) -> None:
        self.term_name = term_name

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": [self.term_name], "scores": [1.0]}
        return {
            "names": [10.0, 20.0],
            "scores": [0.0],
        }


class WideRangeTermExplanation:
    """Minimal InterpretML-like explanation with values spanning orders of magnitude."""

    def __init__(self, term_name: str) -> None:
        self.term_name = term_name

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": [self.term_name], "scores": [1.0]}
        return {
            "names": [0.5, 5.0, 50.0],
            "scores": [-0.2, 0.0, 0.2],
        }


class NarrowRangeTermExplanation:
    """Minimal InterpretML-like explanation with positive values in a narrow range."""

    def __init__(self, term_name: str) -> None:
        self.term_name = term_name

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": [self.term_name], "scores": [1.0]}
        return {
            "names": [3.6, 4.0, 4.4],
            "scores": [-0.2, 0.0, 0.2],
        }


class ClinicalCovariateExplanation:
    """InterpretML-like explanation for clinical covariates that stay linear."""

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {
                "names": ["time_since_diagnosis_years", "age_at_diagnosis"],
                "scores": [1.0, 0.8],
            }
        return {
            "names": [1.0, 10.0, 100.0],
            "scores": [-0.2, 0.0, 0.2],
        }


class MultiPointTermExplanation:
    """Minimal InterpretML-like explanation with several positive numeric points."""

    def __init__(self, term_name: str) -> None:
        self.term_name = term_name

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": [self.term_name], "scores": [1.0]}
        return {
            "names": [0.6, 1.0, 2.0],
            "scores": [-0.2, 0.0, 0.2],
        }


class TimeSinceDiagnosisExplanation:
    """Time-since-diagnosis explanation spanning several five-year intervals."""

    def __init__(self, term_name: str) -> None:
        self.term_name = term_name

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": [self.term_name], "scores": [1.0]}
        return {
            "names": [0.5, 6.0, 12.0],
            "scores": [-0.2, 0.0, 0.2],
        }


class ExtremeUncertainTermExplanation:
    """Univariate term whose hazard-ratio uncertainty spans many orders of magnitude."""

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": ["S-CRP|||mg/l"], "scores": [1.0]}
        return {
            "names": [1.0, 10.0, 100.0],
            "scores": [0.0, 20.0],
            "lower_bounds": [-1.0, 15.0],
            "upper_bounds": [1.0, 25.0],
        }


class GridTermExplanation:
    """InterpretML-like explanation with ordered univariate terms and one interaction."""

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {
                "names": [
                    "P-Magnesium|||mmol/l",
                    "interaction",
                    "P-ALAT|||u/l",
                    "S-CRP|||mg/l",
                    "P-Kalsium|||mmol/l",
                ],
                "scores": [0.4, 99.0, 1.5, 0.1, 0.8],
            }
        if term_index == 1:
            return {
                "type": "interaction",
                "names": ["interaction"],
                "left_names": [0.0, 1.0],
                "right_names": [0.0, 1.0],
                "scores": [[0.0]],
            }
        return {
            "names": [0.0, 1.0, 2.0],
            "scores": [-0.2, 0.0],
            "lower_bounds": [-0.4, -0.2],
            "upper_bounds": [0.0, 0.2],
        }


class UncertainGridTermExplanation:
    """InterpretML-like univariate explanation with outer-bag uncertainty bounds."""

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {"names": ["marker|||u/l"], "scores": [1.0]}
        return {
            "names": [1.0, 2.0, 4.0, 8.0],
            "scores": [-0.2, 0.0, 0.2],
            "lower_bounds": [-0.4, -0.2, 0.0],
            "upper_bounds": [0.0, 0.2, 0.4],
        }


def grid_style_snapshot(figure: plt.Figure) -> dict[str, object]:
    collections = [collection for axis in figure.axes for collection in axis.collections]
    line_segments = [
        {
            "axis_index": axis_index,
            "color": line.get_color(),
            "orientation": line_orientation(line),
            "position": line_position(line),
        }
        for axis_index, axis in enumerate(figure.axes)
        for line in axis.lines
    ]
    return {
        "patch_labels": [patch.get_label() for axis in figure.axes for patch in axis.patches],
        "patch_edge_alphas": [
            patch.get_edgecolor()[-1] for axis in figure.axes for patch in axis.patches
        ],
        "line_colors": {line.get_color() for axis in figure.axes for line in axis.lines},
        "line_segments": line_segments,
        "markers": [line.get_marker() for axis in figure.axes for line in axis.lines],
        "fill_count": sum(isinstance(collection, FillBetweenPolyCollection) for collection in collections),
        "error_bar_count": sum(isinstance(collection, LineCollection) for collection in collections),
    }


def gridline_snapshot(line: Line2D) -> dict[str, object]:
    return {
        "color": line.get_color(),
        "linestyle": line.get_linestyle(),
        "linewidth": line.get_linewidth(),
        "visible": line.get_visible(),
    }


def grid_axis_snapshot(axis: plt.Axes) -> dict[str, object]:
    minor_ticks = axis.yaxis.get_minor_ticks()
    return {
        "title": axis.get_title(),
        "xlabel": axis.get_xlabel(),
        "ylim": axis.get_ylim(),
        "y_major_ticks": axis.get_yticks(),
        "y_minor_ticks": axis.get_yticks(minor=True),
        "y_major_tick_length": axis.yaxis.get_major_ticks()[
            0
        ].tick1line.get_markersize(),
        "y_minor_tick_length": minor_ticks[0].tick1line.get_markersize()
        if minor_ticks
        else None,
        "y_labels": [
            label.get_text() for label in axis.get_yticklabels() if label.get_text()
        ],
        "visible": axis.axison,
        "has_legend": axis.get_legend() is not None,
    }


def grid_legend_snapshot(figure: plt.Figure) -> dict[str, object]:
    axis_legends = [
        legend
        for axis in figure.axes
        if (legend := axis.get_legend()) is not None
    ]
    return {
        "figure_legend_labels": [
            text.get_text()
            for legend in figure.legends
            for text in legend.get_texts()
        ],
        "figure_legend_line_count": sum(
            len(legend.get_lines()) for legend in figure.legends
        ),
        "figure_legend_patch_count": sum(
            len(legend.get_patches()) for legend in figure.legends
        ),
        "axis_legend_labels": [
            text.get_text() for legend in axis_legends for text in legend.get_texts()
        ],
        "axis_legend_line_count": sum(
            len(legend.get_lines()) for legend in axis_legends
        ),
        "axis_legend_patch_count": sum(
            len(legend.get_patches()) for legend in axis_legends
        ),
    }


def line_orientation(line: Line2D) -> str | None:
    x_data = np.asarray(line.get_xdata(), dtype=float)
    y_data = np.asarray(line.get_ydata(), dtype=float)
    if len(x_data) < 2 or len(y_data) < 2:
        return None
    if np.allclose(x_data, x_data[0]):
        return "vertical"
    if np.allclose(y_data, y_data[0]):
        return "horizontal"
    return None


def line_position(line: Line2D) -> float | None:
    orientation = line_orientation(line)
    if orientation == "horizontal":
        return float(np.asarray(line.get_ydata(), dtype=float)[0])
    if orientation == "vertical":
        return float(np.asarray(line.get_xdata(), dtype=float)[0])
    return None


def axis_snapshot(axis: plt.Axes) -> dict[str, object]:
    return {
        "x_labels": [label.get_text() for label in axis.get_xticklabels() if label.get_text()],
        "y_labels": [label.get_text() for label in axis.get_yticklabels() if label.get_text()],
        "y_ticks": axis.get_yticks(),
    }
