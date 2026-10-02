from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.legend import Legend

from src.data.reference_intervals import reference_interval_label
from src.figure_headings import manuscript_scale, visible_text_bounds
from src.model.ebm_benchmark_top_features import create_ebm_top_feature_grid
from src.model.ebm_visualizations import (
    HazardRatioGridStyle,
    draw_hazard_ratio_grid_term,
    write_univariate_hazard_ratio_grid,
    write_univariate_hazard_ratio_plot,
)
from src.model.normal_range_plotting import NUMERIC_REFERENCE_GID, fit_numeric_reference_labels
from test.test_publication_style import synthetic_ebm_rows
from test.test_recurrence_reference_statistics import reference_interval


def numeric_legend(axis: plt.Axes) -> Legend:
    return next(artist for artist in axis.artists
                if isinstance(artist, Legend) and artist.get_gid() == NUMERIC_REFERENCE_GID)


@pytest.mark.parametrize("term_count", [4, 26])
def test_grid_title_clears_first_row_at_short_and_manuscript_heights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, term_count: int,
) -> None:
    names = [f"Marker {index}|||u" for index in range(term_count)]
    ranges = tuple(reference_interval(test_name="Marker", unit="u", age_group=age) for age in (
        "18-39 years", "40-69 years", ">=70 years",
    ))
    explanation = Mock()
    explanation.data.side_effect = lambda index=None: (
        {"names": names, "scores": list(range(term_count))} if index is None
        else {"names": [0.1, 1, 10], "scores": [-0.2, 0.2]}
    )
    checked: list[Path] = []

    def check_layout(figure: plt.Figure, path: Path) -> None:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        heading = figure.texts[0].get_window_extent(renderer)
        title_bottom, _ = visible_text_bounds(figure.texts[0], renderer)
        panel_top = max(visible_text_bounds(axis.title, renderer)[1] for axis in figure.axes[:4])
        scale = manuscript_scale("terms_hazard_ratio_grid", figure.get_figwidth() * 72, figure.get_figheight() * 72)
        assert (title_bottom - panel_top) * 72 / figure.dpi * scale == pytest.approx(3.0)
        assert heading.y1 <= figure.bbox.y1
        checked.append(path)

    monkeypatch.setattr(plt.Figure, "savefig", check_layout)
    write_univariate_hazard_ratio_grid(
        Mock(), explanation, tmp_path / "grid.png", tmp_path / "grid.svg",
        include_missing_bins=False, normal_ranges_by_feature=dict.fromkeys(names, ranges),
        test_name_translation={}, x_axis_scale="log", columns=4, y_limits=(0.2, 5),
        show_x_markers=False, show_uncertainty_area=True,
    )
    assert len(checked) == 2


@pytest.mark.parametrize(("changes", "expected"), [
    ({"lower": 35, "upper": 105, "age_group": ">18 years"}, "35–105 (>18 years)"),
    ({"lower": 1, "upper": 2, "lower_operator": "≥", "upper_operator": "≤"}, "1–2 (all ages)"),
    ({"lower": 1, "upper": 2, "lower_operator": ">", "upper_operator": "<="}, "1 < x ≤ 2 (all ages)"),
    ({"lower": None, "upper": 5, "upper_operator": "<", "age_group": "adult"}, "<5 (≥18 years)"),
    ({"lower": 90, "upper": None, "lower_operator": ">=", "age_group": "adult G1"}, "≥90 (≥18 years; G1)"),
])
def test_shared_reference_formatter_preserves_original_bounds(changes: dict[str, object], expected: str) -> None:
    assert reference_interval_label(reference_interval(**changes)) == expected


def test_grid_panel_labels_full_intervals_outside_visible_x_range_without_changing_curves() -> None:
    ranges = (
        reference_interval(test_name="Albumin", unit="g/l", lower=36, upper=48, age_group="18-39 years"),
        reference_interval(test_name="Albumin", unit="g/l", lower=36, upper=45, age_group="40-69 years"),
        reference_interval(test_name="Albumin", unit="g/l", lower=34, upper=45, age_group=">=70 years"),
    )
    style = HazardRatioGridStyle({"Albumin|||g/l": ranges}, {}, "linear", (0.3, 4), False, True)
    figure, axis = plt.subplots(figsize=(4, 3), dpi=160)
    try:
        draw_hazard_ratio_grid_term(axis, {"names": [50, 60, 70], "scores": [0.2, 0.5]},
                                    "Albumin|||g/l", style, show_ylabel=True)
        figure.tight_layout()
        fit_numeric_reference_labels(axis)
        figure.canvas.draw()
        legend = numeric_legend(axis)
        assert [text.get_text() for text in legend.get_texts()] == [
            "36–48 (18–39 years)", "36–45 (40–69 years)", "34–45 (≥70 years)",
        ]
        assert axis.get_xlim()[0] > 48
        np.testing.assert_allclose(axis.lines[0].get_ydata(), np.exp([0.2, 0.5]))
        assert axis.get_xlabel() == "g/L"
        renderer = figure.canvas.get_renderer()
        bounds = legend.get_window_extent(renderer)
        assert axis.bbox.y0 < bounds.y0 < bounds.y1 < axis.bbox.y1
        assert axis.bbox.x0 <= bounds.x0 < bounds.x1 <= axis.bbox.x1
        assert not legend.get_frame_on()
        assert (bounds.x0 - axis.bbox.x0) / axis.bbox.width < 0.1
        assert (axis.bbox.y1 - bounds.y1) / axis.bbox.height < 0.1
    finally:
        plt.close(figure)


def test_standalone_hazard_plot_exports_numeric_limits_and_sex_as_selectable_text(tmp_path: Path) -> None:
    female = reference_interval(test_name="Marker", unit="u/l", lower=None, upper=5, upper_operator="<")
    male = replace(female, sex="male", upper=7, upper_operator="<=")
    output = tmp_path / "term.svg"
    with plt.rc_context({"svg.fonttype": "none"}):
        write_univariate_hazard_ratio_plot({"names": [1, 5, 10], "scores": [0.0, 0.2]},
                                           "Marker|||u/l", "Hazard ratio", output,
                                           normal_ranges=(female, male))
    document = ElementTree.parse(output)
    labels = ["".join(node.itertext()) for node in document.iter("{http://www.w3.org/2000/svg}text")]
    assert "<5 (Female, all ages)" in labels
    assert "≤7 (Male, all ages)" in labels


def test_top_feature_reference_labels_fit_inside_upper_left_corner() -> None:
    ranges = tuple(reference_interval(test_name="A", unit="u", age_group=age) for age in (
        "adult", "<40 years", ">=40 years", "18-39 years", "40-69 years", "18-69 years", ">=70 years",
    ))
    rows = tuple(replace(row, style=replace(row.style, normal_ranges_by_feature={"A|||u": ranges}))
                 for row in synthetic_ebm_rows())
    figure = create_ebm_top_feature_grid(rows, "Top EBM feature effects")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in figure.axes[::4]:
            bounds = numeric_legend(axis).get_window_extent(renderer)
            assert axis.bbox.y0 < bounds.y0 < bounds.y1 < axis.bbox.y1
            assert axis.bbox.x0 < bounds.x0 < bounds.x1 <= axis.bbox.x1
            assert not numeric_legend(axis).get_frame_on()
            assert (bounds.x0 - axis.bbox.x0) / axis.bbox.width < 0.1
            assert (axis.bbox.y1 - bounds.y1) / axis.bbox.height < 0.1
            assert axis.bbox.height / figure.dpi > 0.5
    finally:
        plt.close(figure)
