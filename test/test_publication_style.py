from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure
from matplotlib.text import Text

from src.data.cohort_figure import AnnualCurveTable, CohortFigureData, annual_incidence_table
from src.data.population_statistics import PopulationStatisticsRow, PopulationStatisticsTable
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, RecurrenceTimeWindow
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_site_common import RecurrenceSiteRule
from src.figure_style import FigureKind, publication_style, recurrence_site_colors
from src.model.ebm_benchmark_top_features import (
    EbmTopFeatureRow, SAVE_PADDING_INCHES, create_ebm_top_feature_grid,
)
from src.model.ebm_visualizations import HazardRatioGridStyle
from src.model.normal_range_plotting import normal_range_color
from src.script.plot.dataset_statistics import CONFIG_PATH, build_dataset_statistics_figure, load_config
from src.script.plot.recurrence_blood_test_insights import create_recurrence_blood_test_figure
from test.test_recurrence_blood_test_insights import report_fixture
from test.test_recurrence_reference_statistics import reference_interval
from test.test_single_bloodtest_ebm_top_features import FakeExplanation

SITES = ("Bone", "Brain", "Liver", "Lung")


def synthetic_dataset_figure() -> Figure:
    """Exercise the six-panel renderer without requiring any patient files."""
    config = load_config(CONFIG_PATH)
    sites = (*SITES, "Other", "Any")
    incidence = pd.DataFrame([
        {"site": site, "years_since_diagnosis": year, "cumulative_incidence_percent": year * (index + 1) / 20}
        for index, site in enumerate(sites) for year in range(21)
    ])
    survival = pd.DataFrame([
        {"site": site, "years_since_origin": year, "survival_percent": 100 - year * (index + 1)}
        for index, site in enumerate(sites) for year in range(11)
    ])
    population = PopulationStatisticsTable(
        tuple(PopulationStatisticsRow(label, ("100",) * 4) for label in ("Patients", "Site occurrences*", *SITES)),
        ("Train", "Validation", "Test", "Overall"),
    )
    cohort = CohortFigureData(
        population, ("2000–2020",) * 4,
        annual_incidence_table(incidence, sites, config.recurrence_site_incidence.max_years_since_diagnosis),
        AnnualCurveTable(tuple(range(11)), sites, (("100 (10)",) * 11,) * len(sites)),
    )
    return build_dataset_statistics_figure(
        replace(config, cohort_figure=cohort), recurrence_incidence_df=incidence, recurrence_survival_df=survival,
    )


def synthetic_trajectories(tmp_path: Path, columns: int) -> tuple[pd.DataFrame, RecurrenceBloodTestConfig]:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config, site_rules=config.site_rules[:1], other_label=None, decimals=3,
        windows=(
            RecurrenceTimeWindow("6–<18 months", 180, 539), RecurrenceTimeWindow("3–<6 months", 90, 179),
            RecurrenceTimeWindow("1–<3 months", 30, 89), RecurrenceTimeWindow("1 week–<1 month", 7, 29),
            RecurrenceTimeWindow("<1 week", 0, 6),
        ),
        reference_ranges=tuple(reference_interval(age_group=age) for age in ("18-39 years", "40-69 years", ">=70 years")),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    frame = frame.loc[frame["site"].eq("Liver") & frame["test_key"].eq("B-Hb|||g/dL")].copy()
    frame[["q1", "median", "q3"]] = [100.7, 130, 152.75]
    frame[["change_q1", "median_change", "change_q3"]] = [-44, 3, 48.5]
    frame.loc[frame["test_status"].ne("baseline"), "p_value_bh"] = 0.01
    frames = [frame] + [frame.assign(test_key=f"Test {index}|||g/dL", blood_test=f"Example test {index}")
                        for index in range(1, columns)]
    return pd.concat(frames, ignore_index=True), config


def synthetic_ebm_rows(*, markers: bool = False) -> tuple[EbmTopFeatureRow, ...]:
    style = HazardRatioGridStyle(
        {"A|||u": tuple(reference_interval(test_name="A", unit="u", age_group=age)
                        for age in ("18-39 years", "40-69 years", ">=70 years"))},
        {"A": "Example test"}, "log", (0.3, 4), markers, not markers,
    )
    return tuple(EbmTopFeatureRow(site, FakeExplanation(), style) for site in SITES)


def synthetic_five_site_trajectories(tmp_path: Path) -> tuple[pd.DataFrame, RecurrenceBloodTestConfig]:
    """Repeat synthetic panels across Any and four sites to check the full page."""
    frame, config = synthetic_trajectories(tmp_path, 5)
    config = replace(
        config, decimals=2, include_overall=True,
        site_rules=tuple(RecurrenceSiteRule(site, r"C787\d*") for site in SITES),
    )
    return pd.concat([frame.assign(site=site) for site in config.sites], ignore_index=True), config


def printed_size(text: Text, figure: Figure, kind: FigureKind) -> float:
    width = figure.get_figwidth()
    if kind == "ebm_top_features":
        figure.canvas.draw()
        width = figure.get_tightbbox(figure.canvas.get_renderer()).width + 2 * SAVE_PADDING_INCHES
    return text.get_fontsize() * publication_style().printed_width_inches(kind) / width


@pytest.mark.parametrize(("columns", "decimals"), [
    (1, 3),
    pytest.param(5, 2, marks=pytest.mark.xfail(strict=True, reason="Source baseline: five-column axes extend past canvas; preserve manuscript renderer.")),
    pytest.param(5, 3, marks=pytest.mark.xfail(strict=True, reason="Source baseline: five-column axes extend past canvas; preserve manuscript renderer.")),
])
def test_manuscript_typography_keeps_trajectory_tests_on_one_row(
    tmp_path: Path, columns: int, decimals: int,
) -> None:
    frame, config = synthetic_trajectories(tmp_path, columns)
    config = replace(config, decimals=decimals)
    dataset = synthetic_dataset_figure()
    trajectories = create_recurrence_blood_test_figure(frame, config)
    ebm = create_ebm_top_feature_grid(synthetic_ebm_rows(), "Top EBM feature effects — 6-month horizon")
    try:
        kinds: tuple[FigureKind, ...] = ("dataset_statistics", "recurrence_blood_test_trajectories", "ebm_top_features")
        trajectory_axis = next(axis for axis in trajectories.axes if axis.get_ylabel() == "g/dL")
        panels = (
            next(text for text in dataset.texts if text.get_gid() == "panel-title"),
            next(text for text in trajectories.texts if text.get_text().startswith("1. ")),
            ebm.axes[0].title,
        )
        site_labels = (
            next(text for text in dataset.axes[4].texts if text.get_text() == "Liver"),
            next(text for text in trajectories.texts if text.get_text().startswith("Liver (")),
            next(text for text in ebm.texts if text.get_text() == "Liver"),
        )
        axes = (dataset.axes[4], trajectory_axis, ebm.axes[0])
        trajectory_scale = (printed_size(trajectory_axis.xaxis.label, trajectories, kinds[1])
                            / printed_size(ebm.axes[0].xaxis.label, ebm, kinds[2]))
        if columns == 1:
            assert trajectory_scale == pytest.approx(1)
        else:
            assert 0.7 < trajectory_scale < 1
        assert printed_size(panels[1], trajectories, kinds[1]) == pytest.approx(
            printed_size(site_labels[2], ebm, kinds[2]), rel=0.001,
        )
        for labels in (tuple(axis.xaxis.label for axis in axes),
                       tuple(axis.get_yticklabels()[0] for axis in axes)):
            sizes = [printed_size(label, figure, kind)
                     for label, figure, kind in zip(labels, (dataset, trajectories, ebm), kinds, strict=True)]
            assert sizes == pytest.approx([sizes[0], sizes[0] * trajectory_scale, sizes[0]], rel=0.001)
            assert all(label.get_fontfamily() == ["DejaVu Sans"] for label in labels)
        assert printed_size(site_labels[1], trajectories, kinds[1]) == pytest.approx(
            printed_size(site_labels[2], ebm, kinds[2]), rel=0.001,
        )
        table_label = next(text for text in dataset.axes[2].texts if text.get_text() == "Liver")
        assert site_labels[0].get_fontsize() == table_label.get_fontsize()
        assert all(label.get_fontweight() == "bold" for label in (*panels, *site_labels))
        style = publication_style()
        expected = style.reference_sizes.site_label * style.scale * style.printed_width_inches("ebm_top_features") / 10.32
        assert printed_size(panels[2], ebm, "ebm_top_features") == pytest.approx(expected, rel=0.001)
        assert printed_size(panels[0], dataset, kinds[0]) > expected
        ebm_title = next(text for text in ebm.texts if text.get_gid() != "site-label")
        trajectory_title = next(text for text in trajectories.texts if text.get_gid() == "figure-title")
        assert printed_size(ebm_title, ebm, kinds[2]) > expected
        assert printed_size(trajectory_title, trajectories, kinds[1]) == pytest.approx(
            printed_size(ebm_title, ebm, kinds[2]), rel=0.001,
        )
        bounds = ebm.get_tightbbox(ebm.canvas.get_renderer())
        assert ebm_title.get_position()[0] * ebm.get_figwidth() == pytest.approx((bounds.x0 + bounds.x1) / 2)
        assert len(ebm.axes) == len(SITES) * 4
        trajectories.canvas.draw()
        renderer = trajectories.canvas.get_renderer()
        legend_bottom = trajectories.legends[0].get_window_extent(renderer).y0
        titles = [text for text in trajectories.texts if text.get_text().startswith(("1. ", "2. ", "3. ", "4. ", "5. "))]
        assert len(titles) == columns
        assert [text.get_position()[1] for text in titles] == pytest.approx([titles[0].get_position()[1]] * columns)
        assert all(left.get_position()[0] < right.get_position()[0] for left, right in zip(titles, titles[1:]))
        assert all(text.get_window_extent(renderer).y1 + 2 < legend_bottom for text in titles)
        for axis in trajectories.axes:
            bounds = axis.get_tightbbox(renderer)
            assert trajectories.bbox.x0 <= bounds.x0 < bounds.x1 <= trajectories.bbox.x1
            if axis.axison:
                continue
            boxes = [text.get_window_extent(renderer) for text in axis.texts]
            for text in axis.texts:
                column, _ = text.get_position()
                if column < 0:
                    continue
                cell_left = axis.transData.transform((column - 0.5, 0))[0]
                cell_right = axis.transData.transform((column + 0.5, 0))[0]
                bounds = text.get_window_extent(renderer)
                assert cell_left < bounds.x0 < bounds.x1 < cell_right
            assert all(not a.overlaps(b) for index, a in enumerate(boxes) for b in boxes[index + 1:])
            for box in boxes:
                assert trajectories.bbox.x0 <= box.x0 < box.x1 <= trajectories.bbox.x1
            for other_axis in trajectories.axes:
                if other_axis is axis or other_axis.axison:
                    continue
                others = [text.get_window_extent(renderer) for text in other_axis.texts]
                assert all(not a.overlaps(b) for a in boxes for b in others)
    finally:
        for figure in (dataset, trajectories, ebm):
            plt.close(figure)


@pytest.mark.xfail(strict=True, reason="Source baseline: five-site overview exceeds the historical 210 mm target.")
def test_five_site_trajectories_leave_page_space_for_caption(tmp_path: Path) -> None:
    frame, config = synthetic_five_site_trajectories(tmp_path)
    figure = create_recurrence_blood_test_figure(frame, config)
    full_height = create_recurrence_blood_test_figure(
        frame, replace(config, plot=replace(config.plot, graph_height_scale=1)),
    )
    try:
        print_scale = publication_style().printed_width_inches("recurrence_blood_test_trajectories") / figure.get_figwidth()
        assert figure.get_figheight() * print_scale * 25.4 <= 210
        assert figure.get_figheight() < 0.82 * full_height.get_figheight()
        assert [text.get_fontsize() for text in figure.texts] == [text.get_fontsize() for text in full_height.texts]
        graphs = [axis for axis in figure.axes if axis.get_ylabel() == "g/dL"]
        tables = [axis for axis in figure.axes if not axis.axison]
        full_tables = [axis for axis in full_height.axes if not axis.axison]
        assert len(graphs) == len(tables) == 25
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for index, (graph, table) in enumerate(zip(graphs, tables, strict=True)):
            assert graph.get_position().height > 0
            assert table.get_position().height * figure.get_figheight() == pytest.approx(
                full_tables[index].get_position().height * full_height.get_figheight(),
            )
            assert [text.get_fontsize() for text in table.texts] == [
                text.get_fontsize() for text in full_tables[index].texts
            ]
            labels_bottom = min(label.get_window_extent(renderer).y0 for label in graph.get_xticklabels())
            table_top = max(text.get_window_extent(renderer).y1 for text in table.texts)
            assert labels_bottom > table_top
            if index >= 5:
                previous_bottom = min(text.get_window_extent(renderer).y0 for text in tables[index - 5].texts)
                assert graph.get_tightbbox(renderer).y1 < previous_bottom
            assert all(0 <= text.get_window_extent(renderer).y0 for text in table.texts)
    finally:
        plt.close(figure)
        plt.close(full_height)


@pytest.mark.parametrize("markers", [False, True])
def test_site_colors_cover_ebm_labels_curves_uncertainty_and_importance(markers: bool) -> None:
    figure = create_ebm_top_feature_grid(synthetic_ebm_rows(markers=markers), "Synthetic EBM")
    try:
        colors = recurrence_site_colors(SITES)
        for index, site in enumerate(SITES):
            axes = figure.axes[index * 4:(index + 1) * 4]
            assert next(text for text in figure.texts if text.get_text() == site).get_color() == colors[site]
            for axis in axes[:3]:
                assert axis.lines[0].get_color() == colors[site]
                collection = axis.collections[0]
                color = collection.get_edgecolor()[0] if markers else collection.get_facecolor()[0]
                assert to_hex(color).lower() == colors[site].lower()
            assert all(to_hex(patch.get_facecolor()).lower() == colors[site].lower() for patch in axes[-1].patches)
    finally:
        plt.close(figure)


def test_reference_colors_are_shared_and_independent_of_order(tmp_path: Path) -> None:
    frame, config = synthetic_trajectories(tmp_path, 1)
    expected = [normal_range_color(interval) for interval in config.reference_ranges]
    figure = create_recurrence_blood_test_figure(frame, replace(config, reference_ranges=config.reference_ranges[::-1]))
    try:
        axis = next(axis for axis in figure.axes if axis.get_legend() is not None)
        assert [to_hex(patch.get_facecolor()) for patch in axis.patches[:3]] == expected[::-1]
        assert len(set(expected)) == 3
        np.testing.assert_allclose([patch.get_alpha() for patch in axis.patches[:3]], [0.14] * 3)
        np.testing.assert_allclose([handle.get_facecolor() for handle in axis.get_legend().legend_handles],
                                   [patch.get_facecolor() for patch in axis.patches[:3]])
    finally:
        plt.close(figure)
    assert "Pleura" not in publication_style().sites
    assert recurrence_site_colors(["Other", "Bone"])["Bone"] == recurrence_site_colors(["Bone"])["Bone"]
    assert recurrence_site_colors(["All"])["All"] == recurrence_site_colors(["Any"])["Any"]


def test_ebm_local_intervals_fit_all_configured_female_reference_groups() -> None:
    ranges = tuple(reference_interval(test_name="A", unit="u", age_group=age) for age in (
        "adult", "<40 years", ">=40 years", "18-39 years", "40-69 years", "18-69 years", ">=70 years",
    ))
    rows = tuple(replace(row, style=replace(row.style, normal_ranges_by_feature={"A|||u": ranges}))
                 for row in synthetic_ebm_rows())
    figure = create_ebm_top_feature_grid(rows, "Top EBM feature effects — 6-month horizon")
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        assert figure.legends == []
        axis = figure.axes[0]
        legend = next(artist for artist in axis.artists if artist.get_gid() == "numeric-reference-intervals")
        assert len(legend.get_texts()) == len(ranges)
        bounds = legend.get_window_extent(renderer)
        assert axis.bbox.x0 < bounds.x0 < bounds.x1 <= axis.bbox.x1
        assert axis.bbox.y0 < bounds.y0 < bounds.y1 < axis.bbox.y1
        assert not legend.get_frame_on()
        assert len(figure.axes) == 4 * len(SITES)
        assert (bounds.x0 - axis.bbox.x0) / axis.bbox.width < 0.1
        assert (axis.bbox.y1 - bounds.y1) / axis.bbox.height < 0.1
    finally:
        plt.close(figure)
