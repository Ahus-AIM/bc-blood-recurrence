from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml
from matplotlib.patches import StepPatch

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_config import RecurrenceTimeWindow, load_recurrence_blood_test_config
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_blood_test_plot_config import RecurrenceBloodTestPlotConfig
from src.data.recurrence_blood_test_selection import select_recurrence_plot_rows
from src.data.recurrence_reference_exchange import export_reference_package, prepare_reference_report
from src.reporting import latex_escape
from src.figure_style import figure_typography
from src.script.plot.recurrence_blood_test_insights import create_recurrence_blood_test_figure
from src.script.plot.recurrence_blood_test_panel import numerical_rows
from src.script.plot.recurrence_blood_test_panel import reference_label
from src.script.plot.recurrence_blood_test_panel import trajectory_typography
from src.script.plot.dataset_statistics import site_color_mapping
from src.script.recurrence_blood_test_reporting import write_recurrence_blood_test_report
from test.test_recurrence_blood_test_insights import report_fixture
from test.test_recurrence_reference_statistics import reference_interval, run_calculation, write_patient_dates


def test_figure_values_tables_reference_axes_and_exports_agree(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    blood.loc[blood["id"].eq("D"), "result"] = 60  # Outside at baseline, with no later measurement.
    blood.to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(dataset.config)
    config = replace(config, reference_ranges=(reference_interval(),), plot=replace(config.plot, dpi=50))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    result_path = run_calculation(folder)
    archive_return(result_path, "reference_percentages")
    frame, _metadata = prepare_reference_report(frame, dataset, config)
    final_liver = (
        frame["site"].eq("Liver")
        & frame["test_key"].eq("B-Hb|||g/dL")
        & frame["window"].eq(config.windows[-1].label)
    )
    frame.loc[final_liver, ["p_value", "p_value_bh"]] = [2.4e-12, 7.2e-10]
    selected = select_recurrence_plot_rows(frame, config)
    figure = create_recurrence_blood_test_figure(frame, config)
    try:
        percentage_axes = [axis for axis in figure.axes if " ".join(axis.get_ylabel().split()) == "Outside reference range (%)"]
        assert len(percentage_axes) == 2
        assert all(axis.get_ylim()[0] == 0 and axis.get_ylim()[1] > 100 for axis in percentage_axes)
        assert all(tuple(axis.get_yticks()) == (0, 50, 100) for axis in percentage_axes)
        original_axes = [axis for axis in figure.axes if axis.get_ylabel() == "g/dL"]
        assert original_axes[0].get_ylim() == original_axes[1].get_ylim()
        line = next(line for line in original_axes[0].lines if line.get_marker() == "o")
        np.testing.assert_allclose(
            line.get_ydata(),
            frame.loc[frame["site"].eq("Liver") & frame["test_key"].eq("B-Hb|||g/dL"), "median"],
            equal_nan=True,
        )
        table = numerical_rows(selected.loc[selected["site"].eq("Liver")], config.decimals)
        assert [label for label, _ in table][4:6] == ["p value", "Adjusted p"]
        assert table[4][1][-1] == "2.4 × 10⁻¹²"
        assert table[5][1][-1] == "7.2 × 10⁻¹⁰"
        assert table[2][1][-1] == "+4.50"
        assert table[-1][1][0] == "33.3%"
        assert table[-1][1][-1] == "0.0%"
        assert table[6][0] == "Tested / paired"
        assert "Liver (4 patients)" in {text.get_text() for text in figure.texts}
        assert "Brain (0 patients)" in {text.get_text() for text in figure.texts}
        legend = " ".join(text.get_text() for text in figure.legends[0].get_texts())
        assert "Outside reference range (%)" in legend
        assert "Baseline: entire tested population for that test/site" in legend
        assert "Later windows: patients paired with baseline" in legend
        area = next(patch for patch in original_axes[0].patches if isinstance(patch, StepPatch))
        assert area.get_data().values[0] == pytest.approx(1 / 3)
        # The new headroom must not detach percentage shading from its right-hand scale.
        for value in (0, 1 / 3, 1):
            shaded_y = area.get_data_transform().transform((0, value))[1]
            axis_y = percentage_axes[0].transData.transform((0, value * 100))[1]
            assert shaded_y == pytest.approx(axis_y)
    finally:
        plt.close(figure)
    write_recurrence_blood_test_report(frame, dataset, config)
    latex = (config.output_dir / "recurrence_blood_test_insights.tex").read_text()
    baseline_label = r"\leftskip=2em\relax " + latex_escape("<12–6 mo (baseline)")
    baseline_line = next(line for line in latex.splitlines() if line.startswith(baseline_label))
    assert r" &  &  &  &  & 1/\allowbreak{}3 (33.3\%)" in baseline_line
    assert "Baseline uses the entire population tested" in latex
    assert r"$2.4 \times 10^{-12}$" in latex
    assert r"\textbf{\boldmath $7.2 \times 10^{-10}$}" in latex
    metadata = json.loads((config.output_dir / "recurrence_blood_test_insights.json").read_text())
    for filename in metadata["figure"]["files"]:
        assert (config.output_dir / filename).stat().st_size > 0
    assert (config.output_dir / "recurrence_blood_test_trajectories.pdf").read_bytes().startswith(b"%PDF")
    svg = (config.output_dir / "recurrence_blood_test_trajectories.svg").read_text()
    assert "<text" in svg
    assert "p value" in svg and "Outside reference range (%)" in svg
    assert "2.4 × 10⁻¹²" in svg and "7.2 × 10⁻¹⁰" in svg
    exported = pd.read_csv(config.output_dir / "recurrence_blood_test_trajectories_values.csv")
    np.testing.assert_allclose(exported["p_value"], selected["p_value"], rtol=1e-12, atol=0, equal_nan=True)
    np.testing.assert_allclose(exported["p_value_bh"], selected["p_value_bh"], equal_nan=True)
    assert metadata["config"]["reference_ranges"][0]["upper"] == 50
    assert metadata["site_patient_counts"] == {"Liver": 4, "Bone": 2, "Brain": 0, "Other": 0}
    assert metadata["figure"]["site_patient_counts"] == metadata["site_patient_counts"]
    assert exported.loc[exported["site"].eq("Liver"), "n_site_patients"].eq(4).all()


def test_config_loads_local_reference_snapshot_and_plot_settings(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    raw_dataset = {key: str(value) if isinstance(value, Path) else value for key, value in vars(dataset.config).items()}
    config.dataset_config_path.write_text(yaml.safe_dump({"dataset": raw_dataset}))
    interval = reference_interval()
    pd.DataFrame([vars(interval)]).to_csv(tmp_path / "ranges.csv", index=False)
    report_path = tmp_path / "report.yml"
    report_path.write_text(
        yaml.safe_dump(
            {
                "source": {"dataset_config_path": "source.yml"},
                "report": {"output_dir": "output"},
                "reference_intervals": {"csv_path": "ranges.csv"},
                "plot": {"top_n": 3, "font_scale": 0.9, "graph_height_scale": 0.6, "formats": ["svg"]},
            }
        )
    )
    parsed = load_recurrence_blood_test_config(report_path)
    assert parsed.reference_ranges_csv_path == tmp_path / "ranges.csv"
    assert parsed.reference_ranges == (interval,)
    assert parsed.plot.top_n == 3
    assert parsed.plot.font_scale == 0.9
    assert parsed.plot.graph_height_scale == 0.6
    assert parsed.plot.formats == ("svg",)
    assert RecurrenceBloodTestPlotConfig().top_n == 5


def test_panels_fit_their_own_values_and_reference_bands_with_dataset_site_colors(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=(reference_interval(),))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    frame.loc[frame["site"].eq("Bone"), "q3"] = 500
    figure = create_recurrence_blood_test_figure(frame, config)
    try:
        axes = [axis for axis in figure.axes if axis.get_ylabel() == "g/dL"]
        # Preserve each panel's lower extent and add clear legend space above its original range.
        assert axes[0].get_ylim()[0] == -4 and axes[0].get_ylim()[1] > 54
        assert axes[1].get_ylim()[0] == -40 and axes[1].get_ylim()[1] > 540
        colors = site_color_mapping(config.site_rules, config.other_label)
        for axis, site in zip(axes, ("Liver", "Bone"), strict=True):
            line = next(line for line in axis.lines if line.get_marker() == "o")
            assert line.get_color() == colors[site]
            assert axis.get_position().height * figure.get_figheight() > (
                0.4 * config.plot.row_height * config.plot.graph_height_scale
            )
        title = figure.texts[0]
        assert title.get_position()[0] == 0.5 and title.get_ha() == "center"
        assert not any("Median/IQR summarize" in text.get_text() for text in figure.texts)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        for axis in axes:
            assert all(tick == int(tick) for tick in axis.get_yticks())
            legend = axis.get_legend()
            legend_box = legend.get_window_extent(renderer)
            assert axis.bbox.contains(legend_box.x0, legend_box.y0)
            assert axis.bbox.contains(legend_box.x1, legend_box.y1)
            for line in axis.lines:
                assert line.get_window_extent(renderer).y1 < legend_box.y0
            for patch in axis.patches:
                assert patch.get_path().transformed(patch.get_transform()).get_extents().y1 < legend_box.y0
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        ({"lower": 35, "upper": 105, "age_group": ">18 years"}, "35–105 (>18 years)"),
        ({"lower": 35, "upper": 105, "lower_operator": ">", "upper_operator": "<"}, "35 < x < 105 (all ages)"),
        ({"lower": None, "upper": 5, "upper_operator": "<", "age_group": "adult"}, "<5 (≥18 years)"),
        ({"lower": 90, "upper": None, "age_group": ">=18 years"}, "≥90 (≥18 years)"),
        ({"age_group": "18-39 years"}, "0–50 (18–39 years)"),
        ({"age_group": "adult fasting serum"}, "0–50 (≥18 years; fasting serum)"),
        ({"age_group": "adult G1"}, "0–50 (≥18 years; G1)"),
        ({"age_group": ">=1 month"}, "0–50 (≥1 month)"),
    ],
)
def test_reference_labels_preserve_inclusive_strict_and_age_bounds(changes: dict[str, object], expected: str) -> None:
    assert reference_label(reference_interval(**changes)) == expected


def test_small_test_scales_retain_informative_decimal_ticks(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    frame[["q1", "median", "q3"]] *= 0.001
    figure = create_recurrence_blood_test_figure(frame, config)
    try:
        axis = next(axis for axis in figure.axes if axis.get_ylabel() == "g/dL")
        lower, upper = axis.get_ylim()
        ticks = [tick for tick in axis.get_yticks() if lower <= tick <= upper]
        assert len(ticks) >= 3
        assert any(tick != int(tick) for tick in ticks)
    finally:
        plt.close(figure)


@pytest.mark.parametrize(("panel_width", "decimals"), [(6.4, 2), (8, 3)])
def test_five_window_columns_fit_labels_and_numeric_values(tmp_path: Path, panel_width: float, decimals: int) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config,
        site_rules=config.site_rules[:1],
        other_label=None,
        decimals=decimals,
        plot=replace(config.plot, panel_width=panel_width),
        windows=(
            RecurrenceTimeWindow("6–<18 months", 180, 539),
            RecurrenceTimeWindow("3–<6 months", 90, 179),
            RecurrenceTimeWindow("1–<3 months", 30, 89),
            RecurrenceTimeWindow("1 week–<1 month", 7, 29),
            RecurrenceTimeWindow("<1 week", 0, 6),
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    frame = frame.loc[frame["site"].eq("Liver") & frame["test_key"].eq("B-Hb|||g/dL")].copy()
    frame[["q1", "median", "q3"]] = [100.7, 130, 152.75]
    frame[["change_q1", "median_change", "change_q3"]] = [-44, 3, 48.5]
    frame["p_value"] = [np.nan, 2.4e-12, 4.94e-324, 9.999e-4, 0.001]
    frame["p_value_bh"] = [np.nan, 7.2e-10, 1.5e-321, 0.003, 0.05]
    frame = pd.concat([frame, frame.assign(test_key="Other|||g/dL", blood_test="Another test")])
    figure = create_recurrence_blood_test_figure(frame, config)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        legend_bottom = figure.legends[0].get_window_extent(renderer).y0
        for title in (text for text in figure.texts if text.get_text().startswith(("1. ", "2. "))):
            assert title.get_window_extent(renderer).y1 + 2 < legend_bottom
        for axis in figure.axes:
            if axis.get_ylabel() == "g/dL":
                labels = axis.get_xticklabels()
                assert [label.get_text() for label in labels] == [
                    "<18–6 mo\n(baseline)", "<6–3 mo", "<3–1 mo", "<1 mo–1 wk", "<1 wk"
                ]
                for left, right in zip(labels, labels[1:]):
                    assert left.get_window_extent(renderer).x1 < right.get_window_extent(renderer).x0
            if not axis.axison:
                for text in axis.texts:
                    column, _ = text.get_position()
                    if column < 0:  # Row headings sit outside the numeric columns.
                        continue
                    cell_left = axis.transData.transform((column - 0.5, 0))[0]
                    cell_right = axis.transData.transform((column + 0.5, 0))[0]
                    bounds = text.get_window_extent(renderer)
                    assert cell_left + 2 < bounds.x0 < bounds.x1 < cell_right - 2
                    assert text.get_fontsize() == pytest.approx(
                        figure_typography("recurrence_blood_test_trajectories", figure.get_figwidth()).table
                    )
    finally:
        plt.close(figure)


@pytest.mark.parametrize("panel_width", [6.4, 8])
def test_three_reference_intervals_clear_the_panel_title(tmp_path: Path, panel_width: float) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config,
        reference_ranges=tuple(
            reference_interval(age_group=age) for age in ("18-39 years", "40-69 years", ">=70 years")
        ),
        plot=replace(config.plot, panel_width=panel_width),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    figure = create_recurrence_blood_test_figure(frame, config)
    try:
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        titles = [text for text in figure.texts if text.get_text().startswith("1. ")]
        axes = [axis for axis in figure.axes if axis.get_legend() is not None]
        for title, axis in zip(titles, axes, strict=True):
            legend = axis.get_legend()
            assert {label.get_fontsize() for label in legend.get_texts()} == {
                trajectory_typography(config, figure.get_figwidth()).legend,
            }
            assert legend.get_window_extent(renderer).y1 + 2 < title.get_window_extent(renderer).y0
            assert axis.get_position().height > 0
    finally:
        plt.close(figure)


@pytest.mark.parametrize(
    "values", [
        {"top_n": 0}, {"top_n": 1.5}, {"formats": ["jpg"]}, {"dpi": 0}, {"panel_width": float("nan")},
        {"font_scale": 0}, {"font_scale": 1.1}, {"font_scale": float("nan")},
        {"graph_height_scale": 0}, {"graph_height_scale": 1.1}, {"graph_height_scale": float("nan")},
    ]
)
def test_invalid_plot_config_is_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RecurrenceBloodTestPlotConfig.from_mapping(values)
