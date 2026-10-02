from __future__ import annotations

from pathlib import Path

import yaml
import pytest
import pandas as pd
import matplotlib.pyplot as plt

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.cohort_figure import (
    AnnualCurveTable, annual_incidence_table, annual_survival_table, composition_table,
    recurrence_patient_counts, risk_count_text, suppress_risk_counts,
)
from src.data.dataset_statistics import prepared_recurrence_site_rows
from src.data.population_statistics import PopulationStatisticsConfig
from src.data.dataset_statistics_export import recurrence_incidence_from_payload, recurrence_survival_from_payload
from src.script.plot.dataset_statistics import CONFIG_PATH, build_dataset_statistics_figure, plot_config_from_payload
from src.script.training_dataset_statistics import write_training_dataset_statistics_png
from src.script.plot.cohort_composition_panels import draw_annual_curve_table
import src.script.plot.overview_layout as cohort_layout
from src.script.plot.overview_layout import create_cohort_axes, fit_cohort_footer, fit_recurrence_tick_spacing
from test.test_dataset_statistics_config import source_mapping, write_minimal_dataset_files
from test.test_population_statistics import population_statistics_dataset


def test_training_exports_selected_cohort_layout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    layout = yaml.safe_load(cohort_layout.LAYOUT_CONFIG_PATH.read_text())
    layout["table_refinements"]["subcategory_indent_mm"] = 3
    layout_path = tmp_path / "designs.yml"
    layout_path.write_text(yaml.safe_dump(layout))
    monkeypatch.setattr(cohort_layout, "LAYOUT_CONFIG_PATH", layout_path)
    write_minimal_dataset_files(tmp_path)
    dataset = PatientBloodTestDataset({
        "bloodtest_csv_path": str(tmp_path / "blood.csv"),
        "endpoint_csv_path": str(tmp_path / "endpoint.csv"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "selected_blood_tests": ["A|||u"],
        "min_blood_test_types_per_day": 1,
    })
    raw = load_yml(CONFIG_PATH)
    raw["dataset"] = source_mapping()
    raw["figure"]["dpi"] = 50
    plot_config = tmp_path / "plot.yml"
    plot_config.write_text(yaml.safe_dump(raw), encoding="utf-8")

    write_training_dataset_statistics_png(dataset, tmp_path / "run", plot_config_path=plot_config)

    svg = (tmp_path / "run/dataset_statistics.svg").read_text()
    assert "Study cohort composition" in svg
    assert "Site occurrences" in svg
    assert "At risk" in svg
    assert "At risk (events)" not in svg
    payload = yaml.safe_load((tmp_path / "run/dataset_statistics_values.yml").read_text())
    assert payload["cohort_figure"]["population"]["rows"][0]["cells"] == ["2", "—", "—", "2"]
    occurrences = next(row for row in payload["cohort_figure"]["population"]["rows"]
                       if row["label"] == "Site occurrences*")
    assert occurrences["cells"] == ["0 (0)", "—", "—", "0 (0)"]
    assert all(cell == "≤5" for row in payload["cohort_figure"]["survival"]["cells"] for cell in row)
    # Re-rendering legacy snapshots must strip events and suppress exact small counts.
    payload["cohort_figure"]["survival"]["cells"][0][0] = "2 (0)"
    saved = plot_config_from_payload(raw, payload, config_dir=tmp_path)
    figure = build_dataset_statistics_figure(saved, recurrence_incidence_df=recurrence_incidence_from_payload(payload),
                                             recurrence_survival_df=recurrence_survival_from_payload(payload))
    try:
        risk_cells = [text.get_text() for text in figure.axes[5].texts
                      if str(text.get_gid()).startswith("annual-survival-")]
        assert risk_cells and set(risk_cells) == {"≤5"}
        assert sum(text.get_text() == "0 (0)" for text in figure.axes[2].texts) == 2
        bone = next(text for text in figure.axes[2].texts if text.get_text() == "Bone")
        assert bone.get_position()[0] == 3
        assert [axis.get_label() for axis in figure.axes] == [f"panel-{letter}" for letter in "abcdef"]
        assert sum(text.get_text() == "Study cohort" for text in figure.axes[0].texts) == 1
        assert not any(text.get_text() in ("Train", "Validation", "Test") for text in figure.axes[0].texts)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        notes = [text.get_window_extent(renderer) for text in figure.axes[0].texts
                 if text.get_gid() == "flow-note"]
        assert notes
        cohort = next(patch.get_window_extent(renderer) for patch in figure.axes[0].patches
                      if patch.get_gid() == "study-cohort")
        composition_title = next(text.get_window_extent(renderer) for text in figure.texts
                                 if text.get_text() == "Study cohort composition")
        assert all(composition_title.y1 < note.y0 < note.y1 < cohort.y0 for note in notes)
        assert all(figure.bbox.contains(*note.min) and figure.bbox.contains(*note.max) for note in notes)
        for axis, table in ((figure.axes[4], saved.cohort_figure.incidence),
                            (figure.axes[5], saved.cohort_figure.survival)):
            assert tuple(axis.get_xticks()) == table.years
            cells = [text for text in axis.texts if str(text.get_gid()).startswith("annual-")]
            assert len(cells) == len(table.sites) * len(table.years)
    finally:
        plt.close(figure)


def test_annual_incidence_samples_steps_without_extrapolation() -> None:
    points = pd.DataFrame({"site": ["Any"] * 4, "years_since_diagnosis": [0., 1., 1.5, 2.],
                           "cumulative_incidence_percent": [0., 20., 30., 30.]})
    table = annual_incidence_table(points, ("Any", "Bone"), 3)
    assert table.years == (0, 1, 2, 3)
    assert table.cells == (("0.0", "20.0", "30.0", "—"), ("—", "—", "—", "—"))


def test_risk_table_counts_patients_once_and_keeps_tick_day_deaths_at_risk() -> None:
    rows = pd.DataFrame([
        {"patient_id": patient, "days_since_diagnosis": recurrence, "code": code,
         "death_day": death, "last_observation_day": end}
        for patient, recurrence, code, death, end in (
            ("a", 10, "C795", 375, 1000), ("a", 10, "C793", 375, 1000),
            ("b", 20, "C795", 20, 1000), ("c", 30, "C793", None, 760),
        )
    ])
    config = PopulationStatisticsConfig.from_mapping(None)
    table = annual_survival_table(rows, config, ("Any", "Bone"), overall_label="Any", max_year=3, min_days=0,
                                  min_patients_at_risk=1)
    assert table.cells == (("3", "2", "≤1", "≤1"), ("2", "≤1", "≤1", "≤1"))


def test_site_totals_include_unique_patients_without_changing_percentages(tmp_path: Path) -> None:
    dataset = population_statistics_dataset(tmp_path)
    config = PopulationStatisticsConfig.from_mapping(None)
    endpoints = prepared_recurrence_site_rows(dataset)
    table = composition_table(dataset, config, None, endpoint_rows=endpoints, min_days=0)
    rows = {row.label: row.cells for row in table.rows}
    # Train's one patient has bone and brain on day 100; later lung is excluded.
    assert rows["Site occurrences*"] == ("2 (1)", "0 (0)", "1 (1)", "3 (2)")
    assert rows["Bone"] == ("1 (50.0%)", "0 (—)", "0 (0.0%)", "1 (33.3%)")
    assert rows["Brain"][0] == "1 (50.0%)"
    assert rows["Lung"][-1] == "0 (0.0%)"
    risk = annual_survival_table(endpoints, config, ("Any",), overall_label="Any", max_year=0,
                                min_days=0, min_patients_at_risk=1)
    assert rows["Site occurrences*"][-1] == f"3 ({risk.cells[0][0]})"


@pytest.mark.parametrize(("min_days", "expected"), [(60, (1, 0, 1, 2)), (61, (1, 0, 0, 1)), (101, (0, 0, 0, 0))])
def test_unique_recurrence_counts_use_the_any_population_and_origin_filter(
    tmp_path: Path, min_days: int, expected: tuple[int, ...],
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    endpoints = prepared_recurrence_site_rows(dataset)
    counts = recurrence_patient_counts(dataset, endpoints, min_days)
    assert counts == expected
    risk = annual_survival_table(endpoints, PopulationStatisticsConfig.from_mapping(None), ("Any",),
                                overall_label="Any", max_year=0, min_days=min_days, min_patients_at_risk=1)
    assert risk_count_text(str(counts[-1]), 1) == risk.cells[0][0]


def test_large_annual_counts_and_dense_integer_ticks_remain_readable() -> None:
    figure = plt.figure(figsize=(200 / 25.4, 266 / 25.4))
    axes = create_cohort_axes(figure)
    incidence, survival = axes[4:]
    incidence.set_xlim(0, 20)
    incidence.set_xticks(range(21))
    incidence.set_ylim(0, 80)
    incidence.set_yticks(range(81))
    incidence.tick_params(labelsize=5.5)
    survival.set_xlim(0, 10)
    survival.set_xticks(range(11))
    survival.set_xlabel("Years since first recurrence")
    table = AnnualCurveTable(tuple(range(11)), ("Any", "Bone"), (("1,000",) * 11,) * 2)
    try:
        fit_recurrence_tick_spacing((incidence, survival))
        draw_annual_curve_table(incidence, AnnualCurveTable(tuple(range(21)), ("Any",), (("15.9",) * 21,)),
                                survival=False)
        draw_annual_curve_table(survival, table, survival=True)
        fit_cohort_footer(figure)
        figure.canvas.draw()
        renderer = figure.canvas.get_renderer()
        ticks = [label.get_window_extent(renderer) for label in incidence.get_yticklabels()]
        assert all(not first.overlaps(second) for first, second in zip(ticks, ticks[1:]))
        cells = [text for text in survival.texts if str(text.get_gid()).startswith("annual-")]
        bounds = [text.get_window_extent(renderer) for text in cells]
        assert all(text.get_text() == "1,000" for text in cells)
        assert all(figure.bbox.contains(*box.min) and figure.bbox.contains(*box.max) for box in bounds)
        assert all(not box.overlaps(other) for index, box in enumerate(bounds) for other in bounds[index + 1:])
        incidence_cells = [text.get_window_extent(renderer) for text in incidence.texts
                           if str(text.get_gid()).startswith("annual-")]
        assert all(not box.overlaps(other) for box, other in zip(incidence_cells, incidence_cells[1:]))
        labels = [*incidence.texts, *incidence.get_xticklabels(), *incidence.get_yticklabels()]
        bounds = [text.get_window_extent(renderer) for text in labels]
        assert all(not box.overlaps(other) for index, box in enumerate(bounds) for other in bounds[index + 1:])
        assert incidence.bbox.height == survival.bbox.height
    finally:
        plt.close(figure)


def test_risk_suppression_bounds_small_counts_and_preserves_missing_values() -> None:
    table = AnnualCurveTable(tuple(range(7)), ("Any", "Bone", "Lung"), (
        ("10 (0)", "8 (2)", "7 (3)", "6 (4)", "5 (5)", "3 (7)", "0 (10)"),
        ("4", "3", "2", "1", "0", "0", "0"), ("—",) * 7,
    ))
    masked = suppress_risk_counts(table, 5)
    assert masked.cells == (("10", "8", "7", "6", "≤5", "≤5", "≤5"), ("≤5",) * 7, ("—",) * 7)
    assert suppress_risk_counts(masked, 5) == masked
    assert table.cells[0][0] == "10 (0)"


def test_risk_suppression_uses_configured_threshold_and_preserves_previous_bounds() -> None:
    table = AnnualCurveTable((0, 1, 2, 3), ("Any",), (("6", "3", "≤5", "—"),))
    assert suppress_risk_counts(table, 3).cells == (("6", "≤3", "≤5", "—"),)
    assert suppress_risk_counts(table, 10).cells == (("≤10", "≤10", "≤10", "—"),)


def test_known_risk_counts_do_not_disappear_after_last_survival_step() -> None:
    table = AnnualCurveTable((4, 5, 6, 7), ("Liver",), (("12", "7", "6", "4"),))
    assert suppress_risk_counts(table, 5).cells == (("12", "7", "6", "≤5"),)
