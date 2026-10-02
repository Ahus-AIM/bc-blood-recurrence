from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import pytest
from matplotlib.colors import to_hex

import src.script.plot.dataset_statistics as dataset_statistics_plot
from src.data import PatientBloodTestDataset
from src.config import load_yml
from src.data.dataset_statistics_export import load_dataset_statistics_payload
from src.script.plot.dataset_statistics import (
    BloodTestCountPlotConfig,
    DatasetStatisticsSourceConfig,
    DiagnosisYearBarChartConfig,
    RecurrenceSiteIncidencePlotConfig,
    apply_integer_percent_ticks,
    apply_year_ticks,
    build_dataset_statistics_figure,
    draw_diagnosis_year_bar_chart,
    load_config,
    prepared_endpoint_rows,
    year_ticks,
)
from src.data.cohort_figure import AnnualCurveTable, CohortFigureData, annual_incidence_table
from src.data.population_statistics import PopulationStatisticsRow, PopulationStatisticsTable
from src.script.plot.overview_layout import cohort_headers, create_cohort_axes
from src.script.training_dataset_statistics import write_training_dataset_statistics_png


def source_mapping() -> dict[str, object]:
    return {
        "endpoint_csv_path": "endpoint.csv",
        "split_csv_path": "split.csv",
        "recurrence_endpoint_name": "distant_recurrence",
        "input_recurrence_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "endpoint_ddfs_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
    }


def test_checked_in_dataset_statistics_config_is_valid() -> None:
    config = load_config(dataset_statistics_plot.CONFIG_PATH)

    assert config.blood_test_counts.max_bars == 24
    assert config.patient_flow.inclusion_box_labels[2:6] == (
        "No other cancer before BC diagnosis",
        "No distant recurrence before BC diagnosis",
        "No other cancer ≤ 30 days after BC diagnosis",
        "No distant recurrence ≤ 30 days after BC diagnosis",
    )
    assert config.patient_flow.exclusion_box_labels[1:5] == (
        "Other cancer before BC diagnosis",
        "Distant recurrence before BC diagnosis",
        "Other cancer ≤ 30 days after BC diagnosis",
        "Distant recurrence ≤ 30 days after BC diagnosis",
    )
    assert (
        config.patient_flow.inclusion_box_labels[-3]
        == "Blood tests on or before (competing) event or censoring"
    )
    assert (
        config.patient_flow.exclusion_box_labels[-3]
        == "Blood tests after (competing) event or censoring¹"
    )
    assert config.patient_flow.inclusion_box_labels[-1] == "Days > 30 days after BC diagnosis\nwith > 12 different blood tests"
    assert config.patient_flow.exclusion_box_labels[-2] == "No days > 30 days after BC diagnosis\nwith > 12 different blood tests"
    assert config.patient_flow.exclusion_box_labels[-1] == "Days > 30 days after BC diagnosis\nwith ≤ 12 different blood tests"
    assert config.diagnosis_years.series_labels[-1] == config.patient_flow.exclusion_box_labels[-2]
    assert config.diagnosis_years.series_labels[:5] == (
        "Included",
        "Other cancer before BC diagnosis",
        "Distant recurrence before BC diagnosis",
        "Other cancer ≤ 30 days after BC diagnosis",
        "Distant recurrence ≤ 30 days after BC diagnosis",
    )


def test_dataset_statistics_source_requires_endpoint_metadata(tmp_path: Path) -> None:
    config = source_mapping()
    del config["recurrence_endpoint_name"]

    with pytest.raises(KeyError, match="recurrence_endpoint_name"):
        DatasetStatisticsSourceConfig.from_mapping(config, config_dir=tmp_path)


def test_dataset_statistics_source_can_use_ebm_dataset_config(tmp_path: Path) -> None:
    ebm_config_path = tmp_path / "ebm_survival.yml"
    ebm_config_path.write_text(
        "\n".join(
            [
                "dataset:",
                '  endpoint_csv_path: "data/endpoint.csv"',
                '  bloodtest_csv_path: "data/blood.csv"',
                '  split_csv_path: "data/split.csv"',
                '  recurrence_endpoint_name: "distant_recurrence"',
                '  input_recurrence_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                '  endpoint_ddfs_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
            ]
        ),
        encoding="utf-8",
    )

    source_config = DatasetStatisticsSourceConfig.from_mapping(
        {"dataset_config_path": "ebm_survival.yml"},
        config_dir=tmp_path,
    )

    assert source_config.dataset_config_path == ebm_config_path.resolve()
    assert source_config.endpoint_csv_path == (tmp_path / "data" / "endpoint.csv").resolve()
    assert source_config.split_csv_path == (tmp_path / "data" / "split.csv").resolve()


def test_dataset_statistics_endpoint_rows_use_dataset_eligibility(tmp_path: Path) -> None:
    write_minimal_dataset_files(tmp_path)
    ebm_config_path = tmp_path / "ebm_survival.yml"
    ebm_config_path.write_text(
        "\n".join(
            [
                "dataset:",
                '  bloodtest_csv_path: "blood.csv"',
                '  endpoint_csv_path: "endpoint.csv"',
                '  split_csv_path: "split.csv"',
                '  bc_diagnosis_days_csv_path: "bc_diagnosis_days.csv"',
                '  diagnosis_procedure_csv_path: "clinical.csv"',
                "  exclude_prior_bc_surgery_patients: true",
                '  selected_blood_tests: ["A|||u"]',
                '  recurrence_endpoint_name: "distant_recurrence"',
                '  input_recurrence_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                '  endpoint_ddfs_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
            ]
        ),
        encoding="utf-8",
    )
    source_config = DatasetStatisticsSourceConfig.from_mapping(
        {"dataset_config_path": "ebm_survival.yml"},
        config_dir=tmp_path,
    )

    endpoint_rows = prepared_endpoint_rows(source_config)

    assert set(endpoint_rows["patient_id"]) == {"keep"}


def test_dataset_statistics_figure_accepts_precomputed_recurrence_frames(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / "dataset_statistics.yml"
    config_path.write_text(
        "\n".join(
            [
                "figure:",
                '  output_path: "figure.pgf"',
                "dataset:",
                '  endpoint_csv_path: "missing_endpoint.csv"',
                '  split_csv_path: "missing_split.csv"',
                '  recurrence_endpoint_name: "distant_recurrence"',
                '  input_recurrence_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                '  endpoint_ddfs_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                "panel_a:",
                '  inclusion_box_labels: ["Start", "Kept"]',
                '  inclusion_box_counts: ["N = 2", "N = 1"]',
                '  exclusion_box_labels: ["Dropped"]',
                '  exclusion_box_counts: ["N = 1"]',
                '  split_box_labels: ["Train", "Validation", "Test"]',
                '  split_box_counts: ["N = 1", "N = 0", "N = 0"]',
                "panel_b:",
                "  diagnosis_years: [2020]",
                '  series_labels: ["Included"]',
                "  series_counts:",
                "    - [1]",
                "panel_c:",
                '  blood_test_names: ["Hemoglobin"]',
                "  blood_test_counts: [1]",
                "panel_d:",
                '  site_labels: ["Bone"]',
                '  site_regexes: ["C79\\\\d*"]',
                "panel_e: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_config(config_path)
    recurrence_incidence = pd.DataFrame(
        {
            "site": ["Any", "Any", "Bone", "Bone"],
            "years_since_diagnosis": [0.0, 10.0, 0.0, 10.0],
            "cumulative_incidence_percent": [0.0, 1.0, 0.0, 1.0],
        }
    )
    recurrence_survival = pd.DataFrame(
        {
            "site": ["Any", "Any", "Bone", "Bone"],
            "years_since_origin": [0.0, 10.0, 0.0, 10.0],
            "survival_percent": [100.0, 90.0, 100.0, 90.0],
        }
    )

    config = replace(config, cohort_figure=CohortFigureData(
        PopulationStatisticsTable((PopulationStatisticsRow("Patients", ("1", "0", "0", "1")),
                                   PopulationStatisticsRow("Site occurrences*", ("1", "0", "0", "1"))),
                                  ("Train", "Validation", "Test", "Overall")),
        ("2020", "—", "—", "2020"),
        annual_incidence_table(recurrence_incidence, ("Any", "Bone"), config.recurrence_site_incidence.max_years_since_diagnosis),
        AnnualCurveTable(tuple(range(config.first_recurrence_site_survival.max_years_since_origin + 1)),
                         ("Any",), (("—",) * (config.first_recurrence_site_survival.max_years_since_origin + 1),)),
    ))
    figure = build_dataset_statistics_figure(
        config,
        recurrence_incidence_df=recurrence_incidence,
        recurrence_survival_df=recurrence_survival,
    )

    panel_titles = tuple(axis.get_title() for axis in figure.axes[:6])
    for axis, source, x_column, y_column in (
        (figure.axes[4], recurrence_incidence, "years_since_diagnosis", "cumulative_incidence_percent"),
        (figure.axes[5], recurrence_survival, "years_since_origin", "survival_percent"),
    ):
        for line in axis.lines:
            rows = source.loc[source["site"] == line.get_label()]
            assert list(line.get_xdata()) == list(rows[x_column])
            assert list(line.get_ydata()) == list(rows[y_column])
    plt.close(figure)
    assert panel_titles == ("", "", "", "", "", "")


def test_training_dataset_statistics_writes_run_local_png(tmp_path: Path) -> None:
    write_minimal_dataset_files(tmp_path)
    plot_config_path = tmp_path / "dataset_statistics.yml"
    plot_config_path.write_text(
        "\n".join(
            [
                "figure:",
                "  figure_width: 4",
                "  figure_height: 6",
                "  dpi: 50",
                "dataset:",
                '  endpoint_csv_path: "endpoint.csv"',
                '  split_csv_path: "split.csv"',
                '  recurrence_endpoint_name: "distant_recurrence"',
                '  input_recurrence_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                '  endpoint_ddfs_code_filter_regex: "(?:C78\\\\d*)|(?:C79\\\\d*)"',
                "panel_a:",
                '  inclusion_box_labels: ["Start", "End"]',
                '  inclusion_box_counts: ["N = 2", "N = 2"]',
                '  exclusion_box_labels: ["Removed"]',
                '  exclusion_box_counts: ["N = 0"]',
                '  split_box_labels: ["Train", "Validation", "Test"]',
                '  split_box_counts: ["N = 2", "N = 0", "N = 0"]',
                "panel_b:",
                "  diagnosis_years: [2020]",
                '  series_labels: ["Included"]',
                "  series_counts: [[2]]",
                "panel_c:",
                "  max_bars: 24",
                "  blood_test_names: []",
                "  blood_test_counts: []",
                "panel_d:",
                '  site_labels: ["Bone"]',
                '  site_regexes: ["C795\\\\d*"]',
                "panel_e: {}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dataset = PatientBloodTestDataset(
        {
            "bloodtest_csv_path": str(tmp_path / "blood.csv"),
            "endpoint_csv_path": str(tmp_path / "endpoint.csv"),
            "split_csv_path": str(tmp_path / "split.csv"),
            "selected_blood_tests": ["A|||u"],
        }
    )
    output_dir = tmp_path / "run"

    metadata = write_training_dataset_statistics_png(
        dataset,
        output_dir,
        plot_config_path=plot_config_path,
    )

    output_path = output_dir / "dataset_statistics.png"
    assert metadata == {
        "path": "dataset_statistics.png",
        "input_fingerprint": dataset.prepared_data.input_fingerprint,
        "pdf": {"path": "dataset_statistics.pdf"},
        "svg": {"path": "dataset_statistics.svg"},
        "values": {"path": "dataset_statistics_values.yml"},
    }
    assert output_path.stat().st_size > 0
    assert output_path.with_suffix(".pdf").read_bytes().startswith(b"%PDF")
    assert "<text" in output_path.with_suffix(".svg").read_text()
    payload = load_dataset_statistics_payload(output_dir / "dataset_statistics_values.yml")
    dataset_statistics_plot.validate_statistics_source(payload, asdict(dataset.config), tmp_path)
    refreshed = dataset_statistics_plot.plot_config_from_payload(
        load_yml(plot_config_path),
        payload,
        config_dir=tmp_path,
    )
    assert list(refreshed.patient_flow.inclusion_box_counts) == payload["panel_a"]["inclusion_box_counts"]
    assert list(refreshed.blood_test_counts.blood_test_counts) == payload["panel_c"]["blood_test_counts"]
    assert not refreshed.blood_test_counts.show_imputation_sources


def test_recurrence_site_plot_defaults_to_input_recurrence_regex(
    tmp_path: Path,
) -> None:
    source_config = DatasetStatisticsSourceConfig.from_mapping(source_mapping(), config_dir=tmp_path)
    plot_config = RecurrenceSiteIncidencePlotConfig.from_mapping(
        {
            "site_labels": ["Lung"],
            "site_regexes": ["C780\\d*"],
        },
        input_recurrence_code_filter_regex=source_config.input_recurrence_code_filter_regex,
    )

    assert plot_config.overall_sanity_check_regex == source_config.input_recurrence_code_filter_regex
    assert plot_config.min_patients_at_risk == 5
    assert plot_config.confidence_level == 0.95
    assert plot_config.random_seed == 42


def test_bloodtest_count_config_requires_matching_manual_lists() -> None:
    with pytest.raises(ValueError, match="blood_test_names"):
        BloodTestCountPlotConfig.from_mapping(
            {
                "blood_test_names": ["Albumin"],
                "blood_test_counts": [],
            }
        )


def test_diagnosis_year_config_accepts_stacked_series() -> None:
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": [2020, 2021],
            "series_labels": ["Included", "Excluded"],
            "series_counts": [[3, 4], [1, 2]],
        }
    )

    assert config.series_labels == ("Included", "Excluded")
    assert config.series_counts == ((3, 4), (1, 2))


def test_diagnosis_year_bar_chart_uses_100_patient_y_ticks() -> None:
    figure, axis = plt.subplots()
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": [2020, 2021],
            "series_labels": ["Included", "Excluded"],
            "series_counts": [[250, 300], [100, 150]],
        }
    )

    draw_diagnosis_year_bar_chart(axis, config)
    figure.canvas.draw()

    y_max = axis.get_ylim()[1]
    visible_ticks = tuple(int(tick) for tick in axis.get_yticks() if 0.0 <= tick <= y_max)
    plt.close(figure)
    assert visible_ticks[0] == 0
    assert all(right - left == 100 for left, right in zip(visible_ticks, visible_ticks[1:]))


def test_diagnosis_year_bar_chart_reserves_space_for_legend() -> None:
    figure, axis = plt.subplots()
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": [2020],
            "series_labels": [
                "Included",
                *[f"Exclusion {index}" for index in range(7)],
            ],
            "series_counts": [[100] for _ in range(8)],
        }
    )

    draw_diagnosis_year_bar_chart(axis, config)

    legend_labels = tuple(text.get_text() for text in axis.get_legend().get_texts())
    plt.close(figure)
    assert axis.get_ylim()[1] > 800
    assert not axis.get_legend().get_frame_on()
    assert legend_labels == (
        *(f"Exclusion {index}" for index in range(7)),
        "Included",
    )


def test_diagnosis_year_legend_fits_above_compact_panel() -> None:
    figure, axis = plt.subplots(figsize=(2.55, 3.2), dpi=100)
    labels = [
        "Included",
        "Other cancer before BC diagnosis",
        "Distant recurrence before BC diagnosis",
        "Other cancer ≤ 30 days after BC diagnosis",
        "Distant recurrence ≤ 30 days after BC diagnosis",
        "Blood test unrelated to BC checkup",
        "No blood tests > 30 days after BC diagnosis",
        "No blood tests before other cancer/recurrence/death/right-censoring",
        "No days with > 12 different blood tests",
    ]
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": [2020],
            "series_labels": labels,
            "series_counts": [[100] for _ in labels],
        }
    )

    draw_diagnosis_year_bar_chart(axis, config)
    figure.canvas.draw()

    renderer = figure.canvas.get_renderer()
    legend_bbox = axis.get_legend().get_window_extent(renderer)
    axis_bbox = axis.get_window_extent(renderer)
    plt.close(figure)
    assert legend_bbox.x1 <= axis_bbox.x1
    assert legend_bbox.y0 > axis_bbox.y1


def test_diagnosis_year_ticks_are_centered_without_moving_bars() -> None:
    figure, axis = plt.subplots()
    years = tuple(range(2004, 2027))
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": list(years),
            "series_labels": ["Included"],
            "series_counts": [[1] * len(years)],
        }
    )

    draw_diagnosis_year_bar_chart(axis, config)

    x_limits = axis.get_xlim()
    x_ticks = tuple(axis.get_xticks())
    x_tick_labels = tuple(label.get_text() for label in axis.get_xticklabels())
    first_bar = axis.containers[0].patches[0]
    second_bar = axis.containers[0].patches[1]
    last_bar = axis.containers[0].patches[-1]
    left_edge_gap = first_bar.get_x() - x_limits[0]
    right_edge_gap = x_limits[1] - (last_bar.get_x() + last_bar.get_width())
    plt.close(figure)
    assert x_ticks == pytest.approx((2005.4, 2010.4, 2015.4, 2020.4, 2025.4))
    assert x_tick_labels == ("2005", "2010", "2015", "2020", "2025")
    assert first_bar.get_x() == 2004
    assert second_bar.get_x() == 2005
    assert left_edge_gap == pytest.approx(right_edge_gap)
    assert x_ticks[0] == pytest.approx(second_bar.get_x() + second_bar.get_width() / 2.0)
    assert first_bar.get_x() + first_bar.get_width() <= x_limits[1]


def test_diagnosis_year_stack_colors_match_legend_order() -> None:
    figure, axis = plt.subplots()
    config = DiagnosisYearBarChartConfig.from_mapping(
        {
            "diagnosis_years": [2020],
            "series_labels": [
                "Included",
                "Prior BC surgery",
                "Recurrence before BC diagnosis",
                "Distant recurrence ≤ 30 days",
                "No blood tests > 30 days",
                "No blood tests before recurrence",
                "No valid landmark/outcome",
            ],
            "series_counts": [[1], [1], [1], [1], [1], [1], [1]],
        }
    )

    draw_diagnosis_year_bar_chart(axis, config)

    colors = tuple(to_hex(container.patches[0].get_facecolor()) for container in axis.containers)
    top_to_bottom = tuple(
        container.get_label()
        for container in sorted(
            axis.containers,
            key=lambda container: container.patches[0].get_y(),
            reverse=True,
        )
    )
    legend_labels = tuple(text.get_text() for text in axis.get_legend().get_texts())
    legend_colors = tuple(to_hex(handle.get_facecolor()) for handle in axis.get_legend().legend_handles)
    colors_by_label = dict(zip(config.series_labels, colors, strict=True))
    plt.close(figure)
    assert len(set(colors)) == len(colors)
    assert top_to_bottom == tuple(label.replace("\n", " ").replace("/ ", "/") for label in legend_labels)
    assert legend_colors == tuple(colors_by_label[label] for label in top_to_bottom)


def test_diagnosis_year_uses_unique_colors_for_all_current_series() -> None:
    figure, axis = plt.subplots()
    config = load_config(dataset_statistics_plot.CONFIG_PATH).diagnosis_years

    draw_diagnosis_year_bar_chart(axis, config)

    colors = {to_hex(container.patches[0].get_facecolor()) for container in axis.containers}
    plt.close(figure)
    assert len(colors) == len(config.series_labels)


def test_recurrence_year_ticks_keep_configured_annual_step() -> None:
    assert year_ticks(20, 1) == tuple(range(21))


def test_recurrence_year_ticks_apply_annual_labels() -> None:
    figure, axis = plt.subplots()

    apply_year_ticks(axis, max_year=20, configured_step=1)
    figure.canvas.draw()

    labels = tuple(label.get_text() for label in axis.get_xticklabels())
    plt.close(figure)
    assert labels == tuple(str(year) for year in range(21))


def test_recurrence_incidence_y_ticks_apply_integer_percent_labels() -> None:
    figure, axis = plt.subplots()

    apply_integer_percent_ticks(axis, y_max=4.8)
    figure.canvas.draw()

    labels = tuple(label.get_text() for label in axis.get_yticklabels())
    plt.close(figure)
    assert labels == tuple(str(percent) for percent in range(5))


@pytest.mark.parametrize("figure_size", [(7.874, 9.252), (6.4, 8.0)])
def test_dataset_statistics_layout_reserves_headers_and_calendar_legend(
    figure_size: tuple[float, float],
) -> None:
    config = load_config(dataset_statistics_plot.CONFIG_PATH)
    labels = tuple(
        label.replace(
            "within follow-up (including event/censoring day)",
            "before other cancer/recurrence/death/right-censoring",
        )
        for label in config.diagnosis_years.series_labels
    )
    figure = plt.figure(figsize=figure_size)
    axes = create_cohort_axes(figure)
    draw_diagnosis_year_bar_chart(axes[1], replace(config.diagnosis_years, series_labels=labels))
    axes[3].plot((0, 1), (0, 1), label="Any")
    axes[4].plot((0, 1), (1, 0), label="Any")
    original = [axis.get_position().bounds for axis in axes]
    cohort_headers(figure, axes, ("Cohort selection", config.diagnosis_years.title, "Study cohort composition",
                                  config.blood_test_counts.title, config.recurrence_site_incidence.title,
                                  config.first_recurrence_site_survival.title))
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    assert all(axis.get_position().bounds == original[index] for index, axis in enumerate(axes) if index != 1)
    assert axes[1].get_position().y0 == original[1][1]
    assert axes[4].get_position().height == pytest.approx(axes[5].get_position().height)
    assert axes[4].get_position().y0 == axes[5].get_position().y0
    assert axes[0].get_position().y0 > axes[2].get_position().y1
    assert axes[2].get_position().y0 > axes[4].get_position().y1
    year_legend = axes[1].get_legend().get_window_extent(renderer)
    assert not axes[1].get_legend().get_frame_on()
    assert year_legend.y0 > axes[1].bbox.y1
    assert year_legend.y1 < figure.texts[2].get_window_extent(renderer).y0
    assert all(figure.bbox.contains(*text.get_window_extent(renderer).max) for text in figure.texts)
    plt.close(figure)


def test_diagnosis_year_config_rejects_mismatched_stacked_series() -> None:
    with pytest.raises(ValueError, match="series_counts"):
        DiagnosisYearBarChartConfig.from_mapping(
            {
                "diagnosis_years": [2020, 2021],
                "series_labels": ["Included"],
                "series_counts": [[3]],
            }
        )


def write_minimal_dataset_files(tmp_path: Path) -> None:
    (tmp_path / "blood.csv").write_text(
        "\n".join(
            [
                "id,days_since_diagnosis,test_name,result,unit",
                "keep,10,A,1,u",
                "drop,10,A,1,u",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "endpoint.csv").write_text(
        "\n".join(
            [
                "id,bc_diagnosis_year,days_from_diagnosis_to_last_entry",
                "keep,2016,100",
                "drop,2016,100",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "split.csv").write_text(
        "\n".join(
            [
                "id,split",
                "keep,train",
                "drop,train",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "clinical.csv").write_text(
        "\n".join(
            [
                "id,days_since_diagnosis,event_type,code,description",
                "drop,-1,procedure,3800,prior surgery",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "bc_diagnosis_days.csv").write_text(
        "\n".join(
            [
                "patient_id,bc_diagnosis_day,code,description",
                "keep,0,C509,Breast cancer",
                "drop,0,C509,Breast cancer",
            ]
        ),
        encoding="utf-8",
    )
