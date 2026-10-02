from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt

from src.data import PatientBloodTestDataset
from src.data.cohort_age_exchange import age_display_summary, cohort_membership, find_age_summary, membership_hash
from src.data.cohort_figure import (
    CohortFigureData,
    annual_incidence_table,
    annual_survival_table,
    composition_table,
    diagnosis_year_ranges,
)
from src.data.population_statistics import PopulationStatisticsConfig
from src.data.dataset_statistics import (
    DatasetStatisticsSummary,
    build_dataset_statistics_summary,
    load_test_name_translation,
    prepared_recurrence_site_rows,
)
from src.data.dataset_statistics_export import dataset_statistics_payload, write_dataset_statistics_payload
from src.model.blood_test_imputation_statistics import (
    BloodTestImputationStatistics,
)
from src.script.plot.dataset_statistics import (
    CONFIG_PATH,
    DatasetStatisticsPlotConfig,
    build_recurrence_incidence_table,
    build_recurrence_survival_table,
    build_dataset_statistics_figure,
    load_config,
    ordered_site_labels,
)
from src.script.plot.overview_style import save_publication_figure

DATASET_STATISTICS_FILENAME = "dataset_statistics.png"


def write_training_dataset_statistics_png(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    *,
    imputation_statistics: BloodTestImputationStatistics | None = None,
    plot_config_path: Path = CONFIG_PATH,
    population_config: PopulationStatisticsConfig | None = None,
) -> dict[str, object]:
    """Export PNG/PDF/SVG and exact re-renderable values from the prepared dataset."""
    plot_config = load_config(plot_config_path)
    cohort_hash = membership_hash(cohort_membership(dataset))
    returned = find_age_summary(cohort_hash)
    plot_config = replace(plot_config, manual_age=age_display_summary(None if returned is None else returned.value))
    translation = load_test_name_translation(plot_config.source.bloodtest_translation_csv_path)
    summary = build_dataset_statistics_summary(
        dataset.config,
        prepared_dataset=dataset,
        test_name_translation=translation,
        max_bars=plot_config.blood_test_counts.max_bars,
        imputation_statistics=imputation_statistics,
    )
    figure_config = plot_config_with_summary(
        plot_config,
        summary,
        show_imputation_sources=imputation_statistics is not None,
    )
    endpoint_rows = prepared_recurrence_site_rows(dataset)
    incidence = build_recurrence_incidence_table(figure_config, endpoint_rows=endpoint_rows)
    survival = build_recurrence_survival_table(figure_config, endpoint_rows=endpoint_rows)
    sites = figure_config.recurrence_site_incidence
    population = (PopulationStatisticsConfig.from_mapping({
        "site_labels": [rule.label for rule in sites.site_rules],
        "site_regexes": [rule.code_regex for rule in sites.site_rules],
        "other_label": sites.other_label,
    }) if population_config is None else population_config)
    survival_sites = figure_config.first_recurrence_site_survival
    cohort_figure = CohortFigureData(
        composition_table(dataset, population, imputation_statistics, endpoint_rows=endpoint_rows,
                          min_days=sites.min_days_since_diagnosis),
        diagnosis_year_ranges(dataset),
        annual_incidence_table(incidence, ordered_site_labels(
            sites.site_rules, sites.other_label, sites.overall_label if sites.show_overall else None),
            sites.max_years_since_diagnosis),
        annual_survival_table(
            endpoint_rows,
            replace(population, site_rules=survival_sites.site_rules, other_label=survival_sites.other_label),
            ordered_site_labels(survival_sites.site_rules, survival_sites.other_label,
                                survival_sites.overall_label if survival_sites.show_overall else None),
            overall_label=survival_sites.overall_label if survival_sites.show_overall else None,
            max_year=survival_sites.max_years_since_origin,
            min_days=sites.min_days_since_diagnosis,
            min_patients_at_risk=survival_sites.min_patients_at_risk,
        ),
    )
    figure_config = replace(figure_config, cohort_figure=cohort_figure)
    figure = build_dataset_statistics_figure(
        figure_config,
        recurrence_incidence_df=incidence,
        recurrence_survival_df=survival,
    )
    output_path = output_dir / DATASET_STATISTICS_FILENAME
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        save_publication_figure(figure, output_path, figure_config.figure.dpi)
    finally:
        plt.close(figure)
    payload = dataset_statistics_payload(
        summary,
        dataset_config=asdict(dataset.config),
        recurrence_incidence_table=incidence,
        recurrence_survival_table=survival,
    )
    payload["metadata"]["show_imputation_sources"] = imputation_statistics is not None
    payload["metadata"]["dataset_config_format"] = "resolved"
    payload["metadata"]["figure_layout"] = "cohort_composition"
    payload["metadata"]["cohort_hash"] = cohort_hash
    payload["metadata"]["age_return_sources"] = [] if returned is None else [str(path) for path in returned.paths]
    payload["cohort_figure"] = asdict(cohort_figure)
    if figure_config.manual_age is not None:
        payload["metadata"]["manual_age_summary"] = asdict(figure_config.manual_age)
    write_dataset_statistics_payload(payload, output_dir / "dataset_statistics_values.yml")
    return {
        "path": output_path.name,
        "input_fingerprint": summary.input_fingerprint,
        "pdf": {"path": output_path.with_suffix(".pdf").name},
        "svg": {"path": output_path.with_suffix(".svg").name},
        "values": {"path": "dataset_statistics_values.yml"},
    }


def plot_config_with_summary(
    config: DatasetStatisticsPlotConfig,
    summary: DatasetStatisticsSummary,
    *,
    show_imputation_sources: bool,
) -> DatasetStatisticsPlotConfig:
    """Insert run-specific cohort counts into the reusable plot configuration."""
    return replace(
        config,
        patient_flow=replace(
            config.patient_flow,
            inclusion_box_labels=summary.inclusion_box_labels,
            inclusion_box_counts=summary.inclusion_box_counts,
            exclusion_box_labels=summary.exclusion_box_labels,
            exclusion_box_counts=summary.exclusion_box_counts,
            split_box_counts=summary.split_box_counts,
        ),
        diagnosis_years=replace(
            config.diagnosis_years,
            diagnosis_years=summary.diagnosis_years,
            series_labels=summary.series_labels,
            series_counts=summary.series_counts,
        ),
        blood_test_counts=replace(
            config.blood_test_counts,
            show_imputation_sources=show_imputation_sources,
            blood_test_names=summary.blood_test_names,
            blood_test_counts=summary.blood_test_counts,
            blood_test_locf_counts=summary.blood_test_locf_counts,
            blood_test_mice_counts=summary.blood_test_mice_counts,
        ),
    )
