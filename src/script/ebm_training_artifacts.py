from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path

from interpret.glassbox import ExplainableBoostingRegressor

from src.data.dataset_config import PatientDatasetConfig
from src.data.timing import timed
from src.model.ebm_visualizations import write_ebm_explanations
from src.script.ebm_survival_config import (
    combined_term_hazard_ratio_columns,
    combined_term_hazard_ratio_show_uncertainty_area,
    combined_term_hazard_ratio_show_x_markers,
    combined_term_hazard_ratio_y_limits,
    explanation_normal_ranges,
    explanation_test_name_translation,
    explanation_term_x_axis_scale,
)

LOGGER = logging.getLogger(__name__)


@timed(LOGGER, "write configured EBM explanations")
def write_configured_ebm_explanations(
    model: ExplainableBoostingRegressor,
    output_dir: Path,
    dataset_config: PatientDatasetConfig,
    output_config: Mapping[str, object],
    config_dir: Path,
    *,
    imputation_enabled: bool,
    fail_fast: bool = False,
) -> dict[str, object]:
    """Write the standard EBM explanation artifacts for one configured run."""
    normal_ranges, normal_ranges_csv_path = explanation_normal_ranges(output_config, config_dir)
    test_name_translation, translation_csv_path = explanation_test_name_translation(output_config, config_dir)
    include_missing_bins = bool(
        output_config.get("explanation_include_missing_bins", True)
    ) and not imputation_enabled
    return write_ebm_explanations(
        model,
        output_dir,
        dataset_config.recurrence_endpoint_name,
        include_missing_bins=include_missing_bins,
        term_x_axis_scale=explanation_term_x_axis_scale(output_config),
        normal_ranges_by_feature=normal_ranges,
        normal_ranges_csv_path=normal_ranges_csv_path,
        test_name_translation=test_name_translation,
        test_name_translation_csv_path=translation_csv_path,
        write_combined_term_hazard_ratio_plot=bool(
            output_config.get("write_combined_term_hazard_ratio_plot", True)
        ),
        combined_term_hazard_ratio_columns=combined_term_hazard_ratio_columns(output_config),
        combined_term_hazard_ratio_y_limits=combined_term_hazard_ratio_y_limits(output_config),
        combined_term_hazard_ratio_show_x_markers=combined_term_hazard_ratio_show_x_markers(output_config),
        combined_term_hazard_ratio_show_uncertainty_area=(
            combined_term_hazard_ratio_show_uncertainty_area(output_config)
        ),
        fail_fast=fail_fast,
    )
