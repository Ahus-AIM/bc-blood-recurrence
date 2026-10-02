from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.config import load_yml
from src.model.ebm_benchmark_top_features import EbmTopFeatureRow, write_ebm_top_feature_grid
from src.model.ebm_plotting_defaults import TOP_FEATURES_TITLE
from src.model.ebm_visualizations import HazardRatioGridStyle
from src.reporting import display_output_path, recurrence_endpoint_label, report_output_path
from src.script.ebm_survival_config import (
    METADATA_FILENAME,
    TRAINING_CONFIG_FILENAME,
    combined_term_hazard_ratio_show_uncertainty_area,
    combined_term_hazard_ratio_show_x_markers,
    combined_term_hazard_ratio_y_limits,
    explanation_normal_ranges,
    explanation_term_x_axis_scale,
    explanation_test_name_translation,
    safe_path_component,
)
from src.script.saved_ebm_artifacts import load_model
from src.script.single_bloodtest_benchmark_config import benchmark_model_output_dir, mapping_sequence
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY

LOGGER = logging.getLogger(__name__)


def write_benchmark_ebm_top_features(
    benchmark: Mapping[str, object],
    output_dir: Path,
    results: Sequence[Mapping[str, object]],
    *,
    fail_fast: bool = False,
) -> list[dict[str, object]]:
    """Plot saved all-tests models across sites, once per configured horizon.

    Current aggregate results exclude failed or unvisited cells even if older
    complete models remain on disk. Split-level metric non-estimability does
    not exclude an otherwise successfully fitted model.
    """
    if not bool(benchmark.get("include_all_blood_tests", True)):
        return []
    figure_config = benchmark.get("ebm_top_features", {})
    if not isinstance(figure_config, Mapping):
        raise TypeError("`benchmark.ebm_top_features` must be a mapping.")
    figure_dir = report_output_path(output_dir, figure_config.get("output_dir", "top_features"))
    available_cells = {
        (str(result["site_label"]), str(result["horizon_label"]))
        for result in results
        if result.get("model_key") == ALL_MODEL_KEY and result.get("status") != "failed"
    }
    sites = mapping_sequence(benchmark["sites"], "sites")
    artifacts: list[dict[str, object]] = []
    for horizon in mapping_sequence(benchmark["horizons"], "horizons"):
        horizon_label = str(horizon["label"])
        rows = tuple(
            load_site_row(
                output_dir, site, horizon,
                available=(str(site["label"]), horizon_label) in available_cells,
                fail_fast=fail_fast,
            )
            for site in sites
        )
        horizon_dir = figure_dir / safe_path_component(horizon_label)
        png_path = horizon_dir / "ebm_top_features.png"
        svg_path = horizon_dir / "ebm_top_features.svg"
        write_ebm_top_feature_grid(rows, png_path, svg_path, TOP_FEATURES_TITLE)
        artifacts.append({
            "horizon_label": horizon_label,
            "horizon_days": horizon["days"],
            "png": display_output_path(png_path, output_dir),
            "svg": display_output_path(svg_path, output_dir),
        })
        LOGGER.info("wrote EBM top-feature figures: %s and %s", png_path, svg_path)
    return artifacts


def load_site_row(
    output_dir: Path,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
    *,
    available: bool,
    fail_fast: bool = False,
) -> EbmTopFeatureRow:
    """Keep a labeled blank row when the all-tests model is unavailable."""
    label = str(site["label"])
    display_label = recurrence_endpoint_label(label)
    blank_row = EbmTopFeatureRow(display_label, None, None)
    if not available:
        if fail_fast:
            raise FileNotFoundError(f"No saved all-tests result for {label}/{horizon['label']}.")
        return blank_row
    model_dir = benchmark_model_output_dir(output_dir, ALL_MODEL_KEY, site, horizon)
    try:
        metadata_path = model_dir / METADATA_FILENAME
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, Mapping):
            raise TypeError(f"Metadata file must contain a JSON object: {metadata_path}")
        if metadata.get("status") != "complete":
            if fail_fast:
                raise ValueError(f"The saved all-tests model is not complete: {model_dir}")
            return blank_row
        model = load_model(model_dir / "model.pkl")
        saved_config = load_yml(model_dir / TRAINING_CONFIG_FILENAME)
        style = saved_plot_style(saved_config["output"], model_dir)
        return EbmTopFeatureRow(display_label, model.explain_global(), style)
    except Exception:
        if fail_fast:
            raise
        LOGGER.exception("leaving EBM top-feature row blank; could not load %s", model_dir)
        return blank_row


def saved_plot_style(
    output_config: Mapping[str, object],
    config_dir: Path,
) -> HazardRatioGridStyle:
    """Use the combined-grid style recorded by the saved EBM run."""
    normal_ranges, _normal_ranges_path = explanation_normal_ranges(output_config, config_dir)
    translations, _translation_path = explanation_test_name_translation(output_config, config_dir)
    return HazardRatioGridStyle(
        normal_ranges_by_feature=normal_ranges,
        test_name_translation=translations,
        x_axis_scale=explanation_term_x_axis_scale(output_config),
        y_limits=combined_term_hazard_ratio_y_limits(output_config),
        show_x_markers=combined_term_hazard_ratio_show_x_markers(output_config),
        show_uncertainty_area=combined_term_hazard_ratio_show_uncertainty_area(output_config),
    )
