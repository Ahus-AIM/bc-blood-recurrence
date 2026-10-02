from __future__ import annotations

import logging
import re
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Mapping

import matplotlib.pyplot as plt
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from src.config import resolve_mapping_path
from src.data import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.recurrence_site_incidence import RecurrenceSiteRule
from src.model.temporal_risk import (
    TemporalRiskSiteConfig,
    assign_bootstrap_mean_quantiles,
    build_temporal_risk_figure,
    build_temporal_risk_landmark_frame,
    summarize_temporal_risk,
    temporal_risk_site_order,
)
from src.script.saved_ebm_artifacts import load_feature_imputer, predict_risk_scores
from src.time_units import DAYS_PER_MONTH

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TemporalRiskConfig:
    """Output, site, month, and typography settings for the temporal-risk figure."""

    output_csv_path: Path
    output_png_path: Path
    output_svg_path: Path
    figure_width: float
    panel_height: float
    dpi: int
    font_size: float
    title_font_size: float
    month_days: int
    site_config: TemporalRiskSiteConfig

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        output_dir: Path,
    ) -> TemporalRiskConfig:
        labels = string_tuple(config, "site_labels")
        regexes = string_tuple(config, "site_regexes")
        validate_site_rules(labels, regexes)
        site_config = TemporalRiskSiteConfig(
            site_rules=tuple(
                RecurrenceSiteRule(label, regex)
                for label, regex in zip(labels, regexes, strict=True)
            ),
            include_overall=bool(config.get("include_overall", True)),
            overall_label=str(config.get("overall_label", "Any")),
            other_label=optional_text(config.get("other_label", "Other")),
        )
        parsed = cls(
            output_csv_path=resolve_mapping_path(
                config,
                "output_csv_path",
                "temporal_risk_quantiles.csv",
                output_dir,
            ),
            output_png_path=resolve_mapping_path(
                config,
                "output_png_path",
                "temporal_risk_quantiles.png",
                output_dir,
            ),
            output_svg_path=resolve_mapping_path(
                config,
                "output_svg_path",
                "temporal_risk_quantiles.svg",
                output_dir,
            ),
            figure_width=float(config.get("figure_width", 7.0)),
            panel_height=float(config.get("panel_height", 1.75)),
            dpi=int(config.get("dpi", 300)),
            font_size=float(config.get("font_size", 8.0)),
            title_font_size=float(config.get("title_font_size", 9.0)),
            month_days=int(config.get("month_days", DAYS_PER_MONTH)),
            site_config=site_config,
        )
        validate_temporal_risk_config(parsed)
        return parsed


def build_temporal_risk_statistics(
    model: ExplainableBoostingRegressor,
    raw_config: Mapping[str, Mapping[str, object]],
    dataset_config: PatientDatasetConfig,
    temporal_config: TemporalRiskConfig,
    bootstrap_risk_scores: pd.DataFrame,
    output_dir: Path,
    *,
    prepared_dataset: PatientBloodTestDataset | None = None,
) -> pd.DataFrame:
    if prepared_dataset is None:
        prepared_dataset = PatientBloodTestDataset(dataset_config)
    test_dataset = PatientBloodTestDataset(
        replace(dataset_config, split="test"),
        prepared_data=prepared_dataset.prepared_data,
    )
    test_table = test_dataset.to_landmark_table(bootstrap=False)
    feature_imputer = load_feature_imputer(raw_config, output_dir, test_table)
    risk_scores = predict_risk_scores(model, feature_imputer, test_table)
    landmark_df = build_temporal_risk_landmark_frame(
        test_table,
        risk_scores,
        tuple(test_dataset.items),
        temporal_config.site_config,
        input_recurrence_code_filter_regex=dataset_config.input_recurrence_code_filter_regex,
        censoring_horizon_days=dataset_config.censoring_horizon_days,
        month_days=temporal_config.month_days,
        include_recurrence_day=dataset_config.include_recurrence_day_blood_tests,
    )
    quantile_df = assign_bootstrap_mean_quantiles(landmark_df, bootstrap_risk_scores)
    return summarize_temporal_risk(
        quantile_df,
        site_order=temporal_risk_site_order(temporal_config.site_config),
    )


def write_temporal_risk_outputs(
    stats_df: pd.DataFrame,
    temporal_config: TemporalRiskConfig,
    dataset_config: PatientDatasetConfig,
) -> None:
    temporal_config.output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    stats_df.to_csv(temporal_config.output_csv_path, index=False)
    figure = build_temporal_risk_figure(
        stats_df,
        site_order=temporal_risk_site_order(temporal_config.site_config),
        max_month=dataset_config.censoring_horizon_days // temporal_config.month_days,
        figure_width=temporal_config.figure_width,
        panel_height=temporal_config.panel_height,
        font_size=temporal_config.font_size,
        title_font_size=temporal_config.title_font_size,
    )
    temporal_config.output_png_path.parent.mkdir(parents=True, exist_ok=True)
    temporal_config.output_svg_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(temporal_config.output_png_path, dpi=temporal_config.dpi)
    figure.savefig(temporal_config.output_svg_path, dpi=temporal_config.dpi)
    plt.close(figure)
    LOGGER.info("wrote %s", temporal_config.output_csv_path)
    LOGGER.info("wrote %s", temporal_config.output_png_path)
    LOGGER.info("wrote %s", temporal_config.output_svg_path)


def require_test_evaluation_outputs(
    output_config: Mapping[str, object],
    output_dir: Path,
) -> None:
    if not bool(output_config.get("evaluate_test", False)):
        raise ValueError("`output.evaluate_test` must be true before plotting temporal risk.")
    for path in (output_dir / "test_risk_scores.csv", output_dir / "model.pkl"):
        if not path.exists():
            raise FileNotFoundError(f"Required temporal-risk input not found: {path}")


def load_test_risk_scores(csv_path: Path) -> pd.DataFrame:
    risk_scores = pd.read_csv(csv_path)
    required_columns = {"bootstrap_run", "risk_score"}
    missing = required_columns.difference(risk_scores.columns)
    if missing:
        joined = ", ".join(sorted(missing))
        raise ValueError(f"Test risk-score CSV is missing columns: {joined}.")
    if risk_scores.empty:
        raise ValueError("Test risk-score CSV is empty.")
    return risk_scores


def string_tuple(config: Mapping[str, object], key: str) -> tuple[str, ...]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"`temporal_risk.{key}` must be a list.")
    return tuple(str(item) for item in value)


def optional_text(value: object) -> str | None:
    if value is None or str(value).strip() == "":
        return None
    return str(value)


def validate_site_rules(labels: tuple[str, ...], regexes: tuple[str, ...]) -> None:
    if len(labels) != len(regexes):
        raise ValueError("`temporal_risk.site_regexes` must match `site_labels` length.")
    for regex in regexes:
        re.compile(regex)


def validate_temporal_risk_config(config: TemporalRiskConfig) -> None:
    if config.month_days < 1:
        raise ValueError("`temporal_risk.month_days` must be at least 1.")
    if config.figure_width <= 0.0:
        raise ValueError("`temporal_risk.figure_width` must be positive.")
    if config.panel_height <= 0.0:
        raise ValueError("`temporal_risk.panel_height` must be positive.")
    if config.dpi < 1:
        raise ValueError("`temporal_risk.dpi` must be at least 1.")
