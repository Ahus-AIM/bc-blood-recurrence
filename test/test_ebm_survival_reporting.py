from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import src.script.survival_dataset_artifacts as dataset_artifacts
from src.config import load_yml
from src.data import PatientLandmarkTable
from src.data.dataset_config import PatientDatasetConfig
from src.model.imputation import (
    FeatureImputationResult,
    fit_feature_imputer,
    transform_landmark_table,
)
from src.model.imputation_config import ImputationConfig
from src.script.ebm_survival_config import (
    TRAINING_CONFIG_FILENAME,
    combined_term_hazard_ratio_show_uncertainty_area,
    combined_term_hazard_ratio_show_x_markers,
    combined_term_hazard_ratio_y_limits,
    write_training_config_snapshot,
)
from src.model import survival_evaluation
from src.model.survival_evaluation import (
    SurvivalEvaluationCache,
    evaluate_prepared_table,
    prepare_evaluation_table,
    report_c_indexes,
    summarize_bootstrap_runs,
)
from src.script.survival_model_input import model_input_frame
from src.script.survival_training import (
    FittedSurvivalTraining,
    PreparedSurvivalTraining,
    fitted_run_metadata,
)


def test_survival_dataset_artifacts_include_population_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        dataset_artifacts,
        "write_eligible_patient_folds_csv",
        lambda *_args: calls.append("eligible_patient_folds") or {"path": "eligible_patient_folds.csv"},
    )
    monkeypatch.setattr(
        dataset_artifacts,
        "write_training_dataset_statistics_png",
        lambda *_args, **_kwargs: calls.append("dataset_statistics") or {"path": "dataset.png"},
    )
    monkeypatch.setattr(
        dataset_artifacts,
        "write_training_landmark_event_counts",
        lambda *_args: calls.append("landmark_event_counts") or {"latex_path": "counts.tex"},
    )
    monkeypatch.setattr(
        dataset_artifacts,
        "write_training_recurrence_blood_test_insights",
        lambda *_args: calls.append("recurrence_blood_test_insights") or {"path": "trajectories.png"},
    )
    monkeypatch.setattr(
        dataset_artifacts,
        "write_population_statistics_latex",
        lambda *_args, **_kwargs: calls.append("population_statistics") or {"path": "population.tex"},
    )

    monkeypatch.setattr(
        dataset_artifacts,
        "write_training_cohort_comparison",
        lambda *_args: calls.append("cohort_comparison") or {"csv_path": "cohort_comparison.csv"},
    )
    metadata = dataset_artifacts.write_survival_dataset_artifacts(
        object(),  # type: ignore[arg-type]
        tmp_path,
        {},
        landmark_event_counts_config={},
    )

    expected_keys = (
        "eligible_patient_folds", "dataset_statistics", "recurrence_blood_test_insights",
        "landmark_event_counts", "population_statistics", "cohort_comparison",
    )
    assert tuple(calls) == expected_keys
    assert tuple(metadata) == expected_keys


def test_bootstrap_summary_reports_mean_and_percentile_ci() -> None:
    summary = summarize_bootstrap_runs(
        [
            {"row_count": 10, "event_count": 2, "patient_count": 8, "harrell_c_index": 0.2},
            {"row_count": 10, "event_count": 3, "patient_count": 7, "harrell_c_index": 0.4},
            {"row_count": 10, "event_count": 4, "patient_count": 6, "harrell_c_index": 0.8},
        ],
        sample_count=10,
    )

    assert summary["bootstrap_runs"] == 3
    assert summary["bootstrap_sample_count"] == 10
    assert summary["harrell_c_index"] == pytest.approx((0.2 + 0.4 + 0.8) / 3)
    assert summary["harrell_c_index_valid_runs"] == 3
    assert summary["harrell_c_index_ci95_lower"] == pytest.approx(0.21)
    assert summary["harrell_c_index_ci95_upper"] == pytest.approx(0.78)
    assert summary["event_count_mean"] == pytest.approx(3.0)


def test_prepared_evaluation_table_matches_direct_evaluation() -> None:
    table = reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64))
    imputer = fit_feature_imputer(
        table.features,
        table.feature_names,
        ImputationConfig.from_mapping({"enabled": False}),
    )
    model = type("Model", (), {"predict": lambda _self, features: features[:, 0]})()

    direct_summary, direct_scores = survival_evaluation.evaluate_table(
        model,
        table,
        imputer,
        3,
        feature_names=("A|||u",),
    )
    prepared = prepare_evaluation_table(table, imputer, 3)
    prepared_summary, prepared_scores = evaluate_prepared_table(
        model,
        prepared,
        feature_names=("A|||u",),
    )

    assert prepared_summary == direct_summary
    assert prepared_scores.equals(direct_scores)


def test_evaluation_scores_report_full_bootstrap_selection_probability() -> None:
    table = replace(
        reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64)),
        bootstrap_selection_probabilities=np.asarray([0.25, 0.75]),
    )
    imputer = fit_feature_imputer(
        table.features,
        table.feature_names,
        ImputationConfig.from_mapping({"enabled": False}),
    )
    model = type("Model", (), {"predict": lambda _self, features: features[:, 0]})()

    _summary, scores = survival_evaluation.evaluate_table(model, table, imputer, 3)

    assert scores["bootstrap_selection_probability"].tolist() == [0.25, 0.75]


def test_survival_evaluation_cache_prepares_each_split_once(monkeypatch: pytest.MonkeyPatch) -> None:
    prepared_split = object()
    calls: list[str] = []

    def prepare(split, *_args, **_kwargs):
        calls.append(split)
        return prepared_split

    monkeypatch.setattr(survival_evaluation, "prepare_split_evaluation", prepare)
    cache = SurvivalEvaluationCache(
        dataset_config=object(),  # type: ignore[arg-type]
        feature_imputer=object(),  # type: ignore[arg-type]
        landmarks=object(),  # type: ignore[arg-type]
    )

    assert cache.get("test") is prepared_split
    assert cache.get("test") is prepared_split
    assert calls == ["test"]


def test_report_c_indexes_includes_train_bootstrap(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO)

    report_c_indexes(
        {
            "train": {"row_count": 30},
            "train_bootstrap": {
                "harrell_c_index": 0.7,
                "harrell_c_index_ci95_lower": 0.6,
                "harrell_c_index_ci95_upper": 0.8,
                "bootstrap_runs": 10,
            },
            "validation": {
                "harrell_c_index": 0.5,
                "harrell_c_index_ci95_lower": 0.4,
                "harrell_c_index_ci95_upper": 0.6,
                "bootstrap_runs": 10,
            },
        }
    )

    lines = [record.getMessage() for record in caplog.records if record.name == "src.model.survival_evaluation"]
    assert lines[0] == "train_bootstrap harrell_c_index=0.700000 ci95=[0.600000, 0.800000] bootstrap_runs=10"
    assert lines[1] == ("validation harrell_c_index=0.500000 ci95=[0.400000, 0.600000] bootstrap_runs=10")


def test_model_input_frame_contains_targets_weights_and_exact_features() -> None:
    table = replace(
        reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64)),
        sample_weight_interval_days=np.asarray([10, 20], dtype=np.int32),
        target_end_reasons=np.asarray(["configured_recurrence", "other_cancer"], dtype=object),
        right_censoring_days=np.asarray([30, 30], dtype=np.float64),
        right_censoring_reasons=np.asarray(["last_observation", "other_cancer"], dtype=object),
    )

    frame = model_input_frame("train", table)

    assert list(frame.columns) == [
        "split",
        "patient_id",
        "landmark_day",
        "duration_days",
        "event_observed",
        "right_censoring_day",
        "right_censoring_reason",
        "target_end_reason",
        "signed_duration_days",
        "prior_non_endpoint_recurrence",
        "sample_weight",
        "sample_weight_interval_days",
        "A|||u",
        "recent_diagnosis_D1__thing",
    ]
    assert frame["split"].tolist() == ["train", "train"]
    assert frame["sample_weight"].tolist() == [2.0, 3.0]
    assert frame["sample_weight_interval_days"].tolist() == [10, 20]
    assert frame["signed_duration_days"].tolist() == [10.0, -20.0]
    assert frame[["right_censoring_day", "right_censoring_reason", "target_end_reason"]].to_dict("list") == {
        "right_censoring_day": [30.0, 30.0],
        "right_censoring_reason": ["last_observation", "other_cancer"],
        "target_end_reason": ["configured_recurrence", "other_cancer"],
    }
    assert frame["recent_diagnosis_D1__thing"].isna().tolist() == [True, False]


def test_model_input_frame_leaves_non_train_weights_blank() -> None:
    table = replace(
        reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64)),
        sample_weight_interval_days=np.asarray([10, 20], dtype=np.int32),
    )

    frame = model_input_frame("val", table)

    assert frame["split"].tolist() == ["val", "val"]
    assert frame["sample_weight"].isna().all()
    assert frame["sample_weight_interval_days"].isna().all()


def test_model_input_frame_leaves_inactive_weight_intervals_blank() -> None:
    table = reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64))

    frame = model_input_frame("train", table)

    assert frame["sample_weight_interval_days"].isna().all()


def test_model_input_frame_marks_imputed_values_inline() -> None:
    table = reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64))
    imputation_result = FeatureImputationResult(
        features=np.asarray([[0.0, 4.25], [2.5, 1.0]], dtype=np.float32),
        imputed_mask=np.asarray([[True, True], [False, False]], dtype=bool),
        clipped_mask=np.asarray([[True, False], [False, False]], dtype=bool),
    )

    frame = model_input_frame("train", table, imputation_result)

    assert frame["A|||u"].tolist() == ["0 mice clipped", 2.5]
    assert frame["recent_diagnosis_D1__thing"].tolist() == ["4.25 mice", 1.0]


def test_model_input_frame_marks_imputation_sources_inline() -> None:
    table = reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64))
    imputation_result = FeatureImputationResult(
        features=np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32),
        imputed_mask=np.ones((2, 2), dtype=bool),
        clipped_mask=np.zeros((2, 2), dtype=bool),
        carried_forward_mask=np.asarray([[True, False], [False, False]], dtype=bool),
        median_fallback_mask=np.asarray([[False, True], [False, False]], dtype=bool),
        mice_fallback_mask=np.asarray([[False, False], [True, False]], dtype=bool),
    )

    frame = model_input_frame("train", table, imputation_result)

    assert frame["A|||u"].tolist() == ["7 locf", "9 fallback mice"]
    assert frame["recent_diagnosis_D1__thing"].tolist() == ["8 fallback median", "10 mice"]


def test_fitted_ebm_metadata_keeps_config_fields_at_legacy_location() -> None:
    table = reporting_landmark_table(np.asarray([2.0, 3.0], dtype=np.float64))
    imputer = fit_feature_imputer(
        table.features,
        table.feature_names,
        ImputationConfig.from_mapping({"enabled": False}),
    )
    prepared = PreparedSurvivalTraining(
        dataset=object(),  # type: ignore[arg-type]
        train_table=table,
        imputer=imputer,
        train_imputation=transform_landmark_table(imputer, table),
    )
    fitted = FittedSurvivalTraining(
        prepared=prepared,
        model=object(),  # type: ignore[arg-type]
        model_metadata={
            "library": "interpret",
            "version": "test",
            "config": {"interactions": 3},
            "patient_grouped_outer_bags_enabled": False,
        },
        split_summaries={},
    )

    metadata = fitted_run_metadata(fitted, {})

    assert metadata["ebm"]["interactions"] == 3
    assert metadata["ebm"]["config"] == {"interactions": 3}


def test_training_config_snapshot_resolves_run_paths(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    output_dir = tmp_path / "output" / "distant_recurrence"
    output_dir.mkdir(parents=True)
    raw_config = {
        "dataset": {
            "bloodtest_csv_path": "../data/blood.csv",
            "endpoint_csv_path": "../data/endpoint.csv",
            "split_csv_path": "../data/split.csv",
            "selected_blood_tests": [],
        },
        "imputation": {"enabled": False},
        "output": {
            "output_dir": "../output",
            "normal_ranges_csv_path": "normal_ranges.csv",
            "bloodtest_translation_csv_path": "translations.csv",
        },
    }
    dataset_config = PatientDatasetConfig.from_mapping(raw_config["dataset"], config_dir=config_dir)

    metadata = write_training_config_snapshot(raw_config, dataset_config, output_dir, config_dir)

    snapshot = load_yml(output_dir / TRAINING_CONFIG_FILENAME)
    assert metadata == {"path": TRAINING_CONFIG_FILENAME}
    assert snapshot["dataset"]["bloodtest_csv_path"] == str(data_dir / "blood.csv")
    assert snapshot["dataset"]["endpoint_csv_path"] == str(data_dir / "endpoint.csv")
    assert snapshot["dataset"]["split_csv_path"] == str(data_dir / "split.csv")
    assert snapshot["output"]["output_dir"] == str(tmp_path / "output")
    assert snapshot["output"]["normal_ranges_csv_path"] == str(config_dir / "normal_ranges.csv")
    assert snapshot["output"]["bloodtest_translation_csv_path"] == str(config_dir / "translations.csv")


def test_combined_hazard_ratio_grid_style_config_has_requested_defaults_and_overrides() -> None:
    assert combined_term_hazard_ratio_show_x_markers({}) is False
    assert combined_term_hazard_ratio_show_uncertainty_area({}) is True
    assert combined_term_hazard_ratio_show_x_markers({"combined_term_hazard_ratio_show_x_markers": True}) is True
    assert (
        combined_term_hazard_ratio_show_uncertainty_area({"combined_term_hazard_ratio_show_uncertainty_area": False})
        is False
    )


def test_combined_hazard_ratio_grid_y_limits_default_and_overrides() -> None:
    assert combined_term_hazard_ratio_y_limits({}) == (0.3, 4.0)
    assert combined_term_hazard_ratio_y_limits(
        {
            "combined_term_hazard_ratio_y_min": 0.4,
            "combined_term_hazard_ratio_y_max": 2.5,
        }
    ) == (0.4, 2.5)


def reporting_landmark_table(sample_weights: np.ndarray | None) -> PatientLandmarkTable:
    durations = np.asarray([10, 20], dtype=np.int32)
    events = np.asarray([True, False], dtype=bool)
    return PatientLandmarkTable(
        features=np.asarray([[1.5, np.nan], [2.5, 1.0]], dtype=np.float32),
        signed_durations=np.where(events, durations, -durations).astype(np.float64),
        durations=durations,
        events=events,
        sample_weights=sample_weights,
        feature_names=("A|||u", "recent_diagnosis_D1__thing"),
        patient_ids=np.asarray(["P0", "P1"], dtype=object),
        landmark_days=np.asarray([3, 4], dtype=np.int32),
        prior_non_endpoint_recurrence=np.asarray([False, True], dtype=bool),
        metadata={},
    )
