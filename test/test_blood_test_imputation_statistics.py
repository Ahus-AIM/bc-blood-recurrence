from __future__ import annotations

from pathlib import Path

from src.data import PatientBloodTestDataset
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.population_statistics import (
    PopulationStatisticsConfig,
    build_population_statistics_table,
    population_statistics_latex,
)
from src.model.blood_test_imputation_statistics import (
    build_blood_test_imputation_statistics,
)
from src.script.survival_training import prepare_survival_training
from test.test_patient_bloodtest import blood_row, config as dataset_config, endpoint_row
from test.test_population_statistics import population_statistics_dataset


def test_blood_test_imputation_statistics_count_each_value_source(
    tmp_path: Path,
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    train_table = dataset.to_landmark_table(bootstrap=False)
    prepared = prepare_survival_training(
        dataset,
        train_table,
        {
            "enabled": True,
            "method": "locf",
            "locf_max_age_days": 30,
            "locf_fallback_method": "mice",
            "max_iter": 10,
            "tol": 0.001,
            "random_state": 42,
        },
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )

    statistics = build_blood_test_imputation_statistics(
        dataset,
        prepared.imputer,
    )

    assert statistics.by_test_key["A|||u"].as_dict() == {
        "measured": 4,
        "locf": 0,
        "mice": 0,
    }
    assert statistics.by_test_key["B|||u"].as_dict() == {
        "measured": 1,
        "locf": 1,
        "mice": 2,
    }
    assert statistics.by_split["train"].as_dict() == {
        "measured": 3,
        "locf": 1,
        "mice": 0,
    }
    assert statistics.by_split["val"].as_dict() == {
        "measured": 1,
        "locf": 0,
        "mice": 1,
    }
    assert statistics.by_split["test"].as_dict() == {
        "measured": 1,
        "locf": 0,
        "mice": 1,
    }
    assert statistics.overall.as_dict() == {
        "measured": 5,
        "locf": 1,
        "mice": 2,
    }


def test_dataset_statistics_summary_reports_each_imputation_source(
    tmp_path: Path,
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    prepared = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {
            "enabled": True,
            "method": "locf",
            "locf_max_age_days": 30,
            "locf_fallback_method": "mice",
        },
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )
    statistics = build_blood_test_imputation_statistics(dataset, prepared.imputer)

    summary = build_dataset_statistics_summary(
        dataset.config,
        prepared_dataset=dataset,
        imputation_statistics=statistics,
    )

    assert summary.blood_test_names == ("A", "B")
    assert summary.blood_test_counts == (4, 1)
    assert summary.blood_test_locf_counts == (0, 1)
    assert summary.blood_test_mice_counts == (0, 2)


def test_population_statistics_table_indents_blood_test_value_sources(
    tmp_path: Path,
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    prepared = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {
            "enabled": True,
            "method": "locf",
            "locf_max_age_days": 30,
            "locf_fallback_method": "mice",
        },
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )
    statistics = build_blood_test_imputation_statistics(dataset, prepared.imputer)
    config = PopulationStatisticsConfig.from_mapping(None)

    table = build_population_statistics_table(
        dataset,
        config,
        imputation_statistics=statistics,
    )
    latex = population_statistics_latex(table, config)

    assert "Landmark days & 2 & 1 & 1 & 4 \\\\" in latex
    assert "Blood tests & 4 & 2 & 2 & 8 \\\\" in latex
    assert r"\quad Measured & 3 (75.0\%) & 1 (50.0\%) & 1 (50.0\%) & 5 (62.5\%) \\" in latex
    assert r"\quad LOCF & 1 (25.0\%) & 0 (0.0\%) & 0 (0.0\%) & 1 (12.5\%) \\" in latex
    assert r"\quad MICE & 0 (0.0\%) & 1 (50.0\%) & 1 (50.0\%) & 2 (25.0\%) \\" in latex


def test_blood_test_imputation_statistics_allow_empty_temporal_splits(
    tmp_path: Path,
) -> None:
    train_only_config = dataset_config(
        tmp_path,
        [
            blood_row("train", 40, test_name="A", unit="u"),
            blood_row("train", 40, test_name="B", unit="u"),
            blood_row("train", 50, test_name="A", unit="u"),
        ],
        [endpoint_row("train", last_entry_days=100)],
        split_by_patient={"train": "train"},
    )
    train_only_config["selected_blood_tests"] = ["A|||u", "B|||u"]
    train_only = PatientBloodTestDataset(train_only_config)
    prepared = prepare_survival_training(
        train_only,
        train_only.to_landmark_table(bootstrap=False),
        {
            "enabled": True,
            "method": "locf",
            "locf_max_age_days": 30,
            "locf_fallback_method": "mice",
        },
        nonnegative_feature_names=tuple(train_only.selected_test_keys),
    )

    statistics = build_blood_test_imputation_statistics(
        train_only,
        prepared.imputer,
    )

    assert statistics.by_split["val"].total == 0
    assert statistics.by_split["test"].total == 0
