from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from src.data import PatientBloodTestDataset
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.dataset_statistics_export import dataset_statistics_payload
from src.data.population_statistics import PopulationStatisticsConfig, build_population_statistics_table
from src.model.blood_test_imputation_statistics import (
    build_blood_test_imputation_statistics,
    build_blood_test_imputation_statistics_from_tables,
)
from src.model.imputation import transform_landmark_table
from src.model.imputation_metadata import imputation_result_counts
from src.script.survival_training import prepare_survival_training
from test.test_patient_bloodtest import blood_row, config, endpoint_row


def past_imputation_dataset(tmp_path: Path) -> PatientBloodTestDataset:
    """Reuse observed training history; validation needs both current and past A."""
    records = {
        "P": ((-10, "A", 3), (-10, "B", 2), (40, "A", 10), (40, "B", 8), (50, "B", 9), (60, "B", 10)),
        "V": ((40, "B", 8), (50, "B", 9)),
    }
    cfg = config(
        tmp_path,
        [
            blood_row(patient, day, test_name=test, result=value, unit="u")
            for patient, measurements in records.items()
            for day, test, value in measurements
        ],
        [endpoint_row(patient, last_entry_days=100) for patient in records],
        split_by_patient={"P": "train", "V": "val"},
    )
    cfg.update(
        min_landmark_day=31,
        selected_blood_tests=["A|||u", "B|||u"],
        include_blood_test_change=True,
    )
    return PatientBloodTestDataset(cfg)


@pytest.mark.parametrize(
    ("enabled", "method", "fallback", "expected_locf", "expected_mice"),
    [
        (True, "locf", "mice", 2, 5),
        (True, "mice", "mice", 0, 7),
        (True, "locf", "median", 2, 0),
        (False, "locf", "mice", 0, 0),
    ],
)
def test_statistics_count_each_completed_past_input_and_each_actual_measurement_once(
    tmp_path: Path,
    enabled: bool,
    method: str,
    fallback: str,
    expected_locf: int,
    expected_mice: int,
) -> None:
    dataset = past_imputation_dataset(tmp_path)
    tables = {
        split: PatientBloodTestDataset(
            replace(dataset.config, split=split), prepared_data=dataset.prepared_data
        ).to_landmark_table(bootstrap=False)
        for split in ("train", "val", "test")
    }
    prepared = prepare_survival_training(
        dataset,
        tables["train"],
        {"enabled": enabled, "method": method, "locf_fallback_method": fallback},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    results = {split: transform_landmark_table(prepared.imputer, tables[split]) for split in ("train", "val")}
    cached = build_blood_test_imputation_statistics_from_tables(dataset, prepared.imputer, tables, results)
    assert cached == build_blood_test_imputation_statistics(dataset, prepared.imputer)
    assert cached.overall.as_dict() == {"measured": 8, "locf": expected_locf, "mice": expected_mice}
    assert cached.by_test_key["A|||u"].measured == 2  # Reusing the day-40 result adds no Measured counts.
    assert cached.by_split["test"].total == 0
    expected_past = 3 if enabled and fallback == "mice" else 0
    assert imputation_result_counts(results["val"])["past_mice_cell_count"] == expected_past
    if expected_past:
        # Two validation A inputs each need current and past MICE; B needs one past MICE.
        assert cached.by_split["val"].mice == 5
        assert cached.by_test_key["B|||u"].mice == 1


def test_figure_and_population_totals_include_past_mice_in_percentage_denominators(tmp_path: Path) -> None:
    dataset = past_imputation_dataset(tmp_path)
    prepared = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {"enabled": True, "method": "locf", "locf_fallback_method": "mice"},
        nonnegative_feature_names=dataset.selected_test_keys,
    )
    statistics = build_blood_test_imputation_statistics(dataset, prepared.imputer)
    summary = build_dataset_statistics_summary(
        dataset.config, prepared_dataset=dataset, imputation_statistics=statistics
    )
    assert summary.blood_test_names == ("B", "A")
    assert summary.blood_test_counts == (6, 2)
    assert summary.blood_test_locf_counts == (0, 2)
    assert summary.blood_test_mice_counts == (1, 4)
    assert summary.inclusion_box_counts[-1] == "N = 2\nblood tests = 8"  # Panel A counts observations.
    payload = dataset_statistics_payload(summary)
    assert payload["panel_c"]["blood_test_mice_counts"] == [1, 4]
    assert [row["mice_count"] for row in payload["tables"]["bloodtest_count_table"]] == [1, 4]

    population = build_population_statistics_table(
        dataset, PopulationStatisticsConfig.from_mapping({}), imputation_statistics=statistics
    )
    rows = {row.label: row.cells for row in population.rows}
    assert rows["Landmark days"] == ("3", "2", "--", "5")
    assert rows["Blood tests"] == ("8", "7", "--", "15")
    assert rows["Measured"][-1] == r"8 (53.3\%)"
    assert rows["LOCF"][-1] == r"2 (13.3\%)"
    assert rows["MICE"][1] == r"5 (71.4\%)"
    assert rows["MICE"][-1] == r"5 (33.3\%)"
