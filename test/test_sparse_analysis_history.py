from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.cohort_selection import (
    MEASUREMENT_EXCLUSION_STAGE_COLUMN,
    STAGE_EARLY_MODEL_MEASUREMENTS,
    STAGE_MIN_BLOOD_TEST_TYPES,
    STAGE_SPARSE_MODEL_MEASUREMENTS,
)
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.measurement_selection import selected_recurrence_measurements
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, RecurrenceTimeWindow
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_site_common import RecurrenceSiteRule
from test.test_patient_bloodtest import blood_row, config, endpoint_row


@pytest.mark.parametrize("changes", [False, True])
@pytest.mark.parametrize("include_end", [False, True])
def test_sparse_days_restore_analysis_without_changing_model_inputs(
    tmp_path: Path, changes: bool, include_end: bool,
) -> None:
    rows = [blood_row("keep", day, test_name=test, result=value, unit="u") for day, test, value in (
        (-10, "A", 4), (0, "A", 5), (31, "A", 10), (31, "B", 100), (40, "A", 999),
        (60, "A", 20), (60, "B", 200), (100, "A", 30), (100, "B", 300), (110, "A", 40), (110, "B", 400),
    )]
    rows += [blood_row(patient, day, test_name=test, unit="u") for patient, day, test in (
        ("early_dense", 0, "A"), ("early_dense", 0, "B"), ("early_dense", 40, "A"),
        ("late_dense", 40, "A"), ("late_dense", 110, "A"), ("late_dense", 110, "B"),
    )]
    cfg = config(tmp_path, rows, [endpoint_row(patient, diagnosis_to_ddfs=100, ddfs_code="C780", last_entry_days=200)
                                 for patient in ("keep", "early_dense", "late_dense")])
    cfg.update(selected_blood_tests=["A|||u", "B|||u"], min_landmark_day=31, min_blood_test_types_per_day=2,
               include_blood_test_change=changes, include_recurrence_day_blood_tests=include_end,
               build_patient_timeline_tensors=True)
    dataset = PatientBloodTestDataset(cfg)
    assert dataset.patient_ids == ("keep",)
    assert dataset.items[0].valid_landmark_days == ((31, 60, 100) if include_end else (31, 60))
    analysis = selected_recurrence_measurements(dataset.prepared_data)
    assert set(analysis["patient_id"]) == {"keep"}
    assert set(analysis["days_since_diagnosis"]) == ({-10, 0, 31, 40, 60, 100} if include_end else {-10, 0, 31, 40, 60})
    assert analysis["source_index"].is_unique
    stages = dataset.prepared_data.cohort_stages
    eligibility = next(stage for stage in stages if stage.key == STAGE_MIN_BLOOD_TEST_TYPES)
    assert eligibility.patient_ids == stages[-1].patient_ids == {"keep"}
    assert eligibility.bloodtest_count == len(analysis)
    assert stages[-1].key == STAGE_SPARSE_MODEL_MEASUREMENTS
    assert stages[-2].bloodtest_count - stages[-1].bloodtest_count == 1
    assert stages[-2].patient_ids == stages[-1].patient_ids
    assert (stages[-2].key == STAGE_EARLY_MODEL_MEASUREMENTS) is (not changes)
    excluded = dataset.prepared_data.excluded_measurement_df
    sparse = excluded.loc[excluded[MEASUREMENT_EXCLUSION_STAGE_COLUMN] == STAGE_SPARSE_MODEL_MEASUREMENTS]
    assert sparse["days_since_diagnosis"].tolist() == [40]
    summary = build_dataset_statistics_summary(dataset.config, prepared_dataset=dataset)
    assert summary.exclusion_box_counts[-1] == "N = 0\nblood tests = 1"
    assert summary.exclusion_box_labels[-1] not in summary.series_labels
    assert sum(summary.blood_test_counts) == stages[-1].bloodtest_count
    assert_models_match_without_sparse_row(dataset, tmp_path)
    if changes:
        table = dataset.to_landmark_table(bootstrap=False)
        # Day 40's value 999 must not become the past input for the day-60 change.
        row = table.features[table.landmark_days == 60][0]
        assert row[table.feature_names.index("A|||u change")] == 10
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml", dataset.config, (RecurrenceSiteRule("Lung", r"C780\d*"),),
        None, None, tmp_path / "report",
        windows=(RecurrenceTimeWindow("Baseline", 30, 99), RecurrenceTimeWindow("Final", 0, 29)),
    )
    frame = build_recurrence_blood_test_insights(dataset, report, {})
    baseline = frame.loc[(frame["test_key"] == "A|||u") & (frame["window"] == "Baseline")]
    assert baseline["median"].tolist() == [20]  # Median of days 31, 40 and 60; the sparse day contributes.


def assert_models_match_without_sparse_row(dataset: PatientBloodTestDataset, tmp_path: Path) -> None:
    """Removing a report-only row must not affect features, labels, weights or bootstrap samples."""
    source = pd.read_csv(dataset.config.bloodtest_csv_path)
    sparse = source["id"].eq("keep") & source["days_since_diagnosis"].eq(40)
    path = tmp_path / "without_sparse.csv"
    source.loc[~sparse].to_csv(path, index=False)
    reference = PatientBloodTestDataset(replace(dataset.config, bloodtest_csv_path=path))
    for bootstrap in (False, True):
        actual = dataset.to_landmark_table(bootstrap=bootstrap)
        expected = reference.to_landmark_table(bootstrap=bootstrap)
        for field in ("features", "patient_ids", "landmark_days", "durations", "events", "sample_weights"):
            np.testing.assert_array_equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("minimum", [None, 1])
@pytest.mark.parametrize("changes", [False, True])
def test_disabled_or_single_test_minimum_has_no_visible_sparse_exclusions(
    tmp_path: Path, minimum: int | None, changes: bool,
) -> None:
    cfg = config(tmp_path, [blood_row("P", day) for day in (-1, 31, 40)], [endpoint_row("P")])
    cfg.update(min_landmark_day=31, min_blood_test_types_per_day=minimum, include_blood_test_change=changes)
    dataset = PatientBloodTestDataset(cfg)
    assert dataset.items[0].valid_landmark_days == (31, 40)
    assert selected_recurrence_measurements(dataset.prepared_data)["days_since_diagnosis"].tolist() == [-1, 31, 40]
    summary = build_dataset_statistics_summary(dataset.config, prepared_dataset=dataset)
    assert not any("different blood tests" in label for label in summary.inclusion_box_labels)
