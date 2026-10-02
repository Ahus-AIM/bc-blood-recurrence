from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import (
    BenchmarkLandmarks,
    benchmark_target_dataset,
    benchmark_washout_dataset,
)


def test_one_canonical_table_supplies_every_target_and_bootstrap(tmp_path: Path) -> None:
    canonical = PatientBloodTestDataset(canonical_config(tmp_path))
    landmarks = BenchmarkLandmarks(canonical)
    all_table = landmarks.table(canonical.config)
    targets = (
        replace(canonical.config, censoring_horizon_days=20),
        replace(
            canonical.config,
            recurrence_endpoint_name="liver",
            endpoint_ddfs_code_filter_regex=r"C787",
            censoring_horizon_days=100,
        ),
        replace(
            canonical.config,
            recurrence_endpoint_name="brain",
            endpoint_ddfs_code_filter_regex=r"C793",
        ),
    )
    for target in targets:
        projected = landmarks.table(target)
        fresh_dataset = PatientBloodTestDataset(target)
        fresh = fresh_dataset.to_landmark_table(bootstrap=False)

        assert projected.features is all_table.features
        assert projected.patient_ids is all_table.patient_ids
        assert projected.landmark_days is all_table.landmark_days
        assert projected.events.tolist() == fresh.events.tolist()
        assert projected.durations.tolist() == fresh.durations.tolist()
        assert projected.target_end_reasons.tolist() == fresh.target_end_reasons.tolist()
        assert projected.right_censoring_reasons.tolist() == fresh.right_censoring_reasons.tolist()
        np.testing.assert_equal(projected.right_censoring_days, fresh.right_censoring_days)
        assert projected.metadata["input_fingerprint"] == fresh_dataset.prepared_data.input_fingerprint

    brain_config = targets[-1]
    all_draw = landmarks.table(canonical.config, bootstrap_run=0)
    brain_draw = landmarks.table(brain_config, bootstrap_run=0)
    direct_draw = canonical.to_landmark_table(bootstrap=True, bootstrap_run=0)

    assert all_draw.features is brain_draw.features
    assert all_draw.patient_ids is brain_draw.patient_ids
    assert all_draw.landmark_days is brain_draw.landmark_days
    assert all_draw.bootstrap_selection_probabilities is brain_draw.bootstrap_selection_probabilities
    np.testing.assert_equal(all_draw.features, direct_draw.features)
    np.testing.assert_equal(all_draw.patient_ids, direct_draw.patient_ids)
    np.testing.assert_equal(all_draw.landmark_days, direct_draw.landmark_days)
    np.testing.assert_equal(
        all_draw.bootstrap_selection_probabilities,
        direct_draw.bootstrap_selection_probabilities,
    )


def test_site_dataset_is_only_a_config_view_of_the_canonical_cohort(tmp_path: Path) -> None:
    canonical = PatientBloodTestDataset(canonical_config(tmp_path))

    site = benchmark_target_dataset(
        canonical,
        replace(
            canonical.config,
            recurrence_endpoint_name="Brain",
            endpoint_ddfs_code_filter_regex=r"C793",
        ),
    )

    assert site.items is canonical.items
    assert site.prepared_data.measurement_df is canonical.prepared_data.measurement_df
    assert site.prepared_data.input_fingerprint != canonical.prepared_data.input_fingerprint


def test_washout_dataset_excludes_inclusive_days_before_observed_boundary(
    tmp_path: Path,
) -> None:
    canonical = PatientBloodTestDataset(canonical_config(tmp_path))

    zero_dataset = benchmark_washout_dataset(canonical, 0)
    zero_days = zero_dataset.items[0].valid_landmark_days
    thirty_days = benchmark_washout_dataset(canonical, 30).items[0].valid_landmark_days

    assert zero_dataset is canonical
    assert zero_days == (10,)
    assert thirty_days == ()


@pytest.mark.parametrize(("horizon_days", "washout_days"), [(30, 0), (30, 29), (180, 30), (180, 60)])
def test_washout_matches_final_event_or_censoring_duration_before_bootstrap(
    tmp_path: Path,
    horizon_days: int,
    washout_days: int,
) -> None:
    canonical = every_outcome_dataset(tmp_path)
    original = BenchmarkLandmarks(canonical)
    liver_config = replace(canonical.config, recurrence_endpoint_name="liver", endpoint_ddfs_code_filter_regex=r"C787")
    full_table = original.table(liver_config)
    assert set(full_table.target_end_reasons) == {
        "configured_recurrence",
        "death",
        "last_observation",
        "other_cancer",
        "non_endpoint_recurrence",
    }
    assert sorted(full_table.durations.tolist()) == [29] * 5 + [30] * 5 + [31] * 5

    filtered = BenchmarkLandmarks(benchmark_washout_dataset(canonical, washout_days))
    all_config = replace(canonical.config, censoring_horizon_days=horizon_days)
    for target in (all_config, replace(liver_config, censoring_horizon_days=horizon_days)):
        before = original.table(target)
        after = filtered.table(target)
        eligible = before.durations > washout_days
        np.testing.assert_equal(after.patient_ids, before.patient_ids[eligible])
        np.testing.assert_equal(after.landmark_days, before.landmark_days[eligible])
        np.testing.assert_equal(after.durations, before.durations[eligible])
        np.testing.assert_equal(after.events, before.events[eligible])
        if not eligible.any():
            continue
        for run in range(3):
            draw = filtered.table(target, bootstrap_run=run)
            all_draw = filtered.table(all_config, bootstrap_run=run)
            assert len(draw.durations) == len(np.unique(after.patient_ids))
            assert (draw.durations > washout_days).all()
            np.testing.assert_equal(draw.patient_ids, all_draw.patient_ids)
            np.testing.assert_equal(draw.landmark_days, all_draw.landmark_days)


def every_outcome_dataset(tmp_path: Path) -> PatientBloodTestDataset:
    """Provide landmarks 29, 30 and 31 days before each event/censoring reason."""
    config = canonical_config(tmp_path)
    outcomes = {
        "recurrence": {"diagnosis_to_ddfs_in_days": 41, "ddfs_code": "C787"},
        "death": {"days_from_diagnosis_to_death": 41},
        "last_observation": {"days_from_diagnosis_to_last_entry": 41},
        "other_cancer": {},
        "other_site": {"diagnosis_to_ddfs_in_days": 41, "ddfs_code": "C780"},
    }
    pd.DataFrame(
        [
            {"id": patient, "days_since_diagnosis": day, "test_name": "A", "result": 1.0, "unit": "u"}
            for patient in outcomes
            for day in (10, 11, 12)
        ]
    ).to_csv(config["bloodtest_csv_path"], index=False)
    pd.DataFrame(
        [
            {
                "id": patient,
                "bc_diagnosis_year": 2022,
                "diagnosis_to_ddfs_in_days": "",
                "ddfs_code": "",
                "ddfs_description": "",
                "days_from_diagnosis_to_death": "",
                "days_from_diagnosis_to_last_entry": 100,
                **outcome,
            }
            for patient, outcome in outcomes.items()
        ]
    ).to_csv(config["endpoint_csv_path"], index=False)
    pd.DataFrame([{"id": patient, "split": "test"} for patient in outcomes]).to_csv(
        config["split_csv_path"], index=False
    )
    clinical_path = tmp_path / "clinical.csv"
    pd.DataFrame(
        [
            {
                "id": "other_cancer",
                "days_since_diagnosis": 41,
                "event_type": "diagnosis",
                "code": "C80",
                "description": "Other cancer",
            }
        ]
    ).to_csv(clinical_path, index=False)
    return PatientBloodTestDataset(
        {
            **config,
            "split": "test",
            "test_bootstrap_samples": 10,
            "diagnosis_procedure_csv_path": str(clinical_path),
            "exclude_and_censor_other_cancer_patients": True,
            "other_cancer_code_filter_regex": r"C80",
        }
    )


def canonical_config(tmp_path: Path) -> dict[str, object]:
    blood_path = tmp_path / "blood.csv"
    endpoint_path = tmp_path / "endpoint.csv"
    split_path = tmp_path / "split.csv"
    pd.DataFrame(
        [{"id": "P0", "days_since_diagnosis": 10, "test_name": "A", "result": 1.0, "unit": "u"}]
    ).to_csv(blood_path, index=False)
    pd.DataFrame(
        [
            {
                "id": "P0",
                "bc_diagnosis_year": 2016,
                "diagnosis_to_ddfs_in_days": 40,
                "ddfs_code": "C787",
                "ddfs_description": "Liver recurrence",
                "days_from_diagnosis_to_death": "",
                "days_from_diagnosis_to_last_entry": 100,
            }
        ]
    ).to_csv(endpoint_path, index=False)
    pd.DataFrame([{"id": "P0", "split": "train"}]).to_csv(split_path, index=False)
    return {
        "split": "train",
        "bloodtest_csv_path": str(blood_path),
        "endpoint_csv_path": str(endpoint_path),
        "split_csv_path": str(split_path),
        "selected_blood_tests": ["A|||u"],
        "input_recurrence_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "endpoint_ddfs_code_filter_regex": r"(?:C78\d*)|(?:C79\d*)",
        "recurrence_endpoint_name": "distant_recurrence",
        "censoring_horizon_days": 3650,
        "train_bootstrap_samples": 4,
    }
