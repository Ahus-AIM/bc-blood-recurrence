from __future__ import annotations

import os
import shutil
from dataclasses import fields, replace
from pathlib import Path

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import export_reference_package, prepare_reference_report
from src.data.recurrence_reference_import import REFERENCE_COLUMNS
from test.report_return_helpers import archive_return
from test.test_recurrence_blood_test_insights import measured, report_fixture
from test.test_recurrence_reference_statistics import age_reference_intervals, run_calculation, write_patient_dates
from test.test_sample_weighting import endpoint_row


@pytest.mark.parametrize("change", ["timestamps", "paths", "model_settings"])
def test_return_survives_changes_unrelated_to_reference_calculation(tmp_path: Path, change: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, original = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    archive_return(run_calculation(folder), "reference_percentages")
    expected, _metadata = prepare_reference_report(frame, dataset, config)
    changed_config = dataset.config
    if change == "timestamps":
        for field in fields(changed_config):
            path = getattr(changed_config, field.name)
            if isinstance(path, Path):
                stat = path.stat()
                os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
    elif change == "paths":
        destination = tmp_path / "moved"
        destination.mkdir()
        changed_paths = {}
        for field in fields(changed_config):
            path = getattr(changed_config, field.name)
            if isinstance(path, Path):
                new_path = destination / path.name
                shutil.copyfile(path, new_path)
                changed_paths[field.name] = new_path
        changed_config = replace(changed_config, **changed_paths)
    else:
        changed_config = replace(
            changed_config, censoring_horizon_days=180, random_seed=7, train_bootstrap_samples=100,
            recurrence_endpoint_name="Same cohort", build_patient_timeline_tensors=False,
        )
    reloaded = PatientBloodTestDataset(changed_config)
    config = replace(config, dataset=changed_config)
    frame = build_recurrence_blood_test_insights(reloaded, config, {})
    actual, metadata = prepare_reference_report(frame, reloaded, config)
    assert metadata["data_hash"] == original["data_hash"]
    assert metadata["status"] == "verified"
    pd.testing.assert_frame_equal(actual[REFERENCE_COLUMNS], expected[REFERENCE_COLUMNS])


def test_changed_measurements_invalidate_return_even_with_original_file_timestamp(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, original = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    archive_return(run_calculation(folder), "reference_percentages")
    path = dataset.config.bloodtest_csv_path
    stat = path.stat()
    blood = pd.read_csv(path)
    blood.loc[0, "result"] = 11
    blood.to_csv(path, index=False)
    assert path.stat().st_size == stat.st_size
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    reloaded = PatientBloodTestDataset(dataset.config)
    assert reloaded.prepared_data.input_fingerprint == dataset.prepared_data.input_fingerprint
    frame = build_recurrence_blood_test_insights(reloaded, config, {})
    _actual, metadata = prepare_reference_report(frame, reloaded, config)
    assert metadata["measurements_sha256"] != original["measurements_sha256"]
    assert metadata["data_hash"] != original["data_hash"]
    assert metadata["status"] == "pending_remote"


def test_hash_tracks_eligible_patients_without_measurements_in_report_windows(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    _, original = export_reference_package(frame, dataset, config)
    extra_endpoint = {
        **endpoint_row("F", recurrence_day=500, right_censoring_day=1000),
        "ddfs_code": "C787", "bc_diagnosis_year": 2016,
    }
    additions = (
        (dataset.config.bloodtest_csv_path, measured("F", 400, 10)),
        (dataset.config.endpoint_csv_path, extra_endpoint),
        (dataset.config.split_csv_path, {"id": "F", "split": "train"}),
    )
    for path, row in additions:
        pd.concat([pd.read_csv(path), pd.DataFrame([row])], ignore_index=True).to_csv(path, index=False)
    reloaded = PatientBloodTestDataset(dataset.config)
    frame = build_recurrence_blood_test_insights(reloaded, config, {})
    _, changed = export_reference_package(frame, reloaded, config)
    assert changed["measurements_sha256"] == original["measurements_sha256"]
    assert changed["groups"] == original["groups"]
    assert changed["analysis_sha256"] != original["analysis_sha256"]
    assert changed["data_hash"] != original["data_hash"]
