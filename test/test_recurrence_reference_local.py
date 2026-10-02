from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import prepare_reference_report
from src.data.recurrence_reference_import import REFERENCE_COLUMNS
from src.script.calculate_recurrence_reference_percentages import MEASUREMENT_COLUMNS, RESULT_COLUMNS
from src.script.recurrence_blood_test_reporting import write_recurrence_blood_test_report
from test.test_recurrence_blood_test_insights import measured, report_fixture, row_for
from test.test_recurrence_reference_statistics import (
    age_reference_intervals,
    load_transfer,
    reference_interval,
    run_calculation,
    write_patient_dates,
)


@pytest.mark.parametrize("age_group", ["all ages", ">=70 years", "unsupported age label"])
def test_single_interval_is_calculated_locally_without_dates_or_a_remote_return(tmp_path: Path, age_group: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=(reference_interval(age_group=age_group, upper=20),))
    stale = tmp_path / "stale_remote.csv"
    stale.write_text("previous remote file is irrelevant when no test needs remote calculation")
    archive_return(stale, "reference_percentages")
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    prepared, metadata = prepare_reference_report(frame, dataset, config)
    final = row_for(prepared, "Liver", 3)
    assert final["n_patients"] == 3  # Unpaired C remains in descriptive counts.
    assert final["n_pairs"] == final["n_reference_patients"] == 2
    assert final["outside_score_sum"] == 1
    assert final["outside_reference_percent"] == 50
    assert final["reference_status"] == "ok"
    baseline = row_for(prepared, "Liver", 0)
    assert baseline["n_patients"] == baseline["n_reference_patients"] == 3
    assert baseline["n_pairs"] == 0 and baseline["outside_score_sum"] == 1
    assert baseline["outside_reference_percent"] == pytest.approx(100 / 3)
    assert baseline["reference_status"] == "ok"
    assert row_for(prepared, "Liver", 2)["outside_reference_percent"] == 0  # Zero-change pairs still count.
    assert row_for(prepared, "Brain", 3)["reference_status"] == "no_measurements"
    assert metadata["status"] == "not_required"
    assert metadata["local_test_keys"] == ["B-Hb|||g/dL"]
    assert metadata["remote_test_keys"] == []
    folder = Path(metadata["package_dir"])
    exported, manifest = load_transfer(folder / "measurements.csv")
    assert exported.empty and manifest["groups"] == []
    transfer = pd.read_csv(folder / "measurements.csv", dtype=str, keep_default_na=False)
    assert transfer.empty and transfer.columns.tolist() == MEASUREMENT_COLUMNS
    assert pd.read_csv(folder / "reference_thresholds.csv").empty
    assert not (folder / "patients.csv").exists()
    returned = pd.read_csv(run_calculation(folder))
    assert returned.empty and returned.columns.tolist() == RESULT_COLUMNS
    pd.testing.assert_frame_equal(prepared.drop(columns=REFERENCE_COLUMNS), frame.drop(columns=REFERENCE_COLUMNS))


@pytest.mark.parametrize(
    ("lower", "upper", "operator", "expected"),
    [(0, None, ">", 50), (0, None, ">=", 0), (None, 2, "<", 50), (None, 2, "<=", 0)],
)
def test_local_single_interval_respects_zero_one_sided_limits_and_strict_bounds(
    tmp_path: Path, lower: float | None, upper: float | None, operator: str, expected: float
) -> None:
    dataset, config = report_fixture(tmp_path)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    for patient, value in (("A", 0), ("B", 2)):
        blood.loc[blood["id"].eq(patient) & blood["days_since_diagnosis"].eq(500), "result"] = value
    blood.to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(dataset.config)
    interval = reference_interval(lower=lower, upper=upper, lower_operator=operator, upper_operator=operator)
    config = replace(config, reference_ranges=(interval,))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    prepared, _metadata = prepare_reference_report(frame, dataset, config)
    assert row_for(prepared, "Liver", 3)["outside_reference_percent"] == expected


@pytest.mark.parametrize("reference_case", ["missing", "single", "male", "duplicate", "unusable", "multiple"])
def test_only_multiple_usable_female_intervals_export_patient_medians(tmp_path: Path, reference_case: str) -> None:
    dataset, config = report_fixture(tmp_path)
    single = reference_interval(upper=20)
    intervals = {
        "missing": (),
        "single": (single,),
        "male": (single, replace(single, sex="male", upper=30)),
        "duplicate": (single, single),
        "unusable": (single, replace(single, lower=30, upper=20)),
        "multiple": age_reference_intervals(upper=20),
    }
    config = replace(config, reference_ranges=intervals[reference_case])
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    prepared, metadata = prepare_reference_report(frame, dataset, config)
    rows, manifest = load_transfer(Path(metadata["package_dir"]) / "measurements.csv")
    expected_keys = {"B-Hb|||g/dL"} if reference_case == "multiple" else set()
    assert set(rows["test_key"]) == expected_keys
    thresholds = pd.read_csv(Path(metadata["package_dir"]) / "reference_thresholds.csv")
    assert set(thresholds["test_key"]) == expected_keys
    assert {group["test_key"] for group in manifest["groups"]} == expected_keys
    final = row_for(prepared, "Liver", 3)
    if reference_case == "multiple":
        assert final["reference_status"] == "pending_remote" and pd.isna(final["outside_reference_percent"])
    elif reference_case == "missing":
        assert final["reference_status"] == "unavailable_reference" and pd.isna(final["outside_reference_percent"])
    else:
        assert final["reference_status"] == "ok" and final["outside_reference_percent"] == 50
    assert row_for(prepared, "Liver", 0)["reference_status"] == final["reference_status"]


def test_report_combines_local_and_remote_percentages_without_changing_other_statistics(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    additional = [{**measured(patient, 0, value), "test_name": "Other"} for patient, value in (("A", 9), ("C", 999))]
    pd.concat([blood, pd.DataFrame(additional)], ignore_index=True).to_csv(
        dataset.config.bloodtest_csv_path, index=False
    )
    dataset = PatientBloodTestDataset(dataset.config)
    config = replace(
        config,
        reference_ranges=(*age_reference_intervals(upper=20), reference_interval(test_name="Other", upper=8)),
        plot=replace(config.plot, dpi=35, formats=("svg",)),
    )
    original = build_recurrence_blood_test_insights(dataset, config, {})
    pending, metadata = prepare_reference_report(original, dataset, config)
    assert row_for(pending, "Liver", 3, "Other|||g/dL")["outside_reference_percent"] == 100
    assert pd.isna(row_for(pending, "Liver", 3)["outside_reference_percent"])
    folder = Path(metadata["package_dir"])
    exported, _manifest = load_transfer(folder / "measurements.csv")
    assert set(exported["test_key"]) == {"B-Hb|||g/dL"}
    write_patient_dates(folder)
    result_path = run_calculation(folder)
    assert set(pd.read_csv(result_path)["test_key"]) == {"B-Hb|||g/dL"}
    archived_path = archive_return(result_path, "reference_percentages")
    verified, metadata = prepare_reference_report(pending, dataset, config)
    assert metadata["status"] == "verified"
    assert row_for(verified, "Liver", 3)["outside_reference_percent"] == 50
    assert row_for(verified, "Liver", 3, "Other|||g/dL")["outside_reference_percent"] == 100
    pd.testing.assert_frame_equal(verified.drop(columns=REFERENCE_COLUMNS), original.drop(columns=REFERENCE_COLUMNS))
    write_recurrence_blood_test_report(verified, dataset, config)
    latex = (config.output_dir / "recurrence_blood_test_insights.tex").read_text()
    assert r"50.0\%" in latex and r"100.0\%" in latex
    plotted = pd.read_csv(config.output_dir / "recurrence_blood_test_trajectories_values.csv")
    assert row_for(plotted, "Liver", 3)["outside_reference_percent"] == 50
    assert row_for(plotted, "Liver", 3, "Other|||g/dL")["outside_reference_percent"] == 100
    # Reusing the report without a return file must clear only its remote percentages.
    archived_path.unlink()
    pending_again, _metadata = prepare_reference_report(
        verified, dataset, config
    )
    assert pd.isna(row_for(pending_again, "Liver", 3)["outside_reference_percent"])
    assert row_for(pending_again, "Liver", 3, "Other|||g/dL")["outside_reference_percent"] == 100


def test_local_threshold_change_preserves_remote_return_and_recalculates_local_scores(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    local = reference_interval(test_name="Other", upper=8)
    remote = age_reference_intervals(upper=20)
    config = replace(config, reference_ranges=(*remote, local))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    original, metadata = prepare_reference_report(frame, dataset, config)
    folder = Path(metadata["package_dir"])
    write_patient_dates(folder)
    archive_return(run_calculation(folder), "reference_percentages")

    changed = replace(config, reference_ranges=(*remote, replace(local, upper=100)))
    actual, updated = prepare_reference_report(frame, dataset, changed)
    assert updated["data_hash"] == metadata["data_hash"]
    assert updated["status"] == "verified"
    assert row_for(actual, "Liver", 3)["outside_reference_percent"] == 50
    assert row_for(original, "Liver", 0, "Other|||g/dL")["outside_reference_percent"] == 100
    assert row_for(actual, "Liver", 0, "Other|||g/dL")["outside_reference_percent"] == 0
    exported = pd.read_csv(folder / "measurements.csv")
    thresholds = pd.read_csv(folder / "reference_thresholds.csv")
    assert set(thresholds["test_key"]) == set(exported["test_key"]) == {"B-Hb|||g/dL"}


def test_remote_thresholds_exclude_tests_without_measurements_in_report_windows(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    blood.loc[blood["test_name"].eq("Other"), "days_since_diagnosis"] = 0
    blood.to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(dataset.config)
    config = replace(config, reference_ranges=(
        *age_reference_intervals(), *age_reference_intervals(test_name="Other"),
    ))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    prepared, metadata = prepare_reference_report(frame, dataset, config)
    folder = Path(metadata["package_dir"])
    exported, manifest = load_transfer(folder / "measurements.csv")
    thresholds = pd.read_csv(folder / "reference_thresholds.csv")
    assert set(thresholds["test_key"]) == set(exported["test_key"]) == {"B-Hb|||g/dL"}
    assert manifest["context"]["remote_test_keys"] == ["B-Hb|||g/dL"]
    assert prepared.loc[prepared["test_key"].eq("Other|||g/dL"), "reference_status"].eq("no_measurements").all()
    changed = replace(config, reference_ranges=(
        *age_reference_intervals(), *age_reference_intervals(test_name="Other", upper=20),
    ))
    _, updated = prepare_reference_report(frame, dataset, changed)
    assert updated["data_hash"] == metadata["data_hash"]
