from __future__ import annotations

import runpy
from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import CALCULATOR_PATH, export_reference_package, prepare_reference_report
from src.data.reference_intervals import reference_age_bounds, reference_ranges_for_test
from src.model.normal_ranges import NormalRange, load_normal_ranges
from src.script.calculate_recurrence_reference_percentages import (
    age_applicability,
    aggregate_reference_patients,
    classify_measurements,
)
from test.test_recurrence_blood_test_insights import measured, report_fixture, row_for
from test.test_sample_weighting import dataset_config, endpoint_row


def remote_calculator(directory: Path) -> dict[str, Any]:
    """Exercise the exported script with its bundled settings, without running its main block."""
    return runpy.run_path(str(directory / CALCULATOR_PATH.name))


def run_calculation(directory: Path) -> Path:
    return remote_calculator(directory)["run_calculation"](directory)


def load_transfer(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    return remote_calculator(path.parent)["load_transfer"](path)


def reference_interval(**changes: object) -> NormalRange:
    interval = NormalRange("B-Hb", "g/dl", "female", "all ages", 0, 50, ">=", "<=", "", "")
    return replace(interval, **changes)


def age_reference_intervals(**changes: object) -> tuple[NormalRange, NormalRange]:
    """Two disjoint female age bands with identical result limits for exercising remote calculation."""
    interval = reference_interval(**changes)
    return replace(interval, age_group="<40 years"), replace(interval, age_group=">=40 years")


def write_patient_dates(directory: Path, ids: str = "ABCDE") -> Path:
    path = directory / "patients.csv"
    pd.DataFrame(
        {
            "id": list(ids),
            "birth date": ["1960-06-15"] * len(ids),
            "day of first bc diagnosis": ["2010-01-01"] * len(ids),
        }
    ).to_csv(path, index=False)
    return path


@pytest.mark.parametrize(
    ("group", "birth", "draws", "expected"),
    [
        (
            "18-39 years",
            "1980-06-15",
            ["1998-06-14", "1998-06-15", "2020-06-14", "2020-06-15"],
            [False, True, True, False],
        ),
        (
            "40-69 years",
            "1950-06-15",
            ["1990-06-14", "1990-06-15", "2020-06-14", "2020-06-15"],
            [False, True, True, False],
        ),
        (">=70 years", "1950-06-15", ["2020-06-14", "2020-06-15"], [False, True]),
        ("adult fasting serum", "2000-06-15", ["2018-06-14", "2018-06-15"], [False, True]),
        (">18 years", "2000-06-15", ["2018-06-15", "2018-06-16"], [False, True]),
        (">=1 month", "2020-01-31", ["2020-02-28", "2020-02-29"], [False, True]),
        (">=1 year", "2000-02-29", ["2001-02-27", "2001-02-28"], [False, True]),
        ("all ages", "2000-01-01", ["2000-01-01", "2099-01-01"], [True, True]),
        ("unsupported", "2000-01-01", ["2020-01-01"], [False]),
    ],
)
def test_reference_age_intervals_use_exact_calendar_birthdays(
    group: str,
    birth: str,
    draws: list[str],
    expected: list[bool],
) -> None:
    rows = pd.DataFrame({"birth_date": pd.to_datetime([birth] * len(draws)), "draw_date": pd.to_datetime(draws)})
    assert age_applicability(rows, reference_age_bounds(group)).tolist() == expected


def test_reference_bounds_respect_strictness_missing_limits_units_and_ambiguity() -> None:
    rows = pd.DataFrame({"blood test value": [4.9, 5, 6]})
    upper = {"age_bounds": [], "lower": None, "upper": 5, "lower_operator": "", "upper_operator": "<"}
    assert classify_measurements(rows, [upper]).tolist() == [0, 1, 1]
    lower = {**upper, "lower": 5, "upper": None, "lower_operator": ">="}
    assert classify_measurements(rows, [lower]).tolist() == [1, 0, 0]
    assert classify_measurements(rows, [upper, upper]).isna().all()
    interval = reference_interval()
    assert reference_ranges_for_test((interval, replace(interval, sex="male")), "B-Hb|||g/dL") == (interval,)
    assert reference_ranges_for_test((interval,), "B-Hb|||g/L") == ()


def test_every_checked_in_female_reference_has_supported_calendar_age_bounds() -> None:
    path = Path(__file__).resolve().parents[1] / "src/config/ahus_blood_test_normal_ranges.csv"
    for ranges in load_normal_ranges(path).values():
        for interval in ranges:
            if interval.sex == "female":
                assert reference_age_bounds(interval.age_group) is not None, interval.age_group


def test_round_trip_uses_window_specific_baseline_pairs_for_reference_percentages(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals(upper=20))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    assert frame["outside_reference_percent"].isna().all()
    assert row_for(frame, "Liver", 0)["reference_status"] == "pending_remote"
    folder, manifest = export_reference_package(frame, dataset, config)
    measurements = pd.read_csv(folder / "measurements.csv")
    assert set(measurements["test_key"]) == {"B-Hb|||g/dL"}  # Other has no reference interval.
    write_patient_dates(folder)
    result_path = run_calculation(folder)
    archive_return(result_path, "reference_percentages")
    frame, metadata = prepare_reference_report(frame, dataset, config)
    assert metadata["status"] == "verified"
    assert metadata["data_hash"] == manifest["data_hash"]
    baseline, final = row_for(frame, "Liver", 0), row_for(frame, "Liver", 3)
    assert baseline["n_patients"] == baseline["n_reference_patients"] == 3
    assert baseline["n_pairs"] == 0
    assert baseline["outside_score_sum"] == 1  # Baseline-only D counts; A's isolated high draw does not.
    assert baseline["outside_reference_percent"] == pytest.approx(100 / 3)
    assert baseline["reference_status"] == "ok"
    assert final["n_patients"] == 3  # Descriptive summaries still include unpaired C.
    assert final["n_reference_patients"] == final["n_pairs"] == 2
    assert final["outside_score_sum"] == 1  # B is outside; unpaired C is excluded.
    assert final["outside_reference_percent"] == 50
    assert final["reference_status"] == "ok"
    earlier = row_for(frame, "Liver", 2)
    assert earlier["n_pairs"] == earlier["n_reference_patients"] == 1  # Only A, including its zero change.
    assert earlier["outside_reference_percent"] == 0
    assert row_for(frame, "Bone", 3)["n_pairs"] == 1
    assert row_for(frame, "Bone", 3)["outside_reference_percent"] == 0  # Unpaired E is excluded.
    assert row_for(frame, "Brain", 3)["reference_status"] == "no_measurements"
    assert np.isnan(row_for(frame, "Brain", 3)["outside_reference_percent"])
    assert row_for(frame, "Liver", 0, "Other|||g/dL")["reference_status"] == "unavailable_reference"
    returned = pd.read_csv(result_path)
    assert not {"id", "birth date", "day of first bc diagnosis", "draw_date"} & set(returned.columns)


def test_three_to_six_month_percentage_classifies_paired_patient_medians_not_single_high_results(
    tmp_path: Path,
) -> None:
    dataset, config = report_fixture(tmp_path)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    additional = [
        measured("A", 100, 10),
        measured("A", 101, 10),
        measured("A", 102, 100),  # A's median is inside despite one high result.
        measured("B", 100, 80),
        measured("C", 100, 1000),  # C has no baseline for this test.
        {**measured("C", 300, 10), "test_name": "Other"},
        {**measured("B", 100, 10), "test_name": "Other"},  # B's baseline Hb cannot pair with Other.
    ]
    pd.concat([blood, pd.DataFrame(additional)], ignore_index=True).to_csv(
        dataset.config.bloodtest_csv_path, index=False
    )
    dataset = PatientBloodTestDataset(dataset.config)
    config = replace(
        config,
        reference_ranges=(reference_interval(),),
        windows=(
            replace(config.windows[0], label="6–18 months", max_days=config.windows[0].max_days + 180),
            *config.windows[1:],
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    result = run_calculation(folder)
    archive_return(result, "reference_percentages")
    verified, _metadata = prepare_reference_report(frame, dataset, config)
    comparison = row_for(verified, "Liver", 1)
    assert comparison["n_patients"] == 3
    assert comparison["n_pairs"] == comparison["n_reference_patients"] == 2
    assert comparison["outside_score_sum"] == 1
    assert comparison["outside_reference_percent"] == 50
    # A belongs to Bone too, but B's Liver pair cannot contribute to Bone's denominator.
    assert row_for(verified, "Bone", 1)["n_pairs"] == 1
    assert row_for(verified, "Bone", 1)["outside_reference_percent"] == 0
    unpaired = row_for(verified, "Liver", 1, "Other|||g/dL")
    assert unpaired["n_patients"] == 1
    assert unpaired["n_pairs"] == unpaired["n_reference_patients"] == 0
    assert unpaired["reference_status"] == "no_pairs"
    assert np.isnan(unpaired["outside_reference_percent"])


def test_exact_dates_replace_binned_age_and_export_respects_same_day_toggle(tmp_path: Path) -> None:
    ages = tmp_path / "ages.csv"
    pd.DataFrame({"patient_id": ["P"], "age_at_diagnosis": [40]}).to_csv(ages, index=False)
    raw = dataset_config(
        tmp_path,
        [measured("P", 300, 60), measured("P", 0, 60)],
        [{**endpoint_row("P", recurrence_day=500, right_censoring_day=1000), "ddfs_code": "C787"}],
        include_recurrence_day_blood_tests=True,
        bc_diagnosis_patient_age_csv_path=str(ages),
    )
    dataset = PatientBloodTestDataset(raw)
    fixture_path = tmp_path / "fixture"
    fixture_path.mkdir()
    _, config = report_fixture(fixture_path)
    config = replace(
        config,
        dataset=dataset.config,
        reference_ranges=(
            reference_interval(age_group="<40 years", lower=10, upper=45),
            reference_interval(age_group=">=40 years", lower=10, upper=75),
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    pd.DataFrame({"id": ["P"], "birth date": ["1970-04-01"], "day of first bc diagnosis": ["2009-01-01"]}).to_csv(
        folder / "patients.csv",
        index=False,
    )
    result = run_calculation(folder)
    archive_return(result, "reference_percentages")
    frame, _metadata = prepare_reference_report(frame, dataset, config)
    assert row_for(frame, "Liver", 0)["outside_reference_percent"] == 100  # Younger baseline age uses upper=45.
    assert row_for(frame, "Liver", 3)["outside_reference_percent"] == 0
    excluded = PatientBloodTestDataset({**raw, "include_recurrence_day_blood_tests": False})
    excluded_config = replace(config, dataset=excluded.config)
    excluded_frame = build_recurrence_blood_test_insights(excluded, excluded_config, {})
    export_reference_package(excluded_frame, excluded, excluded_config)
    draw_days = pd.read_csv(folder / "measurements.csv")["draw days since diagnosis"]
    assert not draw_days.eq(500).any()


@pytest.mark.parametrize("patient", ["B", "C", "D"])
def test_nearest_age_fallback_retains_full_baseline_and_paired_populations(tmp_path: Path, patient: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config,
        reference_ranges=(reference_interval(age_group="50-69 years"), reference_interval(age_group=">=70 years")),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    path = write_patient_dates(folder)
    patients = pd.read_csv(path)
    patients.loc[patients["id"].eq(patient), "birth date"] = "1980-01-01"
    patients.to_csv(path, index=False)
    returned = pd.read_csv(run_calculation(folder))
    liver = returned.loc[returned["recurrence site"].eq("Liver")]
    baseline = liver.loc[liver["interval"].eq(config.windows[0].label)].iloc[0]
    final = liver.loc[liver["interval"].eq(config.windows[-1].label)].iloc[0]
    assert baseline["n_reference_patients"] == baseline["n_patients"] == 3
    assert final["n_reference_patients"] == final["n_pairs"] == 2
    assert final["n_patients"] == 3
    assert baseline["reference_status"] == final["reference_status"] == "ok"
    assert baseline["percentage outside reference area"] == final["percentage outside reference area"] == 0


@pytest.mark.parametrize(
    ("values", "classified", "percentage"),
    [((20, 20), 1, 0), ((20, 100), 1, 50), ((80, 80), 1, 100)],
)
@pytest.mark.parametrize("window_index", [0, 3])
def test_middle_draws_crossing_an_age_boundary_average_their_individual_flags(
    tmp_path: Path, values: tuple[int, int], classified: int, percentage: float, window_index: int
) -> None:
    dataset, config = report_fixture(tmp_path)
    baseline = window_index == 0
    days = (305, 300) if baseline else (5, 0)
    rows = [
        measured("A", 0 if baseline else 300, 60),
        measured("A", days[0], values[0]),
        measured("A", days[1], values[1]),
    ]
    pd.DataFrame(rows).to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=("B-Hb|||g/dL",)))
    config = replace(
        config,
        dataset=dataset.config,
        reference_ranges=(
            reference_interval(age_group="<40 years", lower=10, upper=45),
            reference_interval(age_group=">=40 years", lower=10, upper=75),
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    birth = "1969-07-18" if baseline else "1970-05-14"
    pd.DataFrame({"id": ["A"], "birth date": [birth], "day of first bc diagnosis": ["2009-01-01"]}).to_csv(
        folder / "patients.csv", index=False
    )
    result = run_calculation(folder)
    archive_return(result, "reference_percentages")
    verified, _metadata = prepare_reference_report(frame, dataset, config)
    comparison = row_for(verified, "Liver", window_index)
    assert comparison["n_pairs"] == (0 if baseline else 1)
    assert comparison["n_reference_patients"] == classified
    np.testing.assert_allclose(comparison["outside_reference_percent"], percentage, equal_nan=True)


def test_patient_scores_average_selected_flags_and_require_complete_coverage() -> None:
    measured_rows = pd.DataFrame(
        {
            "id": ["A", "A", "B", "B", "C", "C", "D", "D"],
            "test_key": ["Hb"] * 8,
            "interval": ["comparison"] * 8,
            "recurrence site": ["Liver"] * 8,
            "outside": [1, np.nan, 0, 1, 1, 1, 0, 0],
            "blood test value": [10] * 8,
        }
    )
    counts = aggregate_reference_patients(measured_rows, "baseline").iloc[0]
    assert counts["n_pairs"] == 4
    assert counts["n_reference_patients"] == 3
    assert counts["outside_score_sum"] == 1.5
