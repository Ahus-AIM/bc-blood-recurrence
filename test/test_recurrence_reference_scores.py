from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import export_reference_package, prepare_reference_report
from src.model.normal_ranges import load_normal_ranges
from src.script.recurrence_blood_test_reporting import outside_reference_cell
from test.report_return_helpers import archive_return
from test.test_recurrence_blood_test_insights import measured, report_fixture, row_for
from test.test_recurrence_reference_statistics import (
    age_reference_intervals,
    reference_interval,
    run_calculation,
    write_patient_dates,
)


@pytest.mark.parametrize("remote", [False, True])
def test_tied_middle_draws_contribute_fractional_scores_with_equal_patient_weights(
    tmp_path: Path, remote: bool
) -> None:
    dataset, config = report_fixture(tmp_path)
    values = {"A": [34, 35, 35, 36, 36, 36], "B": [40, 40, 40], "D": [30]}
    rows = [
        measured(patient, start - index, value)
        for patient, results in values.items()
        for start in (300, 5)
        for index, value in enumerate(results)
    ]
    rows.append(measured("C", 0, 100))  # No baseline: exclude C from later outside scores.
    pd.DataFrame(rows).to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=("B-Hb|||g/dL",)))
    ranges = age_reference_intervals(lower=36, upper=45) if remote else (reference_interval(lower=36, upper=45),)
    config = replace(config, dataset=dataset.config, reference_ranges=ranges)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    if remote:
        folder, _manifest = export_reference_package(frame, dataset, config)
        write_patient_dates(folder)
        archive_return(run_calculation(folder), "reference_percentages")
    result, _metadata = prepare_reference_report(frame, dataset, config)
    for index in (0, 3):
        row = row_for(result, "Liver", index)
        assert row["outside_score_sum"] == pytest.approx(1.4)  # A=2/5, B=0, D=1.
        assert row["n_reference_patients"] == 3
        assert row["outside_reference_percent"] == pytest.approx(100 * 1.4 / 3)
        assert row["reference_status"] == "ok"
    final = row_for(result, "Liver", 3)
    assert final["n_patients"] == 4 and final["n_pairs"] == 3
    assert outside_reference_cell(final.to_dict()) == r"1.4/\allowbreak{}3 (46.7\%)"


@pytest.mark.parametrize(("values", "expected"), [([4, 6], 0.5), ([1, 4, 4, 6, 8], 0), ([1, 5, 5, 5, 8], 1)])
def test_single_interval_scores_actual_middle_draws_at_strict_limit(
    tmp_path: Path,
    values: list[float],
    expected: float,
) -> None:
    dataset, config = report_fixture(tmp_path)
    pd.DataFrame([measured("A", 300 - index, value) for index, value in enumerate(values)]).to_csv(
        dataset.config.bloodtest_csv_path,
        index=False,
    )
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=("B-Hb|||g/dL",)))
    config = replace(
        config, dataset=dataset.config, reference_ranges=(reference_interval(lower=None, upper=5, upper_operator="<"),)
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    result, _metadata = prepare_reference_report(frame, dataset, config)
    assert row_for(result, "Liver", 0)["outside_reference_percent"] == expected * 100


@pytest.mark.parametrize(
    ("draws", "birth_date", "expected"),
    [
        ([(195, 35), (200, 35)], "1939-07-18", 50),
        ([(195, 34), (200, 35), (201, 38)], "1939-07-18", 0),
        ([(190, 34), (195, 35), (196, 35), (200, 35), (201, 38)], "1939-07-18", 100 * 2 / 3),
        ([(197, 47), (198, 47)], "1969-07-18", 50),  # Exact 40th birthday switches bands.
        ([(195, 47)], "1992-07-18", 0),  # Under 18: closest band is 18-39, upper=48.
    ],
)
def test_albumin_uses_actual_draw_ages_and_all_median_ties(
    tmp_path: Path,
    draws: list[tuple[int, float]],
    birth_date: str,
    expected: float,
) -> None:
    dataset, config = report_fixture(tmp_path)
    rows = [{**measured("A", 500 - day, value), "test_name": "P-Albumin", "unit": "g/l"} for day, value in draws]
    pd.DataFrame(rows).to_csv(dataset.config.bloodtest_csv_path, index=False)
    key = "P-Albumin|||g/l"
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=(key,)))
    ranges = load_normal_ranges(Path(__file__).resolve().parents[1] / "src/config/ahus_blood_test_normal_ranges.csv")
    config = replace(config, dataset=dataset.config, reference_ranges=tuple(ranges[key]))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    pd.DataFrame({"id": ["A"], "birth date": [birth_date], "day of first bc diagnosis": ["2009-01-01"]}).to_csv(
        folder / "patients.csv", index=False
    )
    archive_return(run_calculation(folder), "reference_percentages")
    result, _metadata = prepare_reference_report(frame, dataset, config)
    baseline = row_for(result, "Liver", 0, key)
    assert baseline["reference_status"] == "ok"
    assert baseline["n_reference_patients"] == 1
    assert baseline["outside_reference_percent"] == pytest.approx(expected)
    assert pd.read_csv(folder / "reference_diagnostics.csv").empty
