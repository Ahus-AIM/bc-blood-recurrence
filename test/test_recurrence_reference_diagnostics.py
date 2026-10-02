from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import export_reference_package, render_remote_calculator
from src.script.calculate_recurrence_reference_percentages import REMOTE_SETTINGS, RESULT_COLUMNS
from test.test_recurrence_blood_test_insights import measured, report_fixture
from test.test_recurrence_reference_statistics import reference_interval, run_calculation


@pytest.mark.parametrize(
    ("age_groups", "birth", "reason", "matches", "classifications"),
    [
        (("<40 years", ">=40 years"), "1969-07-18", "inconsistent_across_draw_ages", [1, 1], ["outside", "inside"]),
        (("<40 years", ">=50 years"), "1969-07-18", "no_applicable_interval", [1, 0], ["outside", "unclassified"]),
        (
            ("<=40 years", ">=40 years"),
            "1969-07-15",
            "multiple_applicable_intervals",
            [2, 1],
            ["unclassified", "inside"],
        ),
    ],
)
def test_remote_diagnostics_only_report_unresolved_classifications(
    tmp_path: Path,
    age_groups: tuple[str, str],
    birth: str,
    reason: str,
    matches: list[int],
    classifications: list[str],
) -> None:
    dataset, config = report_fixture(tmp_path)
    pd.DataFrame(
        [
            measured("A", 305, 60),
            measured("A", 300, 60),
            measured("B", 300, 20),
            measured("A", 0, 60),
            measured("C", 0, 60),  # Unpaired C must not appear in diagnostics.
        ]
    ).to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=("B-Hb|||g/dL",)))
    config = replace(
        config,
        dataset=dataset.config,
        reference_ranges=(
            reference_interval(age_group=age_groups[0], lower=10, upper=45),
            reference_interval(age_group=age_groups[1], lower=10, upper=75),
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    pd.DataFrame(
        {
            "id": ["A", "B", "C"],
            "birth date": [birth, "1950-01-01", "1969-07-18"],
            "day of first bc diagnosis": ["2009-01-01"] * 3,
        }
    ).to_csv(folder / "patients.csv", index=False)

    returned = pd.read_csv(run_calculation(folder))
    baseline = returned.loc[
        returned["recurrence site"].eq("Liver") & returned["interval"].eq(config.windows[0].label)
    ].iloc[0]
    diagnostics = pd.read_csv(folder / "reference_diagnostics.csv", keep_default_na=False)
    assert returned.columns.tolist() == RESULT_COLUMNS
    if reason != "multiple_applicable_intervals":
        assert baseline["reference_status"] == "ok"
        assert baseline["n_reference_patients"] == 2
        assert baseline["percentage outside reference area"] == (
            25 if reason == "inconsistent_across_draw_ages" else 50
        )
        assert diagnostics.empty
        return
    assert baseline["reference_status"] == "incomplete_reference"
    assert baseline["n_patients"] == 2 and baseline["n_reference_patients"] == 1
    assert pd.isna(baseline["percentage outside reference area"])
    assert set(diagnostics["id"]) == {"A"}
    baseline_draws = diagnostics.loc[
        diagnostics["recurrence site"].eq("Liver") & diagnostics["interval"].eq(config.windows[0].label)
    ].sort_values("draw days since diagnosis")
    assert baseline_draws["reason"].tolist() == [reason, reason]
    assert baseline_draws["matching_intervals"].tolist() == matches
    assert baseline_draws["draw_classification"].tolist() == classifications
    assert baseline_draws["draw days since diagnosis"].tolist() == [195, 200]
    assert baseline_draws["draw_date"].tolist() == ["2009-07-15", "2009-07-20"]
    assert baseline_draws["birth_date"].tolist() == [birth, birth]
    assert baseline_draws["blood test value"].tolist() == [60, 60]
    assert baseline_draws["is_median_draw"].tolist() == [True, True]
    intervals = baseline_draws["selected_intervals"].map(json.loads).tolist()
    assert [len(values) for values in intervals] == matches
    assert intervals[0][0]["upper"] == 45
    assert intervals[0][0]["age_group"] == age_groups[0]

    # A fresh successful calculation replaces old failures with a header-only file.
    patients = pd.read_csv(folder / "patients.csv")
    patients["birth date"] = "1950-01-01"
    patients.to_csv(folder / "patients.csv", index=False)
    run_calculation(folder)
    assert pd.read_csv(folder / "reference_diagnostics.csv").empty


@pytest.mark.parametrize(
    "diagnostics_csv", ["patients.csv", "reference_percentages.csv", "reference_percentages.csv.tmp"]
)
def test_diagnostic_output_cannot_overwrite_inputs_or_aggregate_output(tmp_path: Path, diagnostics_csv: str) -> None:
    dataset, config = report_fixture(tmp_path)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    settings = {**REMOTE_SETTINGS, "diagnostics_csv": diagnostics_csv}
    script = folder / "calculate_recurrence_reference_percentages.py"
    script.write_text(render_remote_calculator(settings, manifest))
    patient_path = folder / "patients.csv"
    patient_path.write_text("id,birth date,day of first bc diagnosis\n")
    original = patient_path.read_bytes()
    with pytest.raises(ValueError, match="overwrite|overlap"):
        run_calculation(folder)
    assert patient_path.read_bytes() == original
    assert not (folder / "reference_percentages.csv").exists()
