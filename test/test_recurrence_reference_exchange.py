from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import (
    export_reference_package,
    prepare_reference_report,
    render_remote_calculator,
)
from src.data.recurrence_reference_import import import_reference_percentages
from src.script.calculate_recurrence_reference_percentages import (
    MEASUREMENT_COLUMNS,
    REMOTE_SETTINGS,
    attach_patient_dates,
)
from test.test_recurrence_blood_test_insights import measured, report_fixture
from test.test_recurrence_reference_statistics import (
    age_reference_intervals,
    reference_interval,
    remote_calculator,
    run_calculation,
    write_patient_dates,
)
from test.test_sample_weighting import blood_row, dataset_config, endpoint_row


def test_package_runs_outside_repository_with_pandas_and_supports_configured_dates(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    assert {path.name for path in folder.iterdir()} == {
        "measurements.csv",
        "reference_thresholds.csv",
        "calculate_recurrence_reference_percentages.py",
    }
    measurements = pd.read_csv(folder / "measurements.csv", dtype=str, keep_default_na=False)
    assert measurements.columns.tolist() == MEASUREMENT_COLUMNS
    assert len(measurements) > 1
    embedded = remote_calculator(folder)["EXPORT_MANIFEST"]
    assert embedded == manifest
    assert set(embedded["context"]) == {"remote_test_keys", "windows"}
    assert len(embedded["analysis_sha256"]) == 64
    assert str(dataset.config.bloodtest_csv_path) not in json.dumps(embedded)
    pd.DataFrame({"patient": list("ABCDE"), "dob": ["15/06/1960"] * 5, "bc": ["01.01.2010"] * 5}).to_csv(
        folder / "demographics.csv",
        sep=";",
        index=False,
    )
    script_path = folder / "calculate_recurrence_reference_percentages.py"
    settings = dict(REMOTE_SETTINGS)
    settings.update(
        {
            "patients_csv": "demographics.csv",
            "patients_separator": ";",
            "patient_columns": {"id": "patient", "birth_date": "dob", "diagnosis_date": "bc"},
            "date_formats": {"birth_date": "%d/%m/%Y", "diagnosis_date": "%d.%m.%Y"},
        }
    )
    script_path.write_text(render_remote_calculator(settings, manifest))
    process = subprocess.run(
        [sys.executable, "-I", str(script_path)],
        cwd=tmp_path,
        check=False,
        capture_output=True,
        text=True,
    )
    assert process.returncode == 0, process.stderr
    remote_frame = frame.loc[frame["test_key"].isin(manifest["context"]["remote_test_keys"])]
    returned = import_reference_percentages(
        remote_frame,
        folder / "reference_percentages.csv",
        manifest["data_hash"],
    )
    assert returned["n_patients"].tolist() == remote_frame["n_patients"].tolist()
    assert returned["data_hash"].eq(manifest["data_hash"]).all()


@pytest.mark.parametrize("problem", ["missing", "invalid", "reversed", "conflicting"])
def test_remote_rejects_bad_patient_records_without_writing_results(tmp_path: Path, problem: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    path = write_patient_dates(folder)
    patients = pd.read_csv(path)
    if problem == "missing":
        patients = patients.loc[patients["id"] != "A"]
    elif problem == "invalid":
        patients.loc[patients["id"] == "A", "birth date"] = "not-a-date"
    elif problem == "reversed":
        patients.loc[patients["id"] == "A", "birth date"] = "2020-01-01"
    else:
        patients = pd.concat(
            [
                patients,
                pd.DataFrame(
                    {
                        "id": ["A"],
                        "birth date": ["1961-06-15"],
                        "day of first bc diagnosis": ["2010-01-01"],
                    }
                ),
            ]
        )
    patients.to_csv(path, index=False)
    with pytest.raises(ValueError, match="IDs.*A"):
        run_calculation(folder)
    assert not (folder / "reference_percentages.csv").exists()


def test_patient_join_preserves_leading_zeros_and_literal_na_ids(tmp_path: Path) -> None:
    pd.DataFrame(
        {"id": ["001", "1", "NA"], "birth": ["1960-01-01", "1970-01-01", "1980-01-01"], "bc": ["2010-01-01"] * 3}
    ).to_csv(
        tmp_path / "patients.csv",
        index=False,
    )
    config = {
        "patients_csv": "patients.csv",
        "patient_columns": {"id": "id", "birth_date": "birth", "diagnosis_date": "bc"},
        "date_formats": {"birth_date": "%Y-%m-%d", "diagnosis_date": "%Y-%m-%d"},
    }
    medians = pd.DataFrame({"id": ["001", "1", "NA"], "draw days since diagnosis": [10, 10, 10]})
    joined = attach_patient_dates(medians, config, tmp_path)
    assert joined["birth_date"].dt.year.tolist() == [1960, 1970, 1980]
    assert joined["draw_date"].dt.strftime("%Y-%m-%d").tolist() == ["2010-01-11"] * 3


@pytest.mark.parametrize(
    "modified",
    [
        "measurements",
        "thresholds",
        "hash",
        "missing_hash",
        "empty_measurements",
        "manifest",
        "calculator",
    ],
)
def test_remote_rejects_modified_export_or_calculator(tmp_path: Path, modified: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    script_path = folder / "calculate_recurrence_reference_percentages.py"
    if modified == "calculator":
        script_path.write_bytes(script_path.read_bytes() + b"\n")
    elif modified in {"hash", "missing_hash", "manifest"}:
        if modified == "hash":
            manifest["data_hash"] = "0" * 64
        elif modified == "missing_hash":
            del manifest["data_hash"]
        else:
            manifest["groups"][0]["n_patients"] += 1
        script_path.write_text(render_remote_calculator(REMOTE_SETTINGS, manifest))
    elif modified == "thresholds":
        path = folder / "reference_thresholds.csv"
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
        rows.loc[0, "upper"] = "999999"
        rows.to_csv(path, index=False)
    else:
        path = folder / "measurements.csv"
        rows = pd.read_csv(path, dtype=str, keep_default_na=False)
        if modified == "empty_measurements":
            rows = rows.head(0)
        else:
            rows.loc[len(rows) - 1, "blood test value"] = "999999"
        rows.to_csv(path, index=False)
    process = subprocess.run(
        [sys.executable, "-I", str(script_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert process.returncode != 0
    assert "hash" in process.stderr
    assert not (folder / "patients.csv").exists()  # Hash checks finish before requesting patient dates.
    assert not (folder / "reference_percentages.csv").exists()


def test_returned_hash_is_unchanged_when_remote_patient_dates_change(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config,
        reference_ranges=(
            reference_interval(age_group="<40 years", upper=20),
            reference_interval(age_group=">=40 years", upper=40),
        ),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    patients_path = write_patient_dates(folder)
    original = pd.read_csv(run_calculation(folder))
    patients = pd.read_csv(patients_path)
    patients["birth date"] = "1990-06-15"
    patients["day of first bc diagnosis"] = "2011-01-01"
    patients.to_csv(patients_path, index=False)
    changed = pd.read_csv(run_calculation(folder))
    assert original["data_hash"].eq(manifest["data_hash"]).all()
    assert changed["data_hash"].eq(manifest["data_hash"]).all()
    final = changed["recurrence site"].eq("Liver") & changed["interval"].eq(config.windows[-1].label)
    assert original.loc[final, "percentage outside reference area"].tolist() == [0]
    assert changed.loc[final, "percentage outside reference area"].tolist() == [50]


@pytest.mark.parametrize(
    "problem",
    [
        "hash",
        "missing",
        "duplicate",
        "site",
        "unexpected_test",
        "denominator",
        "paired_denominator",
        "count",
        "negative_score",
        "infinite_score",
        "missing_score",
        "fractional_count",
        "percentage",
        "status",
    ],
)
def test_import_rejects_stale_or_inconsistent_results(tmp_path: Path, problem: str) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    path = run_calculation(folder)
    rows = pd.read_csv(path, keep_default_na=False)
    if problem == "hash":
        rows.loc[0, "data_hash"] = "0" * 64
    elif problem == "missing":
        rows = rows.iloc[:-1]
    elif problem == "duplicate":
        rows.iloc[1] = rows.iloc[0]
    elif problem == "site":
        rows.loc[0, "recurrence site"] = "Different site"
    elif problem == "unexpected_test":
        rows.loc[0, "test_key"] = "Other|||g/dL"
    elif problem == "denominator":
        rows.loc[0, "n_patients"] += 1
    elif problem == "paired_denominator":
        rows.loc[0, "n_pairs"] += 1
    elif problem == "count":
        rows.loc[0, "outside_score_sum"] = rows.loc[0, "n_patients"] + 1
    elif problem in {"negative_score", "infinite_score", "missing_score"}:
        rows["outside_score_sum"] = rows["outside_score_sum"].astype(str)
        rows.loc[0, "outside_score_sum"] = {"negative_score": "-0.5", "infinite_score": "inf", "missing_score": ""}[
            problem
        ]
    elif problem == "fractional_count":
        rows["n_reference_patients"] = rows["n_reference_patients"].astype(float)
        rows.loc[0, "n_reference_patients"] = 0.5
    elif problem == "percentage":
        rows.loc[0, "percentage outside reference area"] = 0.5
    else:
        rows.loc[0, "reference_status"] = "wrong-status"
    rows.to_csv(path, index=False)
    remote_frame = frame.loc[frame["test_key"].isin(manifest["context"]["remote_test_keys"])]
    with pytest.raises(ValueError):
        import_reference_percentages(remote_frame, path, manifest["data_hash"])


def test_hash_tracks_data_ranges_windows_and_cohort_but_not_plot_selection(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    _, original = export_reference_package(frame, dataset, config)
    _, repeated = export_reference_package(frame, dataset, replace(config, plot=replace(config.plot, top_n=1)))
    assert original["data_hash"] == repeated["data_hash"]
    changes = [
        replace(config, reference_ranges=age_reference_intervals(upper=20)),
        replace(config, reference_ranges=(reference_interval(),)),
        replace(config, windows=(replace(config.windows[0], max_days=400), *config.windows[1:])),
        replace(config, dataset=replace(config.dataset, include_recurrence_day_blood_tests=False)),
    ]
    for changed in changes:
        changed_dataset = PatientBloodTestDataset(changed.dataset)
        changed_frame = build_recurrence_blood_test_insights(changed_dataset, changed, {})
        _, manifest = export_reference_package(changed_frame, changed_dataset, changed)
        assert manifest["data_hash"] != original["data_hash"]
    assert len(original["data_hash"]) == 64


def test_missing_return_file_keeps_percentages_unavailable_and_exports_package(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(
        config,
        reference_ranges=age_reference_intervals(),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    prepared, metadata = prepare_reference_report(frame, dataset, config)
    assert prepared["outside_reference_percent"].isna().all()
    assert metadata["status"] == "pending_remote"
    assert (Path(metadata["package_dir"]) / "measurements.csv").is_file()


def test_reexport_with_changed_ranges_ignores_previous_returned_results(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    write_patient_dates(folder)
    returned_path = run_calculation(folder)
    archive_return(returned_path, "reference_percentages")
    changed = replace(config, reference_ranges=age_reference_intervals(upper=20))
    prepared, metadata = prepare_reference_report(frame, dataset, changed)
    assert metadata["status"] == "pending_remote"
    assert prepared["outside_reference_percent"].isna().all()


def test_no_recurrence_measurements_return_empty_group_percentages_without_patient_file(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, config = report_fixture(fixture)
    raw = dataset_config(tmp_path, [blood_row("P", 40)], [endpoint_row("P")])
    dataset = PatientBloodTestDataset(raw)
    config = replace(config, dataset=dataset.config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    path = run_calculation(folder)
    remote_frame = frame.loc[frame["test_key"].isin(manifest["context"]["remote_test_keys"])]
    returned = import_reference_percentages(remote_frame, path, manifest["data_hash"])
    assert returned["reference_status"].eq("no_measurements").all()
    assert returned["outside_reference_percent"].isna().all()


def test_export_only_includes_selected_recurrence_sites_and_configured_time_windows(tmp_path: Path) -> None:
    fixture = tmp_path / "fixture"
    fixture.mkdir()
    _, config = report_fixture(fixture)
    rows = [blood_row("L", day) for day in (139, 200, 501)]
    rows += [blood_row("N", 200), blood_row("B", 200)]
    endpoints = [
        {**endpoint_row("L", recurrence_day=500, right_censoring_day=1000), "ddfs_code": "C787"},
        endpoint_row("N", right_censoring_day=1000),
        {**endpoint_row("B", recurrence_day=500, right_censoring_day=1000), "ddfs_code": "C795"},
    ]
    dataset = PatientBloodTestDataset(dataset_config(tmp_path, rows, endpoints))
    config = replace(
        config,
        dataset=dataset.config,
        site_rules=config.site_rules[:1],
        other_label=None,
        reference_ranges=age_reference_intervals(),
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, _manifest = export_reference_package(frame, dataset, config)
    exported = pd.read_csv(folder / "measurements.csv")
    assert exported["id"].tolist() == ["L"]
    assert exported["draw days since diagnosis"].tolist() == [200]
    assert exported["recurrence site"].tolist() == ["Liver"]


@pytest.mark.parametrize("values", [[8], [2, 100, 10], [100, 2, 10, 14], [10, 10, 100, 10]])
def test_export_preserves_every_draw_value_and_day_including_median_ties(tmp_path: Path, values: list[int]) -> None:
    dataset, config = report_fixture(tmp_path)
    rows = [measured("A", 300, 5), *[measured("A", 5 - index, value) for index, value in enumerate(values)]]
    pd.DataFrame(rows).to_csv(dataset.config.bloodtest_csv_path, index=False)
    dataset = PatientBloodTestDataset(replace(dataset.config, selected_blood_tests=("B-Hb|||g/dL",)))
    config = replace(config, dataset=dataset.config, reference_ranges=age_reference_intervals())
    frame = build_recurrence_blood_test_insights(dataset, config, {"B-Hb": "Hemoglobin"})
    folder, _manifest = export_reference_package(frame, dataset, config)
    exported = pd.read_csv(folder / "measurements.csv", keep_default_na=False)
    assert len(exported) == 2 * (1 + len(values))  # Synchronous Liver/Bone sites.
    assert not exported.duplicated(["id", "recurrence site", "test_key", "interval", "draw days since diagnosis"]).any()
    final = exported.loc[exported["interval"].eq(config.windows[-1].label)]
    for _site, draws in final.groupby("recurrence site"):
        assert draws["blood test value"].tolist() == values
        assert draws["draw days since diagnosis"].tolist() == list(range(495, 495 + len(values)))
    assert set(exported["blood test name"]) == {"Hemoglobin"}
    counts = exported.groupby(["recurrence site", "test_key", "interval"])["id"].nunique()
    expected_counts = frame.set_index(["site", "test_key", "window"])["n_patients"]
    assert counts.reindex(expected_counts.index, fill_value=0).tolist() == expected_counts.tolist()


def test_reference_thresholds_are_exported_once_per_band_separate_from_patient_draws(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    younger = reference_interval(age_group="<40 years", lower=None, upper=45, upper_operator="<")
    older = reference_interval(age_group=">=40 years", lower=0, upper=None, source_url="https://example.org/ranges")
    config = replace(config, reference_ranges=(younger, older, younger, replace(older, sex="male")))
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    thresholds = pd.read_csv(folder / "reference_thresholds.csv", keep_default_na=False).set_index("age_group")
    assert len(thresholds) == 2
    assert set(thresholds["test_key"]) == {"B-Hb|||g/dL"}
    assert set(thresholds["sex"]) == {"female"}
    assert thresholds.loc["<40 years", ["lower", "upper", "upper_operator"]].tolist() == ["", "45", "<"]
    assert thresholds.loc[">=40 years", ["lower", "upper"]].tolist() == ["0", ""]
    assert json.loads(thresholds.loc[">=40 years", "age_bounds"]) == [{"months": 480, "operator": ">="}]
    assert "reference_ranges" not in manifest["context"]
    assert manifest["format_version"] == 7
    assert len(manifest["reference_thresholds_sha256"]) == 64
