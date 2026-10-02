from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_config import (
    DEFAULT_WINDOWS,
    RecurrenceBloodTestConfig,
    RecurrenceTimeWindow,
    load_recurrence_blood_test_config,
    parse_windows,
    validate_report_config,
)
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights, patient_window_medians
from src.data.recurrence_blood_test_insights import site_patient_counts
from src.data.recurrence_site_common import RecurrenceSiteRule
from src.script import export_recurrence_blood_test_insights as script
from test.test_sample_weighting import blood_row, dataset_config, endpoint_row


def report_fixture(tmp_path: Path) -> tuple[PatientBloodTestDataset, RecurrenceBloodTestConfig]:
    rows = [
        measured("A", 300, 10),
        measured("A", 299, 10),
        measured("A", 298, 100),
        measured("B", 300, 20),
        measured("D", 300, 40),
        measured("A", 89, 10),
        measured("A", 0, 15),
        measured("B", 0, 24),
        measured("C", 0, 30),
        measured("E", 0, 50),
        measured("A", -1, 10000),
        {**measured("A", 300, 99), "test_name": "Other"},
    ]
    endpoints = [endpoint_row(patient, recurrence_day=500, right_censoring_day=1000) for patient in "ABCDE"]
    for row in endpoints:
        row["ddfs_code"] = "C787"
    endpoints[0].update(diagnosis_to_ddfs_in_days="[500,500,600]", ddfs_code="['C787','C795','C793']")
    endpoints[-1].update(diagnosis_to_ddfs_in_days="[500,600]", ddfs_code="['C795','C787']")
    config = dataset_config(
        tmp_path,
        rows,
        endpoints,
        split_by_patient={"A": "train", "B": "val", "C": "test", "D": "train", "E": "train"},
        selected_blood_tests=["B-Hb|||g/dL", "Other|||g/dL"],
        include_recurrence_day_blood_tests=True,
    )
    dataset = PatientBloodTestDataset(config)
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml",
        dataset.config,
        (
            RecurrenceSiteRule("Liver", r"C787\d*"),
            RecurrenceSiteRule("Bone", r"C795\d*"),
            RecurrenceSiteRule("Brain", r"C793\d*"),
        ),
        "Other",
        None,
        tmp_path / "output",
    )
    return dataset, report


def measured(patient: str, days_before: int, value: float) -> dict[str, object]:
    return {**blood_row(patient, 500 - days_before), "result": value}


def row_for(frame: pd.DataFrame, site: str, window_index: int, test: str = "B-Hb|||g/dL") -> pd.Series:
    return frame.loc[
        (frame["site"] == site) & (frame["test_key"] == test) & (frame["window"] == DEFAULT_WINDOWS[window_index].label)
    ].iloc[0]


def test_patient_medians_pairing_missingness_and_global_adjustment(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    frame = build_recurrence_blood_test_insights(dataset, config, {"B-Hb": "Hemoglobin"})
    baseline, final = row_for(frame, "Liver", 0), row_for(frame, "Liver", 3)
    assert baseline["blood_test"] == "Hemoglobin"
    assert baseline["n_patients"] == 3
    assert baseline[["q1", "median", "q3"]].tolist() == [15, 20, 30]
    assert final["n_patients"] == 3
    assert final["median"] == 24
    assert final["n_pairs"] == 2
    assert final[["change_q1", "median_change", "change_q3"]].tolist() == [4.25, 4.5, 4.75]
    assert final["p_value"] == 0.5  # Two positive paired differences: exact two-sided probability.
    assert final["p_value_bh"] == 1.0  # Adjustment includes other organs and windows.
    assert row_for(frame, "Bone", 3)["n_patients"] == 2
    assert row_for(frame, "Brain", 3)["n_patients"] == 0  # A's later brain recurrence is ignored.
    other = row_for(frame, "Liver", 3, "Other|||g/dL")
    assert other["n_patients"] == other["n_pairs"] == 0  # No LOCF or fallback filling.
    assert other["test_status"] == "no_pairs"
    assert np.isnan(other["p_value"])
    zero = row_for(frame, "Liver", 2)
    assert zero["n_pairs"] == 1  # No arbitrary minimum count.
    assert zero["n_nonzero_pairs"] == 0
    assert zero["median_change"] == 0
    assert zero["p_value"] == 1


@pytest.mark.parametrize(
    ("configured_windows", "days", "expected_medians"),
    [
        (False, [360, 359, 180, 179, 90, 89, 7, 6, 0, -1], [269.5, 134.5, 48, 3]),
        (True, [540, 539, 180, 179, 90, 89, 30, 29, 7, 6, 0, -1], [359.5, 134.5, 59.5, 18, 3]),
    ],
)
def test_elapsed_window_boundaries_and_recurrence_day_toggle(
    tmp_path: Path,
    configured_windows: bool,
    days: list[int],
    expected_medians: list[float],
) -> None:
    config = dataset_config(
        tmp_path,
        [measured("P", day, day) for day in days],
        [{**endpoint_row("P", recurrence_day=500, right_censoring_day=1000), "ddfs_code": "C787"}],
        selected_blood_tests=["B-Hb|||g/dL"],
        include_recurrence_day_blood_tests=True,
    )
    dataset = PatientBloodTestDataset(config)
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml", dataset.config, (RecurrenceSiteRule("Liver", "C787"),), None, None, tmp_path / "output"
    )
    if configured_windows:
        path = Path(__file__).resolve().parents[1] / "src/config/recurrence_blood_test_insights.yml"
        report = replace(report, windows=parse_windows(load_yml(path)["report"]["windows"]))
    validate_report_config(report)
    medians = patient_window_medians(dataset, report).sort_values("window_index")
    assert medians["result"].tolist() == expected_medians
    excluded = PatientBloodTestDataset({**config, "include_recurrence_day_blood_tests": False})
    excluded_report = replace(report, dataset=excluded.config)
    assert patient_window_medians(excluded, excluded_report).sort_values("window_index")["result"].tolist()[-1] == 6


@pytest.mark.parametrize("include_overall", [False, True])
def test_report_without_any_recurrences_keeps_empty_cells(tmp_path: Path, include_overall: bool) -> None:
    config = dataset_config(tmp_path, [blood_row("P", 40)], [endpoint_row("P")])
    dataset = PatientBloodTestDataset(config)
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml",
        dataset.config,
        (RecurrenceSiteRule("Liver", "C787"),),
        None,
        None,
        tmp_path / "output",
        include_overall=include_overall,
    )
    frame = build_recurrence_blood_test_insights(dataset, report, {})
    assert len(frame) == 4 * (1 + include_overall)
    assert not frame["n_patients"].any()
    assert frame["p_value_bh"].isna().all()
    assert frame["n_site_patients"].eq(0).all()


@pytest.mark.parametrize("include_overall", [False, True])
def test_site_totals_include_patients_without_window_measurements_and_deduplicate_sites(
    tmp_path: Path,
    include_overall: bool,
) -> None:
    dataset, config = report_fixture(tmp_path)
    config = replace(config, include_overall=include_overall)
    blood = pd.read_csv(dataset.config.bloodtest_csv_path)
    endpoints = pd.read_csv(dataset.config.endpoint_csv_path)
    splits = pd.read_csv(dataset.config.split_csv_path)
    extra_blood = [measured("F", 400, 30), measured("G", 400, 30), measured("H", -1, 30)]
    extra_endpoints = [
        {**endpoint_row(patient, recurrence_day=500, right_censoring_day=1000),
         "ddfs_code": code, "bc_diagnosis_year": year}
        for patient, code, year in (("F", "C787", 2022), ("G", "C793", 2020), ("H", "C787", 2016))
    ]
    endpoints.loc[endpoints["id"].eq("A"), "diagnosis_to_ddfs_in_days"] = "[500,500,500,600]"
    endpoints.loc[endpoints["id"].eq("A"), "ddfs_code"] = "['C787','C787','C795','C793']"
    pd.concat([blood, pd.DataFrame(extra_blood)], ignore_index=True).to_csv(
        dataset.config.bloodtest_csv_path, index=False
    )
    pd.concat([endpoints, pd.DataFrame(extra_endpoints)], ignore_index=True).to_csv(
        dataset.config.endpoint_csv_path, index=False
    )
    pd.concat([splits, pd.DataFrame({"id": ["F", "G", "H"], "split": ["test", "val", "train"]})]).to_csv(
        dataset.config.split_csv_path,
        index=False,
    )
    dataset = PatientBloodTestDataset(dataset.config)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    # F and G are eligible but outside all report windows; H has only a post-recurrence result.
    expected_totals = {"Liver": 5, "Bone": 2, "Brain": 1, "Other": 0}
    if include_overall:
        expected_totals = {"Any": 7, **expected_totals}
    assert site_patient_counts(frame) == expected_totals
    assert row_for(frame, "Liver", 0)["n_patients"] == 3
    assert row_for(frame, "Liver", 3)["n_pairs"] == 2
    assert row_for(frame, "Brain", 0)["n_patients"] == 0
    assert row_for(frame, "Brain", 0)["n_site_patients"] == 1
    config = replace(config, plot=replace(config.plot, dpi=35, formats=("svg",)))
    script.write_recurrence_blood_test_report(frame, dataset, config)
    svg = (config.output_dir / "recurrence_blood_test_trajectories.svg").read_text()
    assert "Brain (1 patient)" in svg  # Site totals survive top-test selection with no calculable pairs.
    latex = (config.output_dir / "recurrence_blood_test_insights.tex").read_text()
    assert r"\textbf{Liver} ($N = 5$ patients)" in latex


@pytest.mark.parametrize(
    "windows",
    [
        (RecurrenceTimeWindow("baseline", 181, 360), RecurrenceTimeWindow("overlap", 90, 181)),
        (RecurrenceTimeWindow("baseline", 181, 360), RecurrenceTimeWindow("gap", 0, 179)),
        (RecurrenceTimeWindow("negative", -1, 360),),
    ],
)
def test_invalid_report_windows_are_rejected(tmp_path: Path, windows: tuple[RecurrenceTimeWindow, ...]) -> None:
    _dataset, config = report_fixture(tmp_path)
    with pytest.raises(ValueError):
        validate_report_config(replace(config, windows=windows))


@pytest.mark.parametrize(
    ("source_other_label", "local_site_config", "expected_other_label"),
    [
        (None, {}, None),
        ("Other", {}, "Other"),
        ("Other", {"other_label": None}, None),
        ("Other", {"other_label": "Unmatched"}, "Unmatched"),
    ],
)
def test_standalone_script_writes_csv_latex_and_reproducibility_metadata(
    tmp_path: Path,
    source_other_label: str | None,
    local_site_config: dict[str, str | None],
    expected_other_label: str | None,
) -> None:
    dataset, config = report_fixture(tmp_path)
    raw_dataset = {key: str(value) if isinstance(value, Path) else value for key, value in vars(dataset.config).items()}
    source = {
        "dataset": raw_dataset,
        "population_statistics": {
            "site_labels": ["Liver", "Brain"],  # Bone events exercise unmatched-site handling.
            "site_regexes": ["C787", "C793"],
            "other_label": source_other_label,
        },
        "imputation": {"enabled": True, "method": "mice"},
    }
    config.dataset_config_path.write_text(yaml.safe_dump(source), encoding="utf-8")
    report_path = tmp_path / "report.yml"
    report = {"source": {"dataset_config_path": "source.yml"}, "report": {"output_dir": "output"}}
    if local_site_config:
        report["population_statistics"] = local_site_config
    report_path.write_text(yaml.safe_dump(report), encoding="utf-8")
    parsed = load_recurrence_blood_test_config(report_path)
    assert parsed.dataset.include_recurrence_day_blood_tests
    assert parsed.other_label == expected_other_label
    script.export_recurrence_report(report_path)
    csv = pd.read_csv(config.output_dir / "recurrence_blood_test_insights.csv")
    assert row_for(csv, "Liver", 3)["median_change"] == 4.5
    expected_sites = ["Liver", "Brain"]
    if expected_other_label is not None:
        expected_sites.append(expected_other_label)
        assert row_for(csv, expected_other_label, 3)["n_patients"] == 2
    assert csv["site"].unique().tolist() == expected_sites
    latex = (config.output_dir / "recurrence_blood_test_insights.tex").read_text()
    assert r"\begin{longtable}" in latex
    assert "g/dL" in latex
    assert "No minimum paired count" in latex
    for label in ("Other", "Unmatched"):
        assert (rf"\textbf{{{label}}}" in latex) == (label == expected_other_label)
    metadata = json.loads((config.output_dir / "recurrence_blood_test_insights.json").read_text())
    assert metadata["config"]["other_label"] == expected_other_label
    assert metadata["number_of_hypotheses"] == int(csv["p_value"].notna().sum())
    assert "exact conditional sign distribution" in metadata["p_value_method"]
    assert "tied average ranks" in metadata["p_value_method"]
    assert metadata["eligible_patients_by_split"] == {"train": 3, "val": 1, "test": 1}
