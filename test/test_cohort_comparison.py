from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import canonical_benchmark_dataset
from src.data.cohort_age_exchange import AGE_FOLDS
from src.data.cohort_comparison import build_comparison_payload, comparison_rows, write_comparison_outputs
from src.data.dataset import load_patient_data_sources, validate_final_cohort_outcomes
from src.data.dataset_config import PatientDatasetConfig
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.population_statistics import PopulationStatisticsConfig
from src.script.ebm_survival_config import serializable_value
from src.script.export_cohort_age_membership import export_saved_cohort_age_membership
from src.script import export_cohort_age_membership, update_cohort_comparison
from src.script.training_cohort_comparison import write_training_cohort_comparison
from src.script.training_dataset_statistics import CONFIG_PATH, write_training_dataset_statistics_png
from src.script.update_cohort_comparison import refresh_cohort_comparison
from test.report_return_helpers import archive_return


@pytest.fixture
def dataset(tmp_path: Path) -> PatientBloodTestDataset:
    patients = [
        ("I", 2008, 730, 5000, [100, 200, 300, 900], ["C795", "C780", "C795", "C799"]),
        ("V", 2017, None, 365, [], []),
        ("T", 2022, None, 1095, [], []),
        ("old", 2000, None, 3650, [], []),
        ("missing_age", 2010, None, 365, [], []),
        ("zero", 2015, None, None, [], []),
        ("other", 2011, None, 365, [], []),
        ("met", 2012, None, 365, [-10], ["C795"]),
    ]
    pd.DataFrame(patients, columns=[
        "id", "diagnosis_year", "diagnosis_to_death", "last_entry_days", "diagnosis_to_ddfs", "ddfs_code",
    ]).to_csv(tmp_path / "endpoints.csv", index=False)
    pd.DataFrame([(row[0], "val" if row[0] == "V" else "test" if row[0] == "T" else "train") for row in patients],
                 columns=["id", "split"]).to_csv(tmp_path / "splits.csv", index=False)
    pd.DataFrame([(row[0], 60) for row in patients if row[0] != "missing_age"],
                 columns=["id", "age"]).to_csv(tmp_path / "ages.csv", index=False)
    blood = [("I", -10, "A", 1), ("I", 40, "A", 2), ("I", 40, "A", 3), ("I", 40, "B", 4),
             ("I", 200, "A", 5), ("I", 900, "A", "not numeric"), ("I", 1000, "X", 6)]
    blood.extend((row[0], 40, "A", 1) for row in patients if row[0] not in {"I", "zero"})
    blood.append(("zero", 40, "X", 1))
    frame = pd.DataFrame(blood, columns=["id", "day", "test_name", "result"])
    frame["unit"] = "u"
    frame.to_csv(tmp_path / "blood.csv", index=False)
    pd.DataFrame([
        ("I", 1000, "diagnosis", "C18.0"), ("I", 1001, "diagnosis", "C180"),
        ("other", -5, "diagnosis", "C180"), ("other", 500, "diagnosis", "C44"),
        ("V", -5, "procedure", "C180"), ("V", 0, "diagnosis", "C50"),
        ("V", 1, "diagnosis", "C77"), ("V", 2, "diagnosis", "D05"),
    ], columns=["id", "day", "event_type", "code"]).to_csv(tmp_path / "clinical.csv", index=False)
    return PatientBloodTestDataset(PatientDatasetConfig.from_mapping({
        "endpoint_csv_path": "endpoints.csv", "bloodtest_csv_path": "blood.csv", "split_csv_path": "splits.csv",
        "bc_diagnosis_patient_age_csv_path": "ages.csv", "diagnosis_procedure_csv_path": "clinical.csv",
        "require_diagnosis_age": True, "include_diagnosis_age": True,
        "selected_blood_tests": ["A|||u", "B|||u"], "min_diagnosis_year": 2004, "min_landmark_day": 31,
        "exclude_and_censor_other_cancer_patients": True,
        "other_cancer_code_filter_regex": r"C(?!(?:50|7[7-9]|9[89]))[0-9]{2}\d*",
        "endpoint_ddfs_code_filter_regex": r"C787\d*",
    }, config_dir=tmp_path))


@pytest.fixture
def population_config() -> PopulationStatisticsConfig:
    return PopulationStatisticsConfig.from_mapping({
        "site_labels": ["Bone", "Brain", "Liver", "Lung"],
        "site_regexes": [r"C795\d*", r"C793\d*", r"C787\d*", r"C780\d*"],
    })


def test_full_history_uses_exact_partition_and_shared_panel(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, tmp_path: Path,
) -> None:
    payload, membership = build_comparison_payload(dataset, population_config)
    assert set(membership.loc[membership["population"] == "included", "id"]) == {"I", "V", "T"}
    assert set(membership.loc[membership["population"] == "excluded", "id"]) == {
        "old", "zero", "other", "met",
    }
    included, excluded = (payload["columns"][group] for group in ("included", "excluded"))
    assert included["patients"] == 3
    assert excluded["patients"] == 4
    assert included["diagnosis_year"] == {"median": 2017, "q1": 2012.5, "q3": 2019.5}
    assert included["observation_years"] == {"median": 2, "q1": 1.5, "q3": 2.5}
    assert excluded["observation_missing"] == 1
    assert included["other_cancer"] == excluded["other_cancer"] == 1
    assert included["distant_metastasis"] == excluded["distant_metastasis"] == 1
    assert included["sites"] == {"Bone": 1, "Brain": 0, "Liver": 0, "Lung": 1, "Other": 1}
    assert excluded["sites"]["Bone"] == 1
    assert included["blood_tests"] == 6  # Numeric history, distinct types/day, before follow-up filtering.
    assert included["blood_tests_per_patient"] == {"median": 1, "q1": 1, "q3": 2.5}
    assert excluded["blood_tests"] == 3
    assert excluded["blood_tests_per_patient"] == {"median": 1, "q1": 0.75, "q3": 1}
    rows = dict(comparison_rows(payload))
    assert rows["Any registered distant recurrence, n (%)"] == ("1 (33.3%)", "1 (25.0%)")
    assert rows["BC diagnosis year"][0] == "2017 (2012.5–2019.5)"
    assert rows["Observation time, years"][0] == "2.0 (1.5–2.5)"
    assert rows["Age at BC diagnosis, years"] == ("—", "—")
    assert "Age missing, n (%)" not in rows
    write_comparison_outputs(payload, tmp_path)
    latex = (tmp_path / "cohort_comparison.tex").read_text()
    assert r"1 (33.3\%)" in latex
    assert "Included (N=3)" in latex and "Excluded (N=4)" in latex
    assert "Missing, nonnumeric and imputed results do not count" in latex


@pytest.mark.parametrize("require_age", [False, True])
def test_numeric_comparison_totals_reconcile_with_initial_figure(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, require_age: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    path = dataset.config.bloodtest_csv_path
    blood = pd.read_csv(path)
    extra = pd.DataFrame([
        ("I", 40, "A", "Sample could not be analysed", "u"),  # Preserve the earlier numeric duplicate.
        ("I", 800, "A", 0, "u"),  # Zero and post-death numeric results still count.
        ("zero", 40, "A", "Sample could not be analysed", "u"),
        ("old", 50, "A", None, "u"),
        ("missing_age", 50, "A", "not numeric", "u"),
        ("unrelated", 40, "A", 1, "u"),  # Not in the endpoint population.
    ], columns=["id", "day", "test_name", "result", "unit"])
    pd.concat([blood, extra], ignore_index=True).to_csv(path, index=False)
    config = replace(dataset.config, require_diagnosis_age=require_age, include_diagnosis_age=require_age)
    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        prepared = PatientBloodTestDataset(config)
    payload, _ = build_comparison_payload(prepared, population_config)
    summary = build_dataset_statistics_summary(config, prepared_dataset=prepared)
    columns = payload["columns"].values()

    assert sum(column["blood_tests"] for column in columns) == (10 if require_age else 11)
    assert sum(column["patients"] for column in columns) == (7 if require_age else 8)
    assert summary.inclusion_box_counts[0] == (
        "N = 7\nblood tests = 10" if require_age else "N = 8\nblood tests = 11"
    )
    assert len(prepared.prepared_data.measurement_df) == (4 if require_age else 5)
    patients = {patient.patient_id: patient for patient in prepared.prepared_data.cohort_comparison_patients}
    assert patients["I"].blood_test_count == 5
    assert patients["zero"].blood_test_count == 0
    assert "Missing diagnosis age" not in summary.exclusion_box_labels
    assert ("missing_age" in patients) is not require_age
    if require_age:
        assert caplog.messages.count("patients removed for missing diagnosis age: 1") == 1


@pytest.mark.parametrize("death_day", [None, -10, 0, 4000])
def test_recorded_deaths_count_all_source_dates_in_both_populations(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig,
    tmp_path: Path, death_day: int | None,
) -> None:
    endpoints = pd.read_csv(dataset.config.endpoint_csv_path, dtype=object)
    endpoints.loc[endpoints["id"].isin(["old", "zero", "missing_age"]), "diagnosis_to_death"] = death_day
    endpoints.to_csv(dataset.config.endpoint_csv_path, index=False)
    prepared = PatientBloodTestDataset(dataset.config)
    payload, _ = build_comparison_payload(prepared, population_config)

    # I dies after recurrence; excluded patients include zero-test and pre-BC deaths.
    # Missing-age patients remain outside both columns, even if a death is recorded.
    assert payload["columns"]["included"]["deaths"] == 1
    assert payload["columns"]["excluded"]["deaths"] == (0 if death_day is None else 2)
    cells = ("1 (33.3%)", "0 (0.0%)" if death_day is None else "2 (50.0%)")
    assert dict(comparison_rows(payload))["Recorded death, n (%)"] == cells

    write_comparison_outputs(payload, tmp_path)
    exported = pd.read_csv(tmp_path / "cohort_comparison.csv").set_index("Characteristic")
    assert tuple(exported.loc["Recorded death, n (%)"]) == cells
    saved = json.loads((tmp_path / "cohort_comparison_values.json").read_text())
    assert saved["columns"]["excluded"]["deaths"] == payload["columns"]["excluded"]["deaths"]
    latex = (tmp_path / "cohort_comparison.tex").read_text()
    expected_row = " & ".join(("Recorded death, n (%)", *cells)).replace("%", r"\%")
    assert expected_row in latex
    assert "including before BC diagnosis and after model follow-up" in latex


@pytest.mark.parametrize("version", [1, 2, 3])
def test_age_refresh_rejects_old_comparison_definitions_before_writing(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, tmp_path: Path,
    version: int,
) -> None:
    write_training_cohort_comparison(dataset, tmp_path, population_config)
    values_path = tmp_path / "cohort_comparison_values.json"
    payload = json.loads(values_path.read_text())
    payload["version"] = version
    values_path.write_text(json.dumps(payload))
    before = {path: path.read_bytes() for path in tmp_path.glob("cohort_comparison*")}

    with pytest.raises(ValueError, match="regenerate.*comparison"):
        refresh_cohort_comparison(tmp_path)

    assert all(path.read_bytes() == content for path, content in before.items())


@pytest.mark.parametrize("missing", [False, True])
def test_missing_age_removals_are_logged_for_preloaded_training_sources(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig,
    missing: bool, caplog: pytest.LogCaptureFixture,
) -> None:
    path = dataset.config.bc_diagnosis_patient_age_csv_path
    ages = pd.read_csv(path, dtype=object)
    ages.loc[ages["id"] == "V", "age"] = 0
    if missing:
        ages.loc[ages["id"] == "old", "age"] = ""
        ages.loc[ages["id"] == "zero", "age"] = None
    else:
        ages.loc[len(ages)] = ["missing_age", 0]
    ages.to_csv(path, index=False)
    sources = load_patient_data_sources(dataset.config)

    with caplog.at_level(logging.INFO, logger="src.data.dataset"):
        prepared = canonical_benchmark_dataset(dataset.config, dataset.selected_test_keys, sources)
    payload, membership = build_comparison_payload(prepared, population_config)

    removed_ids = {"missing_age", "old", "zero"} if missing else set()
    assert set(membership["id"]) == set(sources.endpoint_patients) - removed_ids
    assert "V" in set(membership["id"])  # Numeric age zero is present, not missing.
    assert caplog.messages.count(f"patients removed for missing diagnosis age: {len(removed_ids)}") == 1
    assert sum(column["patients"] for column in payload["columns"].values()) == (5 if missing else 8)


@pytest.mark.parametrize(("patient_id", "last_observation_day", "death_day"), [
    ("V", -10, None),
    ("V", -10, 365),
    ("V", 365, -10),
    ("V", None, -10),
    ("V", -10, -5),
    ("old", -10, None),
    ("missing_age", -10, None),
])
def test_excluded_pre_diagnosis_observation_end_does_not_abort_or_warn(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig,
    patient_id: str, last_observation_day: int | None, death_day: int | None,
    caplog: pytest.LogCaptureFixture,
) -> None:
    sources = load_patient_data_sources(dataset.config)
    patient = replace(sources.endpoint_patients[patient_id],
                      last_observation_day=last_observation_day, death_day=death_day)
    sources.endpoint_patients[patient_id] = patient

    canonical = canonical_benchmark_dataset(dataset.config, dataset.selected_test_keys, sources)

    payload, membership = build_comparison_payload(canonical, population_config)
    included_ids = {"I", "V", "T"} - {patient_id}
    assert set(membership.loc[membership["population"] == "included", "id"]) == included_ids
    assert set(membership["id"]) == set(sources.endpoint_patients) - {"missing_age"}
    assert payload["columns"]["excluded"]["observation_missing"] == (1 if patient_id == "missing_age" else 2)
    assert payload["columns"]["excluded"]["observation_years"] == {
        "median": 1, "q1": 1, "q3": {"V": 5.5, "old": 1, "missing_age": 5.5}[patient_id],
    }
    summaries = {row.patient_id: row for row in canonical.prepared_data.cohort_comparison_patients}
    if patient_id == "missing_age":
        assert patient_id not in summaries
    else:
        assert summaries[patient_id].observation_day is None
        assert summaries[patient_id].blood_test_count == 1
    assert sources.endpoint_patients[patient_id] == patient
    assert canonical.prepared_data.status_by_patient[patient_id].right_censoring_day == -10
    assert "observation end before BC diagnosis" not in caplog.text


@pytest.mark.parametrize("patient_id", ["I", "V", "T"])
@pytest.mark.parametrize(("last_observation_day", "death_day"), [(-10, 365), (365, -10)])
def test_final_cohort_rejects_source_observation_ends_before_diagnosis(
    dataset: PatientBloodTestDataset, patient_id: str, last_observation_day: int, death_day: int,
) -> None:
    sources = load_patient_data_sources(dataset.config)
    sources.endpoint_patients[patient_id] = replace(
        sources.endpoint_patients[patient_id], last_observation_day=last_observation_day, death_day=death_day,
    )
    # Validate source dates even when the selected items and resolved outcomes look valid.
    prepared = dataset.prepared_data
    with pytest.raises(ValueError, match=f"Patient {patient_id} has an observation end before BC diagnosis") as error:
        validate_final_cohort_outcomes(prepared.items_by_split, prepared.status_by_patient, sources.endpoint_patients)
    assert f"last_observation_day={last_observation_day}, death_day={death_day}" in str(error.value)


@pytest.mark.parametrize(("last_observation_day", "death_day", "observation_day"), [
    (0, None, 0), (None, 0, 0), (0, 365, 0), (365, 0, 0),
    (365, None, 365), (None, 365, 365), (730, 365, 365), (365, 730, 365),
    (None, None, None),
])
def test_comparison_preserves_zero_missing_and_death_capped_observation_times(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig,
    last_observation_day: int | None, death_day: int | None, observation_day: int | None,
) -> None:
    sources = load_patient_data_sources(dataset.config)
    sources.endpoint_patients["old"] = replace(sources.endpoint_patients["old"],
                                             last_observation_day=last_observation_day, death_day=death_day)

    canonical = canonical_benchmark_dataset(dataset.config, dataset.selected_test_keys, sources)

    summaries = {row.patient_id: row for row in canonical.prepared_data.cohort_comparison_patients}
    assert summaries["old"].observation_day == observation_day
    payload, _ = build_comparison_payload(canonical, population_config)
    assert payload["columns"]["excluded"]["observation_missing"] == (2 if observation_day is None else 1)
    if observation_day == 0:
        assert payload["columns"]["excluded"]["observation_years"] == {"median": 1, "q1": 0.5, "q3": 1}


def write_age_return(package: Path) -> Path:
    membership = pd.read_csv(package / "cohort_membership.csv", dtype=str, keep_default_na=False)
    quartiles = [(48, 48, 48), (47, 47, 47), (62, 62, 62), (48, 47.5, 55), (48, 47.5, 55), (51, 50.5, 51.5)]
    result_path = package / "age_summaries.csv"
    pd.DataFrame([
        (fold, *summary, membership["hash"].iloc[0])
        for fold, summary in zip(AGE_FOLDS, quartiles, strict=True)
    ], columns=["fold", "median", "lower_q", "upper_q", "hash"]).to_csv(result_path, index=False)
    return result_path


def test_csv_age_export_covers_excluded_and_refreshes_both_outputs_without_sources(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = tmp_path / "run"
    metadata = write_training_cohort_comparison(dataset, run, population_config)
    package = run / "cohort_age_export"
    membership = pd.read_csv(package / "cohort_membership.csv", dtype=str, keep_default_na=False)
    assert list(membership.columns) == ["id", "fold", "hash"]
    assert set(membership["hash"]) == {metadata["cohort_hash"]}
    assert set(membership.loc[membership["id"] == "I", "fold"]) == {"train", "overall", "included"}
    assert set(membership.loc[membership["fold"] == "excluded", "id"]) == {"old", "zero", "other", "met"}
    assert "missing_age" not in set(membership["id"])
    assert not (package / "src").exists()
    result_path = write_age_return(package)
    archived = archive_return(result_path, "ages")
    write_training_cohort_comparison(dataset, run, population_config)
    training_values = json.loads((run / "cohort_comparison_values.json").read_text())
    assert training_values["ages"]["age_years"]["overall"]["median"] == 48
    # Export a real subplot c, then remove patient sources before importing ages.
    raw = load_yml(CONFIG_PATH)
    raw["figure"].update({"dpi": 30, "manual_age_config_path": None})
    raw["dataset"] = {
        "endpoint_csv_path": str(dataset.config.endpoint_csv_path),
        "split_csv_path": str(dataset.config.split_csv_path),
        "recurrence_endpoint_name": "distant_recurrence",
        "input_recurrence_code_filter_regex": r"C7[89]\d*", "endpoint_ddfs_code_filter_regex": r"C787\d*",
    }
    plot_config = tmp_path / "plot.yml"
    plot_config.write_text(yaml.safe_dump(raw))
    monkeypatch.setattr(update_cohort_comparison, "PLOT_CONFIG_PATH", plot_config)
    write_training_dataset_statistics_png(dataset, run, plot_config_path=plot_config, population_config=population_config)
    (run / "training_config.yml").write_text(yaml.safe_dump({"dataset": serializable_value(asdict(dataset.config))}))
    for path in tmp_path.glob("*.csv"):
        path.unlink()
    refresh_config = tmp_path / "refresh.yml"
    refresh_config.write_text(yaml.safe_dump({"report": {"run_dir": str(run), "refresh_dataset_statistics_figure": True}}))
    update_cohort_comparison.refresh_cohort_comparison(run)
    updated = json.loads((run / "cohort_comparison_values.json").read_text())
    assert updated["ages"]["age_years"]["overall"] == updated["ages"]["age_years"]["included"]
    assert dict(comparison_rows(updated))["Recorded death, n (%)"] == ("1 (33.3%)", "0 (0.0%)")
    assert "counts" not in updated["ages"]
    assert "Age missing, n (%)" not in dict(comparison_rows(updated))
    assert dict(comparison_rows(updated))["Age at BC diagnosis, years"] == ("48.0 (47.5–55.0)", "51.0 (50.5–51.5)")
    statistics = yaml.safe_load((run / "dataset_statistics_values.yml").read_text())
    assert statistics["metadata"]["manual_age_summary"]["age_years"]["overall"]["median"] == 48
    assert "48 (47.5–55)" in (run / "dataset_statistics.svg").read_text()
    assert (run / "cohort_age_summary.yml").is_file()
    assert all((run / f"dataset_statistics.{suffix}").is_file() for suffix in ("png", "pdf", "svg"))
    assert "48.0 (47.5--55.0)" in (run / "cohort_comparison.tex").read_text()
    figure_before = (run / "dataset_statistics.svg").read_bytes()
    returned = pd.read_csv(result_path, dtype={"median": float})
    returned.loc[returned["fold"].isin(["overall", "included"]), "median"] = 49.123
    result_path = tmp_path / "returned.csv"
    returned.to_csv(result_path, index=False)
    archived.unlink()
    archived = archive_return(result_path, "ages")
    refresh_cohort_comparison(run, refresh_figure=False)
    assert (run / "dataset_statistics.svg").read_bytes() == figure_before
    updated = json.loads((run / "cohort_comparison_values.json").read_text())
    assert updated["ages"]["age_years"]["overall"]["median"] == 49.123
    assert yaml.safe_load((run / "cohort_age_summary.yml").read_text())["age_years"]["overall"]["median"] == 49.1
    archived.unlink()
    refresh_cohort_comparison(run, refresh_figure=False)
    updated = json.loads((run / "cohort_comparison_values.json").read_text())
    assert updated["ages"] is None  # Old run-local CSVs and saved ages cannot supply fallback values.
    assert dict(comparison_rows(updated))["Age at BC diagnosis, years"] == ("—", "—")


@pytest.mark.parametrize("mutation", ["hash", "quartiles", "missing", "groups", "duplicate", "pooled"])
def test_age_import_rejects_incompatible_returns_before_writing(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, tmp_path: Path, mutation: str,
) -> None:
    write_training_cohort_comparison(dataset, tmp_path, population_config)
    package = tmp_path / "cohort_age_export"
    result_path = write_age_return(package)
    result = pd.read_csv(result_path, dtype=str, keep_default_na=False)
    if mutation == "hash":
        result.loc[0, "hash"] = "another cohort"
    elif mutation == "quartiles":
        result.loc[0, "lower_q"] = "99"
    elif mutation == "missing":
        result.loc[0, "median"] = ""
    elif mutation == "groups":
        result = result.iloc[:-1]
    elif mutation == "duplicate":
        result.loc[0, "fold"] = "validation"
    elif mutation == "pooled":
        result.loc[result["fold"] == "included", "median"] = "49"
    result.to_csv(result_path, index=False)
    archive_return(result_path, "ages")
    before = {path: path.read_bytes() for path in tmp_path.glob("cohort_comparison*")}
    with pytest.raises((ValueError, FileNotFoundError)):
        refresh_cohort_comparison(tmp_path)
    assert all(path.read_bytes() == content for path, content in before.items())
    assert yaml.safe_load((tmp_path / "cohort_age_summary.yml").read_text())["age_years"]["overall"] is None


def test_missing_clinical_source_is_unavailable_and_empty_groups_are_valid(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig,
) -> None:
    patients = tuple(replace(patient, other_cancer=None) for patient in dataset.prepared_data.cohort_comparison_patients)
    empty = replace(dataset.prepared_data, items_by_split={"train": (), "val": (), "test": ()},
                    cohort_comparison_patients=patients)
    payload, _ = build_comparison_payload(PatientBloodTestDataset(dataset.config, prepared_data=empty), population_config)
    assert payload["columns"]["included"]["patients"] == 0
    assert payload["columns"]["included"]["diagnosis_year"] is None
    assert payload["columns"]["excluded"]["other_cancer"] is None
    assert dict(comparison_rows(payload))["Age at BC diagnosis, years"] == ("—", "—")


def test_existing_run_exports_new_csv_without_sources_or_changes_to_reports(
    dataset: PatientBloodTestDataset, population_config: PopulationStatisticsConfig, tmp_path: Path,
) -> None:
    run = tmp_path / "run"
    payload, membership = build_comparison_payload(dataset, population_config)
    write_comparison_outputs(payload, run)
    package = run / "cohort_age_export"
    package.mkdir()
    membership.to_csv(package / "cohort_membership.csv", index=False)  # Legacy unique-patient format.
    before = {path: path.read_bytes() for path in run.glob("cohort_comparison*")}
    for path in tmp_path.glob("*.csv"):
        path.unlink()

    config = tmp_path / "export.yml"
    config.write_text(yaml.safe_dump({"report": {"run_dir": "run"}}))
    export_cohort_age_membership.export_saved_cohort_age_membership(run)
    exported_path = package / "cohort_membership.csv"
    exported = pd.read_csv(exported_path, dtype=str)
    assert list(exported.columns) == ["id", "fold", "hash"]
    assert len(exported) == 13  # Three included patients with three memberships each, plus four excluded.
    assert set(exported["hash"]) == {payload["cohort_hash"]}
    exported_before = exported_path.read_bytes()
    assert export_saved_cohort_age_membership(run) == exported_path
    assert exported_path.read_bytes() == exported_before
    assert all(path.read_bytes() == content for path, content in before.items())
