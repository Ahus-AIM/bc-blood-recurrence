from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import Mock

import pandas as pd
import pytest
import yaml

from src.data import PatientBloodTestDataset
from src.data import dataset as dataset_module
from src.script.ebm_survival_config import write_training_config_snapshot
from src.script import export_recurrence_blood_test_insights
from src.script.survival_benchmark_cache import artifact_paths
from src.script.training_recurrence_blood_test_insights import write_training_recurrence_blood_test_insights
from test.test_recurrence_blood_test_insights import report_fixture, row_for
from test.test_recurrence_reference_statistics import age_reference_intervals, run_calculation, write_patient_dates
from test.report_return_helpers import archive_return


def training_report_fixture(tmp_path: Path) -> tuple[PatientBloodTestDataset, Path, Path]:
    dataset, config = report_fixture(tmp_path)
    output = tmp_path / "training"
    output.mkdir()
    source = {
        "dataset": {},
        "population_statistics": {
            "site_labels": [rule.label for rule in config.site_rules],
            "site_regexes": [rule.code_regex for rule in config.site_rules],
            "other_label": None,
        },
        "imputation": {"enabled": True, "method": "locf", "locf_fallback_method": "mice"},
    }
    write_training_config_snapshot(source, dataset.config, output, tmp_path)
    pd.DataFrame([vars(interval) for interval in age_reference_intervals()]).to_csv(
        tmp_path / "ranges.csv", index=False
    )
    path = tmp_path / "report.yml"
    path.write_text(
        yaml.safe_dump(
            {
                "source": {"dataset_config_path": "unavailable_standalone_source.yml"},
                "report": {"output_dir": "unused_standalone_output"},
                "reference_intervals": {"csv_path": "ranges.csv", "percentages_csv_path": None},
                "plot": {"dpi": 35, "formats": ["png", "pdf", "svg"]},
            }
        )
    )
    return dataset, path, output


def test_training_reports_prepared_observed_cohort_without_loading_standalone_data(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, path, output = training_report_fixture(tmp_path)
    reload = Mock(side_effect=AssertionError("The training cohort must be reused."))
    monkeypatch.setattr(dataset_module, "load_patient_data_sources", reload)
    artifacts = write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    assert artifacts["reference_status"] == "pending_remote"
    assert Path("remote_reference/reference_thresholds.csv") in set(artifact_paths(artifacts))
    for artifact in artifact_paths(artifacts):
        assert (output / artifact).is_file()
    for suffix in ("png", "pdf", "svg"):
        assert (output / f"recurrence_blood_test_trajectories.{suffix}").stat().st_size > 0
    frame = pd.read_csv(output / "recurrence_blood_test_insights.csv")
    assert row_for(frame, "Liver", 3)["n_pairs"] == 2
    assert row_for(frame, "Liver", 3)["median_change"] == 4.5
    assert row_for(frame, "Liver", 3, "Other|||g/dL")["n_patients"] == 0
    metadata = json.loads((output / "recurrence_blood_test_insights.json").read_text())
    assert metadata["config"]["dataset_config_path"] == str(output / "training_config.yml")
    assert metadata["input_fingerprint"] == dataset.prepared_data.input_fingerprint
    assert metadata["eligible_patients_by_split"] == {"train": 3, "val": 1, "test": 1}
    assert not (tmp_path / "unused_standalone_output").exists()
    reload.assert_not_called()


@pytest.mark.parametrize("remote_state", ("matching", "stale", "old_definition", "invalid_counts"))
def test_new_training_run_only_uses_matching_valid_shared_percentages(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
    remote_state: str,
) -> None:
    dataset, path, first_output = training_report_fixture(tmp_path)
    write_training_recurrence_blood_test_insights(dataset, first_output, plot_config_path=path)
    folder = first_output / "remote_reference"
    write_patient_dates(folder)
    result_path = run_calculation(folder)
    remote = pd.read_csv(result_path)
    if remote_state in {"stale", "old_definition"}:
        remote["data_hash"] = "different-cohort"
        if remote_state == "old_definition":
            remote = remote.drop(columns="n_pairs")
    elif remote_state == "invalid_counts":
        remote.loc[remote["n_patients"].gt(0), "outside_score_sum"] = 10000
    remote.to_csv(result_path, index=False)
    archived = archive_return(result_path, "reference_percentages")
    raw = yaml.safe_load(path.read_text())
    raw["reference_intervals"]["percentages_csv_path"] = str(result_path)
    path.write_text(yaml.safe_dump(raw))
    output = tmp_path / "next_training"
    output.mkdir()
    shutil.copy2(first_output / "training_config.yml", output / "training_config.yml")
    if remote_state == "invalid_counts":
        with pytest.raises(ValueError, match="denominators"):
            write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
        return
    artifacts = write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    frame = pd.read_csv(output / "recurrence_blood_test_insights.csv")
    if remote_state == "matching":
        assert artifacts["reference_status"] == "verified"
        assert row_for(frame, "Liver", 0)["outside_reference_percent"] == 0
        assert row_for(frame, "Liver", 3)["n_reference_patients"] == 2
        assert row_for(frame, "Liver", 3)["outside_reference_percent"] == 0
        assert not (output / "remote_reference/reference_percentages.csv").exists()
        metadata = json.loads((output / "recurrence_blood_test_insights.json").read_text())
        assert metadata["reference_exchange"]["percentages_csvs"] == [str(archived)]
    else:
        assert artifacts["reference_status"] == "pending_remote"
        assert frame["outside_reference_percent"].isna().all()
        assert "No matching report return" in caplog.text
        assert not (output / "remote_reference/reference_percentages.csv").exists()
        assert (output / "recurrence_blood_test_trajectories.png").is_file()


def test_training_ignores_run_local_return(tmp_path: Path) -> None:
    dataset, path, output = training_report_fixture(tmp_path)
    write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    folder = output / "remote_reference"
    write_patient_dates(folder)
    result_path = run_calculation(folder)
    remote = pd.read_csv(result_path)
    remote["data_hash"] = "wrong-local-export"
    remote.to_csv(result_path, index=False)
    metadata = write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    assert metadata["reference_status"] == "pending_remote"


@pytest.mark.parametrize("conflict", [False, True])
def test_training_and_standalone_discover_duplicates_by_parsed_values(
    tmp_path: Path, conflict: bool,
) -> None:
    dataset, path, output = training_report_fixture(tmp_path)
    write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    folder = output / "remote_reference"
    write_patient_dates(folder)
    archived = archive_return(run_calculation(folder), "reference_percentages")
    second = pd.read_csv(archived)
    if conflict:
        index = second.index[second["n_reference_patients"].gt(0)][0]
        second.loc[index, "outside_score_sum"] = 1
        second.loc[index, "percentage outside reference area"] = 100 / second.loc[index, "n_reference_patients"]
    second["n_pairs"] = second["n_pairs"].astype(float)
    second.iloc[::-1].to_csv(archived.with_name("another_filename.csv"), index=False)
    before = (output / "recurrence_blood_test_insights.csv").read_bytes()
    if conflict:
        with pytest.raises(ValueError, match="Conflicting report returns"):
            write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
        assert (output / "recurrence_blood_test_insights.csv").read_bytes() == before
        return
    artifacts = write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
    assert artifacts["reference_status"] == "verified"
    standalone = tmp_path / "standalone"
    raw = yaml.safe_load(path.read_text())
    raw["source"]["dataset_config_path"] = str(output / "training_config.yml")
    raw["report"]["output_dir"] = str(standalone)
    path.write_text(yaml.safe_dump(raw))
    export_recurrence_blood_test_insights.export_recurrence_report(path)
    metadata = json.loads((standalone / "recurrence_blood_test_insights.json").read_text())
    assert metadata["reference_exchange"]["status"] == "verified"
    assert len(metadata["reference_exchange"]["percentages_csvs"]) == 2
