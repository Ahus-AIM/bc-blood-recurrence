from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return

import pandas as pd
import pytest
import yaml

from src.data.recurrence_blood_test_config import load_recurrence_blood_test_config, validate_report_config
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights, site_patient_counts
from src.data.recurrence_reference_exchange import export_reference_package
from src.script.recurrence_blood_test_reporting import write_recurrence_blood_test_report
from test.test_recurrence_blood_test_insights import report_fixture, row_for
from test.test_recurrence_reference_statistics import (
    age_reference_intervals,
    reference_interval,
    run_calculation,
    write_patient_dates,
)


def test_overall_counts_synchronous_and_unmatched_recurrences_once(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    # E's first recurrence is Bone only; A has synchronous Liver and Bone events.
    config = replace(config, site_rules=config.site_rules[:1], other_label=None)
    original = build_recurrence_blood_test_insights(dataset, config, {})
    config = replace(config, include_overall=True)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    assert config.sites == ("Any", "Liver")
    assert site_patient_counts(frame) == {"Any": 5, "Liver": 4}
    baseline, final = row_for(frame, "Any", 0), row_for(frame, "Any", 3)
    assert baseline["n_patients"] == 3
    assert baseline["median"] == 20
    assert final["n_patients"] == 4
    assert final["median"] == 27
    assert final["n_pairs"] == 2
    assert final["median_change"] == 4.5
    # Site-specific summaries are preserved; adjustment includes the new comparisons.
    pd.testing.assert_frame_equal(
        original.drop(columns="p_value_bh"),
        frame.loc[frame["site"].eq("Liver")].drop(columns="p_value_bh").reset_index(drop=True),
    )


@pytest.mark.parametrize("remote", [False, True])
def test_overall_is_exported_with_paired_reference_percentages(tmp_path: Path, remote: bool) -> None:
    dataset, config = report_fixture(tmp_path)
    references = age_reference_intervals(upper=20) if remote else (reference_interval(upper=20),)
    config = replace(
        config, other_label=None, reference_ranges=references, plot=replace(config.plot, formats=("svg",), dpi=35)
    )
    original = build_recurrence_blood_test_insights(dataset, config, {})
    _, previous_manifest = export_reference_package(original, dataset, config)
    config = replace(config, include_overall=True)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    folder, manifest = export_reference_package(frame, dataset, config)
    assert manifest["data_hash"] != previous_manifest["data_hash"]
    if remote:
        measurements = pd.read_csv(folder / "measurements.csv")
        overall = measurements.loc[measurements["recurrence site"].eq("Any")]
        assert not overall.duplicated(["id", "test_key", "interval", "draw days since diagnosis"]).any()
        assert set(overall["id"]) == set("ABCDE")
        write_patient_dates(folder)
        archive_return(run_calculation(folder), "reference_percentages")
    write_recurrence_blood_test_report(frame, dataset, config)
    exported = pd.read_csv(config.output_dir / "recurrence_blood_test_insights.csv")
    baseline, final = row_for(exported, "Any", 0), row_for(exported, "Any", 3)
    assert baseline["n_reference_patients"] == 3
    assert baseline["outside_reference_percent"] == pytest.approx(100 / 3)
    assert final["n_patients"] == 4
    assert final["n_reference_patients"] == final["n_pairs"] == 2
    assert final["outside_reference_percent"] == 50
    selected = pd.read_csv(config.output_dir / "recurrence_blood_test_trajectories_values.csv")
    assert selected.iloc[0]["site"] == "Any"
    assert selected.loc[selected["site"].eq("Any"), "n_site_patients"].eq(5).all()
    svg = (config.output_dir / "recurrence_blood_test_trajectories.svg").read_text()
    assert "Any (5 patients)" in svg
    latex = (config.output_dir / "recurrence_blood_test_insights.tex").read_text()
    assert r"\textbf{Any} ($N = 5$ patients)" in latex
    metadata = json.loads((config.output_dir / "recurrence_blood_test_insights.json").read_text())
    assert metadata["site_patient_counts"]["Any"] == 5
    assert metadata["number_of_hypotheses"] == int(exported["p_value"].notna().sum())


@pytest.mark.parametrize("enabled", [False, True])
def test_report_config_can_override_inherited_overall_setting(tmp_path: Path, enabled: bool) -> None:
    dataset, config = report_fixture(tmp_path)
    raw_dataset = {key: str(value) if isinstance(value, Path) else value for key, value in vars(dataset.config).items()}
    config.dataset_config_path.write_text(
        yaml.safe_dump(
            {
                "dataset": raw_dataset,
                "population_statistics": {"include_overall": not enabled, "overall_label": "All sites"},
            }
        )
    )
    report_path = tmp_path / "report.yml"
    report_path.write_text(
        yaml.safe_dump(
            {
                "source": {"dataset_config_path": "source.yml"},
                "report": {"output_dir": "output"},
                "population_statistics": {"include_overall": enabled, "other_label": None},
            }
        )
    )
    parsed = load_recurrence_blood_test_config(report_path)
    assert parsed.include_overall is enabled
    assert parsed.overall_label == "All sites"
    assert ("All sites" in parsed.sites) is enabled
    assert parsed.other_label is None


@pytest.mark.parametrize("label", ["", " ", "Liver", "Other"])
def test_overall_label_must_be_nonempty_and_distinct(tmp_path: Path, label: str) -> None:
    _, config = report_fixture(tmp_path)
    with pytest.raises(ValueError, match="nonempty and unique"):
        validate_report_config(replace(config, include_overall=True, overall_label=label))
