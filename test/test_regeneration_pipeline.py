from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import yaml
import pytest

from src.data.recurrence_blood_test_selection import RANKING_DESCRIPTION
from src.model.risk_score_calibration import TRAINING_RISK_REFERENCE_FILENAME
from src.regeneration.runner import run_regeneration
from src.script.survival_benchmark_artifacts import COMPLETION_FILENAME
from test.regeneration_helpers import regeneration_config
from test.test_brier_benchmark import saved_models


def test_all_reuses_fits_and_redraws_work_without_patient_sources(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    benchmark.update(
        table_caption="Synthetic benchmark",
        table_label="tab:synthetic",
        output_csv_path="benchmark.csv",
        output_latex_path="benchmark.tex",
    )
    rows = [{**source, "split": split} for source in sources for split in ("train_bootstrap", "validation", "test")]
    for source in sources:
        directory = Path(source["output_dir"])
        path = directory / "training_config.yml"
        raw = yaml.safe_load(path.read_text())
        raw["output"] = {"evaluate_test": True}
        raw["landmark_event_counts"] = {
            "horizons": [{"label": "6m", "days": 180}],
            "site_labels": ["Liver"],
            "site_regexes": ["C787"],
        }
        raw["dataset"].update(
            train_bootstrap_runs=2,
            val_bootstrap_runs=2,
            test_bootstrap_runs=3,
            train_bootstrap_samples=12,
            val_bootstrap_samples=12,
            test_bootstrap_samples=12,
        )
        raw["temporal_risk"] = {"dpi": 30, "site_labels": ["Liver"], "site_regexes": ["C787"]}
        path.write_text(yaml.safe_dump(raw))
        (directory / "metadata.json").write_text(json.dumps({"status": "complete"}))
        (directory / COMPLETION_FILENAME).write_text(
            json.dumps(
                {
                    "run_fingerprint": "saved-fit",
                    "model_input_fingerprint": "saved-input",
                    "results": [row for row in rows if row["model_key"] == source["model_key"]],
                }
            )
        )
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "selected_blood_tests": dataset.selected_test_keys,
                "results": rows,
            }
        )
    )
    immutable = {
        path: path.read_bytes()
        for source in sources
        for path in Path(source["output_dir"]).iterdir()
        if path.suffix == ".pkl" or path.name == "training_config.yml"
    }
    config = regeneration_config(tmp_path, benchmark, "all", allow_competing_model_fit=True)
    with patch("src.script.survival_training.fit_ebm_model", side_effect=AssertionError("Recurrence refit")):
        with patch("src.model.imputation.fit_feature_imputer", side_effect=AssertionError("Imputer refit")):
            run_regeneration(config)
    summary = json.loads((tmp_path / "regeneration.json").read_text())
    assert summary["status"] == "complete"
    assert all(value == "complete" for value in summary["tasks"].values())
    assert all(path.read_bytes() == contents for path, contents in immutable.items())
    for source in sources:
        directory = Path(source["output_dir"])
        completion = json.loads((directory / COMPLETION_FILENAME).read_text())
        assert completion["run_fingerprint"] == "saved-fit"
        assert completion["model_input_fingerprint"] == "saved-input"
        assert all("harrell_c_index" in row for row in completion["results"])
        for filename in (
            "dataset_statistics.svg",
            "cohort_comparison.tex",
            "population_statistics.tex",
            "landmark_event_counts.csv",
            "recurrence_blood_test_insights.csv",
            "explanations/terms_hazard_ratio_grid.svg",
            TRAINING_RISK_REFERENCE_FILENAME,
            "temporal_risk_quantiles.svg",
            "test_risk_scores.csv",
        ):
            assert (directory / filename).is_file(), filename
    assert (tmp_path / "benchmark.tex").is_file()
    assert (tmp_path / "washout/cohort_counts.json").is_file()
    assert (tmp_path / "brier/complete_case_brier_washout.tex").is_file()
    record = json.loads((tmp_path / "regeneration_artifacts.json").read_text())
    expected_tables = {
        f"table/{stem}.{extension}"
        for stem in ("cindex_benchmark_washout", "complete_case_brier_washout", "cohort_counts")
        for extension in ("csv", "tex")
    }
    assert {item["destination"] for item in record["files"]} == expected_tables

    saved_analyses = {}
    saved_selections = {}
    for source in sources:
        directory = Path(source["output_dir"])
        analysis = directory / "recurrence_blood_test_insights.csv"
        saved_analyses[analysis] = analysis.read_bytes()
        report = directory / "recurrence_blood_test_insights.json"
        payload = json.loads(report.read_text())
        saved_selections[analysis] = payload["figure"]["selected_tests"]
        payload["figure"]["ranking"] = "previous minimum adjusted p ordering"
        payload["figure"]["selected_tests"] = [{"test_key": "obsolete"}]
        payload["config"]["fdr_alpha"] = 0.01
        payload["config"]["plot"]["top_n"] = 99
        report.write_text(json.dumps(payload))
    for path in tmp_path.glob("*.csv"):
        path.unlink()
    config = regeneration_config(
        tmp_path,
        benchmark,
        [
            "dataset_figure",
            "recurrence_figures",
            "benchmark_table",
            "washout_table",
            "brier_table",
            "temporal_risk_figure",
            "age_returns",
            "age_membership",
        ],
    )
    with patch("src.regeneration.context.saved_washout_landmarks", side_effect=AssertionError("Read source data")):
        run_regeneration(config)
    assert json.loads((tmp_path / "regeneration.json").read_text())["status"] == "complete"
    redraw_record = json.loads((tmp_path / "regeneration_artifacts.json").read_text())
    assert {item["destination"] for item in redraw_record["files"]} == {
        name for name in expected_tables if not name.startswith("table/cohort_counts.")
    }
    for analysis, contents in saved_analyses.items():
        assert analysis.read_bytes() == contents
        payload = json.loads(analysis.with_suffix(".json").read_text())
        assert payload["figure"]["ranking"] == RANKING_DESCRIPTION
        assert payload["figure"]["selected_tests"] == saved_selections[analysis]
        assert payload["config"]["fdr_alpha"] == 0.05
        assert payload["config"]["plot"]["top_n"] == 5
    saved_report = Path(sources[0]["output_dir"]) / "recurrence_blood_test_insights.json"
    payload = json.loads(saved_report.read_text())
    payload["config"]["windows"][0]["min_days"] += 1
    saved_report.write_text(json.dumps(payload))
    config = regeneration_config(tmp_path, benchmark, ["recurrence_figures"])
    with pytest.raises(ValueError, match="run recurrence_statistics"):
        run_regeneration(config)


def test_recurrence_redraw_preserves_saved_float_precision(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    (tmp_path / "metadata.json").write_text(json.dumps({
        "selected_blood_tests": dataset.selected_test_keys, "results": sources,
    }))
    config = regeneration_config(tmp_path, benchmark, ["recurrence_statistics"])
    with patch("src.script.recurrence_blood_test_reporting.write_recurrence_blood_test_figure",
               return_value={"files": []}):
        run_regeneration(config)
    expected_median = 2.4050000000000002
    expected_q = 0.04999999999999999
    for source in sources:
        path = Path(source["output_dir"]) / "recurrence_blood_test_insights.csv"
        frame = pd.read_csv(path, float_precision="round_trip")
        frame.loc[0, "median"] = expected_median
        frame.loc[0, "p_value_bh"] = expected_q
        frame.to_csv(path, index=False)
    config = regeneration_config(tmp_path, benchmark, ["recurrence_figures"])
    with patch("src.regeneration.context.saved_washout_landmarks", side_effect=AssertionError("Read source data")):
        with patch("src.regeneration.cohort_reports.write_recurrence_blood_test_figure",
                   return_value={"files": []}) as render:
            run_regeneration(config)
    assert render.call_count == len(sources)
    for call in render.call_args_list:
        row = call.args[0].iloc[0]
        assert row["median"] == expected_median
        assert f"{row['median']:.2f}" == "2.41"
        assert row["p_value_bh"] == expected_q
        assert row["p_value_bh"] < 0.05
