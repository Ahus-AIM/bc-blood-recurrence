from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import pytest

from src.regeneration import runner
from src.regeneration.config import ALL_TASKS, load_regeneration_config
from src.regeneration.context import load_context
from src.script import brier_model_artifacts
from src.script.single_bloodtest_benchmark_reporting import (
    ALL_MODEL_KEY,
    CLINICAL_BASELINE_MODEL_KEY,
    merge_result_records,
)
from test.regeneration_helpers import regeneration_config
from test.test_brier_benchmark import saved_models
from test.test_single_bloodtest_ebm_top_features import FakeModel, benchmark_config, save_model_cell
from src.script import single_bloodtest_ebm_top_features
from src.model.ebm_visualizations import write_interpret_html


def inventory(tmp_path: Path, tasks: list[str] | str, **settings: object) -> Path:
    benchmark = {
        "sites": [{"label": site} for site in ("Bone", "Lung")],
        "horizons": [{"label": "6m", "days": 180}],
    }
    rows = [
        {
            "site_label": site,
            "horizon_label": "6m",
            "horizon_days": 180,
            "model_key": model,
            "split": "test",
            "status": "complete",
        }
        for site in ("Bone", "Lung")
        for model in (CLINICAL_BASELINE_MODEL_KEY, ALL_MODEL_KEY)
    ]
    (tmp_path / "metadata.json").write_text(json.dumps({"selected_blood_tests": [], "results": rows}))
    return regeneration_config(tmp_path, benchmark, tasks, **settings)


def test_selection_uses_one_benchmark_without_loading_models_or_patient_data(tmp_path: Path) -> None:
    path = inventory(tmp_path, ["explanations"], sites=["Bone"], horizons=["6m"], models=["all_blood_tests"])
    context = load_context(load_regeneration_config(path))
    assert [cell.key for cell in context.selected] == [("Bone", "6m", ALL_MODEL_KEY)]
    assert [cell.key for cell in context.evaluation_cells()] == [
        ("Bone", "6m", CLINICAL_BASELINE_MODEL_KEY),
        ("Bone", "6m", ALL_MODEL_KEY),
    ]
    assert context.figure_benchmark()["sites"] == [{"label": "Bone"}]
    assert not list(tmp_path.glob("**/training_config.yml"))


@pytest.mark.parametrize(
    "settings",
    [
        {"sites": ["Brain"]},
        {"horizons": ["1m"]},
        {"models": ["unknown"]},
        {"sites": []},
        {"models": ["all_blood_tests", "all_blood_tests"]},
        {"allow_competing_model_fit": "false"},
        {"task": ["importance"]},
    ],
)
def test_invalid_selection_fails_before_writing_outputs(tmp_path: Path, settings: dict[str, object]) -> None:
    path = inventory(tmp_path, ["explanations"], **settings)
    with pytest.raises((TypeError, ValueError)):
        runner.run_regeneration(path)
    assert not (tmp_path / "regeneration.json").exists()


def test_all_avoids_duplicate_producers_and_inapplicable_composites(tmp_path: Path) -> None:
    path = inventory(tmp_path, "all", models=["clinical_baseline"])
    context = load_context(load_regeneration_config(path))
    assert context.settings.tasks == ALL_TASKS
    assert "importance" not in runner.task_plan(context)
    assert "top_features" not in runner.task_plan(context)
    context.settings = replace(context.settings, tasks=("cohort_statistics", "dataset_figure", "age_returns"))
    assert runner.task_plan(context) == ("cohort_statistics",)


def test_first_failure_stops_subsequent_tasks_and_records_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = inventory(tmp_path, ["explanations", "importance", "top_features"])
    visited = []

    def task(context) -> None:
        visited.append(context.output_dir)
        raise ValueError("missing required input")

    monkeypatch.setitem(runner.TASKS, "explanations", task)
    monkeypatch.setitem(runner.TASKS, "importance", lambda _context: pytest.fail("Ran after failure"))
    with pytest.raises(ValueError, match="missing required input"):
        runner.run_regeneration(path)
    assert visited == [tmp_path]
    summary = json.loads((tmp_path / "regeneration.json").read_text())
    assert summary["status"] == "failed"
    assert summary["tasks"] == {"explanations": "failed", "importance": "pending", "top_features": "pending"}


def test_partial_result_updates_preserve_other_cells_splits_and_washouts() -> None:
    base = {"model_key": "A", "site_label": "Bone", "horizon_label": "6m", "split": "test"}
    previous = [
        {**base, "score": 1},
        {**base, "split": "validation", "score": 2},
        {**base, "site_label": "Lung", "score": 3},
        {**base, "washout_days": 30, "score": 4},
    ]
    merged = merge_result_records(previous, [{**base, "score": 5}])
    assert [row["score"] for row in merged] == [5, 2, 3, 4]


def test_filtered_composites_use_only_selected_saved_models(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [save_model_cell(tmp_path, site, horizon) for site in ("Bone", "Lung") for horizon in ("1m", "6m")]
    (tmp_path / "metadata.json").write_text(json.dumps({"selected_blood_tests": ["A|||u"], "results": rows}))
    loaded: list[Path] = []

    def load_model(path: Path) -> FakeModel:
        loaded.append(path)
        return FakeModel()

    monkeypatch.setattr(single_bloodtest_ebm_top_features, "load_model", load_model)
    config = regeneration_config(
        tmp_path,
        benchmark_config(),
        ["importance", "top_features"],
        sites=["Bone"],
        horizons=["6m"],
        models=["all_blood_tests"],
    )
    run = tmp_path / "Bone" / "6m" / ALL_MODEL_KEY
    originals = {path: path.read_bytes() for path in run.iterdir()}
    runner.run_regeneration(config)
    assert loaded == [run / "model.pkl", run / "model.pkl"]
    assert (tmp_path / "global_importance/6m/global_importance_bone.svg").is_file()
    assert (tmp_path / "top_features/6m/ebm_top_features.svg").is_file()
    assert not (tmp_path / "top_features/1m").exists()
    assert not (tmp_path / "global_importance/6m/global_importance_lung.svg").exists()
    assert all(path.read_bytes() == content for path, content in originals.items())
    record = json.loads((tmp_path / "regeneration_artifacts.json").read_text())
    assert {item["destination"] for item in record["files"]} == {
        f"figure/{stem}.{extension}"
        for stem in ("global_importance_bone", "ebm_top_features") for extension in ("svg", "png")
    }


def test_optional_html_export_obeys_fail_fast(tmp_path: Path) -> None:
    with patch("src.model.ebm_visualizations.preserve", side_effect=RuntimeError("HTML export failed")):
        with pytest.raises(RuntimeError, match="HTML export failed"):
            write_interpret_html(object(), tmp_path / "explanation.html", fail_fast=True)


def test_brier_fit_requires_opt_in_and_then_reuses_matching_companions(tmp_path: Path) -> None:
    dataset, sources, benchmark = saved_models(tmp_path)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "selected_blood_tests": dataset.selected_test_keys,
                "results": sources,
            }
        )
    )
    path = regeneration_config(tmp_path, benchmark, ["brier"])
    originals = {file: file.read_bytes() for file in tmp_path.rglob("*.pkl")}
    with patch.object(brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected fitting")):
        with pytest.raises(ValueError, match="allow_competing_model_fit"):
            runner.run_regeneration(path)
    assert not list((tmp_path / "brier").rglob("competing_model.pkl"))
    path = regeneration_config(tmp_path, benchmark, ["brier"], allow_competing_model_fit=True)
    runner.run_regeneration(path)
    companions = {file: file.read_bytes() for file in (tmp_path / "brier").rglob("competing_model.pkl")}
    assert len(companions) == len(sources)
    path = regeneration_config(tmp_path, benchmark, ["brier"])
    with patch.object(brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected refitting")):
        runner.run_regeneration(path)
    assert all(file.read_bytes() == content for file, content in (originals | companions).items())
    metadata_path = next((tmp_path / "brier").rglob("competing_model.json"))
    metadata = json.loads(metadata_path.read_text())
    metadata["training_fingerprint"] = "stale"
    metadata_path.write_text(json.dumps(metadata))
    with patch.object(brier_model_artifacts, "fit_ebm_model", side_effect=AssertionError("Unexpected refitting")):
        with pytest.raises(ValueError, match="outdated"):
            runner.run_regeneration(path)
