from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from src.regeneration import runner
from src.regeneration.artifact_run import RECORD_FILENAME
from src.regeneration.config import load_regeneration_config
from src.regeneration.context import RegenerationContext
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY
from test.regeneration_helpers import regeneration_config


def write_file(root: Path, name: str, contents: str) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents)
    return path


def bundle_fixture(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    sites = ("All", "Bone", "Brain", "Liver", "Lung")
    benchmark = {
        "sites": [{"label": site} for site in sites],
        "horizons": [{"label": "6m", "days": 180}],
        "ebm_top_features": {"output_dir": "custom_top"},
        "washout": {"output_latex_path": "custom/cindex.tex", "output_csv_path": "custom/cindex.csv"},
        "washout_cohort_counts": {"output_latex_path": "custom/counts.tex", "output_csv_path": "custom/counts.csv"},
        "brier": {"output_dir": "custom_brier"},
    }
    rows = [{"site_label": site, "horizon_label": "6m", "horizon_days": 180,
             "model_key": ALL_MODEL_KEY, "split": "test", "status": "complete"} for site in sites]
    write_file(tmp_path, "metadata.json", json.dumps({"selected_blood_tests": [], "results": rows}))
    config = regeneration_config(tmp_path, benchmark, ["paper_bundle"], sites=["Bone"])
    expected: dict[str, str] = {}

    def add(source: str, destination: str) -> None:
        write_file(tmp_path, source, source)
        expected[destination] = source

    for extension in ("svg", "png"):
        add(f"custom_top/6m/ebm_top_features.{extension}", f"figure/ebm_top_features.{extension}")
        add(f"All/6m/{ALL_MODEL_KEY}/dataset_statistics.{extension}", f"figure/dataset_statistics.{extension}")
        for suffix in ("", "_any", "_bone", "_brain", "_liver", "_lung"):
            name = f"recurrence_blood_test_trajectories{suffix}.{extension}"
            add(f"All/6m/{ALL_MODEL_KEY}/{name}", f"figure/{name}")
        for site in sites:
            suffix = "" if site == "All" else f"_{site.lower()}"
            add(f"{site}/6m/{ALL_MODEL_KEY}/explanations/terms_hazard_ratio_grid.{extension}",
                f"figure/terms_hazard_ratio_grid{suffix}.{extension}")
            name = f"global_importance{suffix}.{extension}"
            source = (f"global_importance/6m/{name}" if site == "All"
                      else f"{site}/6m/{ALL_MODEL_KEY}/explanations/global_importance.{extension}")
            add(source, f"figure/{name}")
    for extension in ("tex", "csv"):
        add(f"custom/cindex.{extension}", f"table/cindex_benchmark_washout.{extension}")
        add(f"custom/counts.{extension}", f"table/cohort_counts.{extension}")
        add(f"custom_brier/complete_case_brier_washout.{extension}",
            f"table/complete_case_brier_washout.{extension}")
        add(f"All/6m/{ALL_MODEL_KEY}/cohort_comparison.{extension}", f"table/cohort_comparison.{extension}")
    write_file(tmp_path, f"All/6m/{ALL_MODEL_KEY}/model_input.csv", "patient-level data")
    write_file(tmp_path, f"All/1m/{ALL_MODEL_KEY}/dataset_statistics.svg", "wrong horizon")
    return config, expected


def configure_bundle(path: Path, **settings: object) -> None:
    config = yaml.safe_load(path.read_text())
    config["paper_bundle"] = settings
    path.write_text(yaml.safe_dump(config))


def set_tasks(path: Path, tasks: list[str]) -> None:
    config = yaml.safe_load(path.read_text())
    config["regenerate"]["tasks"] = tasks
    path.write_text(yaml.safe_dump(config))


def produce(
    config: Path, monkeypatch: pytest.MonkeyPatch, outputs: dict[str, str], task: str = "explanations",
) -> None:
    def redraw(context: RegenerationContext) -> tuple[Path, ...]:
        for source, content in outputs.items():
            write_file(context.output_dir, source, content)
        return tuple(context.output_dir / source for source in outputs)

    monkeypatch.setitem(runner.TASKS, task, redraw)
    set_tasks(config, [task])
    runner.run_regeneration(config)


def read_manifest(root: Path) -> dict[str, object]:
    return json.loads((root / "paper_bundle/manifest.json").read_text())


def test_replay_copies_recorded_outputs_without_models_and_preserves_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    # Tracking also works when automatic collection is disabled.
    produce(config, monkeypatch, {
        **{source: source for source in expected.values()},
        f"All/6m/{ALL_MODEL_KEY}/model_input.csv": "new patient-level data",
        f"All/1m/{ALL_MODEL_KEY}/dataset_statistics.svg": "new plot at another horizon",
    })
    assert not (tmp_path / "paper_bundle").exists()
    set_tasks(config, ["paper_bundle"])
    originals = {path: path.read_bytes() for path in tmp_path.rglob("*")
                 if path.is_file() and path.name not in {"regeneration.json", "regenerate.log"}}
    with patch("src.regeneration.context.load_model", side_effect=AssertionError("Loaded model")), patch(
        "src.regeneration.context.saved_washout_landmarks", side_effect=AssertionError("Loaded patient data"),
    ):
        runner.run_regeneration(config)
        runner.run_regeneration(config)
    bundle = tmp_path / "paper_bundle"
    files = {str(path.relative_to(bundle)): path.read_text() for path in bundle.rglob("*")
             if path.is_file() and path.name != "manifest.json"}
    assert len(files) == 44
    assert files == expected
    assert all(path.read_bytes() == data for path, data in originals.items())
    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "complete"
    assert set(manifest["refreshed"]) == set(expected)
    assert manifest["retained"] == manifest["missing"] == []
    assert all(len(record["sha256"]) == 64 and record["last_refreshed_utc"] for record in manifest["files"])


def test_bundle_covers_figures_referenced_by_both_manuscripts(tmp_path: Path) -> None:
    _, expected = bundle_fixture(tmp_path)
    inventory = Path(__file__).resolve().parents[1] / "src/config/manuscript_inventory.json"
    figures = set(json.loads(inventory.read_text())["figures"])
    assert len(figures) == 18
    assert {f"figure/{name}.{extension}" for name in figures for extension in ("svg", "png")} == {
        name for name in expected if name.startswith("figure/")
    }


def test_auto_collection_copies_only_written_outputs_regardless_of_existing_mtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True, output_dir="collected")
    old = write_file(tmp_path, "global_importance/6m/global_importance_bone.svg", "old composite")
    os.utime(old, (4_000_000_000, 4_000_000_000))
    name = "figure/global_importance_bone.svg"
    produce(config, monkeypatch, {expected[name]: "new plot"})
    bundle = tmp_path / "collected"
    assert (bundle / name).read_text() == "new plot"
    assert list((bundle / "figure").iterdir()) == [bundle / name]
    manifest = json.loads((bundle / "manifest.json").read_text())
    assert manifest["refreshed"] == [name]
    assert len(manifest["missing"]) == 43
    summary = json.loads((tmp_path / "regeneration.json").read_text())
    assert summary["tasks"] == {"explanations": "complete", "paper_bundle": "complete"}
    assert summary["status"] == "complete"


def test_identical_rewrite_with_preserved_mtime_is_collected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    name = "figure/dataset_statistics.svg"
    source = tmp_path / expected[name]
    os.utime(source, ns=(1, 1))

    def redraw(_context: RegenerationContext) -> tuple[Path, ...]:
        source.write_bytes(source.read_bytes())
        os.utime(source, ns=(1, 1))
        return (source,)

    monkeypatch.setitem(runner.TASKS, "dataset_figure", redraw)
    set_tasks(config, ["dataset_figure"])
    runner.run_regeneration(config)
    assert (tmp_path / "paper_bundle" / name).read_bytes() == source.read_bytes()
    assert read_manifest(tmp_path)["refreshed"] == [name]


def test_partial_runs_retain_older_copies_and_refresh_times(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    produce(config, monkeypatch, {source: source for source in expected.values()})
    previous = {row["destination"]: row for row in read_manifest(tmp_path)["files"]}
    old_name = "figure/global_importance_bone.svg"
    (tmp_path / expected[old_name]).unlink()
    note = write_file(tmp_path, "paper_bundle/my_notes.txt", "keep")
    new_name = "figure/dataset_statistics.svg"
    produce(config, monkeypatch, {expected[new_name]: "updated dataset"}, "dataset_figure")
    manifest = read_manifest(tmp_path)
    assert manifest["status"] == "complete"
    assert manifest["refreshed"] == [new_name]
    assert set(manifest["retained"]) == set(expected) - {new_name}
    for row in manifest["files"]:
        if row["destination"] != new_name:
            assert row["last_refreshed_utc"] == previous[row["destination"]]["last_refreshed_utc"]
    assert (tmp_path / "paper_bundle" / old_name).read_text() == expected[old_name]
    assert note.read_text() == "keep"


@pytest.mark.parametrize("source_state", ["missing", "modified", "failed_cell"])
def test_replay_keeps_previous_copy_when_recorded_source_is_unusable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_state: str,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    name = "figure/global_importance_bone.svg"
    produce(config, monkeypatch, {expected[name]: "good plot"})
    before = read_manifest(tmp_path)["files"][0]
    source = tmp_path / expected[name]
    if source_state == "missing":
        source.unlink()
    elif source_state == "modified":
        source.write_text("unfinished replacement")
    else:
        path = tmp_path / "metadata.json"
        metadata = json.loads(path.read_text())
        metadata["results"][1]["status"] = "failed"
        path.write_text(json.dumps(metadata))
    set_tasks(config, ["paper_bundle"])
    runner.run_regeneration(config)
    manifest = read_manifest(tmp_path)
    assert (tmp_path / "paper_bundle" / name).read_text() == "good plot"
    assert manifest["status"] == "incomplete"
    assert name in manifest["uncollected"]
    assert manifest["files"][0]["last_refreshed_utc"] == before["last_refreshed_utc"]


def test_completed_task_is_collected_before_failure_and_replay_excludes_failed_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    overwritten = "figure/global_importance_bone.svg"
    intact = "figure/terms_hazard_ratio_grid_bone.svg"
    partial = "figure/ebm_top_features.svg"

    def complete(context: RegenerationContext) -> tuple[Path, ...]:
        for name in (overwritten, intact):
            write_file(context.output_dir, expected[name], "complete plot")
        return tuple(context.output_dir / expected[name] for name in (overwritten, intact))

    def fail(context: RegenerationContext) -> None:
        assert (tmp_path / "paper_bundle" / overwritten).read_text() == "complete plot"
        assert read_manifest(tmp_path)["regeneration_status"] == "running"
        for name in (overwritten, partial):
            write_file(context.output_dir, expected[name], "failed task's partial plot")
        raise ValueError("cannot finish")

    monkeypatch.setitem(runner.TASKS, "explanations", complete)
    monkeypatch.setitem(runner.TASKS, "importance", fail)
    set_tasks(config, ["explanations", "importance"])
    with pytest.raises(ValueError, match="cannot finish"):
        runner.run_regeneration(config)
    manifest = read_manifest(tmp_path)
    assert manifest["regeneration_status"] == "failed"
    assert manifest["status"] == "incomplete"
    assert manifest["failed_task"] == "importance"
    assert (tmp_path / "paper_bundle" / overwritten).read_text() == "complete plot"
    assert not (tmp_path / "paper_bundle" / partial).exists()
    # A new destination can recover only unchanged successful outputs from that run.
    recorded = (tmp_path / RECORD_FILENAME).read_bytes()
    configure_bundle(config, output_dir="replayed")
    set_tasks(config, ["paper_bundle"])
    runner.run_regeneration(config)
    assert (tmp_path / "replayed" / intact).read_text() == "complete plot"
    assert not (tmp_path / "replayed" / overwritten).exists()
    assert not (tmp_path / "replayed" / partial).exists()
    assert (tmp_path / RECORD_FILENAME).read_bytes() == recorded
    replayed = json.loads((tmp_path / "replayed/manifest.json").read_text())
    assert replayed["regeneration_status"] == "failed"
    assert overwritten in replayed["uncollected"]


def test_noop_run_replaces_replay_inventory_without_sweeping_older_sources(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    name = "figure/dataset_statistics.svg"
    produce(config, monkeypatch, {expected[name]: "previous run"})
    produce(config, monkeypatch, {})
    set_tasks(config, ["paper_bundle"])
    runner.run_regeneration(config)
    manifest = read_manifest(tmp_path)
    assert manifest["files"] == []
    assert len(manifest["missing"]) == 44


def test_first_task_failure_preserves_old_bundle_without_collecting_partial_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    name = "figure/dataset_statistics.svg"
    old = write_file(tmp_path, f"paper_bundle/{name}", "old copy")

    def fail(context: RegenerationContext) -> None:
        write_file(context.output_dir, expected[name], "partial plot")
        raise ValueError("cannot regenerate")

    monkeypatch.setitem(runner.TASKS, "explanations", fail)
    set_tasks(config, ["explanations"])
    with pytest.raises(ValueError, match="cannot regenerate"):
        runner.run_regeneration(config)
    assert old.read_text() == "old copy"
    manifest = read_manifest(tmp_path)
    assert manifest["retained"] == [name]
    assert manifest["refreshed"] == []
    assert manifest["regeneration_status"] == "failed"
    assert manifest["files"][0]["last_refreshed_utc"] is None


def test_collection_failure_records_generated_outputs_for_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected = bundle_fixture(tmp_path)
    configure_bundle(config, enabled=True)
    name = "figure/dataset_statistics.svg"
    with patch("src.regeneration.paper_bundle.shutil.copy2", side_effect=OSError("copy failed")):
        with pytest.raises(OSError, match="copy failed"):
            produce(config, monkeypatch, {expected[name]: "complete plot"})
    assert read_manifest(tmp_path)["regeneration_status"] == "failed"
    summary = json.loads((tmp_path / "regeneration.json").read_text())
    assert summary["tasks"] == {"explanations": "complete", "paper_bundle": "failed"}
    set_tasks(config, ["paper_bundle"])
    runner.run_regeneration(config)
    assert (tmp_path / "paper_bundle" / name).read_text() == "complete plot"


def test_replay_requires_a_recorded_run(tmp_path: Path) -> None:
    config, _ = bundle_fixture(tmp_path)
    with pytest.raises(FileNotFoundError, match="No recorded regeneration outputs"):
        runner.run_regeneration(config)
    assert not (tmp_path / "paper_bundle").exists()


def test_replay_rejects_a_different_horizon(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config, _ = bundle_fixture(tmp_path)
    produce(config, monkeypatch, {})
    configure_bundle(config, horizon="1m")
    set_tasks(config, ["paper_bundle"])
    with pytest.raises(ValueError, match="recorded regeneration horizon differs"):
        runner.run_regeneration(config)


@pytest.mark.parametrize("directory", [".", "All", "../"])
def test_bundle_cannot_replace_original_output_directories(tmp_path: Path, directory: str) -> None:
    config, _ = bundle_fixture(tmp_path)
    configure_bundle(config, output_dir=directory)
    original = (tmp_path / "metadata.json").read_bytes()
    with pytest.raises(ValueError, match="paper bundle"):
        runner.run_regeneration(config)
    assert (tmp_path / "metadata.json").read_bytes() == original


@pytest.mark.parametrize("settings", [{"enabled": "false"}, {"output_dir": None}, {"horizon": ""}, {"site": "Bone"}])
def test_invalid_bundle_settings_fail_early(tmp_path: Path, settings: dict[str, object]) -> None:
    config, _ = bundle_fixture(tmp_path)
    configure_bundle(config, **settings)
    with pytest.raises((TypeError, ValueError)):
        load_regeneration_config(config)
