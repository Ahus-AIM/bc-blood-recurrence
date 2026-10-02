from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.landmark_event_counts import LandmarkEventCountsConfig
from src.script import survival_dataset_artifacts
from src.script import train_single_bloodtest_benchmark as single_benchmark
from src.script.ebm_survival_config import dataset_config_mapping
from src.script.single_bloodtest_benchmark_reporting import BloodtestModelRow
from src.script.survival_benchmark_artifacts import write_completion_marker
from src.script.survival_benchmark_cache import TaskDatasetArtifactCache
from test.test_landmark_event_counts import count_row, event_count_dataset_config

RECURRENCE_FIGURES = tuple(
    f"recurrence_blood_test_trajectories_{site}.{suffix}"
    for site in ("any", "bone", "brain", "liver", "lung")
    for suffix in ("png", "svg")
)


@pytest.fixture
def dataset(tmp_path: Path) -> PatientBloodTestDataset:
    config = replace(
        event_count_dataset_config(tmp_path),
        endpoint_ddfs_code_filter_regex=r"C787\d*",
        recurrence_endpoint_name="liver_recurrence",
        censoring_horizon_days=30,
    )
    return PatientBloodTestDataset(config)


@pytest.fixture
def raw_config(dataset: PatientBloodTestDataset) -> dict[str, dict[str, object]]:
    config, _config_dir = single_benchmark.load_benchmark_config()
    config["dataset"] = dataset_config_mapping(dataset.config)
    config["ebm"] = {}
    config["imputation"] = {"enabled": False}
    config["output"] = {"write_model_input_csv": False, "evaluate_test": False}
    config["benchmark"] = {"resume": True, "write_risk_scores": False}
    return config


@pytest.fixture
def report_builder(monkeypatch: pytest.MonkeyPatch) -> Mock:
    def write_png(_dataset: PatientBloodTestDataset, output_dir: Path, **_kwargs: object) -> dict[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "dataset_statistics.png").write_bytes(b"figure")
        return {"path": "dataset_statistics.png"}

    monkeypatch.setattr(survival_dataset_artifacts, "write_training_dataset_statistics_png", write_png)
    monkeypatch.setattr(survival_dataset_artifacts, "write_population_statistics_latex", lambda *_args, **_kwargs: {})
    builder = Mock(wraps=survival_dataset_artifacts.build_landmark_event_count_frame)
    monkeypatch.setattr(survival_dataset_artifacts, "build_landmark_event_count_frame", builder)
    return builder


def assert_full_table(output_dir: Path) -> None:
    counts = pd.read_csv(output_dir / "landmark_event_counts.csv")
    assert set(counts["site_label"]) == {"All", "Bone", "Brain", "Liver", "Lung", "Other"}
    assert set(counts["horizon_days"]) == {30, 90, 180, 365, 730}
    assert count_row(counts, "Lung", "3m")["val_event_patient_count"] == 1
    assert count_row(counts, "All", "1m")["overall_event_patient_count"] == 2
    latex = (output_dir / "landmark_event_counts.tex").read_text(encoding="utf-8")
    assert "Site & Dataset fold & 1m & 3m & 6m & 1y & 2y" in latex
    assert "multiple synchronous distant recurrence sites" in latex
    assert (output_dir / "dataset_statistics.png").is_file()
    assert all((output_dir / name).is_file() for name in RECURRENCE_FIGURES)
    assert not (output_dir / "recurrence_blood_test_insights.tex").exists()
    assert (output_dir / "remote_reference" / "measurements.csv").is_file()
    comparison = pd.read_csv(output_dir / "cohort_comparison.csv")
    assert len(comparison.columns) == 3
    assert "BC diagnosis year" in set(comparison["Characteristic"])
    assert (output_dir / "cohort_comparison.tex").is_file()
    assert (output_dir / "cohort_age_export" / "cohort_membership.csv").is_file()
    membership = pd.read_csv(output_dir / "cohort_age_export" / "cohort_membership.csv", dtype=str)
    assert list(membership.columns) == ["id", "fold", "hash"]
    assert membership["hash"].nunique() == 1


def test_single_benchmark_writes_full_table_before_model_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: PatientBloodTestDataset,
    raw_config: dict[str, dict[str, object]],
    report_builder: Mock,
) -> None:
    cache = TaskDatasetArtifactCache(raw_config, dataset, tmp_path, tmp_path / "shared", None)
    task = single_benchmark.SingleBloodtestTask(
        dataset, dataset.to_landmark_table(bootstrap=False), SimpleNamespace(),
        {"label": "Liver", "endpoint_ddfs_code_filter_regex": r"C787\d*"},
        {"label": "1m", "days": 30}, {}, "model-input-fingerprint", None,
        artifact_cache=cache,
    )
    row: BloodtestModelRow = {
        "key": "A", "label": "A", "features": ("A|||u",),
        "kind": "single_blood_test", "blood_test_key": "A|||u",
    }
    output_dir = single_benchmark.model_output_dir(tmp_path, row, task.site, task.horizon)

    def stop_at_fitting(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        assert_full_table(output_dir)
        return []

    fit = Mock(side_effect=stop_at_fitting)
    monkeypatch.setattr(single_benchmark, "fit_cell", fit)
    assert single_benchmark.train_cell(raw_config, task, row, tmp_path, tmp_path) == []

    fit.assert_called_once()
    assert report_builder.call_count == 1
    assert report_builder.call_args.kwargs["prepared_dataset"] is dataset


def test_benchmark_copies_configured_report_paths_once_for_all_models(
    tmp_path: Path,
    dataset: PatientBloodTestDataset,
    raw_config: dict[str, dict[str, object]],
    report_builder: Mock,
) -> None:
    raw_config["landmark_event_counts"].update(
        output_csv_path="tables/counts.csv",
        output_latex_path="tables/counts.tex",
    )
    cache = TaskDatasetArtifactCache(raw_config, dataset, tmp_path, tmp_path / "shared", None)

    for model_name in ("clinical_baseline", "single_test", "all_tests"):
        output_dir = tmp_path / model_name
        metadata = cache.materialize(output_dir)
        assert metadata["landmark_event_counts"] == {
            "csv_path": "tables/counts.csv",
            "latex_path": "tables/counts.tex",
        }
        for filename in ("counts.csv", "counts.tex"):
            assert (output_dir / "tables" / filename).read_bytes() == (
                cache.shared_dir / "tables" / filename
            ).read_bytes()
        for filename in (*RECURRENCE_FIGURES, "remote_reference/measurements.csv"):
            assert (output_dir / filename).read_bytes() == (cache.shared_dir / filename).read_bytes()
        assert (
            load_yml(output_dir / "training_config.yml")["landmark_event_counts"] == raw_config["landmark_event_counts"]
        )

    assert report_builder.call_count == 1
    assert report_builder.call_args.kwargs["prepared_dataset"] is dataset


@pytest.mark.parametrize("existing_report", (None, "landmark_event_counts.csv", "landmark_event_counts.tex"))
def test_single_benchmark_backfills_completed_cells_without_fitting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    dataset: PatientBloodTestDataset,
    raw_config: dict[str, dict[str, object]],
    report_builder: Mock,
    existing_report: str | None,
) -> None:
    cache = TaskDatasetArtifactCache(raw_config, dataset, tmp_path, tmp_path / "shared", None)
    task = single_benchmark.SingleBloodtestTask(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        None,
        {"label": "Liver", "endpoint_ddfs_code_filter_regex": r"C787\d*"},
        {"label": "1m", "days": 30},
        {},
        "model-input-fingerprint",
        None,
        artifact_cache=cache,
    )
    fit = Mock(side_effect=AssertionError("Completed models must not be refitted."))
    monkeypatch.setattr(single_benchmark, "fit_cell", fit)
    old_config = {key: value for key, value in raw_config.items() if key != "landmark_event_counts"}

    for key in ("A", "all_tests"):
        row: BloodtestModelRow = {
            "key": key,
            "label": key,
            "features": ("A|||u",),
            "kind": "single_blood_test",
            "blood_test_key": "A|||u",
        }
        output_dir = single_benchmark.model_output_dir(tmp_path, row, task.site, task.horizon)
        results = [{"status": "complete", "split": "test", "harrell_c_index": 0.7}]
        write_completion_marker(
            output_dir,
            single_benchmark.cell_run_fingerprint(old_config, task, row),
            task.model_input_fingerprint,
            results,
        )
        marker = output_dir / "benchmark_completion.json"
        marker_bytes = marker.read_bytes()
        (output_dir / "model.pkl").write_bytes(b"existing model")
        if existing_report is not None:
            (output_dir / existing_report).write_text("old report", encoding="utf-8")

        assert single_benchmark.train_cell(raw_config, task, row, tmp_path, tmp_path) == results
        assert_full_table(output_dir)
        assert marker.read_bytes() == marker_bytes
        assert (output_dir / "model.pkl").read_bytes() == b"existing model"

    fit.assert_not_called()
    assert report_builder.call_count == 1


def test_empty_prepared_cohort_still_writes_full_table(
    tmp_path: Path,
    dataset: PatientBloodTestDataset,
    raw_config: dict[str, dict[str, object]],
) -> None:
    empty = PatientBloodTestDataset(
        dataset.config,
        prepared_data=replace(dataset.prepared_data, items_by_split={"train": (), "val": (), "test": ()}),
    )
    survival_dataset_artifacts.write_training_landmark_event_counts(
        empty, tmp_path, raw_config["landmark_event_counts"]
    )
    counts = pd.read_csv(tmp_path / "landmark_event_counts.csv")
    config = LandmarkEventCountsConfig.from_mapping(raw_config["landmark_event_counts"], output_dir=tmp_path)
    assert len(counts) == (len(config.site_rules) + 2) * len(config.horizons)
    assert (counts["overall_event_patient_count"] == 0).all()
    assert counts["overall_event_patient_percent"].isna().all()
    assert (tmp_path / "landmark_event_counts.tex").is_file()
