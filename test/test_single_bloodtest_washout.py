from __future__ import annotations

import json
import inspect
from pathlib import Path
from types import SimpleNamespace

import yaml

from src.script import single_bloodtest_washout
from src.script.regenerate import main
from src.script.single_bloodtest_washout import (
    configured_washout_days,
    evaluate_washout_cell,
    evaluate_saved_washout,
    record_washout,
    washout_table,
    write_washout_outputs,
)


def test_washout_days_default_to_four_agreed_periods() -> None:
    assert configured_washout_days({}) == (0, 30, 60, 90)


def test_retrospective_script_has_no_arguments() -> None:
    assert not inspect.signature(main).parameters


def test_retrospective_evaluator_uses_saved_aggregate_cells(tmp_path: Path, monkeypatch) -> None:
    source = {
        "model_key": "A|||u",
        "site_label": "All",
        "horizon_label": "1m",
        "split": "test",
        "status": "complete",
    }
    (tmp_path / "metadata.json").write_text(
        json.dumps({"selected_blood_tests": ["A|||u"], "results": [source]}),
        encoding="utf-8",
    )
    captured: dict[str, object] = {}
    monkeypatch.setattr(single_bloodtest_washout, "saved_washout_landmarks", lambda *_args: {0: object()})
    monkeypatch.setattr(
        single_bloodtest_washout,
        "evaluate_washout_cell",
        lambda result, _landmarks: [{**result, "washout_days": 0}],
    )
    monkeypatch.setattr(
        single_bloodtest_washout,
        "write_washout_outputs",
        lambda benchmark, selected, results, output_dir, config_dir: captured.update(
            benchmark=benchmark,
            selected=selected,
            results=results,
            output_dir=output_dir,
            config_dir=config_dir,
        ),
    )
    benchmark = {"output_metadata_path": "metadata.json"}

    results = evaluate_saved_washout(benchmark, tmp_path, tmp_path / "config")

    assert results == [
        {
            **source,
            "output_dir": str(tmp_path / "All" / "1m" / "A___u"),
            "washout_days": 0,
        }
    ]
    assert captured["selected"] == ("A|||u",)
    assert captured["output_dir"] == tmp_path


def test_saved_cell_uses_run_config_and_skips_washout_at_or_above_horizon(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "training_config.yml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "bloodtest_csv_path": "blood.csv",
                    "endpoint_csv_path": "endpoint.csv",
                    "split_csv_path": "split.csv",
                },
                "imputation": {"enabled": False},
            }
        ),
        encoding="utf-8",
    )
    source = {
        "model_key": "A|||u",
        "site_label": "All",
        "horizon_label": "1m",
        "horizon_days": 30,
        "endpoint_ddfs_code_filter_regex": ".*",
        "split": "test",
        "status": "complete",
        "output_dir": str(tmp_path),
        "bootstrap_runs_attempted": 1000,
        "zero_event_bootstrap_runs_discarded": 2,
    }
    table = SimpleNamespace(feature_names=("A|||u",))
    zero_pool = SimpleNamespace(
        landmarks=SimpleNamespace(table=lambda _config: table),
        evaluation_features=object(),
    )
    pools = {0: zero_pool, 30: object(), 60: object()}
    captured: dict[str, object] = {}
    model = SimpleNamespace(feature_names_in_=["A|||u"])
    monkeypatch.setattr(single_bloodtest_washout, "load_model", lambda _path: model)
    monkeypatch.setattr(
        single_bloodtest_washout,
        "load_feature_imputer",
        lambda raw_config, model_dir, landmark_table: captured.update(
            raw_config=raw_config,
            model_dir=model_dir,
            landmark_table=landmark_table,
        )
        or object(),
    )
    zero_pool.evaluation_cache = lambda config, _imputer, **_kwargs: SimpleNamespace(
        get=lambda split: captured.update(
            config=config,
            pool=zero_pool.landmarks,
            shared=zero_pool.evaluation_features,
            split=split,
        )
        or object()
    )
    monkeypatch.setattr(
        single_bloodtest_washout,
        "evaluate_prepared_split",
        lambda *_args, **_kwargs: {
            "harrell_c_index": 0.6,
            "harrell_c_index_ci95_lower": 0.5,
            "harrell_c_index_ci95_upper": 0.7,
            "runs": [{"bootstrap_run": 0, "harrell_c_index": 0.6}],
        },
    )

    results = evaluate_washout_cell(source, pools)

    assert [result["status"] for result in results] == ["complete", "skipped", "skipped"]
    assert [result["washout_days"] for result in results] == [0, 30, 60]
    assert captured["raw_config"]["imputation"] == {"enabled": False}
    assert captured["model_dir"] == tmp_path
    assert captured["landmark_table"] is table
    assert captured["config"].split == "test"
    assert captured["config"].censoring_horizon_days == 30
    assert captured["pool"] is zero_pool.landmarks
    assert captured["shared"] is zero_pool.evaluation_features
    assert captured["split"] == "test"


def test_saved_model_failure_continues_with_failed_and_skipped_cells(tmp_path: Path, monkeypatch) -> None:
    (tmp_path / "training_config.yml").write_text(
        yaml.safe_dump(
            {
                "dataset": {
                    "bloodtest_csv_path": "blood.csv",
                    "endpoint_csv_path": "endpoint.csv",
                    "split_csv_path": "split.csv",
                }
            }
        ),
        encoding="utf-8",
    )
    source = {
        "horizon_label": "1m",
        "horizon_days": 30,
        "endpoint_ddfs_code_filter_regex": ".*",
        "status": "complete",
        "output_dir": str(tmp_path),
        "bootstrap_runs_attempted": 1000,
        "zero_event_bootstrap_runs_discarded": 2,
    }
    monkeypatch.setattr(
        single_bloodtest_washout,
        "load_model",
        lambda _path: (_ for _ in ()).throw(OSError("unreadable model")),
    )

    results = evaluate_washout_cell(source, {0: object(), 30: object()})

    assert [result["status"] for result in results] == ["failed", "skipped"]
    assert results[0]["error_type"] == "OSError"
    assert results[0]["bootstrap_runs_attempted"] is None
    assert results[1]["zero_event_bootstrap_runs_discarded"] is None


def test_incremental_write_failure_is_retried_after_next_cell(tmp_path: Path, monkeypatch) -> None:
    benchmark = {"sites": [{"label": "All"}], "horizons": [{"label": "1m"}]}
    results: list[dict[str, object]] = []
    write_sizes: list[int] = []
    monkeypatch.setattr(
        single_bloodtest_washout,
        "evaluate_washout_cell",
        lambda source, _landmarks: [{**source, "washout_days": 0}],
    )

    def write_washout(_benchmark, _tests, current, _output_dir, _config_dir):
        write_sizes.append(len(current))
        if len(write_sizes) == 1:
            raise OSError("temporary write failure")

    monkeypatch.setattr(single_bloodtest_washout, "write_washout_outputs", write_washout)
    state = ({0: object()}, results)
    baseline = {"model_key": "__clinical_baseline__", "split": "test"}
    blood_test = {"model_key": "A|||u", "split": "test"}

    record_washout(benchmark, ("A|||u",), [baseline], tmp_path, tmp_path, state)
    record_washout(benchmark, ("A|||u",), [blood_test], tmp_path, tmp_path, state)

    assert write_sizes == [1, 2]
    assert [result["model_key"] for result in results] == ["__clinical_baseline__", "A|||u"]


def test_washout_table_is_rotated_with_three_ordered_headers(tmp_path: Path) -> None:
    benchmark = {
        "include_all_blood_tests": False,
        "sites": [{"label": "All"}],
        "horizons": [{"label": "1m", "days": 30}, {"label": "6m", "days": 180}],
        "washout": {"days": [0, 30]},
        "table_caption": "Caption",
        "table_label": "tab:test",
    }
    results = [
        {
            "model_key": "__clinical_baseline__",
            "site_label": "All",
            "horizon_label": "1m 0d",
            "split": "test",
            "status": "complete",
            "harrell_c_index": 0.6,
            "harrell_c_index_ci95_lower": 0.55,
            "harrell_c_index_ci95_upper": 0.65,
        }
    ]

    latex = washout_table(benchmark, (), results, tmp_path)

    assert r"\begin{sidewaystable*}[p]" in latex
    assert r"\resizebox{\textheight}{!}{%" in latex
    assert r"\multicolumn{2}{c}{1m}" in latex
    assert "& 0d & 30d & 0d & 30d" in latex
    assert "Clinical baseline & 60.0 [55.0, 65.0] & -- & -- & --" in latex


def test_default_table_has_one_hundred_site_horizon_washout_columns(tmp_path: Path) -> None:
    benchmark = {
        "include_all_blood_tests": False,
        "sites": [{"label": f"Site {index}"} for index in range(5)],
        "horizons": [{"label": f"Horizon {index}"} for index in range(5)],
        "washout": {"days": [0, 30, 60, 90]},
        "table_caption": "Caption",
        "table_label": "tab:test",
    }

    latex = washout_table(benchmark, (), [], tmp_path)

    assert r"\cline{2-101}" in latex
    assert latex.count(r"\multicolumn{20}{c|}{Site") == 5
    assert latex.count(r"\multicolumn{4}{c}{Horizon") == 25
    assert latex.count("0d & 30d & 60d & 90d") == 25


def test_outputs_keep_runs_only_in_json_metadata(tmp_path: Path) -> None:
    benchmark = {
        "include_all_blood_tests": False,
        "sites": [{"label": "All"}],
        "horizons": [{"label": "1m"}],
        "washout": {
            "days": [0],
            "output_csv_path": "washout/cindex_benchmark.csv",
            "output_latex_path": "washout/cindex_benchmark.tex",
            "output_metadata_path": "washout/metadata.json",
            "table_caption": "Caption",
            "table_label": "tab:test",
        },
    }
    result = {
        "model_key": "__clinical_baseline__",
        "site_label": "All",
        "horizon_label": "1m 0d",
        "split": "test",
        "status": "complete",
        "harrell_c_index": 0.6,
        "harrell_c_index_ci95_lower": 0.5,
        "harrell_c_index_ci95_upper": 0.7,
        "bootstrap_runs_attempted": 2,
        "zero_event_bootstrap_runs_discarded": 1,
        "runs": [{"bootstrap_run": 1, "harrell_c_index": 0.6}],
    }

    write_washout_outputs(benchmark, (), [result], tmp_path, tmp_path)

    header = (tmp_path / "washout" / "cindex_benchmark.csv").read_text(encoding="utf-8").splitlines()[0]
    assert "runs" not in header.split(",")
    metadata = json.loads((tmp_path / "washout" / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["results"][0]["runs"] == result["runs"]
    assert metadata["results"][0]["bootstrap_runs_attempted"] == 2
