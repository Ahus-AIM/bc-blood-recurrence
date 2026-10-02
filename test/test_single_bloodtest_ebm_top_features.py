from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from xml.etree import ElementTree

import matplotlib.image as mpimg
import numpy as np
import pytest
import yaml

from src.data.dataset_config import PatientDatasetConfig
from src.model.ebm_benchmark_top_features import EbmTopFeatureRow
from src.script import single_bloodtest_ebm_top_features as top_features
from src.script import train_single_bloodtest_benchmark as training
from src.script.single_bloodtest_benchmark_config import benchmark_model_output_dir
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY, BloodtestModelRow, bloodtest_rows


def benchmark_config() -> dict[str, object]:
    return {
        "sites": [
            {"label": label, "endpoint_ddfs_code_filter_regex": ".*"}
            for label in ("Lung", "All", "Brain", "Bone")
        ],
        "horizons": [{"label": "1m", "days": 30}, {"label": "6m", "days": 180}],
    }


def save_model_cell(
    output_dir: Path,
    site: str,
    horizon: str,
    *,
    model_key: str = ALL_MODEL_KEY,
    status: str = "complete",
) -> dict[str, object]:
    model_dir = benchmark_model_output_dir(output_dir, model_key, {"label": site}, {"label": horizon})
    model_dir.mkdir(parents=True)
    (model_dir / "model.pkl").write_bytes(b"saved model")
    (model_dir / "metadata.json").write_text(json.dumps({"status": status}), encoding="utf-8")
    (model_dir / "training_config.yml").write_text(
        """\
output:
  normal_ranges_csv_path: normal_ranges.csv
  bloodtest_translation_csv_path: translations.csv
  explanation_term_x_axis_scale: log
  combined_term_hazard_ratio_y_min: 0.2
  combined_term_hazard_ratio_y_max: 5.0
  combined_term_hazard_ratio_show_x_markers: false
  combined_term_hazard_ratio_show_uncertainty_area: true
""",
        encoding="utf-8",
    )
    (model_dir / "normal_ranges.csv").write_text(
        "test_name,unit,sex,age_group,lower,upper,lower_operator,upper_operator,source_url,notes\n"
        "A,u,female,18-39 years,0.7,0.9,>=,<=,,\n"
        "A,u,female,40-69 years,1.0,1.3,>=,<=,,\n",
        encoding="utf-8",
    )
    (model_dir / "translations.csv").write_text(
        "norwegian_test_name,english_test_name\nA,Example test\n", encoding="utf-8"
    )
    return {
        "model_key": model_key,
        "site_label": site,
        "horizon_label": horizon,
        "status": status,
        "split": "train",
    }


def test_figures_use_all_tests_for_every_horizon_and_rank_values_changes_and_clinical_terms(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = benchmark_config()
    results = [
        save_model_cell(tmp_path, site, horizon)
        for site in ("All", "Bone")
        for horizon in ("1m", "6m")
    ]
    results.append(save_model_cell(tmp_path, "Brain", "6m", model_key="A|||u"))
    loader = Mock(return_value=FakeModel())
    monkeypatch.setattr(top_features, "load_model", loader)

    artifacts = top_features.write_benchmark_ebm_top_features(benchmark, tmp_path, results)

    assert [artifact["horizon_label"] for artifact in artifacts] == ["1m", "6m"]
    assert [artifact["horizon_days"] for artifact in artifacts] == [30, 180]
    assert {call.args[0].parent for call in loader.call_args_list} == {
        tmp_path / site / horizon / ALL_MODEL_KEY
        for site in ("All", "Bone")
        for horizon in ("1m", "6m")
    }
    assert loader.call_count == 4
    for artifact, months in zip(artifacts, (1, 6), strict=True):
        assert artifact["png"] == f"top_features/{months}m/ebm_top_features.png"
        assert artifact["svg"] == f"top_features/{months}m/ebm_top_features.svg"
        png_path = tmp_path / artifact["png"]
        svg = (tmp_path / artifact["svg"]).read_text(encoding="utf-8")
        document = ElementTree.fromstring(svg)
        text_tag = "{http://www.w3.org/2000/svg}text"
        labels = ["".join(node.itertext()) for node in document.iter(text_tag)]
        assert "Top EBM feature effects" in labels
        assert "Hazard ratio" in labels
        titles = [" ".join("".join(child.itertext()) for child in group.iter(text_tag))
                  for group in document.iter() if group.get("id", "").startswith("ebm-panel-title-")]
        assert titles[:4] == ["Example test", "Example test change", "Age at diagnosis", "Global Importance"]
        assert "interaction" not in labels
        assert "low_feature" not in labels
        assert titles.count("Global Importance") == 2
        assert labels.count("Mean Absolute Score") == 2
        assert "Female 18-39 years reference" not in labels
        assert "Female 40-69 years reference" not in labels
        interval_labels = [
            " ".join("".join(text.itertext()) for text in child.iter(text_tag))
            for group in document.iter() if group.get("id") == "numeric-reference-intervals"
            for child in group if child.get("id", "").startswith("text_")
        ]
        assert interval_labels.count("0.7–0.9 (18–39 years)") == 2
        assert interval_labels.count("1–1.3 (40–69 years)") == 2
        row_positions = [labels.index(label) for label in ("Any", "Bone", "Brain", "Lung")]
        assert row_positions == sorted(row_positions)
        # Less than 2 mm at 160 dpi, including the larger shared font's glyph padding.
        assert maximum_outer_whitespace(png_path) <= 12


def test_regeneration_ignores_stale_failed_models_and_preserves_saved_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = benchmark_config()
    benchmark["output_dir"] = str(tmp_path)
    benchmark["ebm_top_features"] = {"output_dir": "custom_figures"}
    failed = save_model_cell(tmp_path, "Lung", "1m")
    failed["status"] = "failed"
    metric_not_estimable = save_model_cell(tmp_path, "All", "1m")
    metric_not_estimable["status"] = "not_estimable"
    non_estimable = save_model_cell(tmp_path, "Bone", "1m", status="not_estimable")
    missing = save_model_cell(tmp_path, "Brain", "1m")
    (tmp_path / "Brain" / "1m" / ALL_MODEL_KEY / "model.pkl").unlink()
    save_model_cell(tmp_path, "All", "6m")  # Unvisited stale model, absent from current results.
    metadata_path = tmp_path / "metadata.json"
    metadata_path.write_text(json.dumps({
        "selected_blood_tests": ["A|||u"],
        "results": [failed, metric_not_estimable, non_estimable, missing],
    }), encoding="utf-8")
    config_path = tmp_path / "benchmark.yml"
    config_path.write_text(yaml.safe_dump({"benchmark": benchmark}), encoding="utf-8")
    original_files = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    written_rows: list[tuple[Path, dict[str, EbmTopFeatureRow]]] = []

    def render(rows: Sequence[EbmTopFeatureRow], png_path: Path, _svg_path: Path, _title: str) -> None:
        written_rows.append((png_path, {row.label: row for row in rows}))

    def load_model(path: Path) -> FakeModel:
        if not path.is_file():
            raise FileNotFoundError(path)
        return FakeModel()

    monkeypatch.setattr(top_features, "load_model", load_model)
    monkeypatch.setattr(top_features, "write_ebm_top_feature_grid", render)

    top_features.write_benchmark_ebm_top_features(
        benchmark, tmp_path, [failed, metric_not_estimable, non_estimable, missing],
    )

    assert [path for path, _rows in written_rows] == [
        tmp_path / "custom_figures" / horizon / "ebm_top_features.png" for horizon in ("1m", "6m")
    ]
    first_rows = written_rows[0][1]
    assert first_rows["Any"].explanation is not None
    assert first_rows["Any"].style.y_limits == (0.2, 5.0)
    assert first_rows["Any"].style.x_axis_scale == "log"
    assert first_rows["Any"].style.show_x_markers is False
    assert first_rows["Any"].style.show_uncertainty_area is True
    assert all(first_rows[label].explanation is None for label in ("Bone", "Brain", "Lung"))
    assert all(row.explanation is None for row in written_rows[1][1].values())
    assert all(path.read_bytes() == content for path, content in original_files.items())


def test_disabling_all_tests_skips_top_feature_reports(tmp_path: Path) -> None:
    assert top_features.write_benchmark_ebm_top_features({"include_all_blood_tests": False}, tmp_path, []) == []
    assert list(tmp_path.iterdir()) == []


def test_benchmark_generates_top_features_after_all_cells(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = benchmark_config()
    config = PatientDatasetConfig.from_mapping({
        "bloodtest_csv_path": "blood.csv",
        "endpoint_csv_path": "endpoint.csv",
        "split_csv_path": "split.csv",
    })
    canonical = SimpleNamespace(config=config)
    monkeypatch.setattr(training, "canonical_benchmark_dataset", lambda *_args: canonical)
    monkeypatch.setattr(training, "benchmark_target_dataset", lambda *_args: canonical)
    monkeypatch.setattr(training, "prepare_washout_state", lambda *_args: None)
    monkeypatch.setattr(training, "prepare_task", lambda *_args: object())
    completed: list[dict[str, object]] = []

    def train_cell(
        _config: Mapping[str, Mapping[str, object]],
        _task: training.SingleBloodtestTask,
        row: BloodtestModelRow,
        _output_dir: Path,
        _config_dir: Path,
    ) -> list[dict[str, object]]:
        result = {"model_key": row["key"], "status": "complete"}
        completed.append(result)
        return [result]

    def record_results(
        _benchmark: Mapping[str, object],
        _tests: tuple[str, ...],
        results: list[dict[str, object]],
        cell_results: list[dict[str, object]],
        *_args: object,
    ) -> None:
        results.extend(cell_results)

    def write_figures(
        _benchmark: Mapping[str, object], _output_dir: Path, results: list[dict[str, object]],
    ) -> None:
        assert len(completed) == 12  # Four site/horizon pairs per model kind.
        assert results == completed

    monkeypatch.setattr(training, "train_cell", train_cell)
    monkeypatch.setattr(training, "record_results", record_results)
    writer = Mock(side_effect=write_figures)
    monkeypatch.setattr(training, "write_benchmark_ebm_top_features", writer)
    benchmark["sites"] = benchmark["sites"][:2]

    training.run_benchmark(
        {"benchmark": benchmark}, config, ("A|||u",),
        bloodtest_rows(("A|||u",), benchmark, tmp_path), tmp_path, tmp_path, object(),
    )

    writer.assert_called_once()


def maximum_outer_whitespace(png_path: Path) -> int:
    image = mpimg.imread(png_path)
    visible = np.any(image[:, :, :3] < 0.99, axis=2)
    rows, columns = np.where(visible)
    height, width = visible.shape
    return max(int(rows.min()), int(columns.min()), int(height - rows.max() - 1), int(width - columns.max() - 1))


class FakeModel:
    def explain_global(self) -> FakeExplanation:
        return FakeExplanation()


class FakeExplanation:
    """The interaction must be excluded even though it has the highest importance."""

    def data(self, term_index: int | None = None) -> dict[str, object]:
        if term_index is None:
            return {
                "names": ["low_feature", "interaction", "A|||u", "age_at_diagnosis", "A|||u change"],
                "scores": [1.0, 100.0, 4.0, 2.0, 3.0],
            }
        if term_index == 1:
            return {"type": "interaction"}
        return {
            "names": [-1.0, 0.0, 1.0] if term_index == 4 else [0.1, 1.0, 10.0],
            "scores": [-0.2, 0.2],
            "lower_bounds": [-0.3, 0.1],
            "upper_bounds": [-0.1, 0.3],
        }
