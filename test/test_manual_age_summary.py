from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

import src.script.plot.dataset_statistics as dataset_statistics_plot
from src.config import load_yml
from src.data import PatientBloodTestDataset
from src.data.cohort_age_exchange import AGE_FOLDS, cohort_membership, membership_hash
from src.data.report_return_lookup import return_directory
from src.data.dataset_statistics_export import (
    load_dataset_statistics_payload,
    recurrence_incidence_from_payload,
    recurrence_survival_from_payload,
)
from src.manual_age_summary import ManualAgeSummary, load_age_cells
from src.script.ebm_survival_config import serializable_value
from src.script.plot.dataset_statistics import (
    CONFIG_PATH,
    DatasetStatisticsPlotConfig,
    build_dataset_statistics_figure,
    plot_config_from_payload,
)
from src.script.training_dataset_statistics import write_training_dataset_statistics_png
from test.test_dataset_statistics_config import source_mapping, write_minimal_dataset_files


def external_ages() -> dict[str, Any]:
    return {
        "cohort_reference": "Test cohort",
        "source_note": "Exact ages at BC diagnosis, one age per eligible patient; pooled overall.",
        "age_years": {
            "train": {"median": 61.25, "q1": 50.5, "q3": 72.75},
            "validation": None,
            "test": None,
            "overall": {"median": 61.25, "q1": 50.5, "q3": 72.75},
        },
    }


@pytest.mark.parametrize(
    "values",
    [
        {"median": True, "q1": 50, "q3": 70},
        {"median": float("nan"), "q1": 50, "q3": 70},
        {"median": 60, "q1": 50, "q3": float("inf")},
        {"median": 60, "q1": 61, "q3": 70},
        {"median": 60, "q1": -1, "q3": 70},
        {"median": None, "q1": 50, "q3": 70},
    ],
)
def test_external_ages_reject_invalid_statistics(values: dict[str, object]) -> None:
    raw = external_ages()
    raw["age_years"]["train"] = values
    with pytest.raises(ValueError):
        ManualAgeSummary.from_mapping(raw)


def test_production_ages_round_trip_and_refresh_without_source_data(tmp_path: Path) -> None:
    write_minimal_dataset_files(tmp_path)
    dataset = PatientBloodTestDataset({
        "bloodtest_csv_path": str(tmp_path / "blood.csv"),
        "endpoint_csv_path": str(tmp_path / "endpoint.csv"),
        "split_csv_path": str(tmp_path / "split.csv"),
        "selected_blood_tests": ["A|||u"],
    })
    age_path = tmp_path / "ages.yml"
    age_path.write_text(yaml.safe_dump(external_ages()), encoding="utf-8")
    raw = load_yml(CONFIG_PATH)
    raw["dataset"] = source_mapping()
    raw["figure"].update({"manual_age_config_path": "ages.yml", "dpi": 50})
    cohort_hash = membership_hash(cohort_membership(dataset))
    returned = return_directory("ages") / "any_name.csv"
    pd.DataFrame([
        (fold, {"validation": 62, "test": 63, "excluded": 70}.get(fold, 61.25), 50.5, 72.75, cohort_hash)
        for fold in AGE_FOLDS
    ], columns=["fold", "median", "lower_q", "upper_q", "hash"]).to_csv(returned, index=False)
    original_return = returned.read_bytes()
    config_path = tmp_path / "plot.yml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    output_dir = tmp_path / "run"
    write_training_dataset_statistics_png(dataset, output_dir, plot_config_path=config_path)
    payload = load_dataset_statistics_payload(output_dir / "dataset_statistics_values.yml")
    assert payload["metadata"]["cohort_hash"] == cohort_hash
    assert payload["metadata"]["manual_age_summary"]["age_years"]["train"]["median"] == 61.2
    cells, provenance = load_age_cells(age_path)
    assert cells == ["61.25 (50.5–72.75)", "—", "—", "61.25 (50.5–72.75)"]
    assert provenance["source_note"] == external_ages()["source_note"]

    # Legacy manual YAML and saved ages never override the shared folder.
    age_path.unlink()
    saved = plot_config_from_payload(raw, payload, config_dir=tmp_path)
    assert saved.manual_age.cells[0] == "61.2 (50.5–72.8)"
    check_age_table_layout(saved, payload)
    returned.unlink()
    refreshed = plot_config_from_payload(raw, payload, config_dir=tmp_path)
    assert refreshed.manual_age.cells == ["—"] * 4
    assert payload["metadata"]["manual_age_summary"]["age_years"]["train"]["median"] == 61.2
    returned.write_bytes(original_return)

    # Exercise the shared saved-value renderer with source CSVs removed.
    import_config = output_dir / "training_config.yml"
    import_config.write_text(yaml.safe_dump({"dataset": serializable_value(asdict(dataset.config))}), encoding="utf-8")
    raw["dataset"] = {
        "dataset_config_path": str(import_config),
        "statistics_output_path": str(output_dir / "dataset_statistics_values.yml"),
    }
    raw["figure"]["output_path"] = str(tmp_path / "refreshed.png")
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    for path in tmp_path.glob("*.csv"):
        path.unlink()
    dataset_statistics_plot.render_saved_dataset_figure(output_dir, plot_config_path=config_path)
    assert (output_dir / "dataset_statistics.png").stat().st_size > 0
    assert "Age in years" in (output_dir / "dataset_statistics.svg").read_text(encoding="utf-8")


def check_age_table_layout(config: DatasetStatisticsPlotConfig, payload: dict[str, Any]) -> None:
    frames = {
        "recurrence_incidence_df": recurrence_incidence_from_payload(payload),
        "recurrence_survival_df": recurrence_survival_from_payload(payload),
    }
    original = build_dataset_statistics_figure(replace(config, manual_age=None), **frames)
    figure = build_dataset_statistics_figure(config, **frames)
    try:
        original.canvas.draw()
        figure.canvas.draw()
        assert figure.get_figheight() == pytest.approx(original.get_figheight())
        for before, after in zip(original.axes, figure.axes, strict=True):
            np.testing.assert_allclose(before.bbox.size, after.bbox.size, rtol=1e-12)
        renderer = figure.canvas.get_renderer()
        table_axis = figure.axes[2]
        assert table_axis.get_label() == "panel-c"
        texts = [text for text in table_axis.texts if text.get_text() == "61.2 (50.5–72.8)"]
        assert len(texts) == 2
        for text in texts:
            bounds = text.get_window_extent(renderer)
            assert table_axis.bbox.contains(*bounds.min) and table_axis.bbox.contains(*bounds.max)
    finally:
        plt.close(original)
        plt.close(figure)


def test_historical_ages_are_not_enabled_for_production_by_default() -> None:
    raw = load_yml(CONFIG_PATH)
    assert raw["figure"]["manual_age_config_path"] is None
