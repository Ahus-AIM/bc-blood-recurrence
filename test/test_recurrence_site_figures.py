from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import yaml

from src.config import load_yml
from src.data.recurrence_blood_test_config import load_recurrence_blood_test_config, validate_report_config
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_blood_test_plot_config import RecurrenceBloodTestPlotConfig
from src.data.recurrence_blood_test_selection import select_recurrence_plot_rows
from src.script import export_recurrence_blood_test_insights
from src.script.plot.recurrence_blood_test_insights import (
    create_recurrence_site_figure,
    write_recurrence_site_figures,
)
from src.script.survival_benchmark_cache import artifact_paths
from src.script.training_recurrence_blood_test_insights import write_training_recurrence_blood_test_insights
from test.test_recurrence_blood_test_insights import report_fixture
from test.test_training_recurrence_blood_test_insights import training_report_fixture


@pytest.mark.parametrize("entry_point", ["training", "standalone"])
def test_reports_export_overview_and_every_site_instead_of_latex(
    tmp_path: Path, entry_point: str,
) -> None:
    dataset, path, output = training_report_fixture(tmp_path)
    raw = yaml.safe_load(path.read_text())
    active = load_yml(export_recurrence_blood_test_insights.CONFIG_PATH)
    raw["plot"] = {**active["plot"], "dpi": 25}
    raw["report"]["export_latex"] = active["report"]["export_latex"]
    raw["report"]["output_dir"] = str(output)
    raw["source"]["dataset_config_path"] = str(output / "training_config.yml")
    raw["population_statistics"] = {
        "include_overall": True,
        "other_label": None,
        "site_labels": ["Bone", "Brain", "Liver", "Lung"],
        "site_regexes": [r"C795\d*", r"C793\d*", r"C787\d*", r"C780\d*"],
    }
    path.write_text(yaml.safe_dump(raw))
    if entry_point == "training":
        artifacts = write_training_recurrence_blood_test_insights(dataset, output, plot_config_path=path)
        assert all((output / name).is_file() for name in artifact_paths(artifacts))
        assert Path("recurrence_blood_test_insights.tex") not in set(artifact_paths(artifacts))
    else:
        export_recurrence_blood_test_insights.export_recurrence_report(path)
    stems = {f"recurrence_blood_test_trajectories_{site}" for site in ("any", "bone", "brain", "liver", "lung")}
    stems.add("recurrence_blood_test_trajectories")
    expected = {f"{stem}.{suffix}" for stem in stems for suffix in ("png", "svg")}
    images = {file.name for file in output.iterdir() if file.suffix in {".png", ".svg", ".pdf"}}
    assert images == expected
    assert not (output / "recurrence_blood_test_insights.tex").exists()
    assert all((output / name).stat().st_size > 0 for name in expected)
    for stem in stems:
        assert "<text" in (output / f"{stem}.svg").read_text()
    svg = (output / "recurrence_blood_test_trajectories_any.svg").read_text()
    assert "Any recurrence" in svg and "· 5 patients" in svg and "N =" not in svg
    assert "synthetic" not in svg.lower() and "Reading this figure" not in svg
    assert "Other" in svg  # This test has no calculable comparisons and must still be drawn.
    lung = (output / "recurrence_blood_test_trajectories_lung.svg").read_text()
    assert "· 0 patients" in lung and "N/A" in lung and "N =" not in lung
    full = pd.read_csv(output / "recurrence_blood_test_insights.csv")
    plotted = pd.read_csv(output / "recurrence_blood_test_trajectories_per_site_values.csv")
    keys = ["site", "test_key", "min_days_before"]
    pd.testing.assert_frame_equal(
        full.sort_values(keys).reset_index(drop=True), plotted.sort_values(keys).reset_index(drop=True),
    )
    assert plotted.groupby("site")["test_key"].nunique().eq(2).all()
    metadata = json.loads((output / "recurrence_blood_test_insights.json").read_text())
    assert metadata["figure"]["layout"] == "both"
    assert metadata["figure"]["per_site"]["layout"] == "per_site"
    for selection in metadata["figure"]["selected_tests"]:
        assert selection["selection_window"] is None
        assert selection["selection_p_value"] is None
        assert selection["selection_p_value_bh"] is None
    assert metadata["config"]["export_latex"] is False
    assert set(metadata["figure"]["files"]) == expected | {
        "recurrence_blood_test_trajectories_values.csv", "recurrence_blood_test_trajectories_per_site_values.csv",
    }
    overview = pd.read_csv(output / "recurrence_blood_test_trajectories_values.csv")
    config = load_recurrence_blood_test_config(path)
    pd.testing.assert_frame_equal(overview, select_recurrence_plot_rows(full, config), check_dtype=False)
    assert overview["test_key"].unique().tolist() == ["B-Hb|||g/dL"]


@pytest.mark.parametrize("site", ["Liver", "Bone"])
def test_site_panels_and_exported_values_follow_overview_ranking(tmp_path: Path, site: str) -> None:
    dataset, config = report_fixture(tmp_path)
    source = build_recurrence_blood_test_insights(dataset, config, {})
    source = source.loc[source["site"].eq(site) & source["test_key"].eq("B-Hb|||g/dL")]
    cases = {
        "d": ("Delta", 0.1, 0.04),
        "z": ("Zebra", 0.000004 if site == "Liver" else 0.08, 0.000003),
        "e": ("Eta", 0.000005, 0.000001),
        "a2": ("Alpha", 0.02, 0.009),
        "b": ("Beta", 0.02, 0.008),
        "a1": ("Alpha", 0.02, 0.009),
        "c": ("Gamma", 0.02, 0.009),
        "u2": ("Unavailable Z", np.nan, np.nan),
        "u1": ("Unavailable A", np.nan, np.nan),
    }
    rows = []
    for key, (name, adjusted, raw) in cases.items():
        test = source.assign(test_key=key, blood_test=name, p_value_bh=adjusted, p_value=raw)
        test.loc[test["test_status"].eq("baseline"), ["p_value", "p_value_bh"]] = np.nan
        if np.isfinite(adjusted):
            # Significance must persist from the second comparison through the final window.
            test.loc[test["window"].eq(config.windows[1].label), ["p_value", "p_value_bh"]] = 0.8
        rows.append(test)
    frame = pd.concat(rows).sample(frac=1, random_state=42)
    expected = ["z", "e", "a1", "a2", "b", "c", "d", "u1", "u2"]
    if site == "Bone":
        expected = ["e", "a1", "a2", "b", "c", "z", "d", "u1", "u2"]
    config = replace(
        config, site_rules=tuple(rule for rule in config.site_rules if rule.label == site), other_label=None,
        plot=replace(config.plot, layout="per_site", formats=("svg",), dpi=25),
    )
    overview = select_recurrence_plot_rows(frame, config)
    assert overview["test_key"].unique().tolist() == expected[:5]
    figure = create_recurrence_site_figure(frame, config)
    try:
        titles = [text.get_text() for text in figure.texts if text.get_text().partition(". ")[0].isdigit()]
        assert titles == [f"{index}. {cases[key][0]}" for index, key in enumerate(expected, start=1)]
    finally:
        plt.close(figure)
    config.output_dir.mkdir()
    write_recurrence_site_figures(frame, config)
    plotted = pd.read_csv(config.output_dir / "recurrence_blood_test_trajectories_values.csv")
    assert plotted["test_key"].unique().tolist() == expected
    for _, windows in plotted.groupby("test_key", sort=False):
        assert windows["window"].tolist() == [window.label for window in config.windows]
    pd.testing.assert_frame_equal(
        plotted.sort_values(["test_key", "min_days_before"]).reset_index(drop=True),
        frame.sort_values(["test_key", "min_days_before"]).reset_index(drop=True),
        check_dtype=False,
    )


def test_site_grid_draws_all_24_tests_in_four_columns_with_all_numeric_rows(tmp_path: Path) -> None:
    dataset, config = report_fixture(tmp_path)
    source = build_recurrence_blood_test_insights(dataset, config, {})
    source = source.loc[source["site"].eq("Liver") & source["test_key"].eq("B-Hb|||g/dL")]
    # All tests are deliberately unrankable; a top-five selector would omit them.
    frame = pd.concat([
        source.assign(test_key=f"Test {index:02d}", blood_test=f"Test {index:02d}", p_value_bh=np.nan)
        for index in reversed(range(24))
    ])
    config = replace(config, plot=replace(config.plot, layout="per_site", columns=4))
    figure = create_recurrence_site_figure(frame, config)
    try:
        graphs = [axis for axis in figure.axes if axis.get_ylabel() == "g/dL"]
        tables = [axis for axis in figure.axes if not axis.axison]
        assert len(graphs) == len(tables) == 24
        assert len({round(axis.get_position().x0, 8) for axis in graphs}) == 4
        assert len({round(axis.get_position().y0, 8) for axis in graphs}) == 6
        titles = [text.get_text() for text in figure.texts if ". Test" in text.get_text()]
        assert titles == [f"{index + 1}. Test {index:02d}" for index in range(24)]
        for table in tables:
            labels = [text for text in table.texts if text.get_position()[0] < 0]
            assert len(labels) == 8
        line = next(line for line in graphs[0].lines if line.get_marker() == "o")
        np.testing.assert_allclose(line.get_ydata(), source["median"], equal_nan=True)
        assert not any("Reading this figure" in text.get_text() for text in figure.texts)
    finally:
        plt.close(figure)


@pytest.mark.parametrize("values", [{"columns": 0}, {"columns": True}, {"columns": 1.5}, {"layout": "unknown"}])
def test_invalid_site_layout_settings_are_rejected(values: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        RecurrenceBloodTestPlotConfig.from_mapping(values)


def test_latex_toggle_requires_a_boolean(tmp_path: Path) -> None:
    _, config = report_fixture(tmp_path)
    with pytest.raises(ValueError, match="export_latex"):
        validate_report_config(replace(config, export_latex="false"))
