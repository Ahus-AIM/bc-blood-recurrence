from pathlib import Path
from xml.etree import ElementTree

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.colors import to_hex

from src.figure_style import recurrence_site_colors
from src.model.ebm_global_importance import draw_global_importance_table
from src.script import single_bloodtest_ebm_top_features as saved_rows
from src.regeneration.importance import regenerate_importance
from test.test_single_bloodtest_ebm_top_features import FakeModel, save_model_cell


def test_importance_table_keeps_exact_bar_lengths_and_readable_translated_rows() -> None:
    figure, axis = plt.subplots(figsize=(7.5, 8), dpi=160)
    names = ["Albumin-corrected calcium", "ALP", "A long clinical feature with a multiword explanatory suffix"]
    scores = [0.00003127, 0.472839, 0.215671]
    try:
        draw_global_importance_table(axis, names, scores, color="#0072B2", x_max=0.5)
        figure.canvas.draw()
        features = [text for text in axis.texts if str(text.get_gid()).startswith("importance-feature-")]
        numbers = [text for text in axis.texts if str(text.get_gid()).startswith("importance-score-")]
        bars = [patch for patch in axis.patches if str(patch.get_gid()).startswith("importance-bar-")]
        assert [text.get_text().replace("\n", " ") for text in features] == [names[1], names[2], names[0]]
        assert [text.get_text() for text in numbers] == ["0.473", "0.216", "3.13e-05"]
        np.testing.assert_allclose([bar.get_width() for bar in bars], np.asarray(scores)[[1, 2, 0]] / 0.5 * 0.34)
        renderer = figure.canvas.get_renderer()
        for feature, number, bar in zip(features, numbers, bars, strict=True):
            label_bounds = feature.get_window_extent(renderer)
            bar_bounds = bar.get_window_extent(renderer)
            assert axis.bbox.x0 <= label_bounds.x0 < label_bounds.x1 < bar_bounds.x0
            assert bar_bounds.x1 < number.get_window_extent(renderer).x0 < axis.bbox.x1
            assert to_hex(bar.get_facecolor()) == "#0072b2"
    finally:
        plt.close(figure)


def test_zero_importance_keeps_rows_and_does_not_divide_by_zero() -> None:
    figure, axis = plt.subplots()
    try:
        draw_global_importance_table(axis, ["First", "Second"], [0.0, 0.0])
        features = [text.get_text() for text in axis.texts if str(text.get_gid()).startswith("importance-feature-")]
        assert features == ["First", "Second"]
        assert all(patch.get_width() == 0 for patch in axis.patches
                   if str(patch.get_gid()).startswith("importance-bar-"))
    finally:
        plt.close(figure)


def test_benchmark_importance_regeneration_uses_saved_models_and_paper_filenames(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = {"sites": [{"label": "All"}, {"label": "Bone"}], "horizons": [{"label": "6m", "days": 180}]}
    results = [save_model_cell(tmp_path, site, "6m") for site in ("All", "Bone")]
    monkeypatch.setattr(saved_rows, "load_model", lambda _path: FakeModel())

    paths = regenerate_importance(benchmark, tmp_path, results)

    assert {path.name for path in paths} == {
        "global_importance.png", "global_importance.svg", "global_importance_bone.png", "global_importance_bone.svg",
    }
    for site, stem in (("Any", "global_importance"), ("Bone", "global_importance_bone")):
        document = ElementTree.parse(tmp_path / "global_importance/6m" / f"{stem}.svg")
        labels = ["".join(node.itertext()) for node in document.iter("{http://www.w3.org/2000/svg}text")]
        assert f"EBM feature importance — {site} recurrence" in labels
        assert "Example test" in labels  # Translation comes from the saved training config.
        assert "Feature" in labels and "Score" in labels
        assert "Score ≈" not in labels
        svg = ElementTree.tostring(document.getroot(), encoding="unicode").lower()
        assert recurrence_site_colors((site,))[site].lower() in svg


def test_benchmark_importance_fails_before_writing_when_a_model_is_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = {"sites": [{"label": "All"}, {"label": "Bone"}], "horizons": [{"label": "6m", "days": 180}]}
    results = [save_model_cell(tmp_path, "All", "6m")]
    monkeypatch.setattr(saved_rows, "load_model", lambda _path: FakeModel())

    with pytest.raises(FileNotFoundError, match="Bone/6m"):
        regenerate_importance(benchmark, tmp_path, results)

    assert not (tmp_path / "global_importance").exists()
