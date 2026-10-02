from __future__ import annotations

from itertools import combinations

import matplotlib.pyplot as plt
import pytest
from matplotlib.colors import to_hex

from src.reporting import count_percentage_text
from src.script.plot.dataset_statistics import BloodTestCountPlotConfig, draw_bloodtest_count_bar_chart
from src.script.plot.overview_style import BLOODTEST_STACK_COLORS


def test_chart_preserves_measured_counts_without_source_attribution() -> None:
    figure, axis = plt.subplots(figsize=(5, 3))
    config = BloodTestCountPlotConfig.from_mapping({"blood_test_names": ["Albumin"], "blood_test_counts": [80]})
    draw_bloodtest_count_bar_chart(axis, config)
    plot = axis.child_axes[0]
    assert plot.get_xlabel() == "Number of blood tests"
    assert plot.containers[0].patches[0].get_width() == 80
    assert [text.get_text().strip() for text in plot.texts] == ["80"]
    assert plot.get_legend() is None
    plt.close(figure)


def test_chart_preserves_source_counts_denominators_and_rounding() -> None:
    figure, axis = plt.subplots(figsize=(5.5, 3))
    names = ["Albumin", "Hemoglobin", "CRP"]
    sources = ([80, 99, 9999], [15, 1, 1], [5, 0, 0])
    config = BloodTestCountPlotConfig.from_mapping(
        {
            "show_imputation_sources": True,
            "blood_test_names": names,
            "blood_test_counts": sources[0],
            "blood_test_locf_counts": sources[1],
            "blood_test_mice_counts": sources[2],
        }
    )
    draw_bloodtest_count_bar_chart(axis, config)
    for row in range(3):
        y = 6 + 3.54 * row
        row_text = [text.get_text() for text in axis.texts if text.get_position()[1] == y]
        total = sum(counts[row] for counts in sources)
        assert row_text == [names[row], *(count_percentage_text(source[row], total).replace("%", "")
                                          for source in sources)]
        bars = [patch for patch in axis.patches if patch.get_y() == y - .65]
        assert sum(bar.get_width() for bar in bars) == pytest.approx(9)
        for source, bar, color in zip(sources, bars, BLOODTEST_STACK_COLORS, strict=True):
            assert bar.get_width() == pytest.approx(9 * source[row] / total)
            assert to_hex(bar.get_facecolor()) == color.lower()
    plt.close(figure)


def test_dense_table_text_stays_in_columns_and_rows_align() -> None:
    figure, axis = plt.subplots(figsize=(5.3, 3.5), dpi=180)
    names = ["Albumin-corrected calcium", "Neutrophil granulocytes"] * 12
    config = BloodTestCountPlotConfig.from_mapping(
        {
            "show_imputation_sources": True,
            "blood_test_names": names,
            "blood_test_counts": [99999] * 24,
            "blood_test_locf_counts": [1] * 24,
            "blood_test_mice_counts": [0] * 24,
        }
    )
    draw_bloodtest_count_bar_chart(axis, config)
    figure.canvas.draw()
    renderer = figure.canvas.get_renderer()
    bounds = [text.get_window_extent(renderer) for text in axis.texts]
    assert not any(a.overlaps(b) for a, b in combinations(bounds, 2))
    assert all(figure.bbox.contains(*box.min) and figure.bbox.contains(*box.max) for box in bounds)
    for row in range(24):
        texts = [text for text in axis.texts if text.get_position()[1] == 6 + row * 3.5]
        assert len(texts) == 4
    plt.close(figure)
