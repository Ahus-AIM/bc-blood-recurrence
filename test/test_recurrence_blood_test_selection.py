from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, RecurrenceTimeWindow
from src.data.recurrence_blood_test_selection import SELECTION_COLUMNS, rank_recurrence_tests, select_recurrence_plot_rows
from src.script.plot.recurrence_blood_test_insights import ordered_site_rows
from test.test_recurrence_blood_test_insights import report_fixture


@pytest.fixture
def config(tmp_path: Path) -> RecurrenceBloodTestConfig:
    _, report = report_fixture(tmp_path)
    return replace(report, windows=(
        RecurrenceTimeWindow("6–<18 months", 180, 539),
        RecurrenceTimeWindow("3–<6 months", 90, 179),
        RecurrenceTimeWindow("1–<3 months", 30, 89),
        RecurrenceTimeWindow("1 week–<1 month", 7, 29),
        RecurrenceTimeWindow("<1 week", 0, 6),
    ))


def comparison_rows(values: dict[str, list[float]], config: RecurrenceBloodTestConfig) -> pd.DataFrame:
    records = []
    for name, adjusted in values.items():
        for index, (window, p_value) in enumerate(zip(config.windows, [np.nan, *adjusted], strict=True)):
            records.append({
                "site": "Liver",
                "test_key": name,
                "blood_test": name,
                "window": window.label,
                "min_days_before": window.min_days,
                "p_value_bh": p_value,
                "p_value": p_value / 2,
                "test_status": "baseline" if index == 0 else "ok",
            })
    return pd.DataFrame(records)


def test_earlier_persistent_change_precedes_smaller_final_p_and_keeps_all_windows(
    config: RecurrenceBloodTestConfig,
) -> None:
    frame = comparison_rows({
        "Albumin": [0.959, 0.174, 0.068, 0.0001],
        "AST": [0.520, 0.047, 0.003, 0.002],
        "Transient": [0.000001, 0.000001, 0.2, 0.2],
    }, config)
    # The same tests can have different trajectories and independent ranks at another site.
    bone = frame.assign(site="Bone")
    bone.loc[bone["test_key"].eq("AST"), "p_value_bh"] = 0.2
    frame = pd.concat([frame, bone], ignore_index=True).sample(frac=1, random_state=42)
    original = frame.copy(deep=True)
    config = replace(config, plot=replace(config.plot, top_n=2))
    selected = select_recurrence_plot_rows(frame, config)
    liver = selected.loc[selected["site"].eq("Liver")]
    assert liver["test_key"].unique().tolist() == ["AST", "Albumin"]
    assert liver.iloc[0]["selection_window"] == config.windows[2].label
    assert liver.iloc[0]["selection_p_value_bh"] == 0.047
    assert liver.iloc[0]["selection_p_value"] == 0.047 / 2
    assert selected.loc[selected["site"].eq("Bone"), "test_key"].unique().tolist() == ["Albumin", "Transient"]
    for _, rows in selected.groupby(["site", "test_key"]):
        assert rows["window"].tolist() == [window.label for window in config.windows]
    keys = ["site", "test_key", "min_days_before"]
    expected = frame.loc[
        (frame["site"].eq("Liver") & frame["test_key"].isin(["AST", "Albumin"]))
        | (frame["site"].eq("Bone") & frame["test_key"].isin(["Albumin", "Transient"]))
    ]
    pd.testing.assert_frame_equal(
        selected.drop(columns=list(SELECTION_COLUMNS)).sort_values(keys).reset_index(drop=True),
        expected.sort_values(keys).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("gap", [0.05, 0.2, np.nan, np.inf, -np.inf, "absent"])
def test_nonsignificant_or_missing_window_breaks_persistence(
    config: RecurrenceBloodTestConfig, gap: float | str,
) -> None:
    frame = comparison_rows({"Test": [0.001, 0.01, 0.03, 0.02]}, config)
    middle = frame["window"].eq(config.windows[2].label)
    if gap == "absent":
        frame = frame.loc[~middle]
    else:
        frame.loc[middle, "p_value_bh"] = gap
    ranked = rank_recurrence_tests(frame, config)
    assert ranked.iloc[0]["selection_window"] == config.windows[3].label
    assert ranked.iloc[0]["selection_p_value_bh"] == 0.03


@pytest.mark.parametrize("final", [0.05, 0.2, np.nan, "absent"])
def test_earlier_consecutive_significance_does_not_qualify_without_final_window(
    config: RecurrenceBloodTestConfig, final: float | str,
) -> None:
    frame = comparison_rows({"Test": [0.001, 0.01, 0.03, 0.02]}, config)
    last = frame["window"].eq(config.windows[-1].label)
    if final == "absent":
        frame = frame.loc[~last]
    else:
        frame.loc[last, "p_value_bh"] = final
    ranked = rank_recurrence_tests(frame, config)
    assert ranked.iloc[0]["selection_rank"] == 1
    assert ranked[list(SELECTION_COLUMNS[1:])].isna().all().all()


def test_ties_use_full_precision_starting_p_then_name_and_key(config: RecurrenceBloodTestConfig) -> None:
    frame = comparison_rows({
        "Smallest final p": [0.2, 0.000005, 0.001, 1e-12],
        "Smallest starting p": [0.2, 0.000004, 0.001, 0.001],
        "a2": [0.2, 0.02, 0.01, 0.01],
        "Beta": [0.2, 0.02, 0.001, 1e-12],
        "a1": [0.2, 0.02, 0.01, 0.01],
    }, config)
    frame.loc[frame["test_key"].isin(["a1", "a2"]), "blood_test"] = "Alpha"
    frame.loc[frame["test_key"].eq("Beta"), "p_value"] = 1e-12
    ranked = rank_recurrence_tests(frame.sample(frac=1, random_state=42), config)
    assert ranked["test_key"].tolist() == ["Smallest starting p", "Smallest final p", "a1", "a2", "Beta"]


def test_tests_without_sustained_significance_follow_by_minimum_adjusted_p(config: RecurrenceBloodTestConfig) -> None:
    frame = comparison_rows({
        "Transient": [0.000005, 0.04, 0.2, 0.2],
        "Earlier transient": [0.000006, 0.03, 0.2, 0.2],
        "Absent": [np.nan] * 4,
        "Negative": [0.8] * 4,
        "Another negative": [0.8] * 4,
        "Zero p transient": [0, 0.2, 0.2, 0.2],
        "Final only": [0.9, 0.8, 0.7, 0.04],
    }, config)
    frame.loc[frame["test_key"].eq("Negative"), "p_value"] = 1e-12
    ranked = rank_recurrence_tests(frame.sample(frac=1, random_state=42), config)
    expected = ["Final only", "Zero p transient", "Transient", "Earlier transient", "Another negative", "Negative"]
    assert ranked["test_key"].tolist() == expected
    assert ranked.iloc[1:][list(SELECTION_COLUMNS[1:])].isna().all().all()
    assert select_recurrence_plot_rows(frame, config)["test_key"].unique().tolist() == expected[:config.plot.top_n]
    assert ordered_site_rows(frame, config)["test_key"].unique().tolist() == [*expected, "Absent"]
    unavailable = frame.loc[frame["test_key"].eq("Absent")]
    assert select_recurrence_plot_rows(unavailable, config).empty
    assert rank_recurrence_tests(frame.iloc[:0], config).empty


@pytest.mark.parametrize("unavailable", [np.nan, np.inf, -np.inf])
def test_fallback_minimum_excludes_baseline_and_nonfinite_comparisons(
    config: RecurrenceBloodTestConfig, unavailable: float,
) -> None:
    frame = comparison_rows({"Alpha": [unavailable, 0.1, 0.2, 0.2], "Beta": [0.09, 0.2, 0.2, 0.2]}, config)
    frame.loc[frame["test_status"].eq("baseline") & frame["test_key"].eq("Alpha"), "p_value_bh"] = 0
    assert rank_recurrence_tests(frame, config)["test_key"].tolist() == ["Beta", "Alpha"]


def test_custom_alpha_controls_overview_and_site_order(config: RecurrenceBloodTestConfig) -> None:
    frame = comparison_rows({
        "Early at five percent": [0.04, 0.04, 0.004, 0.004],
        "Earlier at one percent": [0.2, 0.009, 0.009, 0.009],
    }, config)
    assert rank_recurrence_tests(frame, config)["test_key"].tolist() == [
        "Early at five percent", "Earlier at one percent",
    ]
    config = replace(config, fdr_alpha=0.01)
    expected = ["Earlier at one percent", "Early at five percent"]
    assert select_recurrence_plot_rows(frame, config)["test_key"].unique().tolist() == expected
    assert ordered_site_rows(frame, config)["test_key"].unique().tolist() == expected
