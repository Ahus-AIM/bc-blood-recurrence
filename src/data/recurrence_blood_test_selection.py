from __future__ import annotations

import numpy as np
import pandas as pd

from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig

SELECTION_COLUMNS = ("selection_rank", "selection_window", "selection_p_value", "selection_p_value_bh")
RANKING_DESCRIPTION = (
    "earliest window with adjusted p < fdr_alpha in every remaining window through recurrence; "
    "ties by adjusted p in that starting window, display name, test key; "
    "calculable tests without a qualifying window follow by minimum adjusted p across comparisons, "
    "then display name and test key"
)


def rank_recurrence_tests(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> pd.DataFrame:
    """Rank by the earliest change significant in every remaining window.

    A final-window-only change qualifies last among sustained changes. Missing
    comparisons break persistence. Calculable tests without a qualifying window
    follow by minimum adjusted p, then name and key. Their selection-window and
    p-value metadata remain missing because no persistent sequence starts.
    """
    windows = [window.label for window in config.windows[1:]]
    comparisons = frame.loc[frame["window"].isin(windows) & frame["test_status"].ne("baseline")]
    records: list[dict[str, object]] = []
    for (site, test_key), rows in comparisons.groupby(["site", "test_key"], sort=False):
        calculable = rows.loc[np.isfinite(rows["p_value_bh"].to_numpy(dtype=float))]
        if calculable.empty:
            continue
        start = sustained_significance_start(rows, config)
        records.append({
            "site": site,
            "test_key": test_key,
            "blood_test": rows.iloc[0]["blood_test"],
            "selection_window": np.nan if start is None else start.name,
            "selection_p_value": np.nan if start is None else start["p_value"],
            "selection_p_value_bh": np.nan if start is None else start["p_value_bh"],
            "_fallback_p_value_bh": calculable["p_value_bh"].min() if start is None else np.nan,
        })
    ranked = pd.DataFrame.from_records(
        records, columns=["site", "test_key", "blood_test", *SELECTION_COLUMNS[1:], "_fallback_p_value_bh"],
    )
    ranked["_window_order"] = ranked["selection_window"].map({label: index for index, label in enumerate(windows)})
    ranked = ranked.sort_values(
        ["_window_order", "selection_p_value_bh", "_fallback_p_value_bh", "blood_test", "test_key"],
        kind="stable", na_position="last",
    )
    ranked["selection_rank"] = ranked.groupby("site", sort=False).cumcount() + 1
    return ranked[["site", "test_key", *SELECTION_COLUMNS]]


def sustained_significance_start(rows: pd.DataFrame, config: RecurrenceBloodTestConfig) -> pd.Series | None:
    """Walk backwards from recurrence, stopping at any nonsignificant or missing window."""
    windows = [window.label for window in config.windows[1:]]
    comparisons = rows.set_index("window").reindex(windows)
    start = None
    for _, row in comparisons.iloc[::-1].iterrows():
        adjusted = float(row["p_value_bh"])
        if not np.isfinite(adjusted) or adjusted >= config.fdr_alpha:
            break
        start = row
    return start


def select_recurrence_plot_rows(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> pd.DataFrame:
    """Return all windows for each site's top-ranked calculable tests.

    Preserve global adjustment; impose no significance or sample-size cutoff.
    Different sites may select the same test.
    """
    best = rank_recurrence_tests(frame, config).groupby("site", sort=False).head(config.plot.top_n)
    selected = frame.merge(
        best, on=["site", "test_key"], validate="many_to_one"
    )
    selected["_site_order"] = selected["site"].map({site: index for index, site in enumerate(config.sites)})
    selected["_window_order"] = selected["window"].map(
        {window.label: index for index, window in enumerate(config.windows)}
    )
    return (
        selected.sort_values(["_site_order", "selection_rank", "_window_order"])
        .drop(
            columns=["_site_order", "_window_order"],
        )
        .reset_index(drop=True)
    )
