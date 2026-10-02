from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

from src.data import PatientBloodTestDataset
from src.data.blood_test_display import feature_display_label
from src.data.dataset_statistics import prepared_recurrence_site_rows
from src.data.exact_signed_rank import exact_signed_rank
from src.data.measurement_selection import selected_recurrence_measurements
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, validate_report_config
from src.data.recurrence_site_incidence import first_recurrence_rows, first_recurrence_rows_any

PATIENT_WINDOW_KEYS = ["site", "test_key", "patient_id", "window_index"]

REPORT_COLUMNS = (
    "site",
    "test_key",
    "blood_test",
    "unit",
    "window",
    "min_days_before",
    "max_days_before",
    "n_site_patients",
    "n_patients",
    "q1",
    "median",
    "q3",
    "n_pairs",
    "n_nonzero_pairs",
    "change_q1",
    "median_change",
    "change_q3",
    "wilcoxon_statistic",
    "p_value",
    "p_value_bh",
    "test_status",
)


def build_recurrence_blood_test_insights(
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
    translation: Mapping[str, str],
) -> pd.DataFrame:
    """Summarize observed patient medians and paired baseline changes across all splits."""
    validate_report_config(config)
    measurements = recurrence_window_measurements(dataset, config)
    medians = aggregate_patient_medians(measurements)
    groups = {
        (str(site), str(test)): group for (site, test), group in medians.groupby(["site", "test_key"], sort=False)
    }
    test_keys = sorted(dataset.selected_test_keys, key=lambda key: feature_display_label(key, translation).full_label)
    records = []
    for site in config.sites:
        for test_key in test_keys:
            group = groups.get((site, test_key), medians.head(0))
            records.extend(test_window_rows(site, test_key, group, config, translation))
    frame = pd.DataFrame.from_records(records, columns=REPORT_COLUMNS)
    counts = recurrence_site_memberships(dataset, config).groupby("site")["patient_id"].nunique()
    frame["n_site_patients"] = frame["site"].map(counts.reindex(config.sites, fill_value=0)).astype(int)
    valid = frame["p_value"].notna()
    if valid.any():
        frame.loc[valid, "p_value_bh"] = false_discovery_control(
            frame.loc[valid, "p_value"].to_numpy(dtype=float),
            method="bh",
        )
    # Reference percentages are populated by the shared local/remote report preparation.
    frame["n_reference_patients"] = 0
    frame["outside_score_sum"] = 0.0
    frame["outside_reference_percent"] = np.nan
    frame["reference_status"] = "pending_remote"
    frame.loc[frame["n_pairs"].eq(0) & frame["test_status"].ne("baseline"), "reference_status"] = "no_pairs"
    frame.loc[frame["n_patients"].eq(0), "reference_status"] = "no_measurements"
    return frame


def patient_window_medians(
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> pd.DataFrame:
    """One observed median per patient, first-day site, test key, and window."""
    return aggregate_patient_medians(recurrence_window_measurements(dataset, config))


def aggregate_patient_medians(measurements: pd.DataFrame) -> pd.DataFrame:
    return measurements.groupby(PATIENT_WINDOW_KEYS, as_index=False, sort=False)["result"].median()


def recurrence_window_measurements(
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> pd.DataFrame:
    """Share eligible observed rows between the paired analysis and reference counts."""
    columns = ["site", "test_key", "patient_id", "window_index", "window", "result", "days_since_diagnosis"]
    memberships = recurrence_site_memberships(dataset, config)
    if memberships.empty:
        return pd.DataFrame(columns=columns)
    measurements = selected_recurrence_measurements(dataset.prepared_data)
    measured = measurements.merge(memberships, on="patient_id", how="inner")
    measured = measured.loc[np.isfinite(measured["result"])].copy()
    days_before = measured["recurrence_day"] - measured["days_since_diagnosis"]
    measured["window_index"] = -1
    for index, window in enumerate(config.windows):
        measured.loc[days_before.between(window.min_days, window.max_days), "window_index"] = index
    measured = measured.loc[measured["window_index"] >= 0]
    measured["window"] = measured["window_index"].map(
        {index: window.label for index, window in enumerate(config.windows)}
    )
    return measured[columns].reset_index(drop=True)


def recurrence_site_memberships(
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> pd.DataFrame:
    """One eligible patient per first-day site and, optionally, once in the overall group."""
    endpoints = prepared_recurrence_site_rows(dataset)
    if endpoints.empty:
        return pd.DataFrame(columns=["patient_id", "site", "recurrence_day"])
    memberships = first_recurrence_rows(endpoints, config.site_rules, 0, config.other_label)
    if config.include_overall:
        overall = first_recurrence_rows_any(endpoints, 0).assign(site=config.overall_label)
        memberships = pd.concat([overall, memberships], ignore_index=True)
    return memberships[["patient_id", "site", "days_since_diagnosis"]].rename(
        columns={"days_since_diagnosis": "recurrence_day"},
    )


def site_patient_counts(frame: pd.DataFrame) -> dict[str, int]:
    """Read cohort totals from the full report, including sites without rankable tests."""
    totals = frame[["site", "n_site_patients"]].drop_duplicates()
    if totals["site"].duplicated().any():
        raise ValueError("Inconsistent recurrence-site patient totals in report.")
    return {str(site): int(count) for site, count in totals.itertuples(index=False, name=None)}


def test_window_rows(
    site: str,
    test_key: str,
    medians: pd.DataFrame,
    config: RecurrenceBloodTestConfig,
    translation: Mapping[str, str],
) -> list[dict[str, object]]:
    display = feature_display_label(test_key, translation)
    patient_values = medians.pivot(index="patient_id", columns="window_index", values="result")
    patient_values = patient_values.reindex(columns=range(len(config.windows)))
    rows = []
    for index, window in enumerate(config.windows):
        values = patient_values[index].dropna().to_numpy(dtype=float)
        q1, median, q3 = quartiles(values)
        row = {
            "site": site,
            "test_key": test_key,
            "blood_test": display.title,
            "unit": display.unit,
            "window": window.label,
            "min_days_before": window.min_days,
            "max_days_before": window.max_days,
            "n_patients": len(values),
            "q1": q1,
            "median": median,
            "q3": q3,
            "n_pairs": 0,
            "n_nonzero_pairs": 0,
            "change_q1": np.nan,
            "median_change": np.nan,
            "change_q3": np.nan,
            "wilcoxon_statistic": np.nan,
            "p_value": np.nan,
            "p_value_bh": np.nan,
            "test_status": "baseline",
        }
        if index != 0:
            pairs = patient_values[[0, index]].dropna()
            row.update(
                paired_change_statistics(
                    pairs[0].to_numpy(dtype=float),
                    pairs[index].to_numpy(dtype=float),
                    significant_digits=config.difference_significant_digits,
                )
            )
        rows.append(row)
    return rows


def quartiles(values: np.ndarray) -> tuple[float, float, float]:
    if len(values) == 0:
        return (float("nan"),) * 3
    q1, median, q3 = np.quantile(values, (0.25, 0.5, 0.75), method="linear")
    return float(q1), float(median), float(q3)


def paired_change_statistics(
    baseline: np.ndarray,
    later: np.ndarray,
    *,
    significant_digits: int,
) -> dict[str, object]:
    """Keep all paired changes in summaries and use exact Wilcoxon inference."""
    if len(baseline) == 0:
        return {"test_status": "no_pairs"}
    differences = later - baseline
    q1, median, q3 = quartiles(differences)
    ranked_differences = rounded_rank_differences(baseline, later, significant_digits)
    nonzero = int(np.count_nonzero(ranked_differences))
    statistic, p_value = exact_signed_rank(ranked_differences)
    status = "all_zero_differences" if nonzero == 0 else "ok"
    return {
        "n_pairs": len(differences),
        "n_nonzero_pairs": nonzero,
        "change_q1": q1,
        "median_change": median,
        "change_q3": q3,
        "wilcoxon_statistic": statistic,
        "p_value": p_value,
        "test_status": status,
    }


def rounded_rank_differences(baseline: np.ndarray, later: np.ndarray, digits: int) -> np.ndarray:
    """Suppress subtraction roundoff before assigning equal-difference Wilcoxon ranks."""
    scale = float(max(np.max(np.abs(baseline)), np.max(np.abs(later))))
    if scale == 0:
        return later - baseline
    decimals = digits - 1 - int(np.floor(np.log10(scale)))
    return np.round(later - baseline, decimals=decimals)
