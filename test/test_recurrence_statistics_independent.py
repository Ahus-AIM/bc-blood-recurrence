from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
from math import comb
from pathlib import Path
from random import Random
from statistics import median

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, RecurrenceTimeWindow
from src.data.recurrence_blood_test_insights import (
    build_recurrence_blood_test_insights,
    paired_change_statistics,
)
from src.data.recurrence_site_common import RecurrenceSiteRule
from test.recurrence_statistics_oracles import bh_reference, linear_quartiles, signed_rank_reference
from test.test_sample_weighting import blood_row, dataset_config, endpoint_row

WINDOWS = (
    RecurrenceTimeWindow("baseline", 180, 539),
    RecurrenceTimeWindow("3-6m", 90, 179),
    RecurrenceTimeWindow("1-3m", 30, 89),
    RecurrenceTimeWindow("1w-1m", 7, 29),
    RecurrenceTimeWindow("last week", 0, 6),
)


@pytest.mark.parametrize("size", [1, 2, 5, 13, 14, 30, 50, 51, 114])
@pytest.mark.parametrize("tied", [False, True])
def test_signed_ranks_match_independent_exact_calculation_at_every_size(size: int, tied: bool) -> None:
    rng = Random(712 + size)
    for _ in range(10):
        differences = [(rng.randint(1, 4) if tied else i + 1) * rng.choice((-1, 1)) for i in range(size)]
        baseline = np.full(size, 200.0)
        actual = paired_change_statistics(baseline, baseline + differences, significant_digits=12)
        expected = signed_rank_reference(differences, exact=True)
        assert actual["wilcoxon_statistic"] == expected[0]
        assert actual["p_value"] == pytest.approx(expected[1], rel=1e-12, abs=1e-15)


@pytest.mark.parametrize("zero_count", [0, 8, 9, 25, 95])
def test_discarded_zeros_cannot_change_signed_rank_p_value(zero_count: int) -> None:
    differences = np.array([1.0] * 5 + [0.0] * zero_count)
    baseline = np.full(len(differences), 20.0)
    actual = paired_change_statistics(baseline, baseline + differences, significant_digits=12)
    assert actual["n_pairs"] == 5 + zero_count
    assert actual["n_nonzero_pairs"] == 5
    assert actual["median_change"] == median(differences)
    assert actual["p_value"] == 0.0625  # Two of the 32 equally likely sign assignments.


def test_decimal_subtraction_ties_match_hand_calculation() -> None:
    baseline = np.array([0.525, 0.775, 0.325, 0.55])
    later = np.array([0.5, 0.825, 0.375, 0.5])
    expected = signed_rank_reference([-0.025, 0.05, 0.05, -0.05], exact=True)
    for scale in (0.001, 1.0, 1000.0):
        actual = paired_change_statistics(baseline * scale, later * scale, significant_digits=12)
        assert actual["wilcoxon_statistic"] == expected[0]
        assert actual["p_value"] == expected[1]


@pytest.mark.parametrize("size", [1, 5, 13, 14, 30, 50, 51])
@pytest.mark.parametrize("tied", [False, True])
def test_zero_omission_preserves_exact_inference_at_every_size(size: int, tied: bool) -> None:
    differences = [(i % 3 + 1 if tied else i + 1) * (-1 if i % 4 == 0 else 1) for i in range(size)]
    expected = signed_rank_reference(differences, exact=True)
    baseline = np.full(size + 100, 200.0)
    later = baseline + np.array([*differences, *([0.0] * 100)])
    actual = paired_change_statistics(baseline, later, significant_digits=12)
    assert actual["wilcoxon_statistic"] == expected[0]
    assert actual["p_value"] == pytest.approx(expected[1], rel=1e-12, abs=1e-15)


def test_large_tied_sample_matches_exact_binomial_tail() -> None:
    baseline = np.full(150, 200.0)
    later = baseline + np.array([1.0] * 100 + [-1.0] * 50)
    expected = 2 * sum(comb(150, k) for k in range(51)) / 2**150
    actual = paired_change_statistics(baseline, later, significant_digits=12)
    assert actual["wilcoxon_statistic"] == 50 * 75.5
    assert actual["p_value"] == expected


def test_exact_tail_preserves_very_small_probabilities_at_clinical_cohort_size() -> None:
    baseline = np.full(535, 200.0)
    actual = paired_change_statistics(baseline, baseline + 1, significant_digits=12)
    assert actual["wilcoxon_statistic"] == 0.0
    assert actual["p_value"] == 2 / 2**535


@pytest.mark.parametrize("changed_patients", [5, 30])
def test_exact_p_values_reach_global_adjustment_through_real_patient_pairs(
    tmp_path: Path, changed_patients: int,
) -> None:
    rows = []
    endpoints = []
    for i in range(30):
        patient = f"P{i:02}"
        endpoints.append({**endpoint_row(patient, recurrence_day=1000, right_censoring_day=1200), "ddfs_code": "C787"})
        rows.extend([
            {**blood_row(patient, 800), "result": 20},
            {**blood_row(patient, 999), "result": 21 if i < changed_patients else 20},
        ])
    dataset = PatientBloodTestDataset(dataset_config(tmp_path, rows, endpoints))
    config = RecurrenceBloodTestConfig(
        tmp_path / "source.yml", dataset.config, (RecurrenceSiteRule("Liver", "C787"),),
        None, None, tmp_path / "output", windows=WINDOWS, include_overall=True,
    )
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    final = frame.loc[frame.window.eq("last week")]
    assert len(final) == 2
    assert final.n_pairs.eq(30).all()
    assert final.n_nonzero_pairs.eq(changed_patients).all()
    assert final.median_change.eq(0 if changed_patients == 5 else 1).all()
    assert final.p_value.eq(2 / 2**changed_patients).all()
    assert final.p_value_bh.eq(2 / 2**changed_patients).all()


def synthetic_report(
    tmp_path: Path, seed: int,
) -> tuple[PatientBloodTestDataset, RecurrenceBloodTestConfig, dict[tuple[str, str, int], float], dict[str, set[str]]]:
    """Use CSV sources and an independent patient/window membership oracle."""
    rng = Random(seed)
    sites = {
        "A": {"Liver", "Bone"}, "B": {"Lung"}, "C": {"Brain"}, "D": {"Liver"},
        "E": {"Bone"}, "F": {"Other"}, "G": {"Lung", "Liver"}, "H": {"Lung"},
    }
    codes = {"Liver": "C78.7", "Bone": "C795", "Lung": "C780", "Brain": "C793", "Other": "C794"}
    recurrence_days = {patient: 300 + 100 * i for i, patient in enumerate(sites)}
    endpoints = []
    rows = []
    for patient, labels in sites.items():
        recurrence = recurrence_days[patient]
        first_codes = [codes[label] for label in sorted(labels)]
        # Repeat a first-site code and add a later brain site; neither adds a patient/window copy.
        endpoints.append({
            **endpoint_row(patient, recurrence_day=recurrence, right_censoring_day=recurrence + 200),
            "diagnosis_to_ddfs_in_days": str([recurrence] * (len(first_codes) + 1) + [recurrence + 20]),
            "ddfs_code": str([*first_codes, first_codes[0], "C793"]),
        })
        for test in ("X", "Y", "Z"):
            for window in WINDOWS:
                if rng.random() < 0.25:
                    continue
                for _ in range(rng.randint(1, 5)):
                    day = recurrence - rng.randint(window.min_days, window.max_days)
                    rows.append({**blood_row(patient, day), "test_name": test, "result": rng.randint(1, 80) / 2})
            # Guarantee an eligible panel day; test both ends and out-of-window exclusions.
            for days_before in (540, 539, 180, 179, 100, 90, 89, 30, 29, 7, 6, 0, -1):
                if days_before == 100 or rng.random() < 0.3:
                    rows.append({**blood_row(patient, recurrence - days_before),
                                 "test_name": test, "result": rng.randint(1, 80) / 2})
    rng.shuffle(rows)
    # Explicit conflicting duplicate: only the final numeric draw is retained.
    rows.extend({**blood_row("A", 50), "test_name": "X", "result": value} for value in (900, 10, "invalid"))
    cfg = dataset_config(
        tmp_path, rows, endpoints, selected_blood_tests=[f"{name}|||g/dL" for name in ("X", "Y", "Z")],
        split_by_patient={p: ("train", "val", "test")[i % 3] for i, p in enumerate(sites)},
        min_landmark_day=31, min_blood_test_types_per_day=2, include_recurrence_day_blood_tests=True,
        build_patient_timeline_tensors=False,
    )
    dataset = PatientBloodTestDataset(cfg)
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml", dataset.config,
        tuple(RecurrenceSiteRule(site, code.replace(".", "")) for site, code in codes.items() if site != "Other"),
        None, None, tmp_path / "output", windows=WINDOWS, include_overall=True,
    )
    deduplicated = {}
    for row in rows:
        if row["result"] != "invalid":
            deduplicated[str(row["id"]), str(row["test_name"]), int(row["days_since_diagnosis"])] = float(row["result"])
    values = defaultdict(list)
    for (patient, test, day), value in deduplicated.items():
        elapsed = recurrence_days[patient] - day
        for index, window in enumerate(WINDOWS):
            if window.min_days <= elapsed <= window.max_days:
                values[patient, test, index].append(value)
    medians = {key: median(group) for key, group in values.items()}
    return dataset, report, medians, sites


@pytest.mark.parametrize("seed", range(12))
def test_full_pipeline_matches_independent_patient_medians_pairing_and_bh(tmp_path: Path, seed: int) -> None:
    dataset, config, medians, sites = synthetic_report(tmp_path, seed)
    frame = build_recurrence_blood_test_insights(dataset, config, {})
    expected_p = []
    p_indices = []
    for index, row in frame.iterrows():
        members = {patient for patient, labels in sites.items() if row.site == "Any" or row.site in labels}
        test = row.test_key.split("|||")[0]
        window = next(i for i, spec in enumerate(WINDOWS) if spec.label == row.window)
        observed = {p: medians[p, test, window] for p in members if (p, test, window) in medians}
        assert row.n_site_patients == len(members)
        assert row.n_patients == len(observed)
        np.testing.assert_allclose(row[["q1", "median", "q3"]].to_numpy(dtype=float),
                                   linear_quartiles(list(observed.values())), equal_nan=True)
        if window == 0:
            assert row.test_status == "baseline"
            assert np.isnan(row.p_value) and np.isnan(row.p_value_bh)
            continue
        differences = [value - medians[p, test, 0] for p, value in observed.items() if (p, test, 0) in medians]
        assert row.n_pairs == len(differences)
        assert row.n_nonzero_pairs == sum(value != 0 for value in differences)
        np.testing.assert_allclose(row[["change_q1", "median_change", "change_q3"]].to_numpy(dtype=float),
                                   linear_quartiles(differences), equal_nan=True)
        if not differences:
            assert row.test_status == "no_pairs" and np.isnan(row.p_value) and np.isnan(row.p_value_bh)
            continue
        statistic, p_value = signed_rank_reference(differences, exact=True)
        assert row.wilcoxon_statistic == statistic
        assert row.p_value == p_value
        expected_p.append(p_value)
        p_indices.append(index)
    np.testing.assert_allclose(frame.loc[p_indices, "p_value_bh"], bh_reference(expected_p), atol=1e-15)
    # Presentation selection cannot alter computed statistics or their adjustment family.
    changed = replace(config, plot=replace(config.plot, top_n=1), fdr_alpha=0.01)
    alternate = build_recurrence_blood_test_insights(dataset, changed, {})
    assert frame.equals(alternate)
