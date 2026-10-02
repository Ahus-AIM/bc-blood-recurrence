from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from test.report_return_helpers import archive_return

import numpy as np
import pandas as pd
import pytest

import src.data.recurrence_measurement_history as history_module
from src.data import PatientBloodTestDataset
from src.data.cohort_selection import STAGE_EARLY_MODEL_MEASUREMENTS
from src.data.dataset_statistics import build_dataset_statistics_summary, selected_model_measurements
from src.data.measurement_selection import recurrence_report_measurements, selected_recurrence_measurements
from src.data.population_statistics import PopulationStatisticsConfig, build_population_statistics_table
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig, RecurrenceTimeWindow
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.data.recurrence_reference_exchange import export_reference_package, prepare_reference_report
from src.data.recurrence_site_common import RecurrenceSiteRule
from src.data.patient_preprocessing import load_blood_tests
from src.model.blood_test_imputation_statistics import build_blood_test_imputation_statistics
from src.script.survival_training import prepare_survival_training
from test.test_patient_bloodtest import bc_diagnosis_row, blood_row, config, endpoint_row
from test.test_recurrence_reference_statistics import age_reference_intervals, run_calculation, write_patient_dates


@pytest.mark.parametrize("timeline_tensors", [False, True])
@pytest.mark.parametrize("minimum_day", [0, 31])
def test_final_cutoff_preserves_patients_landmarks_labels_weights_and_bootstrap(
    tmp_path: Path,
    timeline_tensors: bool,
    minimum_day: int,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P", d, result=d + 50.0) for d in (-20, 0, 30, 31, 60, 100)] + [blood_row("early_only", -10)],
        [endpoint_row("P", last_entry_days=100), endpoint_row("early_only", last_entry_days=100)],
    )
    cfg.update(min_landmark_day=minimum_day, build_patient_timeline_tensors=timeline_tensors)
    model = PatientBloodTestDataset(cfg)
    changes = PatientBloodTestDataset({**cfg, "include_blood_test_change": True})
    assert model.patient_ids == changes.patient_ids == ("P",)
    assert model.prepared_data.measurement_df["days_since_diagnosis"].ge(minimum_day).all()
    before, after = model.prepared_data.cohort_stages[-2:]
    assert after.key == STAGE_EARLY_MODEL_MEASUREMENTS
    assert before.patient_ids == after.patient_ids
    assert after.bloodtest_count == before.bloodtest_count - (3 if minimum_day else 1)
    assert STAGE_EARLY_MODEL_MEASUREMENTS not in {s.key for s in changes.prepared_data.cohort_stages}
    for bootstrap in (False, True):
        base = model.to_landmark_table(bootstrap=bootstrap)
        changed = changes.to_landmark_table(bootstrap=bootstrap)
        for field in ("landmark_days", "patient_ids", "durations", "events", "sample_weights"):
            np.testing.assert_array_equal(getattr(base, field), getattr(changed, field))
        np.testing.assert_array_equal(base.features[:, 0], changed.features[:, 0])
    pd.testing.assert_frame_equal(
        selected_model_measurements(model.prepared_data), selected_model_measurements(changes.prepared_data)
    )
    for dataset in (model, changes):
        report = selected_recurrence_measurements(dataset.prepared_data)
        assert sorted(report["days_since_diagnosis"]) == [-20, 0, 30, 31, 60]
        assert set(report["patient_id"]) == {"P"}


@pytest.mark.parametrize("changes", [False, True])
def test_sparse_early_days_survive_without_rescuing_patients_without_landmarks(tmp_path: Path, changes: bool) -> None:
    patient_days = {
        "keep": (-20, -10, 0, 31, 40),
        "early_only": (-10,),
        "outside_followup": (-10, 40),
        "sparse_late": (-10, 40),
    }
    rows = [
        blood_row(p, d, test_name=t, unit="u")
        for p, days in patient_days.items()
        for d in days
        for t in ("A", "B")
        if not (t == "B" and ((p == "sparse_late" and d == 40) or (p == "keep" and d == 0)))
    ]
    cfg = config(
        tmp_path,
        rows,
        [endpoint_row(p, last_entry_days=35 if p == "outside_followup" else 100) for p in patient_days],
        bc_diagnosis_rows=[bc_diagnosis_row(p, d) for p in patient_days for d in (-10, 0, 31, 40)],
    )
    cfg.update(
        min_landmark_day=31,
        selected_blood_tests=["A|||u", "B|||u"],
        min_blood_test_types_per_day=2,
        restrict_bloodtests_to_bc_diagnosis_window=True,
        bc_diagnosis_window_days_before=0,
        bc_diagnosis_window_days_after=0,
        include_blood_test_change=changes,
    )
    dataset = PatientBloodTestDataset(cfg)
    assert dataset.patient_ids == ("keep",)
    assert dataset.items[0].valid_landmark_days == (31, 40)
    report = selected_recurrence_measurements(dataset.prepared_data)
    assert set(report["days_since_diagnosis"]) == {-10, 0, 31, 40}
    assert len(report) == 7  # The single test on day 0 survives the two-test threshold.
    assert set(report["patient_id"]) == {"keep"}
    summary = build_dataset_statistics_summary(dataset.config, prepared_dataset=dataset)
    assert summary.inclusion_box_counts[-1] == f"N = 1\nblood tests = {7 if changes else 4}"
    assert sum(summary.blood_test_counts) == (7 if changes else 4)
    if not changes:
        assert summary.exclusion_box_counts[-2:] == ("N = 0\nblood tests = 3", "N = 0\nblood tests = 0")
        assert all(label not in summary.series_labels for label in summary.exclusion_box_labels[-2:])


@pytest.mark.parametrize(("prior_day", "expected"), [(-335, np.nan), (-334, 6), (24, 6), (25, np.nan)])
def test_changes_automatically_use_early_history_with_inclusive_age_limits(
    tmp_path: Path,
    prior_day: int,
    expected: float,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P", prior_day, result=4), blood_row("P", 31, result=10)],
        [endpoint_row("P", last_entry_days=100)],
    )
    cfg.update(min_landmark_day=31, include_blood_test_change=True)
    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)
    assert table.landmark_days.tolist() == [31]
    np.testing.assert_allclose(table.features[0], [10, expected], equal_nan=True)


@pytest.mark.parametrize("imputation_enabled", [False, True])
def test_statistics_and_locf_ignore_report_only_history(tmp_path: Path, imputation_enabled: bool) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P", d, test_name=t, result=v, unit="u")
            for d, t, v in ((-10, "B", 900), (30, "B", 999), (31, "A", 1), (40, "A", 2), (40, "B", 5))
        ]
        + [blood_row("V", 0, test_name="A", unit="u"), blood_row("V", 31, test_name="A", unit="u")],
        [endpoint_row("P", last_entry_days=100), endpoint_row("V", last_entry_days=100)],
        split_by_patient={"P": "train", "V": "val"},
    )
    cfg.update(min_landmark_day=31, selected_blood_tests=["A|||u", "B|||u"])
    dataset = PatientBloodTestDataset(cfg)
    prepared = prepare_survival_training(
        dataset,
        dataset.to_landmark_table(bootstrap=False),
        {"enabled": imputation_enabled, "method": "locf", "locf_max_age_days": 30, "locf_fallback_method": "mice"},
        nonnegative_feature_names=tuple(dataset.selected_test_keys),
    )
    statistics = build_blood_test_imputation_statistics(dataset, prepared.imputer)
    assert statistics.overall.measured == 4
    assert statistics.overall.locf == 0
    assert statistics.overall.mice == (2 if imputation_enabled else 0)
    if imputation_enabled:
        assert prepared.train_imputation.features[0, 1] == pytest.approx(5)
        assert not prepared.train_imputation.carried_forward_mask.any()
    for attribution in (None, statistics):
        summary = build_dataset_statistics_summary(
            dataset.config, prepared_dataset=dataset, imputation_statistics=attribution
        )
        assert sum(summary.blood_test_counts) == 4
        assert summary.inclusion_box_counts[-1] == "N = 2\nblood tests = 4"
        assert summary.exclusion_box_counts[-1] == "N = 0\nblood tests = 3"
        population = build_population_statistics_table(
            dataset, PopulationStatisticsConfig.from_mapping({}), imputation_statistics=attribution
        )
        rows = {row.label: row.cells for row in population.rows}
        assert rows["Landmark days"] == ("2", "1", "--", "3")
        assert rows["Blood tests"][-1] == str(4 + (2 if imputation_enabled and attribution else 0))


def test_report_and_reference_exports_share_early_measurements(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [
            blood_row("P", -100, result=60),
            blood_row("P", 10, result=20),
            blood_row("P", 95, result=30),
            blood_row("early_only", -100, result=999),
        ],
        [endpoint_row(p, diagnosis_to_ddfs=100, ddfs_code="C787", last_entry_days=200) for p in ("P", "early_only")],
    )
    cfg.update(min_landmark_day=31)
    dataset = PatientBloodTestDataset(cfg)
    report = RecurrenceBloodTestConfig(
        tmp_path / "source.yml",
        dataset.config,
        (RecurrenceSiteRule("Liver", r"C787\d*"),),
        None,
        None,
        tmp_path / "output",
        windows=(RecurrenceTimeWindow("Baseline", 8, 200), RecurrenceTimeWindow("Final", 0, 7)),
        reference_ranges=age_reference_intervals(),
    )
    assert dataset.prepared_data.measurement_df["days_since_diagnosis"].tolist() == [95]
    with recurrence_report_measurements(dataset.prepared_data):
        frame = build_recurrence_blood_test_insights(dataset, report, {})
        assert frame["median"].tolist() == [40, 30]
        assert frame["n_pairs"].tolist() == [0, 1]
        assert frame["n_site_patients"].tolist() == [1, 1]
        assert frame["median_change"].iloc[1] == -10
        folder, _manifest = export_reference_package(frame, dataset, report)
        exported = pd.read_csv(folder / "measurements.csv")
        assert exported["draw days since diagnosis"].tolist() == [-100, 10, 95]
        assert exported["blood test value"].tolist() == [60, 20, 30]
        write_patient_dates(folder, ids="P")
        result = run_calculation(folder)
        archive_return(result, "reference_percentages")
        verified, _metadata = prepare_reference_report(
            frame, dataset, report
        )
        assert verified["outside_reference_percent"].tolist() == [50, 0]  # Baseline averages the 60/20 draw flags.
    changed = PatientBloodTestDataset({**cfg, "include_blood_test_change": True})
    pd.testing.assert_frame_equal(
        frame, build_recurrence_blood_test_insights(changed, replace(report, dataset=changed.config), {})
    )




@pytest.mark.parametrize("value", [None, False, True, "false", 0, 1])
def test_retired_history_option_is_rejected(tmp_path: Path, value: object) -> None:
    cfg = config(tmp_path, [blood_row("P", 40)], [endpoint_row("P")])
    with pytest.raises(ValueError, match="was removed"):
        PatientBloodTestDataset({**cfg, "include_blood_tests_before_min_landmark_day": value})


def test_lazy_history_fails_if_source_changes(tmp_path: Path) -> None:
    cfg = config(tmp_path, [blood_row("P", -1), blood_row("P", 40)], [endpoint_row("P")])
    dataset = PatientBloodTestDataset(cfg)
    with dataset.config.bloodtest_csv_path.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="source changed"):
        selected_recurrence_measurements(dataset.prepared_data)


def test_reporting_loads_history_once_and_releases_it_even_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = []

    def read(path: Path) -> tuple[pd.DataFrame, dict[str, int], int]:
        calls.append(path)
        return load_blood_tests(path)

    monkeypatch.setattr(history_module, "load_blood_tests", read)
    cfg = config(
        tmp_path, [blood_row("P", -1, result=1), blood_row("P", -1, result=2), blood_row("P", 40)], [endpoint_row("P")]
    )
    dataset = PatientBloodTestDataset(cfg)
    dataset.to_landmark_table(bootstrap=False)
    build_dataset_statistics_summary(dataset.config, prepared_dataset=dataset)
    assert calls == []  # Model inputs and Figure 1 never load reporting-only values.
    with pytest.raises(RuntimeError, match="report failure"):
        with recurrence_report_measurements(dataset.prepared_data):
            for _ in range(2):
                report = selected_recurrence_measurements(dataset.prepared_data)
                assert report.loc[report["days_since_diagnosis"].eq(-1), "result"].tolist() == [2]
            assert len(calls) == 1
            raise RuntimeError("report failure")
    selected_recurrence_measurements(dataset.prepared_data)
    assert len(calls) == 2
