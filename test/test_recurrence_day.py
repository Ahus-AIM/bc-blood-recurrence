from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks, benchmark_washout_dataset
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.observation_window import measurement_end_day
from src.data.survival_targets import model_duration_days, signed_survival_durations
from src.model.ebm_fitting import fit_ebm_model
from src.model.survival_metrics import harrell_c_index_excluding_same_patient
from test.test_sample_weighting import blood_row, dataset_config, endpoint_row
from test.test_ebm_fitting import survival_table


def test_toggle_changes_cohort_measurements_labels_and_first_day_targets(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("both", 60), blood_row("both", 70), blood_row("both", 71), blood_row("same-day-only", 70)],
        [endpoint_row("both", recurrence_day=70), endpoint_row("same-day-only", recurrence_day=70)],
        selected_blood_tests=["B-Hb|||g/dL"],
    )
    excluded = PatientBloodTestDataset(config)
    included = PatientBloodTestDataset({**config, "include_recurrence_day_blood_tests": True})
    old_summary = build_dataset_statistics_summary(excluded.config, prepared_dataset=excluded)
    summary = build_dataset_statistics_summary(included.config, prepared_dataset=included)

    assert excluded.patient_ids == ("both",)
    assert set(included.patient_ids) == {"both", "same-day-only"}
    assert old_summary.blood_test_counts == (1,)
    assert summary.blood_test_counts == (3,)
    assert summary.stage_ids[-1] == frozenset(("both", "same-day-only"))
    assert "on or before (competing) event or censoring" in summary.inclusion_box_labels[-2]
    assert summary.inclusion_box_counts[-1] == "N = 2\nblood tests = 3"
    assert summary.split_box_counts[0] == "2016 - 2016\nN = 2\nblood tests = 3"
    assert excluded.prepared_data.input_fingerprint != included.prepared_data.input_fingerprint
    table = included.to_landmark_table(bootstrap=False)
    assert table.landmark_days.tolist() == [60, 70, 70]
    assert table.durations.tolist() == [10, 0, 0]
    assert table.events.tolist() == [True, True, True]
    assert table.signed_durations.tolist() == [10, 1e-6, 1e-6]


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize(
    ("recurrence_day", "death_day", "last_observation_day", "other_cancer_day", "endpoint_regex", "end_reason"),
    [
        (None, 70, 100, None, r"C78\d*", "death"),
        (None, None, 70, None, r"C78\d*", "last_observation"),
        (None, None, 100, 70, r"C78\d*", "other_cancer"),
        (70, None, 100, None, r"C787", "non_endpoint_recurrence"),
        (70, None, 100, None, r"C78\d*", "configured_recurrence"),
        (70, 70, 100, None, r"C78\d*", "configured_recurrence"),
        (70, None, 70, None, r"C78\d*", "configured_recurrence"),
        (70, 70, 70, 70, r"C78\d*", "other_cancer"),
        (80, 70, 70, None, r"C78\d*", "configured_recurrence"),
        (80, 70, 70, 90, r"C78\d*", "other_cancer"),
        (80, None, 100, 70, r"C78\d*", "other_cancer"),
    ],
)
def test_all_observation_end_days_are_included_with_zero_day_targets(
    tmp_path: Path,
    split: str,
    recurrence_day: int | None,
    death_day: int | None,
    last_observation_day: int,
    other_cancer_day: int | None,
    endpoint_regex: str,
    end_reason: str,
) -> None:
    patient_ids = ("history", "same-day-only")
    clinical_path = tmp_path / "clinical.csv"
    pd.DataFrame([
        {
            "id": patient_id,
            "days_since_diagnosis": 0 if other_cancer_day is None else other_cancer_day,
            "event_type": "diagnosis",
            "code": "C509" if other_cancer_day is None else "C800",
            "description": "",
        }
        for patient_id in patient_ids
    ]).to_csv(clinical_path, index=False)
    config = dataset_config(
        tmp_path,
        [blood_row("history", day) for day in (60, 70, 71)] + [blood_row("same-day-only", 70)],
        [endpoint_row(
            patient_id, recurrence_day=recurrence_day, death_day=death_day,
            right_censoring_day=last_observation_day,
        ) for patient_id in patient_ids],
        split=split,
        split_by_patient=dict.fromkeys(patient_ids, split),
        selected_blood_tests=["B-Hb|||g/dL"],
        endpoint_ddfs_code_filter_regex=endpoint_regex,
        exclude_and_censor_other_cancer_patients=True,
        other_cancer_code_filter_regex=r"C80\d*",
        diagnosis_procedure_csv_path=str(clinical_path),
        min_landmark_day=31,
        training_sample_weighting="person_time_weighted",
        bootstrap_sample_weighting="person_time_weighted",
    )
    excluded = PatientBloodTestDataset(config)
    included = PatientBloodTestDataset({**config, "include_recurrence_day_blood_tests": True})
    assert excluded.patient_ids == ("history",)
    assert excluded.to_landmark_table(bootstrap=False).landmark_days.tolist() == [60]
    assert set(included.patient_ids) == set(patient_ids)
    table = included.to_landmark_table(bootstrap=False)
    is_event = end_reason == "configured_recurrence"
    assert table.landmark_days.tolist() == [60, 70, 70]
    assert table.durations.tolist() == [10, 0, 0]
    assert table.events.tolist() == [is_event] * 3
    assert table.target_end_reasons.tolist() == [end_reason] * 3
    assert table.signed_durations.tolist() == ([10, 1e-6, 1e-6] if is_event else [-10.000001, -2e-6, -2e-6])
    assert model_duration_days(table.durations).tolist() == [10, 1e-6, 1e-6]
    canonical = PatientBloodTestDataset(replace(
        included.config, endpoint_ddfs_code_filter_regex=included.config.input_recurrence_code_filter_regex,
    ))
    projected = BenchmarkLandmarks(canonical).table(included.config)
    np.testing.assert_array_equal(projected.landmark_days, table.landmark_days)
    np.testing.assert_array_equal(projected.signed_durations, table.signed_durations)
    np.testing.assert_array_equal(projected.target_end_reasons, table.target_end_reasons)
    if split == "train":
        assert table.sample_weight_interval_days.tolist() == [10, 1, 1]
    sampled_days: set[int] = set()
    # A cohort-sized draw can omit either day. Check inclusion across replicates.
    for run in range(5):
        draw = included.to_landmark_table(bootstrap=True, bootstrap_run=run)
        sampled_days.update(draw.landmark_days.tolist())
        assert (draw.events == is_event).all()
        assert (draw.signed_durations[draw.landmark_days == 70] == (1e-6 if is_event else -2e-6)).all()
        np.testing.assert_allclose(
            draw.bootstrap_selection_probabilities,
            np.where(draw.patient_ids == "same-day-only", 0.5, np.where(draw.landmark_days == 60, 5 / 11, 1 / 22)),
        )
    assert sampled_days == {60, 70}


@pytest.mark.parametrize(
    ("recurrence", "censor", "expected"),
    [
        (70, 70, 71),
        (70, 69, 70),
        (None, 70, 71),
        (70, None, 71),
        (None, None, None),
    ],
)
def test_inclusive_boundary_uses_earliest_event_or_censoring_day(
    recurrence: int | None,
    censor: int | None,
    expected: int | None,
) -> None:
    assert measurement_end_day(recurrence, censor, include_end_day=True) == expected


def test_time_weights_count_recurrence_day_once_and_bootstrap_uses_same_intervals(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P", 40), blood_row("P", 70)],
        [endpoint_row("P", recurrence_day=70)],
        include_recurrence_day_blood_tests=True,
        training_sample_weighting="person_time_weighted",
        bootstrap_sample_weighting="person_time_weighted",
        train_bootstrap_samples=300,
    )
    dataset = PatientBloodTestDataset(config)
    table = dataset.to_landmark_table(bootstrap=False)
    assert table.sample_weight_interval_days.tolist() == [30, 1]
    np.testing.assert_allclose(table.sample_weights, [60 / 31, 2 / 31])
    draw = dataset.to_landmark_table(bootstrap=True)
    assert 70 in draw.landmark_days
    np.testing.assert_allclose(
        draw.bootstrap_selection_probabilities, np.where(draw.landmark_days == 70, 1 / 31, 30 / 31)
    )

    # With no recurrence-day draw, the final earlier sample covers that calendar day.
    without_final = replace(dataset.items[0], valid_landmark_days=(40,))
    dataset.items = (without_final,)
    assert dataset.to_landmark_table(bootstrap=False).sample_weight_interval_days.tolist() == [31]


def test_benchmark_zero_day_event_and_non_endpoint_censoring_share_features(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P", 70)],
        [endpoint_row("P", recurrence_day=70)],
        include_recurrence_day_blood_tests=True,
        training_sample_weighting="person_time_weighted",
    )
    dataset = PatientBloodTestDataset(config)
    landmarks = BenchmarkLandmarks(dataset)
    overall = landmarks.table(dataset.config)
    liver = landmarks.table(replace(dataset.config, endpoint_ddfs_code_filter_regex=r"C787"))
    assert overall.features is liver.features
    assert overall.durations.tolist() == liver.durations.tolist() == [0]
    assert overall.events.tolist() == [True]
    assert liver.events.tolist() == [False]
    assert overall.signed_durations.tolist() == [1e-6]
    assert liver.signed_durations.tolist() == [-2e-6]
    assert overall.sample_weight_interval_days.tolist() == [1]
    assert liver.target_end_reasons.tolist() == ["non_endpoint_recurrence"]
    assert benchmark_washout_dataset(dataset, 0) is dataset
    assert benchmark_washout_dataset(dataset, 1).items[0].valid_landmark_days == ()


def test_ebm_accepts_epsilon_targets_and_preserves_risk_ordering() -> None:
    table = survival_table()
    durations = np.ceil(table.durations)
    durations[:10] = 0
    events = table.events.copy()
    events[:5], events[5:10] = True, False
    table = replace(
        table, durations=durations, events=events, signed_durations=signed_survival_durations(durations, events)
    )
    config = {"outer_bags": 1, "validation_size": 0, "max_rounds": 20, "n_jobs": 1}
    fitted, _ = fit_ebm_model(table, table.features, config)
    alternative_durations = np.where(durations == 0, 0.5, durations)
    alternate = replace(
        table,
        durations=alternative_durations,
        signed_durations=signed_survival_durations(alternative_durations, events),
    )
    alternate_model, _ = fit_ebm_model(alternate, alternate.features, config)
    scores = fitted.predict(table.features)
    assert np.isfinite(scores).all()
    np.testing.assert_allclose(scores, alternate_model.predict(table.features), atol=1e-12)
    raw_c = harrell_c_index_excluding_same_patient(durations, events, scores, table.patient_ids)
    encoded_c = harrell_c_index_excluding_same_patient(
        model_duration_days(durations), events, scores, table.patient_ids
    )
    assert raw_c == encoded_c


def test_zero_encoding_is_floating_point_and_does_not_change_recorded_days() -> None:
    days = np.array([0, 0, 1, 30], dtype=np.int32)
    encoded = signed_survival_durations(days, np.array([True, False, True, False]))
    assert encoded.tolist() == [1e-6, -2e-6, 1, -30.000001]
    assert days.tolist() == [0, 0, 1, 30]
    assert encoded.astype(np.float32)[0] > 0
    with pytest.raises(ValueError, match="nonnegative"):
        model_duration_days(np.array([-1]))


@pytest.mark.parametrize(
    ("recurrence_day", "duration", "is_event", "signed_duration"),
    [(210, 179, True, 179.0), (211, 180, True, 180.0),
     (212, 180, False, -180.000001), (None, 180, False, -180.000001)],
)
def test_censoring_offset_preserves_six_month_horizon(
    tmp_path: Path,
    recurrence_day: int | None,
    duration: int,
    is_event: bool,
    signed_duration: float,
) -> None:
    config = dataset_config(
        tmp_path,
        [blood_row("P", 31)],
        [endpoint_row("P", recurrence_day=recurrence_day, right_censoring_day=1000)],
        min_landmark_day=31,
        censoring_horizon_days=180,
    )
    dataset = PatientBloodTestDataset(config)
    direct = dataset.to_landmark_table(bootstrap=False)
    projected = BenchmarkLandmarks(dataset).table(dataset.config)
    for table in (direct, projected):
        assert table.landmark_days.tolist() == [31]
        assert table.durations.tolist() == [duration]
        assert table.events.tolist() == [is_event]
        assert table.signed_durations.tolist() == [signed_duration]
        assert table.target_end_reasons.tolist() == ["configured_recurrence" if is_event else "censoring_horizon"]
        assert table.metadata["censoring_horizon_days"] == 180
        assert table.metadata["censoring_offset_model_days"] == 1e-6


def test_c50_and_panel_size_filters_still_apply_on_recurrence_day(tmp_path: Path) -> None:
    config = dataset_config(
        tmp_path,
        [
            blood_row("included", 70),
            blood_row("no-checkup", 70),
            blood_row("sparse", 70),
            {**blood_row("included", 70), "test_name": "Other"},
            {**blood_row("no-checkup", 70), "test_name": "Other"},
        ],
        [endpoint_row(patient, recurrence_day=70) for patient in ("included", "no-checkup", "sparse")],
        selected_blood_tests=["B-Hb|||g/dL", "Other|||g/dL"],
        include_recurrence_day_blood_tests=True,
        min_blood_test_types_per_day=2,
        restrict_bloodtests_to_bc_diagnosis_window=True,
        bc_diagnosis_window_days_before=0,
        bc_diagnosis_window_days_after=0,
        bc_diagnosis_days_csv_path=str(tmp_path / "checkups.csv"),
    )
    pd.DataFrame({"id": ["included", "sparse"], "bc_diagnosis_day": [70, 70]}).to_csv(
        tmp_path / "checkups.csv",
        index=False,
    )
    assert PatientBloodTestDataset(config).patient_ids == ("included",)
