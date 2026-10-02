from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.landmark_event_counts import recurrence_target_item
from src.data.landmarks import patient_targets
from src.data.recurrence_site_incidence import load_item_endpoint_site_rows, survival_durations
from test.test_patient_bloodtest import OTHER_CANCER_CONFIG, blood_row, clinical_row, config, endpoint_row


@pytest.mark.parametrize("endpoint_regex", [r"C7[89]\d*", r"C787\d*", r"C795\d*"])
def test_post_recurrence_survival_counts_death_after_last_clinical_record(
    tmp_path: Path, endpoint_regex: str,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 50)],
        [endpoint_row("P0", diagnosis_to_ddfs=100, ddfs_code="C787", diagnosis_to_death=300, last_entry_days=200)],
    )
    cfg.update(endpoint_ddfs_code_filter_regex=endpoint_regex, min_landmark_day=31)
    dataset = PatientBloodTestDataset(cfg)

    rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    durations, events = survival_durations(rows, "recurrence")

    assert rows["days_since_diagnosis"].tolist() == [100]
    assert rows["right_censoring_day"].tolist() == [200]
    assert rows["right_censoring_reason"].tolist() == ["last_observation"]
    assert durations.tolist() == [200.0]
    assert events.tolist() == [True]
    # The recurrence boundary remains separate from survival to a known death.
    assert dataset.items[0].last_observation_day == 300


def test_landmark_report_preserves_source_censoring(tmp_path: Path) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 50)],
        [endpoint_row("P0", diagnosis_to_death=300, last_entry_days=200)],
    )
    dataset = PatientBloodTestDataset(cfg)
    item = dataset.items[0]
    reported = recurrence_target_item(
        item,
        re.compile(dataset.config.input_recurrence_code_filter_regex),
        re.compile(dataset.config.endpoint_ddfs_code_filter_regex),
        dataset.config,
    )

    reported_targets = patient_targets(reported, np.array(item.valid_landmark_days), dataset.config)
    model = dataset.to_landmark_table(bootstrap=False)

    assert reported.right_censoring_day == 200
    assert reported.right_censoring_reason == "last_observation"
    np.testing.assert_equal(reported_targets[0], model.durations)
    np.testing.assert_equal(reported_targets[1], model.events)
    np.testing.assert_equal(reported_targets[3], model.target_end_reasons)


@pytest.mark.parametrize(("recurrence_day", "death_day"), [(100, ""), ("", 100), (100, 120)])
def test_known_outcome_does_not_replace_missing_source_follow_up(
    tmp_path: Path, recurrence_day: int | str, death_day: int | str,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 50)],
        [endpoint_row(
            "P0", diagnosis_to_ddfs=recurrence_day, ddfs_code="C787",
            diagnosis_to_death=death_day, last_entry_days="",
        )],
    )

    with pytest.raises(ValueError, match="P0.*missing source last observation"):
        PatientBloodTestDataset(cfg)


@pytest.mark.parametrize(("death_day", "other_cancer_day"), [("", None), (250, None), ("", 100)])
def test_first_recurrence_after_source_follow_up_is_fatal(
    tmp_path: Path, death_day: int | str, other_cancer_day: int | None,
) -> None:
    cfg = config(
        tmp_path,
        [blood_row("P0", 50)],
        [endpoint_row("P0", diagnosis_to_ddfs=300, ddfs_code="C787", diagnosis_to_death=death_day,
                      last_entry_days=200)],
        clinical_rows=None if other_cancer_day is None else [clinical_row("P0", other_cancer_day, code="C80")],
    )
    if other_cancer_day is not None:
        cfg.update(OTHER_CANCER_CONFIG)

    with pytest.raises(ValueError, match="P0.*first input recurrence.*source last observation"):
        PatientBloodTestDataset(cfg)


@pytest.mark.parametrize("other_cancer_day", [80, 100])
def test_other_cancer_can_end_analytic_follow_up_before_source_end(
    tmp_path: Path, other_cancer_day: int,
) -> None:
    cfg = config(
        tmp_path, [blood_row("P0", 50)],
        [endpoint_row("P0", diagnosis_to_ddfs=100, ddfs_code="C787", last_entry_days=200)],
        clinical_rows=[clinical_row("P0", other_cancer_day, code="C80")],
    )
    cfg.update(OTHER_CANCER_CONFIG)
    dataset = PatientBloodTestDataset(cfg)
    table = dataset.to_landmark_table(bootstrap=False)
    rows = load_item_endpoint_site_rows(dataset.items, dataset.config)

    assert dataset.items[0].source_last_observation_day == 200
    assert table.durations.tolist() == [other_cancer_day - 50]
    assert table.events.tolist() == [False]
    assert table.target_end_reasons.tolist() == ["other_cancer"]
    assert rows["days_since_diagnosis"].isna().all()
    assert rows["right_censoring_day"].tolist() == [other_cancer_day]


@pytest.mark.parametrize("source_end", ["", 200])
def test_excluded_patient_does_not_trigger_source_follow_up_validation(
    tmp_path: Path, source_end: int | str,
) -> None:
    cfg = config(
        tmp_path, [blood_row("included", 50), blood_row("excluded", 50)],
        [endpoint_row("included"), endpoint_row(
            "excluded", diagnosis_year=1999, diagnosis_to_ddfs=300,
            ddfs_code="C787", last_entry_days=source_end,
        )],
    )
    cfg["min_diagnosis_year"] = 2004

    dataset = PatientBloodTestDataset(cfg)

    assert dataset.patient_ids == ("included",)


@pytest.mark.parametrize(("bone_day", "bone_event"), [(100, True), (150, False)])
def test_first_recurrence_censors_other_sites_but_starts_death_survival(
    tmp_path: Path, bone_day: int, bone_event: bool,
) -> None:
    cfg = config(
        tmp_path, [blood_row("P0", 50)],
        [endpoint_row(
            "P0", diagnosis_to_ddfs=str([100, bone_day]), ddfs_code=str(["C787", "C795"]),
            diagnosis_to_death=300, last_entry_days=200,
        )],
    )
    canonical = PatientBloodTestDataset(cfg)
    landmarks = BenchmarkLandmarks(canonical)
    for regex, expected_event in [(r"C7[89]\d*", True), (r"C787\d*", True), (r"C795\d*", bone_event)]:
        target_config = replace(canonical.config, endpoint_ddfs_code_filter_regex=regex)
        dataset = PatientBloodTestDataset(target_config)
        direct = dataset.to_landmark_table(bootstrap=False)
        projected = landmarks.table(target_config)
        reported = recurrence_target_item(
            canonical.items[0], re.compile(target_config.input_recurrence_code_filter_regex),
            re.compile(regex), target_config,
        )
        reported_targets = patient_targets(reported, np.array(reported.valid_landmark_days), target_config)
        reason = "configured_recurrence" if expected_event else "non_endpoint_recurrence"

        for table in (direct, projected):
            assert table.durations.tolist() == [50]
            assert table.events.tolist() == [expected_event]
            assert table.target_end_reasons.tolist() == [reason]
        assert reported_targets[0].tolist() == [50]
        assert reported_targets[1].tolist() == [expected_event]
        assert reported_targets[3].tolist() == [reason]
        rows = load_item_endpoint_site_rows(dataset.items, target_config)
        assert set(rows["days_since_diagnosis"]) == {100}
        assert set(rows["code"]) == ({"C787", "C795"} if bone_event else {"C787"})
        durations, events = survival_durations(rows, "recurrence")
        assert (durations == 200).all()
        assert events.all()
