from __future__ import annotations

from pathlib import Path

from src.data import PatientBloodTestDataset
from src.data.dataset_statistics import (
    build_dataset_statistics_summary,
    exclusion_stage_ids,
)
from test.test_patient_bloodtest import (
    bc_diagnosis_row,
    blood_row,
    clinical_row,
    config,
    endpoint_row,
)


def test_dataset_statistics_follows_shared_selection_order(tmp_path: Path) -> None:
    patient_ids = (
        "old",
        "pre_recurrence",
        "early_recurrence",
        "no_c50",
        "sparse",
        "before_day_31",
        "on_recurrence",
        "keep",
    )
    blood_rows = [
        blood_row(patient_id, 40, test_name=test_name, result=1.0, unit="u")
        for patient_id in patient_ids
        for test_name in ("A", "B")
        if not (patient_id in {"sparse", "before_day_31", "on_recurrence"} and test_name == "B")
    ]
    blood_rows.append(blood_row("keep", 40, test_name="A", result=2.0, unit="u"))
    blood_rows = [{**row, "days_since_diagnosis": 30} if row["id"] == "before_day_31" else row for row in blood_rows]
    cfg = {
        **config(
            tmp_path,
            blood_rows,
            [
                endpoint_row("old", diagnosis_year=2003, last_entry_days=100),
                endpoint_row(
                    "pre_recurrence",
                    diagnosis_to_ddfs=-1,
                    ddfs_code="C780",
                    last_entry_days=100,
                ),
                endpoint_row(
                    "early_recurrence",
                    diagnosis_to_ddfs=30,
                    ddfs_code="C780",
                    last_entry_days=100,
                ),
                endpoint_row("no_c50", last_entry_days=100),
                endpoint_row("sparse", last_entry_days=100),
                endpoint_row("before_day_31", last_entry_days=100),
                endpoint_row(
                    "on_recurrence",
                    diagnosis_to_ddfs=40,
                    ddfs_code="C780",
                    last_entry_days=100,
                ),
                endpoint_row("keep", last_entry_days=100),
            ],
            split_by_patient={patient_id: "train" for patient_id in patient_ids},
            bc_diagnosis_rows=[
                bc_diagnosis_row(patient_id, 100 if patient_id == "no_c50" else 40)
                for patient_id in patient_ids
                if patient_id != "before_day_31"
            ]
            + [bc_diagnosis_row("before_day_31", 30)],
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
        "min_diagnosis_year": 2004,
        "min_landmark_day": 31,
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "bc_diagnosis_window_days_before": 1,
        "bc_diagnosis_window_days_after": 0,
        "min_blood_test_types_per_day": 2,
    }

    summary = build_dataset_statistics_summary(PatientBloodTestDataset(cfg).config)

    assert summary.inclusion_box_labels == (
        "Females diagnosed with BC at Ahus\n\n1.12.1998 - 08.06.2026",
        "Initial BC diagnosis ≥ 2004",
        "No distant recurrence before BC diagnosis",
        "No distant recurrence ≤ 30 days after BC diagnosis",
        "Blood test on-/one day before a BC checkup",
        "Any blood tests > 30 days after BC diagnosis",
        "Blood tests before (competing) event or censoring",
        "Patients with at least one qualifying day:\ndays > 30 days after BC diagnosis\nwith > 1 different blood tests",
        "Blood tests > 30 days after BC diagnosis",
        "Days with > 1 different blood tests",
    )
    assert summary.inclusion_box_counts == (
        "N = 8\nblood tests = 13",
        "N = 7\nblood tests = 11",
        "N = 6\nblood tests = 9",
        "N = 5\nblood tests = 7",
        "N = 4\nblood tests = 5",
        "N = 3\nblood tests = 4",
        "N = 2\nblood tests = 3",
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 2",
    )
    assert summary.exclusion_box_labels[2] == ("Distant recurrence ≤ 30 days after BC diagnosis")
    assert summary.series_labels[2] == ("Distant recurrence ≤ 30 days after BC diagnosis")
    assert summary.exclusion_box_counts == (
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 2",
        "N = 1\nblood tests = 1",
        "N = 1\nblood tests = 1",
        "N = 1\nblood tests = 1",
        "N = 0\nblood tests = 0",
        "N = 0\nblood tests = 0",
    )
    assert exclusion_stage_ids(summary.stage_ids)[-5:] == (
        {"before_day_31"}, {"on_recurrence"}, {"sparse"}, frozenset(), frozenset(),
    )
    assert summary.series_labels[-3:] == summary.exclusion_box_labels[-5:-2]
    assert summary.stage_ids[-1] == {"keep"}
    assert summary.diagnosis_years == tuple(range(2004, 2017))
    assert summary.series_counts == tuple((0,) * 12 + (1,) for _label in summary.series_labels)
    assert summary.blood_test_counts == (1, 1)


def test_dataset_statistics_summary_uses_model_dataset_cohort(tmp_path: Path) -> None:
    patient_ids = (
        "keep_train",
        "keep_val",
        "prior_bc",
        "pre_recurrence",
        "early_recurrence",
        "no_blood_after_min",
        "no_blood_before_recurrence",
        "no_valid_landmark",
    )
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("keep_train", 40, test_name="A", result=1.0, unit="u"),
                blood_row("keep_val", 40, test_name="A", result=1.0, unit="u"),
                blood_row("prior_bc", 40, test_name="A", result=1.0, unit="u"),
                blood_row("pre_recurrence", 40, test_name="A", result=1.0, unit="u"),
                blood_row("early_recurrence", 40, test_name="A", result=1.0, unit="u"),
                blood_row("no_blood_after_min", 10, test_name="A", result=1.0, unit="u"),
                blood_row("no_blood_before_recurrence", 40, test_name="A", result=1.0, unit="u"),
                blood_row("no_valid_landmark", 40, test_name="A", result=1.0, unit="u"),
            ],
            [
                endpoint_row("keep_train", diagnosis_year=2016, last_entry_days=200),
                endpoint_row("keep_val", diagnosis_year=2020, last_entry_days=200),
                endpoint_row("prior_bc", diagnosis_year=2020, last_entry_days=200),
                endpoint_row(
                    "pre_recurrence",
                    diagnosis_year=2020,
                    diagnosis_to_ddfs=-1,
                    ddfs_code="C780",
                    last_entry_days=200,
                ),
                endpoint_row(
                    "early_recurrence",
                    diagnosis_year=2020,
                    diagnosis_to_ddfs=29,
                    ddfs_code="C780",
                    last_entry_days=200,
                ),
                endpoint_row("no_blood_after_min", diagnosis_year=2020, last_entry_days=200),
                endpoint_row(
                    "no_blood_before_recurrence",
                    diagnosis_year=2020,
                    diagnosis_to_ddfs=35,
                    ddfs_code="C780",
                    last_entry_days=200,
                ),
                endpoint_row("no_valid_landmark", diagnosis_year=2020, last_entry_days=40),
            ],
            split_by_patient={
                "keep_train": "train",
                "keep_val": "val",
                "prior_bc": "train",
                "pre_recurrence": "train",
                "early_recurrence": "excluded",
                "no_blood_after_min": "train",
                "no_blood_before_recurrence": "train",
                "no_valid_landmark": "train",
            },
            clinical_rows=[
                clinical_row("prior_bc", -1, event_type="procedure", code="3800"),
            ],
            bc_diagnosis_rows=[bc_diagnosis_row(patient_id, 0) for patient_id in patient_ids],
        ),
        "selected_blood_tests": ["A|||u"],
        "exclude_prior_bc_surgery_patients": True,
        "min_landmark_day": 30,
    }

    dataset = PatientBloodTestDataset(cfg)
    summary = build_dataset_statistics_summary(
        dataset.config,
        test_name_translation={"A": "Albumin"},
    )
    stage_exclusions = exclusion_stage_ids(summary.stage_ids)
    eligible_ids = {
        item.patient_id
        for split in ("train", "val", "test")
        for item in dataset.prepared_data.items_by_split.get(split, ())
    }

    assert summary.stage_ids[-1] == eligible_ids == {"keep_train", "keep_val"}
    assert summary.inclusion_box_labels == (
        "Females diagnosed with BC at Ahus\n\n1.12.1998 - 08.06.2026",
        "No prior BC surgery",
        "No distant recurrence before BC diagnosis",
        "No distant recurrence ≤ 29 days after BC diagnosis",
        "Any blood tests > 29 days after BC diagnosis",
        "Blood tests before (competing) event or censoring",
        "Blood tests > 29 days after BC diagnosis",
    )
    assert stage_exclusions == (
        frozenset({"prior_bc"}),
        frozenset({"pre_recurrence"}),
        frozenset({"early_recurrence"}),
        frozenset({"no_blood_after_min"}),
        frozenset({"no_blood_before_recurrence", "no_valid_landmark"}),
        frozenset(),
    )
    assert summary.series_counts == ((1, 0, 0, 0, 1), *((0, 0, 0, 0, 1),) * 4, (0, 0, 0, 0, 2))
    assert summary.split_box_counts == (
        "2016 - 2016\nN = 1\nblood tests = 1",
        "2020 - 2020\nN = 1\nblood tests = 1",
        "N = 0\nblood tests = 0",
    )
    assert summary.blood_test_names == ("Albumin",)
    assert summary.blood_test_counts == (2,)


def test_dataset_statistics_applies_bc_window_as_first_blood_test_step(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("keep", 0, test_name="A", result=1.0, unit="u"),
                blood_row("outside_bc_window", 40, test_name="A", result=1.0, unit="u"),
            ],
            [
                endpoint_row("keep", diagnosis_year=2016, last_entry_days=200),
                endpoint_row("outside_bc_window", diagnosis_year=2020, last_entry_days=200),
            ],
            bc_diagnosis_rows=[
                bc_diagnosis_row("keep", 0),
                bc_diagnosis_row("outside_bc_window", 0),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "bc_diagnosis_window_days_before": 7,
        "bc_diagnosis_window_days_after": 7,
        "min_landmark_day": 0,
    }

    summary = build_dataset_statistics_summary(PatientBloodTestDataset(cfg).config)
    stage_exclusions = exclusion_stage_ids(summary.stage_ids)

    assert summary.stage_ids[-5] == {"keep", "outside_bc_window"}
    assert summary.stage_ids[-4] == {"keep"}
    assert summary.inclusion_box_labels[-3] == "Any blood tests on or after BC diagnosis"
    assert summary.stage_ids[-2] == {"keep"}
    assert summary.stage_ids[-1] == {"keep"}
    assert stage_exclusions[-4] == {"outside_bc_window"}


def test_dataset_statistics_applies_min_diagnosis_year_first(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("old", 0, test_name="A", result=1.0, unit="u"),
                blood_row("new", 0, test_name="A", result=1.0, unit="u"),
            ],
            [
                endpoint_row("old", diagnosis_year=2010, last_entry_days=200),
                endpoint_row("new", diagnosis_year=2015, last_entry_days=200),
            ],
            split_by_patient={"old": "train", "new": "train"},
            bc_diagnosis_rows=[
                bc_diagnosis_row("old", 0),
                bc_diagnosis_row("new", 0),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "min_diagnosis_year": 2012,
    }

    summary = build_dataset_statistics_summary(PatientBloodTestDataset(cfg).config)
    stage_exclusions = exclusion_stage_ids(summary.stage_ids)

    assert summary.inclusion_box_labels[:2] == (
        "Females diagnosed with BC at Ahus\n\n1.12.1998 - 08.06.2026",
        "Initial BC diagnosis \u2265 2012",
    )
    assert summary.exclusion_box_labels[0] == "Initial BC diagnosis before 2012"
    assert "Initial BC diagnosis before 2012" not in summary.series_labels
    assert summary.stage_ids[0] == {"old", "new"}
    assert summary.stage_ids[1] == {"new"}
    assert stage_exclusions[0] == {"old"}
    assert summary.diagnosis_years == (2012, 2013, 2014, 2015)
    assert summary.series_counts[0] == (0, 0, 0, 1)


def test_dataset_statistics_splits_other_cancer_around_bc_diagnosis(
    tmp_path: Path,
) -> None:
    patient_ids = (
        "old",
        "pre_diagnosis_other_cancer",
        "pre_diagnosis_recurrence",
        "early_other_cancer",
        "early_recurrence",
        "included",
    )
    cfg = {
        **config(
            tmp_path,
            [
                blood_row(patient_id, 40, test_name="A", unit="u")
                for patient_id in patient_ids
            ],
            [
                endpoint_row("old", diagnosis_year=2003, last_entry_days=100),
                endpoint_row("pre_diagnosis_other_cancer", last_entry_days=100),
                endpoint_row(
                    "pre_diagnosis_recurrence",
                    diagnosis_to_ddfs=-1,
                    ddfs_code="C780",
                    last_entry_days=100,
                ),
                endpoint_row("early_other_cancer", last_entry_days=100),
                endpoint_row(
                    "early_recurrence",
                    diagnosis_to_ddfs=30,
                    ddfs_code="C780",
                    last_entry_days=100,
                ),
                endpoint_row("included", last_entry_days=100),
            ],
            split_by_patient={patient_id: "train" for patient_id in patient_ids},
            clinical_rows=[
                clinical_row("pre_diagnosis_other_cancer", -1, code="C44.9"),
                clinical_row("early_other_cancer", 30, code="C44.9"),
            ],
        ),
        "selected_blood_tests": ["A|||u"],
        "min_diagnosis_year": 2004,
        "min_landmark_day": 31,
        "exclude_and_censor_other_cancer_patients": True,
        "other_cancer_code_filter_regex": r"C44\d*",
    }

    summary = build_dataset_statistics_summary(PatientBloodTestDataset(cfg).config)
    stage_exclusions = exclusion_stage_ids(summary.stage_ids)

    assert summary.exclusion_box_labels[:5] == (
        "Initial BC diagnosis before 2004",
        "Other cancer before BC diagnosis",
        "Distant recurrence before BC diagnosis",
        "Other cancer ≤ 30 days after BC diagnosis",
        "Distant recurrence ≤ 30 days after BC diagnosis",
    )
    assert stage_exclusions[:5] == (
        {"old"},
        {"pre_diagnosis_other_cancer"},
        {"pre_diagnosis_recurrence"},
        {"early_other_cancer"},
        {"early_recurrence"},
    )


def test_dataset_statistics_adds_minimum_test_count_before_final_measurement_cutoff(
    tmp_path: Path,
) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("keep", 40, test_name="A", result=1.0, unit="u"),
                blood_row("keep", 40, test_name="B", result=2.0, unit="u"),
                blood_row("too_missing", 40, test_name="A", result=1.0, unit="u"),
                blood_row("old", 40, test_name="A", result=1.0, unit="u"),
                blood_row("old", 40, test_name="B", result=2.0, unit="u"),
            ],
            [
                endpoint_row("keep", diagnosis_year=2015, last_entry_days=200),
                endpoint_row("too_missing", diagnosis_year=2015, last_entry_days=200),
                endpoint_row("old", diagnosis_year=2010, last_entry_days=200),
            ],
            split_by_patient={
                "keep": "train",
                "too_missing": "train",
                "old": "train",
            },
            bc_diagnosis_rows=[
                bc_diagnosis_row("keep", 40),
                bc_diagnosis_row("too_missing", 40),
                bc_diagnosis_row("old", 40),
            ],
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
        "restrict_bloodtests_to_bc_diagnosis_window": True,
        "bc_diagnosis_window_days_before": 0,
        "bc_diagnosis_window_days_after": 0,
        "min_blood_test_types_per_day": 2,
        "min_diagnosis_year": 2012,
    }

    summary = build_dataset_statistics_summary(PatientBloodTestDataset(cfg).config)
    stage_exclusions = exclusion_stage_ids(summary.stage_ids)

    assert summary.inclusion_box_labels[3:] == (
        "Blood tests within the configured C50 diagnosis window",
        "Any blood tests on or after BC diagnosis",
        "Blood tests before (competing) event or censoring",
        "Patients with at least one qualifying day:\ndays on or after BC diagnosis\nwith > 1 different blood tests",
        "Blood tests on or after BC diagnosis",
        "Days with > 1 different blood tests",
    )
    assert summary.exclusion_box_labels[2:] == (
        "Blood test unrelated to BC checkup",
        "No blood tests on or after BC diagnosis",
        "Blood tests on or after (competing) event or censoring¹",
        "No days on or after BC diagnosis\nwith > 1 different blood tests",
        "Blood tests before BC diagnosis",
        "Days with ≤ 1 different blood tests",
    )
    assert summary.stage_ids[-4] == {"keep", "too_missing"}
    assert summary.stage_ids[-1] == {"keep"}
    assert stage_exclusions[-3] == {"too_missing"}
    assert stage_exclusions[-1] == frozenset()
