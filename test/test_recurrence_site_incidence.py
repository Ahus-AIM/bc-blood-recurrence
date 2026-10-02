from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data import PatientBloodTestDataset
from src.data.recurrence_site_cumulative_incidence import (
    build_recurrence_site_cumulative_incidence,
)
from src.data.recurrence_site_incidence import (
    RecurrenceSiteRule,
    build_first_recurrence_site_kaplan_meier,
    load_item_endpoint_site_rows,
    site_label,
    survival_durations,
)
from test.test_patient_bloodtest import (
    blood_row,
    config,
    endpoint_row,
    single_patient_outcome_dataset,
)


def test_recurrence_site_rule_matching_accepts_dotted_icd_codes() -> None:
    label = site_label("C78.7", (RecurrenceSiteRule("Liver", r"C787\d*"),), "Other")

    assert label == "Liver"


def test_recurrence_site_incidence_uses_all_sites_on_first_recurrence_day() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 100, "C780"),
            endpoint_site_row("P0", 100, "C787"),
            endpoint_site_row("P1", None, ""),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (
            RecurrenceSiteRule("Lung", r"C780\d*"),
            RecurrenceSiteRule("Liver", r"C787\d*"),
            RecurrenceSiteRule("Bone", r"C795\d*"),
        ),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        overall_label="Any",
        other_label="Other",
    )

    one_year = incidence[incidence["years_since_diagnosis"] == 1.0]
    incidence_by_site = dict(zip(one_year["site"], one_year["cumulative_incidence_percent"], strict=True))
    lung = incidence[incidence["site"] == "Lung"]

    assert incidence_by_site == {
        "Lung": 50.0,
        "Liver": 50.0,
        "Bone": 0.0,
        "Other": 0.0,
        "Any": 50.0,
    }
    assert tuple(lung["years_since_diagnosis"]) == (0.0, 100.0 / 365.0, 1.0)
    assert lung.iloc[1]["confidence_interval_lower_percent"] <= 50.0
    assert lung.iloc[1]["confidence_interval_upper_percent"] >= 50.0


def test_recurrence_site_survival_uses_panel_d_site_memberships() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 100, "C780"),
            endpoint_site_row("P0", 100, "C787"),
        ]
    )

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (
            RecurrenceSiteRule("Lung", r"C780\d*"),
            RecurrenceSiteRule("Liver", r"C787\d*"),
            RecurrenceSiteRule("Bone", r"C795\d*"),
        ),
        max_years_since_origin=1,
        min_days_since_diagnosis=0,
        time_origin="recurrence",
        other_label="Other",
        overall_label="Any",
    )

    assert set(survival["site"]) == {"Lung", "Liver", "Any"}


def test_recurrence_site_incidence_keeps_unmatched_codes_as_other() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 50, "C781"),
            endpoint_site_row("P1", None, ""),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        overall_label="Any",
        other_label="Other",
    )

    one_year = incidence[incidence["years_since_diagnosis"] == 1.0]
    incidence_by_site = dict(zip(one_year["site"], one_year["cumulative_incidence_percent"], strict=True))

    assert incidence_by_site["Lung"] == 0.0
    assert incidence_by_site["Any"] == 50.0
    assert incidence_by_site["Other"] == 50.0


def test_recurrence_site_incidence_adjusts_for_unequal_right_censoring() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 200, "C780", death_day=None, last_observation_day=400),
            endpoint_site_row("P1", None, "", death_day=None, last_observation_day=100),
            endpoint_site_row("P2", 300, "C780", death_day=None, last_observation_day=400),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        overall_label="Any",
        other_label="Other",
    )

    lung = incidence[incidence["site"] == "Lung"]
    day_200 = lung[lung["years_since_diagnosis"] == 200.0 / 365.0].iloc[0]
    day_300 = lung[lung["years_since_diagnosis"] == 300.0 / 365.0].iloc[0]

    assert day_200["cumulative_incidence_percent"] == pytest.approx(50.0)
    assert day_200["patients_at_risk"] == 2
    assert day_200["target_events"] == 1
    assert day_300["cumulative_incidence_percent"] == pytest.approx(100.0)


def test_recurrence_site_incidence_treats_death_as_competing() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", None, "", death_day=100, last_observation_day=400),
            endpoint_site_row("P1", 200, "C780", death_day=None, last_observation_day=400),
            endpoint_site_row("P2", None, "", death_day=None, last_observation_day=300),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        overall_label="Any",
        other_label="Other",
    )

    lung = incidence[incidence["site"] == "Lung"]
    death = lung[lung["years_since_diagnosis"] == 100.0 / 365.0].iloc[0]
    recurrence = lung[lung["years_since_diagnosis"] == 200.0 / 365.0].iloc[0]

    assert death["competing_events"] == 1
    assert recurrence["cumulative_incidence_percent"] == pytest.approx(100.0 / 3.0)


def test_site_incidence_treats_an_earlier_other_site_as_competing() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 100, "C780", death_day=None),
            endpoint_site_row("P0", 200, "C787", death_day=None),
            endpoint_site_row("P1", None, "", death_day=None),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (
            RecurrenceSiteRule("Lung", r"C780\d*"),
            RecurrenceSiteRule("Liver", r"C787\d*"),
        ),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        overall_label="Any",
        other_label="Other",
    )

    liver = incidence[incidence["site"] == "Liver"]
    first_recurrence = liver[liver["years_since_diagnosis"] == 100.0 / 365.0].iloc[0]

    assert first_recurrence["competing_events"] == 1
    assert liver["target_events"].sum() == 0


def test_recurrence_site_incidence_preserves_same_day_outcome_precedence() -> None:
    other_cancer_row = endpoint_site_row("other", 100, "C780")
    other_cancer_row.update(
        right_censoring_day=100,
        right_censoring_reason="other_cancer",
    )
    death_row = endpoint_site_row("death", 100, "C780")
    death_row.update(
        right_censoring_day=100,
        right_censoring_reason="death",
    )
    censored_row = endpoint_site_row("censored", None, "", death_day=None)
    censored_row.update(
        right_censoring_day=100,
        right_censoring_reason="last_observation",
    )

    incidence = build_recurrence_site_cumulative_incidence(
        pd.DataFrame([other_cancer_row, death_row, censored_row]),
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        random_seed=4,
        overall_label="Any",
        other_label="Other",
    )

    lung = incidence[incidence["site"] == "Lung"]
    day_100 = lung[lung["years_since_diagnosis"] == 100.0 / 365.0].iloc[0]

    assert day_100["target_events"] == 1
    assert day_100["competing_events"] == 1
    assert day_100["censorings"] == 1
    assert day_100["cumulative_incidence_percent"] == pytest.approx(100.0 / 3.0)


def test_recurrence_site_incidence_stops_at_minimum_risk_set() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", None, "", death_day=None, last_observation_day=100),
            endpoint_site_row("P1", 200, "C780", death_day=None, last_observation_day=300),
            endpoint_site_row("P2", None, "", death_day=None, last_observation_day=300),
        ]
    )

    incidence = build_recurrence_site_cumulative_incidence(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_diagnosis=1,
        min_days_since_diagnosis=0,
        min_patients_at_risk=2,
        overall_label="Any",
        other_label="Other",
    )

    lung = incidence[incidence["site"] == "Lung"]

    assert tuple(lung["years_since_diagnosis"])[-1] == 200.0 / 365.0
    assert tuple(lung["patients_at_risk"])[-1] == 2


def test_first_recurrence_site_survival_keeps_unmatched_first_site_as_other() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 50, "C781"),
            endpoint_site_row("P1", None, ""),
        ]
    )

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_origin=1,
        min_days_since_diagnosis=0,
        time_origin="recurrence",
        other_label="Other",
    )

    assert set(survival["site"]) == {"Other"}


def test_first_recurrence_site_survival_can_include_overall_any() -> None:
    endpoint_rows = pd.DataFrame(
        [
            endpoint_site_row("P0", 50, "C781"),
            endpoint_site_row("P1", None, ""),
        ]
    )

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_origin=1,
        min_days_since_diagnosis=0,
        time_origin="recurrence",
        other_label="Other",
        overall_label="Any",
    )

    assert set(survival["site"]) == {"Any", "Other"}


def test_first_recurrence_site_survival_uses_event_time_steps() -> None:
    endpoint_rows = endpoint_rows_with_two_death_times()

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_origin=2,
        min_days_since_diagnosis=0,
        time_origin="recurrence",
        other_label="Other",
    )

    lung_survival = survival[survival["site"] == "Lung"]
    years = tuple(lung_survival["years_since_origin"])
    percentages = tuple(lung_survival["survival_percent"])

    assert years == (0.0, 183.0 / 365.0, 1.0, 2.0)
    assert percentages[0] == 100.0
    assert percentages[1:] == pytest.approx((200.0 / 3.0, 100.0 / 3.0, 100.0 / 3.0), rel=1e-12, abs=1e-12)


def test_first_recurrence_site_survival_censors_when_too_few_at_risk() -> None:
    endpoint_rows = endpoint_rows_with_two_death_times()

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_origin=2,
        min_days_since_diagnosis=0,
        min_patients_at_risk=3,
        time_origin="recurrence",
        other_label="Other",
    )

    lung_survival = survival[survival["site"] == "Lung"]

    assert tuple(lung_survival["years_since_origin"]) == (0.0, 183.0 / 365.0)
    assert tuple(lung_survival["survival_percent"]) == pytest.approx((100.0, 200.0 / 3.0), rel=1e-12, abs=1e-12)


def test_first_recurrence_site_survival_omits_curve_when_initial_at_risk_too_small() -> None:
    endpoint_rows = endpoint_rows_with_two_death_times()

    survival = build_first_recurrence_site_kaplan_meier(
        endpoint_rows,
        (RecurrenceSiteRule("Lung", r"C780\d*"),),
        max_years_since_origin=2,
        min_days_since_diagnosis=0,
        min_patients_at_risk=4,
        time_origin="recurrence",
        other_label="Other",
    )

    assert survival.empty


def test_item_endpoint_rows_use_prepared_other_cancer_censoring(tmp_path: Path) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        recurrence_day=50,
        recurrence_code="C780",
        other_cancer_day=60,
        other_cancer_enabled=True,
    )
    endpoint_rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    row = endpoint_rows.iloc[0]

    assert "other_cancer_day" not in endpoint_rows
    assert row["days_since_diagnosis"] == 50
    assert row["right_censoring_day"] == 60
    assert row["right_censoring_reason"] == "other_cancer"


def test_item_endpoint_rows_keep_all_sites_on_prepared_first_recurrence_day(
    tmp_path: Path,
) -> None:
    dataset_config = config(
        tmp_path,
        [blood_row("P0", 40, test_name="A", unit="u")],
        [
            endpoint_row(
                "P0",
                diagnosis_to_ddfs=[50, 50, 70],
                ddfs_code=["C780", "C787", "C795"],
                last_entry_days=100,
            )
        ],
    )
    dataset_config.update(
        selected_blood_tests=["A|||u"],
        min_landmark_day=31,
        input_recurrence_code_filter_regex=r"C780\d*",
    )
    dataset = PatientBloodTestDataset(dataset_config)

    endpoint_rows = load_item_endpoint_site_rows(dataset.items, dataset.config)

    assert set(zip(endpoint_rows["days_since_diagnosis"], endpoint_rows["code"], strict=True)) == {
        (50, "C780"),
        (50, "C787"),
    }


def test_item_endpoint_rows_use_post_death_normalized_recurrence_day(
    tmp_path: Path,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        recurrence_day=80,
        recurrence_code="C780",
        death_day=70,
    )

    endpoint_rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    row = endpoint_rows.iloc[0]

    assert row["days_since_diagnosis"] == 70
    assert row["right_censoring_day"] == 70
    assert row["right_censoring_reason"] == "death"
    durations, events = survival_durations(endpoint_rows, "recurrence")
    assert durations.tolist() == [0.0]
    assert events.tolist() == [True]


def test_item_endpoint_rows_exclude_recurrence_censored_by_other_cancer(
    tmp_path: Path,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path,
        recurrence_day=50,
        recurrence_code="C780",
        other_cancer_day=50,
        other_cancer_enabled=True,
    )

    endpoint_rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    row = endpoint_rows.iloc[0]

    assert pd.isna(row["days_since_diagnosis"])
    assert row["right_censoring_day"] == 50
    assert row["right_censoring_reason"] == "other_cancer"


def test_item_endpoint_rows_do_not_censor_follow_up_at_first_recurrence(
    tmp_path: Path,
) -> None:
    dataset_config = config(
        tmp_path,
        [blood_row("P0", 40, test_name="A", unit="u")],
        [
            endpoint_row(
                "P0",
                diagnosis_to_ddfs=50,
                ddfs_code="C780",
                last_entry_days=100,
            )
        ],
    )
    dataset_config.update(
        selected_blood_tests=["A|||u"],
        min_landmark_day=31,
        endpoint_ddfs_code_filter_regex=r"C787\d*",
    )
    dataset = PatientBloodTestDataset(dataset_config)
    assert dataset.items[0].right_censoring_reason == "non_endpoint_recurrence"

    endpoint_rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    row = endpoint_rows.iloc[0]

    assert row["days_since_diagnosis"] == 50
    assert row["right_censoring_day"] == 100
    assert row["right_censoring_reason"] == "last_observation"


def endpoint_site_row(
    patient_id: str,
    day: int | None,
    code: str,
    *,
    death_day: int | None = 500,
    last_observation_day: int = 600,
) -> dict[str, object]:
    return {
        "patient_id": patient_id,
        "diagnosis_year": 2020,
        "days_since_diagnosis": day,
        "code": code,
        "description": "",
        "death_day": death_day,
        "last_observation_day": last_observation_day,
    }


def endpoint_rows_with_two_death_times() -> pd.DataFrame:
    return pd.DataFrame(
        [
            endpoint_site_row("P0", 100, "C780", death_day=283, last_observation_day=830),
            endpoint_site_row("P1", 100, "C780", death_day=None, last_observation_day=830),
            endpoint_site_row("P2", 100, "C780", death_day=465, last_observation_day=830),
        ]
    )
