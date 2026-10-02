from __future__ import annotations

from pathlib import Path

import pytest

from src.data.cohort_figure import survival_counts_at_years
from src.data.recurrence_site_incidence import (
    RecurrenceSiteRule,
    build_first_recurrence_site_kaplan_meier,
    load_item_endpoint_site_rows,
    survival_durations,
)
from test.test_patient_bloodtest import single_patient_outcome_dataset


@pytest.mark.parametrize(
    ("death_day", "expected_duration", "expected_event"),
    [(80, 30.0, True), (60, 10.0, True), (450, 400.0, True), (None, 50.0, False)],
)
def test_survival_continues_after_another_cancer(
    tmp_path: Path, death_day: int | None, expected_duration: float, expected_event: bool,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path, recurrence_day=50, other_cancer_day=60,
        other_cancer_enabled=True, death_day=death_day,
    )
    rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    durations, events = survival_durations(rows, "recurrence")

    assert rows["right_censoring_day"].tolist() == [60]
    assert rows["right_censoring_reason"].tolist() == ["other_cancer"]
    assert durations.tolist() == [expected_duration]
    assert events.tolist() == [expected_event]
    assert survival_counts_at_years(rows, (0, 1)) == (1, int(expected_duration >= 365))

    curves = build_first_recurrence_site_kaplan_meier(
        rows, (RecurrenceSiteRule("Liver", r"C787\d*"),),
        max_years_since_origin=2, min_days_since_diagnosis=0,
        time_origin="recurrence", other_label="Other", overall_label="Any",
    )
    for site in ("Any", "Liver"):
        curve = curves.loc[curves["site"] == site]
        if expected_event:
            assert curve.iloc[-1]["years_since_origin"] == pytest.approx(expected_duration / 365)
            assert curve.iloc[-1]["survival_percent"] == 0.0
        else:
            assert curve["survival_percent"].tolist() == [100.0]


@pytest.mark.parametrize("other_cancer_day", [45, 50])
def test_another_cancer_before_or_at_recurrence_still_prevents_survival_entry(
    tmp_path: Path, other_cancer_day: int,
) -> None:
    dataset = single_patient_outcome_dataset(
        tmp_path, recurrence_day=50, other_cancer_day=other_cancer_day,
        other_cancer_enabled=True, death_day=80,
    )
    rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    curves = build_first_recurrence_site_kaplan_meier(
        rows, (RecurrenceSiteRule("Liver", r"C787\d*"),),
        max_years_since_origin=1, min_days_since_diagnosis=0,
        time_origin="recurrence", other_label="Other", overall_label="Any",
    )

    assert rows["days_since_diagnosis"].isna().all()
    assert curves.empty
