from __future__ import annotations

import numpy as np
import pytest

from src.data import PatientLandmarkTable
from src.model import ebm_fitting
from src.model.ebm_bags import build_patient_grouped_bag_plan, validation_count_from_size
from src.model.ebm_fitting import fit_ebm_model


def test_patient_grouped_outer_bags_keep_patient_rows_together() -> None:
    table = landmark_table(
        ["p0", "p0", "p1", "p1", "p2", "p3", "p3", "p4"],
        [False, True, False, False, True, False, False, False],
    )

    plan = build_patient_grouped_bag_plan(
        table,
        {
            "outer_bags": 5,
            "validation_size": 0.4,
            "patient_grouped_bag_max_attempts": 1_000,
            "random_state": 11,
        },
    )

    assert plan.bags.shape == (8, 5)
    assert plan.metadata["validation_patient_count"] == 2
    assert plan.metadata["event_patient_count"] == 2
    for bag_index in range(plan.bags.shape[1]):
        assert set(np.unique(plan.bags[:, bag_index])) == {-1, 1}
        assert split_event_patients(table, plan.bags[:, bag_index], -1) >= 1
        assert split_event_patients(table, plan.bags[:, bag_index], 1) >= 1
        for patient_id in np.unique(table.patient_ids):
            rows = table.patient_ids == patient_id
            assert len(set(plan.bags[rows, bag_index].tolist())) == 1


def test_patient_grouped_outer_bags_are_deterministic() -> None:
    table = landmark_table(
        ["p0", "p0", "p1", "p2", "p3", "p3"],
        [False, True, True, False, False, False],
    )
    config = {
        "outer_bags": 3,
        "validation_size": 1,
        "patient_grouped_bag_max_attempts": 1_000,
        "random_state": 23,
    }

    first = build_patient_grouped_bag_plan(table, config)
    second = build_patient_grouped_bag_plan(table, config)

    assert np.array_equal(first.bags, second.bags)
    assert first.metadata == second.metadata


def test_ebm_fit_uses_supplied_patient_grouped_bag_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    table = landmark_table(
        ["p0", "p0", "p1", "p2"],
        [False, True, True, False],
    )
    config = {
        "outer_bags": 2,
        "validation_size": 1,
        "patient_grouped_bag_max_attempts": 1_000,
        "random_state": 23,
    }
    bag_plan = build_patient_grouped_bag_plan(table, config)
    captured: dict[str, object] = {}

    class Model:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def fit(
            self,
            _features: np.ndarray,
            _targets: np.ndarray,
            *,
            sample_weight: np.ndarray,
            bags: np.ndarray,
        ) -> None:
            captured["sample_weight"] = sample_weight
            captured["bags"] = bags

    monkeypatch.setattr(ebm_fitting, "ExplainableBoostingRegressor", Model)
    monkeypatch.setattr(
        ebm_fitting,
        "build_patient_grouped_bag_plan",
        lambda *_args: pytest.fail("rebuilt supplied bag plan"),
    )

    _model, metadata = fit_ebm_model(
        table,
        table.features,
        config,
        patient_grouped_bag_plan=bag_plan,
    )

    assert captured["sample_weight"] is table.sample_weights
    assert captured["bags"] is bag_plan.bags
    assert metadata["patient_grouped_outer_bags"] is bag_plan.metadata


def test_patient_grouped_outer_bags_reject_impossible_event_split() -> None:
    table = landmark_table(["p0", "p0", "p1", "p2"], [False, True, False, False])

    with pytest.raises(ValueError, match="at least 2 event patients"):
        build_patient_grouped_bag_plan(
            table,
            {
                "outer_bags": 2,
                "validation_size": 1,
                "patient_grouped_bag_max_attempts": 100,
                "random_state": 5,
            },
        )


def test_validation_size_fraction_uses_ceil_but_exact_count_stays_exact() -> None:
    assert validation_count_from_size(0.15, 103) == 16
    assert validation_count_from_size(20, 103) == 20
    with pytest.raises(ValueError, match="leave at least one training patient"):
        validation_count_from_size(103, 103)


def landmark_table(patient_ids: list[str], events: list[bool]) -> PatientLandmarkTable:
    row_count = len(patient_ids)
    durations = np.arange(1, row_count + 1, dtype=np.int32)
    event_array = np.asarray(events, dtype=bool)
    return PatientLandmarkTable(
        features=np.zeros((row_count, 1), dtype=np.float32),
        signed_durations=np.where(event_array, durations, -durations).astype(np.float64),
        durations=durations,
        events=event_array,
        sample_weights=np.ones(row_count, dtype=np.float64),
        feature_names=("marker",),
        patient_ids=np.asarray(patient_ids, dtype=object),
        landmark_days=np.arange(row_count, dtype=np.int32),
        prior_non_endpoint_recurrence=np.zeros(row_count, dtype=bool),
        metadata={},
    )


def split_event_patients(table: PatientLandmarkTable, bag: np.ndarray, split_value: int) -> int:
    patient_count = 0
    for patient_id in np.unique(table.patient_ids):
        rows = table.patient_ids == patient_id
        if bag[rows][0] == split_value and bool(np.any(table.events[rows])):
            patient_count += 1
    return patient_count
