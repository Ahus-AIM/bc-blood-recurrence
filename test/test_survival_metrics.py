import numpy as np
import pytest

from src.model.survival_metrics import harrell_c_index_excluding_same_patient


@pytest.mark.parametrize(
    ("event_risk", "censored_risk", "expected"),
    ((1.0, 0.0, 1.0), (0.0, 1.0, 0.0), (1.0, 1.0, 0.5)),
)
def test_same_duration_event_and_censor_are_comparable(
    event_risk: float,
    censored_risk: float,
    expected: float,
) -> None:
    c_index = harrell_c_index_excluding_same_patient(
        durations=np.asarray([30, 30]),
        events=np.asarray([True, False]),
        risk_scores=np.asarray([event_risk, censored_risk]),
        patient_ids=np.asarray(["event", "censored"]),
    )

    assert c_index == expected


def test_same_duration_pair_from_one_patient_is_not_comparable() -> None:
    c_index = harrell_c_index_excluding_same_patient(
        durations=np.asarray([30, 30]),
        events=np.asarray([True, False]),
        risk_scores=np.asarray([1.0, 0.0]),
        patient_ids=np.asarray(["patient", "patient"]),
    )

    assert np.isnan(c_index)
