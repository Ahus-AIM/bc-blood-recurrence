from pathlib import Path

import pytest

from src.data import PatientBloodTestDataset
from test.test_patient_bloodtest import blood_row, config, endpoint_row


@pytest.mark.parametrize(
    ("minimum_day", "expected_days"),
    [(30, [30, 31, 60]), (31, [31, 60])],
)
def test_min_landmark_day_filters_early_landmarks(
    tmp_path: Path,
    minimum_day: int,
    expected_days: list[int],
) -> None:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("P0", 0, test_name="A", result=1.0, unit="u"),
                blood_row("P0", 29, test_name="A", result=1.5, unit="u"),
                blood_row("P0", 30, test_name="A", result=2.0, unit="u"),
                blood_row("P0", 31, test_name="A", result=3.0, unit="u"),
                blood_row("P0", 60, test_name="A", result=4.0, unit="u"),
            ],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "min_landmark_day": minimum_day,
    }

    table = PatientBloodTestDataset(cfg).to_landmark_table(bootstrap=False)

    assert table.landmark_days.tolist() == expected_days
    assert table.metadata["min_landmark_day"] == minimum_day


def test_min_landmark_day_discards_patients_without_late_landmarks(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 0, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "min_landmark_day": 31,
    }

    dataset = PatientBloodTestDataset(cfg)

    assert len(dataset) == 0
    assert dataset.discarded_patient_items_by_id["P0"].discard_reasons == ("no valid landmark days",)


def test_min_landmark_day_rejects_negative_config(tmp_path: Path) -> None:
    cfg = {
        **config(
            tmp_path,
            [blood_row("P0", 0, test_name="A", result=1.0, unit="u")],
            [endpoint_row("P0", last_entry_days=100)],
        ),
        "min_landmark_day": -1,
    }

    with pytest.raises(ValueError, match="min_landmark_day"):
        PatientBloodTestDataset(cfg)
