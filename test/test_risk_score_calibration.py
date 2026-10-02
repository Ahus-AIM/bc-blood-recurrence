from pathlib import Path

import numpy as np
import pytest

from src.data import PatientBloodTestDataset
from src.model.risk_score_calibration import (
    TRAINING_RISK_REFERENCE_FILENAME,
    RiskScoreCalibration,
    load_risk_score_calibration,
    write_training_risk_reference,
)
from test.test_patient_bloodtest import blood_row, config, endpoint_row


def test_percentiles_preserve_ties_and_clamp_scores_outside_training_range() -> None:
    calibration = RiskScoreCalibration(np.asarray([3.0, 1.0, 1.0, 2.0]))

    np.testing.assert_array_equal(
        calibration.percentiles([-1.0, 1.0, 1.5, 2.0, 3.0, 4.0]),
        [0.0, 50.0, 50.0, 75.0, 100.0, 100.0],
    )
    assert not calibration.reference_scores.flags.writeable


def test_constant_reference_needs_no_risk_groups() -> None:
    calibration = RiskScoreCalibration(np.asarray([1.0, 1.0]))
    np.testing.assert_array_equal(calibration.percentiles([0.0, 1.0, 2.0]), [0.0, 100.0, 100.0])


@pytest.mark.parametrize("scores", [[], 1.0, [[1.0]], [np.nan], [np.inf]])
def test_invalid_reference_is_rejected(scores: object) -> None:
    with pytest.raises(ValueError, match="calibration"):
        RiskScoreCalibration(np.asarray(scores, dtype=float))


@pytest.mark.parametrize("score", [np.nan, np.inf, -np.inf])
def test_nonfinite_prediction_is_rejected(score: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        RiskScoreCalibration(np.asarray([1.0])).percentiles([score])


def test_saved_reference_preserves_float_precision_and_landmark_multiplicity(tmp_path: Path) -> None:
    dataset = PatientBloodTestDataset(
        config(
            tmp_path,
            [blood_row("P0", day, test_name="A", result=1.0, unit="u") for day in (0, 10, 20)],
            [endpoint_row("P0", last_entry_days=100)],
        )
    )
    table = dataset.to_landmark_table(bootstrap=False)
    score = 0.12345678901234566
    scores = np.asarray([score, score, np.nextafter(score, np.inf)])

    summary = write_training_risk_reference(table, scores, tmp_path)
    calibration = load_risk_score_calibration(tmp_path / TRAINING_RISK_REFERENCE_FILENAME)

    assert summary == {"path": TRAINING_RISK_REFERENCE_FILENAME, "landmark_count": 3, "patient_count": 1}
    assert isinstance(calibration, RiskScoreCalibration)
    np.testing.assert_array_equal(calibration.reference_scores, scores)
    np.testing.assert_allclose(calibration.percentiles(scores), [200 / 3, 200 / 3, 100])
