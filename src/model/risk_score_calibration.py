from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

from src.data import PatientLandmarkTable

TRAINING_RISK_REFERENCE_FILENAME = "training_risk_score_reference.csv"


@dataclass(frozen=True)
class RiskScoreCalibration:
    """Map raw predictions to the unweighted training-landmark distribution.

    ``reference_scores`` contains one finite score per non-bootstrapped training
    landmark. A sorted, read-only copy preserves ties for right-inclusive ranks.
    """

    reference_scores: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        scores = np.asarray(self.reference_scores, dtype=float)
        if scores.ndim != 1 or scores.size == 0:
            raise ValueError("Risk-score calibration requires a non-empty one-dimensional reference.")
        if not bool(np.all(np.isfinite(scores))):
            raise ValueError("Risk-score calibration reference values must be finite.")
        sorted_scores = np.sort(scores)
        sorted_scores.setflags(write=False)
        object.__setattr__(self, "reference_scores", sorted_scores)

    def percentiles(self, risk_scores: npt.ArrayLike) -> npt.NDArray[np.float64]:
        """Return right-inclusive empirical percentiles on the 0–100 scale."""
        scores = np.asarray(risk_scores, dtype=float)
        if not bool(np.all(np.isfinite(scores))):
            raise ValueError("Risk scores converted to percentiles must be finite.")
        ranks = np.searchsorted(self.reference_scores, scores, side="right")
        return ranks.astype(float) * (100.0 / len(self.reference_scores))


def load_risk_score_calibration(training_path: Path) -> RiskScoreCalibration | str:
    """Load the saved reference or explain why percentile display is unavailable."""
    if not training_path.is_file():
        return f"Training percentile calibration scores not found: {training_path}"
    try:
        training_scores = pd.read_csv(training_path, float_precision="round_trip")
        return RiskScoreCalibration(training_scores["risk_score"].to_numpy(dtype=float))
    except (KeyError, OSError, TypeError, ValueError) as error:
        return f"Training percentile calibration is invalid: {error}"


def write_training_risk_reference(
    table: PatientLandmarkTable,
    risk_scores: npt.ArrayLike,
    output_dir: Path,
) -> dict[str, object]:
    """Save one unweighted prediction per training landmark with its row identity."""
    scores = np.asarray(risk_scores, dtype=float)
    RiskScoreCalibration(scores)
    frame = pd.DataFrame(
        {
            "patient_id": table.patient_ids,
            "landmark_day": table.landmark_days,
            "risk_score": scores,
        }
    )
    if frame.duplicated(["patient_id", "landmark_day"]).any():
        raise ValueError("Training risk reference requires unique non-bootstrapped patient landmarks.")
    output_path = output_dir / TRAINING_RISK_REFERENCE_FILENAME
    frame.to_csv(output_path, index=False)
    return {
        "path": output_path.name,
        "landmark_count": len(frame),
        "patient_count": int(frame["patient_id"].nunique()),
    }
