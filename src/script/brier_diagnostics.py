from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Self

import pandas as pd

from src.model.brier_diagnostics import split_diagnostics

LOGGER = logging.getLogger(__name__)


@dataclass
class BrierDiagnostics:
    """Incrementally persist a cell's diagnostics, including failures.

    ``destination`` is the Brier model directory; ``values`` contains only
    aggregate diagnostics and the current stage, plus verbatim failure messages.
    Existing reports are replaced when entering a new calculation.
    """

    destination: Path
    values: dict[str, object] = field(default_factory=dict)

    def __enter__(self) -> Self:
        self.destination.mkdir(parents=True, exist_ok=True)
        (self.destination / "risk_set_diagnostics.csv").unlink(missing_ok=True)
        self.values = {
            "status": "running",
            "stage": "loading_models_and_inputs",
            "notes": [
                "Diagnostics describe unfiltered splits; washout/bootstrap results are in metadata.json.",
                "Score/probability quantiles describe landmarks without weighting.",
                "Complete-case means use patient-balanced weights and exclude unknown outcomes; censoring can bias them.",
                "Effective risk patients measures denominator concentration, not inferential sample size.",
                "Events on the landmark day are allowed; review day-zero event counts and washout results.",
                "Native Cox fitting retains the documented tied-event row-order sensitivity.",
                "Open F6: the audited native Cox gradients mishandle unequal row weights; centering does not fix fitting.",
                "A numerically valid result does not establish calibration or clinical validity.",
            ],
        }
        self.save()
        return self

    def __exit__(
        self, _error_type: type[BaseException] | None, error: BaseException | None, _traceback: TracebackType | None,
    ) -> None:
        if error is not None:
            self.fail(error)
            return
        invalid = self.values.get("invalid_prediction_count", 0)
        self.values["status"] = "invalid_predictions" if invalid else "complete"
        self.save()

    def fail(self, error: BaseException) -> None:
        self.values.update(status="failed", error_type=type(error).__name__, failure_reason=str(error))
        self.save()

    def record(self, key: str, value: object) -> None:
        self.values[key] = value
        self.save()

    def save(self) -> None:
        path = self.destination / "diagnostics.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(self.values, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(path)

    def predictions(self, predictions: pd.DataFrame, invalid_hazards: pd.DataFrame) -> None:
        summaries = {
            split: split_diagnostics(predictions.loc[predictions["split"] == split])
            for split in ("train", "val", "test")
        }
        self.values.update(
            splits=summaries,
            invalid_hazard_increment_count=len(invalid_hazards),
            invalid_prediction_count=sum(summary["invalid_probability_count"] for summary in summaries.values()),
        )
        self.save()
        for split, summary in summaries.items():
            LOGGER.info(
                "%s [%s]: known outcomes=%s/%s, invalid probabilities=%s; see diagnostics.json",
                self.destination, split, summary["known_outcome_count"], summary["landmark_count"],
                summary["invalid_probability_count"],
            )


def record_brier_failure(destination: Path, error: Exception) -> None:
    """Keep earlier diagnostics when validation or washout evaluation fails."""
    path = destination / "diagnostics.json"
    values = json.loads(path.read_text()) if path.is_file() else {"stage": "validating_saved_model"}
    if values["stage"] == "finished":
        values["stage"] = "evaluating_washouts"
    BrierDiagnostics(destination, values).fail(error)
