from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from src.data.patient_preprocessing import load_blood_tests

MEASUREMENT_COLUMNS = ("patient_id", "days_since_diagnosis", "test_key", "result", "source_index")


@dataclass
class RecurrenceMeasurementHistory:
    """Source-row references for early and sparse measurements used only by recurrence reports.

    ``source_path`` and ``source_indices`` identify already cleaned, deduplicated
    rows surviving patient selection and the study observation window, before
    model-only early-row and sparse-day exclusions. ``source_signature`` guards against
    file replacement after preparation. Values are loaded only for reporting;
    ``materialized`` shares one read within a report and releases it afterwards.
    """

    source_path: Path
    source_indices: NDArray[np.int64]
    source_signature: tuple[int, int]
    _frame: pd.DataFrame | None = field(default=None, init=False, repr=False)

    @classmethod
    def from_measurements(cls, source_path: Path, measurements: pd.DataFrame) -> RecurrenceMeasurementHistory:
        return cls(
            source_path,
            measurements["source_index"].to_numpy(dtype=np.int64, copy=True),
            file_signature(source_path),
        )

    def read(self) -> pd.DataFrame:
        """Reload the exact retained rows; fail if the underlying source has changed."""
        if file_signature(self.source_path) != self.source_signature:
            raise ValueError("Blood-test source changed after cohort preparation; prepare the cohort again.")
        if self._frame is not None:
            return self._frame
        measurements, _, _ = load_blood_tests(self.source_path)
        history = measurements.loc[
            measurements["source_index"].isin(self.source_indices), list(MEASUREMENT_COLUMNS)
        ].copy()
        if len(history) != len(self.source_indices) or file_signature(self.source_path) != self.source_signature:
            raise ValueError("Recurrence history no longer matches the prepared blood-test source.")
        return history

    @contextmanager
    def materialized(self) -> Iterator[None]:
        previous = self._frame
        self._frame = self.read()
        try:
            yield
        finally:
            self._frame = previous


def file_signature(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_size, stat.st_mtime_ns
