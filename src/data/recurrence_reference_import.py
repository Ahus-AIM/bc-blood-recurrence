from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.script.calculate_recurrence_reference_percentages import (
    GROUP_KEYS,
    RESULT_COLUMNS,
    reference_denominator,
    reference_percentages_and_status,
)

REFERENCE_COLUMNS = [
    "n_reference_patients",
    "outside_score_sum",
    "outside_reference_percent",
    "reference_status",
]


class ReferenceDataMismatchError(ValueError):
    """The returned percentages belong to a different measurement export or analysis."""


def import_reference_percentages(frame: pd.DataFrame, path: Path, expected_hash: str) -> pd.DataFrame:
    """Reject stale/inconsistent aggregates before attaching them to local statistical results."""
    remote = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "data_hash" in remote and not remote["data_hash"].eq(expected_hash).all():
        raise ReferenceDataMismatchError("Remote reference data hash mismatch; recalculate using the current export.")
    if remote.columns.tolist() != RESULT_COLUMNS:
        raise ValueError("Unexpected remote reference CSV columns; use the bundled calculator.")
    if len(remote) != len(frame):
        raise ValueError("Remote reference row count mismatch; recalculate using the current export.")
    if remote.duplicated(GROUP_KEYS).any():
        raise ValueError("Duplicate remote reference groups.")
    remote = remote.rename(
        columns={
            "recurrence site": "site",
            "interval": "window",
            "blood test": "blood_test",
            "percentage outside reference area": "outside_reference_percent",
        }
    )
    for column in ("n_patients", "n_pairs", "n_reference_patients"):
        values = pd.to_numeric(remote[column], errors="raise")
        if not (np.isfinite(values) & values.ge(0) & values.lt(2**63) & values.mod(1).eq(0)).all():
            raise ValueError(f"Invalid remote reference count: {column}.")
        remote[column] = values.astype("int64")
    scores = pd.to_numeric(remote["outside_score_sum"], errors="raise")
    if not (np.isfinite(scores) & scores.ge(0)).all():
        raise ValueError("Invalid remote outside score sum.")
    remote["outside_score_sum"] = scores.astype(float)
    remote["outside_reference_percent"] = pd.to_numeric(remote["outside_reference_percent"], errors="raise")
    keys = ["site", "test_key", "window", "blood_test", "unit"]
    merged = frame.drop(columns=[*REFERENCE_COLUMNS, "data_hash"], errors="ignore").merge(
        remote,
        on=keys,
        how="left",
        validate="one_to_one",
        suffixes=("", "_remote"),
        indicator=True,
    )
    counts_match = all(merged[column].eq(merged[f"{column}_remote"]).all() for column in ("n_patients", "n_pairs"))
    if merged["_merge"].ne("both").any() or not counts_match:
        raise ValueError("Remote reference groups/patient denominators do not match the current report.")
    validate_reference_counts(merged)
    return merged.drop(columns=["n_patients_remote", "n_pairs_remote", "_merge"])


def validate_reference_counts(frame: pd.DataFrame) -> None:
    baseline = frame["test_status"].eq("baseline")
    counts_valid = frame["outside_score_sum"].le(frame["n_reference_patients"])
    counts_valid &= frame["n_reference_patients"].le(reference_denominator(frame, baseline))
    counts_valid &= frame["n_pairs"].le(frame["n_patients"])
    if not counts_valid.all():
        raise ValueError("Remote outside scores/classified counts exceed their denominators.")
    expected, status = reference_percentages_and_status(frame, baseline)
    if not np.isclose(frame["outside_reference_percent"], expected, rtol=1e-12, atol=1e-12, equal_nan=True).all():
        raise ValueError("Remote percentages are inconsistent with patient counts.")
    if not frame["reference_status"].eq(status).all():
        raise ValueError("Remote reference statuses are inconsistent with patient counts.")
