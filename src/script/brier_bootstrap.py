from __future__ import annotations

import gzip
import io
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PatientLandmarkTable
from src.data.competing_risk_outcomes import HORIZON_DAYS
from src.data.dataset_config import PatientDatasetConfig
from src.data.sample_weighting import patient_uniform_weights
from src.model.complete_case_brier import score_draw, summarize_brier_runs
from src.model.survival_evaluation import bootstrap_run_count
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache

IDENTITY_COLUMNS = ["patient_id", "landmark_day"]


@dataclass(frozen=True)
class BrierWashout:
    """One eligible C-index landmark pool and its exact ordered bootstrap indices.

    ``table`` retains all eligible landmarks and recomputed patient weights;
    each entry in ``draws`` selects one existing C-index bootstrap replicate,
    including replicates without target events.
    """

    table: PatientLandmarkTable
    draws: tuple[np.ndarray, ...]

    @property
    def sample_count(self) -> int:
        """Return selections per replicate, including zero for an empty pool."""
        return len(self.draws[0]) if self.draws else 0


def prepare_brier_washouts(
    pools: Mapping[int, SurvivalBenchmarkFeatureCache],
    config: PatientDatasetConfig,
    destination: Path,
) -> dict[int, BrierWashout]:
    """Reuse the C-index sampler after each washout's eligibility filtering."""
    config = replace(config, split="test", censoring_horizon_days=HORIZON_DAYS)
    result = {}
    destination.mkdir(parents=True, exist_ok=True)
    for day, pool in pools.items():
        if day >= HORIZON_DAYS:
            continue
        table = pool.landmarks.table(config)
        if day > 0 and np.any(table.durations <= day):
            raise ValueError("C-index washout pool includes ineligible landmark durations.")
        table = replace(table, sample_weights=patient_uniform_weights(table.patient_ids))
        draws = tuple(pool.landmarks.bootstrap_indices(config, run) for run in range(bootstrap_run_count(config)))
        result[day] = BrierWashout(table, draws)
        write_draws(result[day], destination, day)
    return result


def write_draws(washout: BrierWashout, destination: Path, day: int) -> None:
    table = washout.table
    probabilities = np.empty(0) if len(table.patient_ids) == 0 else table.sample_weights / len(table.patient_ids)
    pool = pd.DataFrame(
        {
            "patient_id": table.patient_ids,
            "landmark_day": table.landmark_days,
            "duration_days": table.durations,
            "eligible_patient_weight": table.sample_weights,
            "bootstrap_selection_probability": probabilities,
        }
    )
    pool.to_csv(
        destination / f"eligible_landmarks_{day}d.csv.gz", index=False, compression={"method": "gzip", "mtime": 0}
    )
    with gzip.GzipFile(filename=str(destination / f"bootstrap_draws_{day}d.csv.gz"), mode="wb", mtime=0) as compressed:
        with io.TextIOWrapper(compressed, encoding="utf-8") as output:
            for run, indices in enumerate(washout.draws):
                frame = pool.iloc[indices].assign(
                    bootstrap_run=run, draw_position=np.arange(len(indices)), washout_days=day
                )
                frame.to_csv(output, index=False, header=run == 0)


def align_predictions(predictions: pd.DataFrame, table: PatientLandmarkTable) -> pd.DataFrame:
    index = pd.MultiIndex.from_frame(predictions[IDENTITY_COLUMNS])
    if not index.is_unique:
        raise ValueError("Saved probabilities require unique patient/landmark identities.")
    desired = pd.MultiIndex.from_arrays([table.patient_ids, table.landmark_days], names=IDENTITY_COLUMNS)
    positions = index.get_indexer(desired)
    if np.any(positions < 0):
        raise ValueError("A C-index bootstrap landmark has no fixed Brier prediction.")
    aligned = predictions.iloc[positions].reset_index(drop=True)
    if not np.array_equal(aligned["duration_days"].to_numpy(), table.durations):
        raise ValueError("Brier predictions and C-index washout durations differ.")
    return aligned


def evaluate_brier_washout(predictions: pd.DataFrame, washout: BrierWashout) -> dict[str, object]:
    pool = align_predictions(predictions, washout.table)
    invalid = ~np.isfinite(pool["target_probability_180d"]) | ~pool["target_probability_180d"].between(0, 1)
    runs = [score_draw(pool.iloc[indices], run) for run, indices in enumerate(washout.draws)]
    return {
        **summarize_brier_runs(runs, int(invalid.sum())),
        "eligible_patient_count": int(pool["patient_id"].nunique()),
        "eligible_landmark_count": len(pool),
        "bootstrap_sample_count": washout.sample_count,
    }
