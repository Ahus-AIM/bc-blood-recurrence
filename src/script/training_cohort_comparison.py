"""Write the supplementary cohort table and membership CSV for external ages."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from src.data import PatientBloodTestDataset
from src.data.cohort_age_exchange import (
    MEMBERSHIP_FILENAME, PACKAGE_DIR, age_display_summary, build_age_membership, find_age_summary,
)
from src.data.cohort_comparison import build_comparison_payload, write_comparison_outputs
from src.data.population_statistics import PopulationStatisticsConfig


def write_training_cohort_comparison(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    population_config: PopulationStatisticsConfig,
) -> dict[str, Any]:
    payload, membership = build_comparison_payload(dataset, population_config)
    returned = find_age_summary(payload["cohort_hash"])
    payload["ages"] = None if returned is None else returned.value
    payload["age_return_sources"] = [] if returned is None else [str(path) for path in returned.paths]
    metadata = write_comparison_outputs(payload, output_dir)
    age_path = output_dir / "cohort_age_summary.yml"
    age_path.write_text(yaml.safe_dump(asdict(age_display_summary(payload["ages"])), sort_keys=False), encoding="utf-8")
    metadata["age_summary"] = {"path": age_path.name}
    metadata["age_export"] = write_age_export(membership, payload["cohort_hash"], output_dir)
    return metadata


def write_age_export(membership: pd.DataFrame, cohort_hash: str, output_dir: Path) -> list[dict[str, str]]:
    package = output_dir / PACKAGE_DIR
    exported = build_age_membership(membership, cohort_hash)
    package.mkdir(parents=True, exist_ok=True)
    membership_path = package / MEMBERSHIP_FILENAME
    exported.to_csv(membership_path, index=False)
    readme = package / "README.md"
    readme.write_text(
        "# Exact ages for the paper tables\n\n"
        "Send `cohort_membership.csv` to your remote age workflow. Its columns are `id,fold,hash`.\n"
        "Each included patient has three rows: their train/validation/test fold, overall and included. "
        "Other initial-cohort patients have one excluded row. Both populations start after required-age filtering.\n"
        "Treat IDs as text. Keep the membership and its hash unchanged; the same hash covers the full cohort.\n\n"
        "Return `age_summaries.csv` with columns `fold,median,lower_q,upper_q,hash`. "
        "Return exactly one row for each of train, validation, test, overall, included and excluded, "
        "copying the exported hash into every row. Calculate exact age at BC diagnosis in years; "
        "lower_q and upper_q are the 25th and 75th percentiles. Overall and included both pool all included "
        "patients and must have identical summaries. All six rows require finite numeric ages with "
        "0 <= lower_q <= median <= upper_q; missing values fail the import.\n\n"
        "Store the return under data/ages with any .csv filename. Training and plot regeneration "
        "discover matching hashes automatically; shared directories are configured in src/config/report_returns.yml. "
        "For a saved benchmark, select tasks: [age_returns] and any model filters in "
        "src/config/regenerate.yml, then run `uv run python -m src.script.regenerate`. "
        "It verifies the cohort hash and summaries, refreshes the supplementary CSV/LaTeX and the saved "
        "dataset-statistics figure, and writes `cohort_age_summary.yml` for the main table. "
        "No model fitting or patient source CSVs are needed for this refresh.\n",
        encoding="utf-8",
    )
    return [{"path": str(path.relative_to(output_dir))} for path in (membership_path, readme)]
