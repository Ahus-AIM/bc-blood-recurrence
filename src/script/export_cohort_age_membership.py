"""Refresh a saved run's age membership CSV without source data or model fitting."""

from __future__ import annotations

import json
from pathlib import Path

from src.data.cohort_age_exchange import MEMBERSHIP_FILENAME, PACKAGE_DIR, load_cohort_membership
from src.data.cohort_comparison import VALUES_FILENAME, validate_comparison_version
from src.script.training_cohort_comparison import write_age_export


def export_saved_cohort_age_membership(run_dir: Path) -> Path:
    """Convert legacy membership exports too, preserving the saved cohort hash."""
    payload = json.loads((run_dir / VALUES_FILENAME).read_text(encoding="utf-8"))
    validate_comparison_version(payload)
    membership_path = run_dir / PACKAGE_DIR / MEMBERSHIP_FILENAME
    membership = load_cohort_membership(membership_path, payload["cohort_hash"])
    write_age_export(membership, payload["cohort_hash"], run_dir)
    return membership_path
