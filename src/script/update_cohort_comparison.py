"""Import matching remote ages and refresh the paper tables without retraining."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import yaml

from src.config import load_yml
from src.data.cohort_age_exchange import (
    MEMBERSHIP_FILENAME, PACKAGE_DIR, age_display_summary, find_age_summary, load_cohort_membership,
)
from src.data.cohort_comparison import VALUES_FILENAME, validate_comparison_version, write_comparison_outputs
from src.data.dataset_statistics_export import (
    load_dataset_statistics_payload,
    recurrence_incidence_from_payload,
    recurrence_survival_from_payload,
    write_dataset_statistics_payload,
)
from src.script.plot.dataset_statistics import (
    CONFIG_PATH as PLOT_CONFIG_PATH,
    build_dataset_statistics_figure,
    plot_config_from_payload,
)
from src.script.plot.overview_style import save_publication_figure
from src.script.survival_benchmark_cache import artifact_paths


def refresh_cohort_comparison(
    run_dir: Path,
    *,
    refresh_figure: bool = True,
) -> tuple[Path, ...]:
    """Refresh saved cohort reports and return the files written for collection."""
    payload = json.loads((run_dir / VALUES_FILENAME).read_text(encoding="utf-8"))
    validate_comparison_version(payload)
    membership = load_cohort_membership(run_dir / PACKAGE_DIR / MEMBERSHIP_FILENAME, payload["cohort_hash"])
    for population in ("included", "excluded"):
        if payload["columns"][population]["patients"] != int(membership["population"].eq(population).sum()):
            raise ValueError("Saved comparison patient counts do not match cohort membership.")
    returned = find_age_summary(payload["cohort_hash"])
    payload["ages"] = None if returned is None else returned.value
    payload["age_return_sources"] = [] if returned is None else [str(path) for path in returned.paths]
    manual = asdict(age_display_summary(payload["ages"]))
    figures = update_main_table_ages(run_dir, payload, manual, refresh_figure=refresh_figure)
    (run_dir / "cohort_age_summary.yml").write_text(yaml.safe_dump(manual, sort_keys=False), encoding="utf-8")
    metadata = write_comparison_outputs(payload, run_dir)
    return (*figures, *(run_dir / path for path in artifact_paths(metadata)))


def update_main_table_ages(
    run_dir: Path,
    comparison: dict[str, Any],
    manual: dict[str, Any],
    *,
    refresh_figure: bool,
) -> tuple[Path, ...]:
    statistics_path = run_dir / "dataset_statistics_values.yml"
    statistics = load_dataset_statistics_payload(statistics_path)
    if statistics["metadata"]["input_fingerprint"] != comparison["input_fingerprint"]:
        raise ValueError("Dataset statistics and cohort comparison come from different prepared inputs.")
    statistics["metadata"]["manual_age_summary"] = manual
    statistics["metadata"]["cohort_hash"] = comparison["cohort_hash"]
    statistics["metadata"]["age_return_sources"] = comparison["age_return_sources"]
    paths: tuple[Path, ...] = ()
    if refresh_figure:
        raw = load_yml(PLOT_CONFIG_PATH)
        raw["figure"]["manual_age_config_path"] = None
        raw["figure"]["output_path"] = str((run_dir / "dataset_statistics.png").resolve())
        # The saved payload provides all counts/curves; source config only identifies the run.
        raw["dataset"]["dataset_config_path"] = str((run_dir / "training_config.yml").resolve())
        config = plot_config_from_payload(raw, statistics, config_dir=PLOT_CONFIG_PATH.parent)
        figure = build_dataset_statistics_figure(
            config,
            recurrence_incidence_df=recurrence_incidence_from_payload(statistics),
            recurrence_survival_df=recurrence_survival_from_payload(statistics),
        )
        try:
            paths = save_publication_figure(figure, config.figure.output_path, config.figure.dpi)
        finally:
            plt.close(figure)
    write_dataset_statistics_payload(statistics, statistics_path)
    return paths
