"""Regenerate full importance tables from saved benchmark models without training."""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path

from src.figure_style import recurrence_site_colors
from src.model.ebm_global_importance import write_global_importance_plot
from src.model.ebm_plotting_defaults import IMPORTANCE_TITLE, recurrence_figure_title
from src.script.ebm_survival_config import safe_path_component
from src.script.single_bloodtest_benchmark_config import mapping_sequence
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY
from src.script.single_bloodtest_ebm_top_features import load_site_row

LOGGER = logging.getLogger(__name__)


def regenerate_importance(
    benchmark: Mapping[str, object],
    output_dir: Path,
    results: Sequence[Mapping[str, object]],
    *,
    fail_fast: bool = False,
) -> tuple[Path, ...]:
    """Export paper-ready names for every configured all-tests site and horizon.

    Require complete saved models for all requested cells before writing any
    figures. Each model supplies its own scores and saved label translations.
    """
    if not bool(benchmark.get("include_all_blood_tests", True)):
        raise ValueError("Global importance regeneration requires the all-tests models.")
    available_cells = {
        (str(result["site_label"]), str(result["horizon_label"]))
        for result in results
        if result.get("model_key") == ALL_MODEL_KEY and result.get("status") != "failed"
    }
    rows = [
        (
            str(horizon["label"]),
            load_site_row(
                output_dir,
                site,
                horizon,
                available=(str(site["label"]), str(horizon["label"])) in available_cells,
                fail_fast=fail_fast,
            ),
        )
        for horizon in mapping_sequence(benchmark["horizons"], "horizons")
        for site in mapping_sequence(benchmark["sites"], "sites")
    ]
    missing = [f"{row.label}/{horizon}" for horizon, row in rows if row.explanation is None or row.style is None]
    if missing:
        raise FileNotFoundError("Complete saved all-tests models are required: " + ", ".join(missing))
    paths: list[Path] = []
    for horizon, row in rows:
        assert row.explanation is not None and row.style is not None
        destination = output_dir / "global_importance" / safe_path_component(horizon)
        destination.mkdir(parents=True, exist_ok=True)
        suffix = "" if row.label == "Any" else "_" + safe_path_component(row.label).lower()
        for extension in ("png", "svg"):
            path = destination / f"global_importance{suffix}.{extension}"
            write_global_importance_plot(
                row.explanation.data(),
                path,
                test_name_translation=row.style.test_name_translation,
                title=recurrence_figure_title(IMPORTANCE_TITLE, row.label),
                color=recurrence_site_colors((row.label,))[row.label],
            )
            paths.append(path)
            LOGGER.info("wrote %s", path)
    return tuple(paths)
