"""Known manuscript artifact sources and their transfer filenames."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from src.regeneration.context import RegenerationContext
from src.reporting import report_output_path
from src.script.ebm_survival_config import safe_path_component
from src.script.single_bloodtest_benchmark_config import benchmark_model_output_dir
from src.script.single_bloodtest_benchmark_reporting import ALL_MODEL_KEY
from src.script.single_bloodtest_brier import brier_config
from src.script.single_bloodtest_washout import washout_config
from src.script.single_bloodtest_washout_counts import cohort_count_settings

PAPER_SITES = (("All", ""), ("Bone", "_bone"), ("Brain", "_brain"), ("Liver", "_liver"), ("Lung", "_lung"))


@dataclass(frozen=True)
class PaperArtifact:
    """One source file and its relative filename inside the manuscript bundle.

    ``unavailable`` records missing/failed model inventory even if stale source
    files remain on disk. Only explicitly listed aggregate figures/tables enter
    the bundle; models and patient-level exports are never traversed.
    """

    source: Path
    destination: Path
    unavailable: str | None = None


def figure_files(source: Path, stem: str, unavailable: str | None = None) -> Iterator[PaperArtifact]:
    for extension in ("svg", "png"):
        yield PaperArtifact(source.with_suffix(f".{extension}"), Path("figure") / f"{stem}.{extension}", unavailable)


def model_source(context: RegenerationContext, site: str) -> tuple[Path, str | None]:
    horizon = context.settings.paper_bundle.horizon
    cell = next((cell for cell in context.cells if cell.key == (site, horizon, ALL_MODEL_KEY)), None)
    if cell is None:
        path = benchmark_model_output_dir(context.output_dir, ALL_MODEL_KEY, {"label": site}, {"label": horizon})
        return path, f"No saved all-tests cell for {site}/{horizon}"
    if any(row["status"] == "failed" for row in cell.results):
        return cell.path, f"Saved all-tests cell failed: {site}/{horizon}"
    return cell.path, None


def paper_artifacts(context: RegenerationContext) -> tuple[PaperArtifact, ...]:
    """List allowed sources for the 18 figures and four generated tables."""
    root = context.output_dir
    horizon = safe_path_component(context.settings.paper_bundle.horizon)
    cohort, unavailable = model_source(context, "All")
    artifacts = list(figure_files(cohort / "dataset_statistics", "dataset_statistics", unavailable))
    trajectories = "recurrence_blood_test_trajectories"
    for suffix in ("", "_any", "_bone", "_brain", "_liver", "_lung"):
        artifacts.extend(figure_files(cohort / f"{trajectories}{suffix}", f"{trajectories}{suffix}", unavailable))
    top = context.benchmark.get("ebm_top_features", {})
    if not isinstance(top, Mapping):
        raise TypeError("`benchmark.ebm_top_features` must be a mapping.")
    top_dir = report_output_path(root, top.get("output_dir", "top_features")) / horizon
    artifacts.extend(figure_files(top_dir / "ebm_top_features", "ebm_top_features"))
    for site, suffix in PAPER_SITES:
        model, reason = model_source(context, site)
        # Both importance producers draw the same fitted global explanation.
        for extension in ("svg", "png"):
            name = f"global_importance{suffix}.{extension}"
            sources = (root / "global_importance" / horizon / name,
                       model / "explanations" / f"global_importance.{extension}")
            artifacts.extend(PaperArtifact(source, Path("figure") / name, reason) for source in sources)
        artifacts.extend(figure_files(
            model / "explanations" / "terms_hazard_ratio_grid", f"terms_hazard_ratio_grid{suffix}", reason,
        ))
    washout = washout_config(context.benchmark)
    counts = cohort_count_settings(context.benchmark)
    brier_dir = report_output_path(root, brier_config(context.benchmark).get("output_dir", "brier"))
    for extension, key in (("tex", "output_latex_path"), ("csv", "output_csv_path")):
        name = f"cindex_benchmark_washout.{extension}"
        artifacts.append(PaperArtifact(report_output_path(root, washout.get(key, f"washout/{name}")),
                                       Path("table") / name))
        name = f"cohort_counts.{extension}"
        artifacts.append(PaperArtifact(report_output_path(root, counts[key]), Path("table") / name))
        name = f"complete_case_brier_washout.{extension}"
        artifacts.append(PaperArtifact(brier_dir / name, Path("table") / name))
        name = f"cohort_comparison.{extension}"
        artifacts.append(PaperArtifact(cohort / name, Path("table") / name, unavailable))
    return tuple(artifacts)


def bundle_destination(context: RegenerationContext, artifacts: tuple[PaperArtifact, ...]) -> Path:
    destination = report_output_path(context.output_dir, context.settings.paper_bundle.output_dir).resolve()
    if context.output_dir.resolve().is_relative_to(destination):
        raise ValueError("The paper bundle must have its own directory, separate from the benchmark root.")
    if any(artifact.source.resolve().is_relative_to(destination) for artifact in artifacts):
        raise ValueError("The paper bundle directory must not contain original artifacts.")
    if any(not (destination / artifact.destination).resolve().is_relative_to(destination) for artifact in artifacts):
        raise ValueError("Paper bundle files must remain inside the bundle directory.")
    return destination
