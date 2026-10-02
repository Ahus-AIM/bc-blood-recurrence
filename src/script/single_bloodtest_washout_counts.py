from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
import re
from typing import Literal

import numpy as np

from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.landmark_event_counts import has_first_recurrence_site
from src.model.cindex_benchmark import write_benchmark_artifacts
from src.reporting import latex_escape, recurrence_endpoint_label
from src.script.single_bloodtest_benchmark_config import mapping_sequence
from src.script.survival_benchmark_cache import SurvivalBenchmarkFeatureCache
from src.time_units import DAYS_PER_MONTH

DEFAULT_CAPTION = (
    "Test-set bootstrap sampling pool by recurrence endpoint and washout interval. "
    r"Cells show events or censored landmarks / eligible total (\%). "
    "Counts describe the eligible pool before bootstrap sampling and complete-case Brier exclusions. "
    "Patient events count unique patients with at least one retained landmark labelled with the specified "
    "recurrence within the prediction horizon. Landmark events count all such rows. Synchronous sites may overlap. "
    "Censored landmarks have no target event and end strictly before the horizon because of other cancer, "
    "death, another recurrence site or last observation. Landmarks observed through the horizon are excluded "
    "from the censored count. This is recurrence-model censoring; competing events remain known Brier outcomes. "
    "Percentages use the eligible patient or landmark total in the same cell. "
    "Counts are shared by all predictor sets. "
    "Zero washout applies no additional filtering. "
    "Positive washouts require time to event, censoring or the horizon to exceed the washout. "
    "One month equals 30 days. Empty pools have undefined percentages; "
    "dashes denote washouts at or beyond the prediction horizon."
)


@dataclass(frozen=True)
class WashoutCohortCount:
    """Test-pool counts for one endpoint, horizon and washout.

    Labels and the endpoint regex identify the target; day fields retain exact
    intervals. Counts refer to unique patients, all landmarks, event-labelled
    landmarks and unique patients contributing them. ``early_censored_landmark_count``
    counts non-event landmarks ending strictly before the horizon, including
    competing outcomes treated as censoring by the recurrence model. Skipped combinations have
    missing sampling-pool counts. ``recurrence_patient_count`` is the fixed CSV/JSON
    total from the unwashed test cohort's first recurrences over observed follow-up,
    independent of the prediction horizon; it is not a percentage denominator.
    """

    site_label: str
    endpoint_ddfs_code_filter_regex: str
    horizon_label: str
    horizon_days: int
    washout_days: int
    status: Literal["complete", "skipped"]
    recurrence_patient_count: int
    patient_count: int | None = None
    landmark_count: int | None = None
    event_patient_count: int | None = None
    event_landmark_count: int | None = None
    early_censored_landmark_count: int | None = None
    split: Literal["test"] = "test"


def build_washout_cohort_counts(
    benchmark: Mapping[str, object],
    pools: Mapping[int, SurvivalBenchmarkFeatureCache],
) -> list[WashoutCohortCount]:
    """Project the same unresampled test pools used by washout evaluation."""
    rows = []
    for site in mapping_sequence(benchmark["sites"], "sites"):
        recurrence_count = first_recurrence_patient_count(
            pools[0].landmarks, str(site["endpoint_ddfs_code_filter_regex"]),
        )
        rows.extend(
            count_washout_cohort(pool.landmarks, site, horizon, day, recurrence_patient_count=recurrence_count)
            for horizon in mapping_sequence(benchmark["horizons"], "horizons")
            for day, pool in pools.items()
        )
    return rows


def first_recurrence_patient_count(landmarks: BenchmarkLandmarks, endpoint_pattern: str) -> int:
    """Count first-site recurrence patients in the test cohort before washout."""
    dataset = landmarks.dataset
    input_regex = re.compile(dataset.config.input_recurrence_code_filter_regex)
    endpoint_regex = re.compile(endpoint_pattern)
    return len({
        item.patient_id
        for item in dataset.prepared_data.items_by_split.get("test", ())
        if item.valid_landmark_days and has_first_recurrence_site(item, input_regex, endpoint_regex)
    })


def count_washout_cohort(
    landmarks: BenchmarkLandmarks,
    site: Mapping[str, object],
    horizon: Mapping[str, object],
    washout_days: int,
    *,
    recurrence_patient_count: int,
) -> WashoutCohortCount:
    horizon_days = int(horizon["days"])
    row = WashoutCohortCount(
        site_label=str(site["label"]),
        endpoint_ddfs_code_filter_regex=str(site["endpoint_ddfs_code_filter_regex"]),
        horizon_label=str(horizon["label"]),
        horizon_days=horizon_days,
        washout_days=washout_days,
        status="skipped" if washout_days >= horizon_days else "complete",
        recurrence_patient_count=recurrence_patient_count,
    )
    if row.status == "skipped":
        return row
    target = replace(
        landmarks.dataset.config,
        split="test",
        recurrence_endpoint_name=str(site.get("recurrence_endpoint_name", site["label"])),
        endpoint_ddfs_code_filter_regex=row.endpoint_ddfs_code_filter_regex,
        censoring_horizon_days=horizon_days,
    )
    table = landmarks.table(target)
    return replace(
        row,
        patient_count=int(np.unique(table.patient_ids).size),
        landmark_count=len(table.patient_ids),
        event_patient_count=int(np.unique(table.patient_ids[table.events]).size),
        event_landmark_count=int(np.count_nonzero(table.events)),
        early_censored_landmark_count=int(np.count_nonzero(~table.events & (table.durations < horizon_days))),
    )


def write_washout_cohort_outputs(
    benchmark: Mapping[str, object],
    pools: Mapping[int, SurvivalBenchmarkFeatureCache],
    output_dir: Path,
) -> dict[str, str]:
    """Write aggregate CSV, JSON and manuscript-ready LaTeX without fitting models."""
    settings = cohort_count_settings(benchmark)
    rows = build_washout_cohort_counts(benchmark, pools)
    records = [asdict(row) for row in rows]
    return write_benchmark_artifacts(
        output_dir,
        settings,
        records,
        {"format_version": 4, "split": "test", "count_definitions": DEFAULT_CAPTION, "results": records},
        washout_cohort_counts_latex(rows, settings),
        allow_nan=False,
    )


def cohort_count_settings(benchmark: Mapping[str, object]) -> dict[str, object]:
    settings = benchmark.get("washout_cohort_counts", {})
    if not isinstance(settings, Mapping):
        raise TypeError("`benchmark.washout_cohort_counts` must be a mapping.")
    return {
        "output_csv_path": "washout/cohort_counts.csv",
        "output_latex_path": "washout/cohort_counts.tex",
        "output_metadata_path": "washout/cohort_counts.json",
        "table_caption": DEFAULT_CAPTION,
        "table_label": "tab:washout_cohort_counts",
        **settings,
    }


def washout_cohort_counts_latex(
    rows: Sequence[WashoutCohortCount],
    settings: Mapping[str, object],
) -> str:
    """Render an upright table with centred measure groups and explicit horizons."""
    sites = tuple(dict.fromkeys(row.site_label for row in rows))
    horizons = tuple(dict.fromkeys((row.horizon_label, row.horizon_days) for row in rows))
    days = tuple(dict.fromkeys(row.washout_days for row in rows))
    headers = [latex_escape(recurrence_endpoint_label(site)) for site in sites]
    caption = str(settings["table_caption"])
    if len(horizons) == 1:
        caption += f" Prediction horizon: {month_label(horizons[0][1])}."
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{settings['table_label']}}}",
        r"\setlength{\tabcolsep}{4pt}",
        r"\renewcommand{\arraystretch}{1.3}",
        r"\resizebox{\textwidth}{!}{%",
        rf"\begin{{tabular}}{{ll{'c' * len(sites)}}}",
        r"\toprule",
        "Measure & Washout interval & " + " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lookup = {(row.site_label, row.horizon_label, row.washout_days): row for row in rows}
    for horizon_index, (horizon_label, horizon_days) in enumerate(horizons):
        if horizon_index:
            lines.append(r"\midrule")
        if len(horizons) > 1:
            lines.extend((
                rf"\multicolumn{{{len(sites) + 2}}}{{l}}{{Prediction horizon: {month_label(horizon_days)}}} \\",
                r"\midrule",
            ))
        recurrence_measure = rf"\shortstack[l]{{Landmarks with recurrence\\within {month_label(horizon_days)}}}"
        for measure in ("Patients", recurrence_measure, "Censored landmarks"):
            if measure != "Patients":
                lines.append(r"\midrule")
            for index, day in enumerate(days):
                label = rf"\multirow[c]{{{len(days)}}}{{*}}{{{measure}}}" if index == 0 else ""
                cells = [label, "None" if day == 0 else month_label(day)]
                for site in sites:
                    row = lookup[site, horizon_label, day]
                    counts = {
                        "Patients": (row.event_patient_count, row.patient_count),
                        recurrence_measure: (row.event_landmark_count, row.landmark_count),
                        "Censored landmarks": (row.early_censored_landmark_count, row.landmark_count),
                    }[measure]
                    cells.append(count_cell(*counts))
                lines.append(" & ".join(cells) + r" \\")
    lines.extend((r"\bottomrule", r"\end{tabular}%", "}", r"\end{table*}"))
    return "\n".join(lines) + "\n"


def month_label(days: int) -> str:
    months = days / DAYS_PER_MONTH
    return f"{months:g} month" + ("" if months == 1 else "s")


def count_cell(count: int | None, total: int | None) -> str:
    if count is None or total is None:
        return "--"
    percentage = "--" if total == 0 else rf"{100 * count / total:.1f}\%"
    return f"{count:,} / {total:,} ({percentage})"
