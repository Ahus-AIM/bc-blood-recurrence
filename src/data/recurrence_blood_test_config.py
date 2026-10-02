from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.config import load_yml, resolve_path
from src.data.dataset_config import PatientDatasetConfig
from src.data.population_statistics import PopulationStatisticsConfig
from src.data.recurrence_blood_test_plot_config import RecurrenceBloodTestPlotConfig, validate_plot_config
from src.data.recurrence_site_common import RecurrenceSiteRule
from src.model.normal_ranges import NormalRange, load_normal_ranges
from src.time_units import DAYS_PER_MONTH, DAYS_PER_WEEK


@dataclass(frozen=True)
class RecurrenceTimeWindow:
    """Integer days before recurrence, with inclusive ``min_days`` and ``max_days``.

    ``label`` describes the elapsed-time interval. An interval from one to
    under three months is stored as 30–89, preserving the inclusive file format.
    """

    label: str
    min_days: int
    max_days: int


DEFAULT_WINDOWS = (
    RecurrenceTimeWindow("6–<12 months", 6 * DAYS_PER_MONTH, 12 * DAYS_PER_MONTH - 1),
    RecurrenceTimeWindow("3–<6 months", 3 * DAYS_PER_MONTH, 6 * DAYS_PER_MONTH - 1),
    RecurrenceTimeWindow("1 week–<3 months", DAYS_PER_WEEK, 3 * DAYS_PER_MONTH - 1),
    RecurrenceTimeWindow("<1 week", 0, DAYS_PER_WEEK - 1),
)


@dataclass(frozen=True)
class RecurrenceBloodTestConfig:
    """Report settings with shared cohort rules and site defaults.

    ``dataset`` is inherited from ``dataset_config_path`` or the prepared training run. ``site_rules`` and
    ``other_label`` use its population-statistics defaults with report-local
    overrides; an explicit null ``other_label`` excludes unmatched sites.
    ``include_overall`` adds an ``overall_label`` row pooling every observed
    first recurrence once per patient, including unmatched sites.
    The first ``windows`` entry is the baseline, followed by progressively
    closer windows. ``decimals`` affects LaTeX and plots; CSV values are unrounded.
    ``difference_significant_digits`` removes subtraction roundoff for ranks,
    using the scale of the paired measurements; reported changes are unrounded.
    Wilcoxon inference always uses the exact conditional sign distribution,
    accounting for ties and omitting rounded zeros from ranks. Paired counts
    and change summaries retain every pair.
    ``fdr_alpha`` controls highlighting and the sustained-significance ranking;
    all comparisons remain in the full statistical exports.
    ``reference_ranges`` snapshots the configured female reference intervals;
    their source path is retained for provenance. Remote percentages for tests with
    multiple female intervals are discovered by hash in the shared directory
    configured in report_returns.yml. Single-interval tests are
    calculated locally, applying the sole interval to each draw (also the nearest-age fallback).
    Patient outside scores average draw flags over all ties at either middle raw value.
    Baseline percentages use everyone tested; later windows use baseline-paired patients.
    A missing file leaves only remote percentages unavailable. ``plot`` controls rendering
    and optional top-test selection without altering the full statistical analysis.
    ``export_latex`` controls the optional long table; CSV/JSON are always retained.
    """

    dataset_config_path: Path
    dataset: PatientDatasetConfig
    site_rules: tuple[RecurrenceSiteRule, ...]
    other_label: str | None
    translation_path: Path | None
    output_dir: Path
    windows: tuple[RecurrenceTimeWindow, ...] = DEFAULT_WINDOWS
    decimals: int = 2
    difference_significant_digits: int = 12
    fdr_alpha: float = 0.05
    reference_ranges_csv_path: Path | None = None
    reference_ranges: tuple[NormalRange, ...] = ()
    plot: RecurrenceBloodTestPlotConfig = RecurrenceBloodTestPlotConfig()
    include_overall: bool = False
    overall_label: str = "Any"
    export_latex: bool = True

    @property
    def sites(self) -> tuple[str, ...]:
        labels = tuple(rule.label for rule in self.site_rules)
        if self.other_label is not None:
            labels = (*labels, self.other_label)
        return (self.overall_label, *labels) if self.include_overall else labels


def load_recurrence_blood_test_config(
    path: Path,
    *,
    source_config_path: Path | None = None,
    dataset_config: PatientDatasetConfig | None = None,
) -> RecurrenceBloodTestConfig:
    """Read report settings; training supplies its saved source and already-resolved cohort."""
    raw = load_yml(path)
    source_path = (
        resolve_path(str(raw["source"]["dataset_config_path"]), path.parent)
        if source_config_path is None
        else source_config_path
    )
    source = load_yml(source_path)
    dataset = (
        PatientDatasetConfig.from_mapping(source["dataset"], config_dir=source_path.parent)
        if dataset_config is None
        else dataset_config
    )
    if dataset.set_all_bloodtest_values_to_one:
        raise ValueError(
            "Recurrence blood-test summaries require observed values, not set_all_bloodtest_values_to_one."
        )
    site_settings = {**source.get("population_statistics", {}), **raw.get("population_statistics", {})}
    sites = PopulationStatisticsConfig.from_mapping(site_settings)
    output = raw["report"]
    translation = source.get("output", {}).get("bloodtest_translation_csv_path")
    reference_path, reference_ranges = load_report_reference_ranges(raw, source, path, source_path)
    config = RecurrenceBloodTestConfig(
        dataset_config_path=source_path,
        dataset=dataset,
        site_rules=sites.site_rules,
        other_label=sites.other_label,
        translation_path=None if translation is None else resolve_path(str(translation), source_path.parent),
        output_dir=resolve_path(str(output["output_dir"]), path.parent),
        windows=parse_windows(output.get("windows")),
        decimals=int(output.get("decimals", 2)),
        difference_significant_digits=int(output.get("difference_significant_digits", 12)),
        fdr_alpha=float(output.get("fdr_alpha", 0.05)),
        reference_ranges_csv_path=reference_path,
        reference_ranges=reference_ranges,
        plot=RecurrenceBloodTestPlotConfig.from_mapping(raw.get("plot", {})),
        include_overall=bool(site_settings.get("include_overall", False)),
        overall_label=str(site_settings.get("overall_label", "Any")),
        export_latex=output.get("export_latex", True),
    )
    validate_report_config(config)
    return config


def load_report_reference_ranges(
    raw: Mapping[str, object],
    source: Mapping[str, object],
    path: Path,
    source_path: Path,
) -> tuple[Path | None, tuple[NormalRange, ...]]:
    reference = raw.get("reference_intervals", {})
    source_output = source.get("output", {})
    if not isinstance(reference, Mapping) or not isinstance(source_output, Mapping):
        raise TypeError("reference_intervals and source output settings must be mappings.")
    value = reference.get("csv_path", source_output.get("normal_ranges_csv_path"))
    directory = path.parent if "csv_path" in reference else source_path.parent
    csv_path = None if value is None else resolve_path(str(value), directory)
    ranges = tuple(
        interval for group in load_normal_ranges(csv_path).values() for interval in group if interval.sex == "female"
    )
    return csv_path, ranges


def parse_windows(value: object) -> tuple[RecurrenceTimeWindow, ...]:
    if value is None:
        return DEFAULT_WINDOWS
    if not isinstance(value, list):
        raise TypeError("report.windows must be a list of labeled day ranges.")
    windows = []
    for entry in value:
        if not isinstance(entry, Mapping):
            raise TypeError("Each report window must be a mapping.")
        for key in ("min_days", "max_days"):
            if type(entry[key]) is not int:
                raise ValueError(f"Window {key} must be an integer.")
        windows.append(RecurrenceTimeWindow(str(entry["label"]), entry["min_days"], entry["max_days"]))
    return tuple(windows)


def validate_report_config(config: RecurrenceBloodTestConfig) -> None:
    validate_plot_config(config.plot)
    if type(config.export_latex) is not bool:
        raise ValueError("report.export_latex must be a boolean.")
    if len(config.windows) < 2:
        raise ValueError("At least a baseline and one comparison window are required.")
    labels = [window.label for window in config.windows]
    if any(not label.strip() for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("Window labels must be nonempty and unique.")
    for window in config.windows:
        if window.min_days < 0 or window.max_days < window.min_days:
            raise ValueError("Window bounds must be nonnegative and increasing.")
    for older, newer in zip(config.windows, config.windows[1:]):
        if newer.max_days != older.min_days - 1:
            raise ValueError("Windows must run from baseline toward recurrence with no gaps or overlaps.")
    if (
        not config.sites
        or any(not site.strip() for site in config.sites)
        or len(set(config.sites)) != len(config.sites)
    ):
        raise ValueError("Recurrence-site labels must be nonempty and unique.")
    if not 0 < config.fdr_alpha < 1:
        raise ValueError("report.fdr_alpha must lie between zero and one.")
    if not 0 <= config.decimals <= 12:
        raise ValueError("report.decimals must lie between zero and 12.")
    if not 1 <= config.difference_significant_digits <= 15:
        raise ValueError("report.difference_significant_digits must lie between one and 15.")
