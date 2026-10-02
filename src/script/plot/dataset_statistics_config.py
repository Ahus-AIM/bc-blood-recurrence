from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from src.config import load_yml, resolve_path
from src.data.recurrence_site_incidence import RecurrenceSiteRule
from src.data.cohort_figure import CohortFigureData
from src.manual_age_summary import ManualAgeSummary, load_manual_age_summary
from src.script.plot.overview_style import BLOODTEST_STACK_COLORS
from src.script.single_bloodtest_benchmark_config import load_config as load_benchmark_config


@dataclass(frozen=True)
class DatasetStatisticsFigureConfig:
    """Initial export dimensions, layout preset and optional ages; typography is shared."""

    output_path: Path
    figure_width: float
    figure_height: float
    dpi: int
    manual_age_config_path: Path | None = None
    layout: str = "compact"

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        config_dir: Path,
    ) -> DatasetStatisticsFigureConfig:
        return cls(
            output_path=resolve_path(
                str(config.get("output_path", "../../output/dataset_statistics.pgf")),
                config_dir,
            ),
            figure_width=float(config.get("figure_width", 7.874)),
            figure_height=float(config.get("figure_height", 10.4724)),
            dpi=int(config.get("dpi", 600)),
            manual_age_config_path=optional_path(config.get("manual_age_config_path"), config_dir),
            layout=str(config.get("layout", "compact")),
        )


@dataclass(frozen=True)
class DatasetStatisticsSourceConfig:
    """Source files and endpoint metadata used for dataset-statistics plots."""

    dataset_config_path: Path | None
    bloodtest_translation_csv_path: Path | None
    endpoint_csv_path: Path
    split_csv_path: Path
    recurrence_endpoint_name: str
    input_recurrence_code_filter_regex: str
    endpoint_ddfs_code_filter_regex: str

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        config_dir: Path,
    ) -> DatasetStatisticsSourceConfig:
        dataset_config_path = optional_path(config.get("dataset_config_path"), config_dir)
        bloodtest_translation_csv_path = optional_path(
            config.get("bloodtest_translation_csv_path"),
            config_dir,
        )
        source_config = config
        source_config_dir = config_dir
        if dataset_config_path is not None:
            source_config = load_dataset_section(dataset_config_path)
            source_config_dir = dataset_config_path.parent
        required_keys = (
            "endpoint_csv_path",
            "split_csv_path",
            "recurrence_endpoint_name",
            "input_recurrence_code_filter_regex",
            "endpoint_ddfs_code_filter_regex",
        )
        missing_keys = [key for key in required_keys if key not in source_config]
        if missing_keys:
            joined = ", ".join(missing_keys)
            raise KeyError(f"Missing required dataset statistics config keys: {joined}.")
        return cls(
            dataset_config_path=dataset_config_path,
            bloodtest_translation_csv_path=bloodtest_translation_csv_path,
            endpoint_csv_path=resolve_path(str(source_config["endpoint_csv_path"]), source_config_dir),
            split_csv_path=resolve_path(str(source_config["split_csv_path"]), source_config_dir),
            recurrence_endpoint_name=str(source_config["recurrence_endpoint_name"]),
            input_recurrence_code_filter_regex=str(source_config["input_recurrence_code_filter_regex"]),
            endpoint_ddfs_code_filter_regex=str(source_config["endpoint_ddfs_code_filter_regex"]),
        )


def load_dataset_section(config_path: Path) -> dict[str, object]:
    """Read a saved training dataset or resolve the benchmark's inherited settings."""
    config = load_yml(config_path)
    if "base_config_path" in config.get("source", {}):
        config, _ = load_benchmark_config(config_path)
    if "dataset" not in config:
        raise KeyError(f"Missing `dataset` section in {config_path}.")
    return config["dataset"]


@dataclass(frozen=True)
class PatientFlowDiagramConfig:
    """Editable flow/split labels and counts, with optional explanatory footnotes."""

    inclusion_box_labels: tuple[str, ...]
    inclusion_box_counts: tuple[str, ...]
    exclusion_box_labels: tuple[str, ...]
    exclusion_box_counts: tuple[str, ...]
    split_box_labels: tuple[str, ...]
    split_box_counts: tuple[str, ...]
    footnotes: tuple[str, ...] = ()

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> PatientFlowDiagramConfig:
        inclusion_labels = string_tuple_for_section(config, "panel_a", "inclusion_box_labels")
        inclusion_counts = string_tuple_for_section(config, "panel_a", "inclusion_box_counts")
        exclusion_labels = string_tuple_for_section(config, "panel_a", "exclusion_box_labels")
        exclusion_counts = string_tuple_for_section(config, "panel_a", "exclusion_box_counts")
        split_labels = string_tuple_for_section(config, "panel_a", "split_box_labels")
        split_counts = string_tuple_for_section(config, "panel_a", "split_box_counts")
        if len(inclusion_labels) < 2:
            raise ValueError("`panel_a.inclusion_box_labels` must contain at least two boxes.")
        if len(inclusion_counts) != len(inclusion_labels):
            raise ValueError("`panel_a.inclusion_box_counts` must match `inclusion_box_labels` length.")
        if len(exclusion_counts) != len(exclusion_labels):
            raise ValueError("`panel_a.exclusion_box_counts` must match `exclusion_box_labels` length.")
        if len(exclusion_labels) > len(inclusion_labels) - 1:
            raise ValueError("`panel_a.exclusion_box_labels` cannot outnumber inclusion transitions.")
        if len(split_labels) != 3:
            raise ValueError("`panel_a.split_box_labels` must contain train, validation, and test boxes.")
        if len(split_counts) != len(split_labels):
            raise ValueError("`panel_a.split_box_counts` must match `split_box_labels` length.")
        return cls(
            inclusion_box_labels=inclusion_labels,
            inclusion_box_counts=inclusion_counts,
            exclusion_box_labels=exclusion_labels,
            exclusion_box_counts=exclusion_counts,
            split_box_labels=split_labels,
            split_box_counts=split_counts,
            footnotes=string_tuple_for_section(config, "panel_a", "footnotes"),
        )


@dataclass(frozen=True)
class DiagnosisYearBarChartConfig:
    """Editable diagnosis-year stacked patient counts and axis labels."""

    title: str
    x_label: str
    y_label: str
    diagnosis_years: tuple[int, ...]
    series_labels: tuple[str, ...]
    series_counts: tuple[tuple[int, ...], ...]

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> DiagnosisYearBarChartConfig:
        years = int_tuple(config, "diagnosis_years", section="panel_b")
        if not years:
            raise ValueError("`panel_b.diagnosis_years` must contain at least one year.")
        labels = string_tuple_for_section(config, "panel_b", "series_labels")
        series_counts = int_tuple_sequence(config, "series_counts", section="panel_b")
        if not labels and not series_counts:
            labels = ("Patients",)
            series_counts = (int_tuple(config, "patient_counts", section="panel_b"),)
        validate_stacked_year_counts(years, labels, series_counts)
        return cls(
            title=str(config.get("title", "Patients diagnosed by year")),
            x_label=str(config.get("x_label", "Diagnosis year")),
            y_label=str(config.get("y_label", "Patients")),
            diagnosis_years=years,
            series_labels=labels,
            series_counts=series_counts,
        )


@dataclass(frozen=True)
class RecurrenceSiteIncidencePlotConfig:
    """Panel D Aalen-Johansen competing-risk and plot settings."""

    title: str
    x_label: str
    y_label: str
    max_years_since_diagnosis: int
    year_step: int
    min_days_since_diagnosis: int
    min_patients_at_risk: int
    confidence_level: float
    random_seed: int
    show_overall: bool
    overall_label: str
    overall_sanity_check_regex: str | None
    other_label: str | None
    site_rules: tuple[RecurrenceSiteRule, ...]

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        input_recurrence_code_filter_regex: str,
    ) -> RecurrenceSiteIncidencePlotConfig:
        labels = string_tuple_for_section(config, "panel_d", "site_labels")
        regexes = string_tuple_for_section(config, "panel_d", "site_regexes")
        if len(labels) != len(regexes):
            raise ValueError("`panel_d.site_regexes` must match `site_labels` length.")
        return cls(
            title=str(config.get("title", "Cumulative distant recurrence incidence by site")),
            x_label=str(config.get("x_label", "Years since diagnosis")),
            y_label=str(config.get("y_label", "Cumulative incidence (%)")),
            max_years_since_diagnosis=int(config.get("max_years_since_diagnosis", 10)),
            year_step=int(config.get("year_step", 1)),
            min_days_since_diagnosis=int(config.get("min_days_since_diagnosis", 0)),
            min_patients_at_risk=int(config.get("min_patients_at_risk", 5)),
            confidence_level=float(config.get("confidence_level", 0.95)),
            random_seed=int(config.get("random_seed", 42)),
            show_overall=bool(config.get("show_overall", True)),
            overall_label=str(config.get("overall_label", "Any")),
            overall_sanity_check_regex=optional_text(
                config.get(
                    "overall_sanity_check_regex",
                    input_recurrence_code_filter_regex,
                )
            ),
            other_label=optional_text(config.get("other_label")),
            site_rules=tuple(RecurrenceSiteRule(label, regex) for label, regex in zip(labels, regexes, strict=True)),
        )


@dataclass(frozen=True)
class FirstRecurrenceSiteSurvivalPlotConfig:
    """Panel f survival settings (legacy panel_e), grouped by first recurrence site.

    ``min_patients_at_risk`` suppresses curve points and exact table counts at
    or below its value; displayed death steps require a larger pre-death risk set.
    """

    title: str
    x_label: str
    y_label: str
    max_years_since_origin: int
    year_step: int
    min_patients_at_risk: int
    time_origin: str
    show_overall: bool
    overall_label: str
    other_label: str | None
    site_rules: tuple[RecurrenceSiteRule, ...]

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, object],
        *,
        site_rules: tuple[RecurrenceSiteRule, ...],
        other_label: str | None,
    ) -> FirstRecurrenceSiteSurvivalPlotConfig:
        return cls(
            title=str(config.get("title", "Survival by first recurrence site")),
            x_label=str(config.get("x_label", "Years since first recurrence")),
            y_label=str(config.get("y_label", "Survival (%)")),
            max_years_since_origin=int(config.get("max_years_since_origin", 10)),
            year_step=int(config.get("year_step", 1)),
            min_patients_at_risk=int(config.get("min_patients_at_risk", 5)),
            time_origin=str(config.get("time_origin", "recurrence")),
            show_overall=bool(config.get("show_overall", True)),
            overall_label=str(config.get("overall_label", "Any")),
            other_label=other_label,
            site_rules=site_rules,
        )


@dataclass(frozen=True)
class BloodTestCountPlotConfig:
    """Panel C measured, LOCF, and MICE stacked-bar settings.

    `blood_test_names` and the three aligned count tuples define one bar per
    displayed test. `blood_test_counts` retains its legacy name and contains
    the Measured counts, including enabled earlier history.
    `blood_test_mice_counts` includes completed current and past change inputs.
    `show_imputation_sources` switches from the legacy count bars to normalized
    source stacks using `source_x_label`.
    """

    title: str
    x_label: str
    source_x_label: str
    y_label: str
    max_bars: int | None
    show_imputation_sources: bool
    bar_color: str
    locf_color: str
    mice_color: str
    blood_test_names: tuple[str, ...]
    blood_test_counts: tuple[int, ...]
    blood_test_locf_counts: tuple[int, ...]
    blood_test_mice_counts: tuple[int, ...]

    @classmethod
    def from_mapping(cls, config: Mapping[str, object]) -> BloodTestCountPlotConfig:
        max_bars = config.get("max_bars", 24)
        blood_test_names = string_tuple_for_section(config, "panel_c", "blood_test_names")
        blood_test_counts = int_tuple(config, "blood_test_counts", section="panel_c")
        if bool(blood_test_names) != bool(blood_test_counts):
            raise ValueError("`panel_c.blood_test_names` and `blood_test_counts` must both be empty or populated.")
        if len(blood_test_names) != len(blood_test_counts):
            raise ValueError("`panel_c.blood_test_counts` must match `blood_test_names` length.")
        locf_counts = optional_count_tuple(
            config,
            "blood_test_locf_counts",
            len(blood_test_counts),
        )
        mice_counts = optional_count_tuple(
            config,
            "blood_test_mice_counts",
            len(blood_test_counts),
        )
        return cls(
            title=str(config.get("title", "Blood test counts")),
            x_label=str(config.get("x_label", "Number of blood tests")),
            source_x_label=str(config.get("source_x_label", "Blood test values (%)")),
            y_label=str(config.get("y_label", "Blood test")),
            max_bars=None if max_bars is None else int(max_bars),
            show_imputation_sources=bool(config.get("show_imputation_sources", False)),
            bar_color=str(config.get("bar_color", BLOODTEST_STACK_COLORS[0])),
            locf_color=str(config.get("locf_color", BLOODTEST_STACK_COLORS[1])),
            mice_color=str(config.get("mice_color", BLOODTEST_STACK_COLORS[2])),
            blood_test_names=blood_test_names,
            blood_test_counts=blood_test_counts,
            blood_test_locf_counts=locf_counts,
            blood_test_mice_counts=mice_counts,
        )


def optional_count_tuple(
    config: Mapping[str, object],
    key: str,
    expected_length: int,
) -> tuple[int, ...]:
    counts = int_tuple(config, key, section="panel_c") if key in config else (0,) * expected_length
    if len(counts) != expected_length:
        raise ValueError(f"`panel_c.{key}` must match `blood_test_names` length.")
    return counts


@dataclass(frozen=True)
class DatasetStatisticsPlotConfig:
    """Figure settings and external ages, independent of model/dataset features.

    ``age_return_sources`` records the shared CSVs that supplied the displayed ages.
    """

    figure: DatasetStatisticsFigureConfig
    source: DatasetStatisticsSourceConfig
    patient_flow: PatientFlowDiagramConfig
    diagnosis_years: DiagnosisYearBarChartConfig
    recurrence_site_incidence: RecurrenceSiteIncidencePlotConfig
    first_recurrence_site_survival: FirstRecurrenceSiteSurvivalPlotConfig
    blood_test_counts: BloodTestCountPlotConfig
    manual_age: ManualAgeSummary | None = None
    cohort_figure: CohortFigureData | None = None
    age_return_sources: tuple[Path, ...] = ()

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, dict[str, object]],
        *,
        config_dir: Path,
    ) -> DatasetStatisticsPlotConfig:
        source_config = DatasetStatisticsSourceConfig.from_mapping(
            config.get("dataset", {}),
            config_dir=config_dir,
        )
        incidence_config = RecurrenceSiteIncidencePlotConfig.from_mapping(
            config.get("panel_d", {}),
            input_recurrence_code_filter_regex=source_config.input_recurrence_code_filter_regex,
        )
        figure = DatasetStatisticsFigureConfig.from_mapping(config.get("figure", {}), config_dir=config_dir)
        return cls(
            figure=figure,
            source=source_config,
            patient_flow=PatientFlowDiagramConfig.from_mapping(config.get("panel_a", {})),
            diagnosis_years=DiagnosisYearBarChartConfig.from_mapping(config.get("panel_b", {})),
            recurrence_site_incidence=incidence_config,
            first_recurrence_site_survival=FirstRecurrenceSiteSurvivalPlotConfig.from_mapping(
                config.get("panel_e", {}),
                site_rules=incidence_config.site_rules,
                other_label=incidence_config.other_label,
            ),
            blood_test_counts=BloodTestCountPlotConfig.from_mapping(config.get("panel_c", {})),
            manual_age=(
                None if figure.manual_age_config_path is None else load_manual_age_summary(figure.manual_age_config_path)
            ),
        )


def string_tuple_for_section(config: Mapping[str, object], section: str, key: str) -> tuple[str, ...]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"`{section}.{key}` must be a list.")
    return tuple(str(item) for item in value)


def int_tuple(config: Mapping[str, object], key: str, *, section: str) -> tuple[int, ...]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"`{section}.{key}` must be a list.")
    return tuple(int(item) for item in value)


def int_tuple_sequence(config: Mapping[str, object], key: str, *, section: str) -> tuple[tuple[int, ...], ...]:
    value = config.get(key, [])
    if not isinstance(value, list):
        raise ValueError(f"`{section}.{key}` must be a list.")
    rows: list[tuple[int, ...]] = []
    for row in value:
        if not isinstance(row, list):
            raise ValueError(f"`{section}.{key}` must contain lists.")
        rows.append(tuple(int(item) for item in row))
    return tuple(rows)


def validate_stacked_year_counts(
    years: tuple[int, ...],
    labels: tuple[str, ...],
    series_counts: tuple[tuple[int, ...], ...],
) -> None:
    if len(labels) != len(series_counts):
        raise ValueError("`panel_b.series_labels` must match `series_counts` length.")
    for counts in series_counts:
        if len(counts) != len(years):
            raise ValueError("`panel_b.series_counts` rows must match `diagnosis_years` length.")


def optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null"}:
        return None
    return text


def optional_path(value: object, config_dir: Path) -> Path | None:
    text = optional_text(value)
    return None if text is None else resolve_path(text, config_dir)
