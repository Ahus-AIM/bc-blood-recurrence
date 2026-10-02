from __future__ import annotations

import json
from dataclasses import asdict
from importlib.metadata import version

import numpy as np
import pandas as pd

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig
from src.data.recurrence_blood_test_display import window_display_label
from src.data.recurrence_blood_test_insights import site_patient_counts
from src.data.recurrence_reference_exchange import prepare_reference_report
from src.reporting import latex_escape, p_value_text, percentage_text
from src.script.ebm_survival_config import serializable_value
from src.script.plot.recurrence_blood_test_insights import write_recurrence_blood_test_figure
from src.script.plot.recurrence_blood_test_panel import OUTSIDE_REFERENCE_LABEL


def write_recurrence_blood_test_report(
    frame: pd.DataFrame,
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> dict[str, object]:
    """Write the observed-data report and return output-relative paths for training artifact copies."""
    config.output_dir.mkdir(parents=True, exist_ok=True)
    frame, reference_metadata = prepare_reference_report(frame, dataset, config)
    frame.to_csv(config.output_dir / "recurrence_blood_test_insights.csv", index=False)
    if config.export_latex:
        (config.output_dir / "recurrence_blood_test_insights.tex").write_text(
            recurrence_blood_test_latex(frame, config),
            encoding="utf-8",
        )
    metadata = {
        "config": serializable_value(asdict(config)),
        "input_fingerprint": dataset.prepared_data.input_fingerprint,
        "selected_blood_tests": dataset.selected_test_keys,
        "site_patient_counts": site_patient_counts(frame),
        "site_patient_count_definition": (
            "Unique eligible patients with that site at first recurrence, including patients without "
            "measurements in the report windows; synchronous sites overlap. "
            "When enabled, the overall group counts every eligible patient with observed recurrence once, "
            "including unmatched sites."
        ),
        "eligible_patients_by_split": {
            split: len(items) for split, items in dataset.prepared_data.items_by_split.items()
        },
        "measurement_selection": (
            "All observed selected measurements in the eligible patient cohort within follow-up, "
            "before model-only early-row and sparse-day exclusions."
        ),
        "scipy_version": version("scipy"),
        "p_value_method": (
            "two-sided Wilcoxon signed-rank; zero_method=wilcox; "
            "exact conditional sign distribution; tied average ranks; integer subset-sum counts"
        ),
        "multiple_testing": "Benjamini-Hochberg across all calculable site/test/window comparisons",
        "number_of_hypotheses": int(frame["p_value"].notna().sum()),
        "status_counts": {str(key): int(count) for key, count in frame["test_status"].value_counts().items()},
        "reference_status_counts": {
            str(key): int(count) for key, count in frame["reference_status"].value_counts().items()
        },
        "outside_reference_method": (
            "100 times the mean patient outside score over everyone tested at baseline or baseline-paired "
            "patients later. Each draw is classified at its exact age, using the nearest age band when "
            "uncovered. The patient score averages 0/1 flags across every draw tied with either middle "
            "raw value. Sole intervals are calculated locally; multiple intervals use hash-verified "
            "remote dates. Missing references or incomplete selected-draw classification give no percentage."
        ),
        "reference_exchange": reference_metadata,
        "figure": serializable_value(write_recurrence_blood_test_figure(frame, config)),
    }
    (config.output_dir / "recurrence_blood_test_insights.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    paths = [
        "recurrence_blood_test_insights.csv",
        *(["recurrence_blood_test_insights.tex"] if config.export_latex else []),
        "recurrence_blood_test_insights.json",
        *metadata["figure"].get("files", []),
        "remote_reference/measurements.csv",
        "remote_reference/reference_thresholds.csv",
        "remote_reference/calculate_recurrence_reference_percentages.py",
    ]
    return {
        "files": [{"path": path} for path in paths],
        "reference_status": reference_metadata["status"],
        "data_hash": reference_metadata["data_hash"],
    }


def recurrence_blood_test_latex(frame: pd.DataFrame, config: RecurrenceBloodTestConfig) -> str:
    """Keep the multipage table within the document's line width using wrapping columns."""
    header = (
        r"Window & $n$ & Median [Q1--Q3] & Paired $n$ & Change [Q1--Q3] & $p$ & Adj. $p$ & "
        + r"Outside score/$N$ (\%)"
        + r" \\"
    )
    # Fractions sum to one; subtract both sides of each column's padding.
    widths = (0.19, 0.07, 0.14, 0.10, 0.15, 0.11, 0.11, 0.13)
    columns = []
    for index, width in enumerate(widths):
        alignment = "raggedright" if index == 0 else "raggedleft"
        columns.append(rf">{{\{alignment}\arraybackslash}}p{{\dimexpr {width:.2f}\linewidth-2\tabcolsep\relax}}")
    heading_column = r"p{\dimexpr\linewidth-2\tabcolsep\relax}"
    lines = [
        "% Requires \\usepackage{array,booktabs,longtable}; generated from observed measurements only.",
        r"\begingroup",
        r"\small",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\setlength{\LTcapwidth}{\linewidth}",
        rf"\begin{{longtable}}{{{''.join(columns)}}}",
        r"\caption{Blood-test changes before first distant recurrence.}\label{tab:recurrence_blood_test_insights}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\bottomrule",
        r"\endfoot",
    ]
    for site, site_rows in frame.groupby("site", sort=False):
        count = int(site_rows.iloc[0]["n_site_patients"])
        patients = "patient" if count == 1 else "patients"
        site_heading = rf"\textbf{{{latex_escape(str(site))}}} ($N = {count:,}$ {patients})"
        lines.append(rf"\multicolumn{{8}}{{{heading_column}}}{{{site_heading}}}\\*")
        for _test, rows in site_rows.groupby("test_key", sort=False):
            first = rows.iloc[0]
            title = str(first["blood_test"])
            if first["unit"]:
                title += f" ({first['unit']})"
            lines.append(rf"\multicolumn{{8}}{{{heading_column}}}{{\hspace*{{1em}}{latex_escape(title)}}}\\*")
            for index, row in enumerate(rows.to_dict("records")):
                ending = r" \\*" if index < len(rows) - 1 else r" \\"
                lines.append(" & ".join(latex_cells(row, config)) + ending)
            lines.append(r"\addlinespace")
    lines.extend((r"\end{longtable}", report_notes(config), r"\endgroup"))
    return "\n".join(lines) + "\n"


def latex_cells(row: dict[str, object], config: RecurrenceBloodTestConfig) -> list[str]:
    label = window_display_label(str(row["window"]))
    is_baseline = row["test_status"] == "baseline"
    if is_baseline:
        label += " (baseline)"
    cells = [rf"\leftskip=2em\relax {latex_escape(label)}", str(row["n_patients"]), interval_cell(row, config.decimals)]
    outside = outside_reference_cell(row)
    if is_baseline:
        return [*cells, "", "", "", "", outside]
    change = interval_cell(row, config.decimals, change=True)
    adjusted = p_value_cell(float(row["p_value_bh"]))
    if float(row["p_value_bh"]) < config.fdr_alpha:
        change = rf"\textbf{{{change}}}"
        # Scientific notation uses math mode, which textbf alone does not bold.
        adjusted = rf"\textbf{{\boldmath {adjusted}}}"
    return [*cells, str(row["n_pairs"]), change, p_value_cell(float(row["p_value"])), adjusted, outside]


def outside_reference_cell(row: dict[str, object]) -> str:
    """Show fractional outside-score sums over the full tested/paired population."""
    percentage = float(row["outside_reference_percent"])
    if not np.isfinite(percentage):
        return "N/A"
    score = float(row["outside_score_sum"])
    score_text = f"{score:,.2f}".rstrip("0").rstrip(".")
    denominator = int(row["n_patients"] if row["test_status"] == "baseline" else row["n_pairs"])
    return latex_escape(f"{score_text}/") + r"\allowbreak{}" + latex_escape(
        f"{denominator:,} ({percentage_text(percentage)})"
    )


def interval_cell(row: dict[str, object], decimals: int, *, change: bool = False) -> str:
    keys = ("median_change", "change_q1", "change_q3") if change else ("median", "q1", "q3")
    values = [float(row[key]) for key in keys]
    if not np.isfinite(values).all():
        return "---"
    median, q1, q3 = values
    return f"{median:.{decimals}f} [{q1:.{decimals}f}, {q3:.{decimals}f}]"


def p_value_cell(value: float) -> str:
    return p_value_text(value, latex=True, missing="---")


def report_notes(config: RecurrenceBloodTestConfig) -> str:
    windows = "; ".join(
        f"{window_display_label(window.label)}: {window.min_days}–{window.max_days} days" for window in config.windows
    )
    same_day = "included" if config.dataset.include_recurrence_day_blood_tests else "excluded"
    overall_note = (
        f"{config.overall_label} pools all first recurrences once per patient, including unmatched sites. "
        if config.include_overall
        else ""
    )
    return (
        r"\noindent\footnotesize "
        + latex_escape(
            "Time before recurrence is recurrence day minus measurement day; recurrence is day 0. "
            f"Weeks (wk) use 7 days and months (mo) use 30 days. Inclusive integer-day bounds are {windows}. "
        )
        + f"Recurrence-day measurements are {same_day}. "
        + "Eligible train, validation, and test patients are pooled. Only observed measurements are used. "
        + "All available selected measurements within follow-up are used, including pre-diagnosis "
        + "measurements and sparse days excluded only from model data. "
        + "Patient eligibility requires at least one landmark day meeting the configured test-count minimum. "
        + "Each patient contributes one median per test/window; cells summarize these medians. "
        + "Patients with synchronous first-recurrence sites contribute to each applicable organ; later sites are ignored. "
        + latex_escape(overall_note)
        + "Site N includes every eligible patient at that site, even without measurements in these windows; "
        + "window n counts patients measured for that test in that window. "
        + "Window populations may differ. Changes are later minus baseline within patients measured in both windows; "
        + "they are not differences between the displayed group medians. "
        + "Outside is reported as score/N (percentage), where score is the sum of patient outside scores "
        + "and N is the full tested or paired denominator. Each draw is flagged 0 inside or 1 outside "
        + "its female reference interval. Each patient score averages the flags of all draws tied with "
        + "either middle raw value, including every tie, and can be fractional. Scores are displayed "
        + "to at most two decimals; percentages use unrounded scores and give each patient equal weight. "
        + "The trajectory figure shows the same percentage as "
        + latex_escape(OUTSIDE_REFERENCE_LABEL)
        + ". "
        + "Baseline uses the entire population tested for that blood test at that recurrence site in the baseline "
        + "window, including patients without later measurements. Later windows use all patients paired with baseline "
        + "for that comparison. A sole usable female interval is applied locally to each draw. "
        + "Tests with multiple intervals use exact draw ages remotely, selecting the nearest age band "
        + "when none applies. "
        + "Missing or incomplete reference coverage, no measurements, or no pairs gives N/A. "
        + "Two-sided Wilcoxon signed-rank tests assume symmetric paired differences. "
        + "Zero differences count in paired n and change summaries but are omitted from Wilcoxon ranks; "
        + "p-values always use the exact conditional sign distribution, accounting for tied ranks. "
        + "An all-zero comparison has p = 1. No minimum paired count is imposed. "
        + "Dashes indicate no measurements/pairs or an uncomputable test; the CSV records the reason. "
        + "Benjamini--Hochberg adjustment covers all calculable site/test/window comparisons together, "
        + "including the overall group when enabled. "
        + f"Bold changes and adjusted p-values meet adjusted p < {config.fdr_alpha:g}. "
        + "These within-patient associations do not establish organ specificity or causation.\n"
    )
