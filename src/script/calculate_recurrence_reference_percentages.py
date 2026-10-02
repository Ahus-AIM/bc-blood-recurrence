"""Copy this script, measurements.csv, and reference_thresholds.csv to the remote system.

Only pandas and the remote patient CSV are required. Edit REMOTE_SETTINGS below,
then run without arguments. Relative paths start at this script's directory.
Measurements contain individual draw values/days for tests with multiple female
reference intervals. Classify each draw at its exact age (nearest age band if
uncovered), then average flags over all draws tied with either middle raw value.
Group percentages give each patient's resulting outside score equal weight.
Single-interval tests are calculated locally and are absent from the returned CSV.
Baseline percentages use everyone tested in that window; later windows use patients
paired with baseline for the same test and recurrence site.
The bundled settings contain data_hash and checksums for both CSVs, verified before
patient dates are read. The same hash is returned unchanged in the aggregate CSV.
reference_diagnostics.csv explains unclassified scores with draw dates, selected
limits and median-draw flags. Keep this patient-level file on the remote system.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, TypedDict

import pandas as pd

# BEGIN REMOTE SETTINGS
REMOTE_SETTINGS: dict[str, Any] = {
    "measurements_csv": "measurements.csv",
    "reference_thresholds_csv": "reference_thresholds.csv",
    "patients_csv": "patients.csv",
    "output_csv": "reference_percentages.csv",
    "diagnostics_csv": "reference_diagnostics.csv",
    "patient_columns": {"id": "id", "birth_date": "birth date", "diagnosis_date": "day of first bc diagnosis"},
    "date_formats": {"birth_date": "%Y-%m-%d", "diagnosis_date": "%Y-%m-%d"},
    "patients_separator": ",",
    "patients_encoding": "utf-8-sig",
}
# END REMOTE SETTINGS

# BEGIN EXPORT MANIFEST
EXPORT_MANIFEST: dict[str, Any] = {}
# END EXPORT MANIFEST

GROUP_KEYS = ["test_key", "interval", "recurrence site"]
GROUP_COLUMNS = [*GROUP_KEYS, "blood test", "unit", "n_patients", "n_pairs"]
MEASUREMENT_COLUMNS = [
    "id",
    "draw days since diagnosis",
    "recurrence site",
    "interval",
    "blood test name",
    "blood test value",
    "unit",
    "test_key",
]
REFERENCE_COLUMNS = [
    "test_key",
    "test_name",
    "unit",
    "sex",
    "age_group",
    "lower",
    "upper",
    "lower_operator",
    "upper_operator",
    "source_url",
    "notes",
    "age_bounds",
]
RESULT_COLUMNS = [
    "blood test",
    "interval",
    "recurrence site",
    "percentage outside reference area",
    "n_patients",
    "n_pairs",
    "outside_score_sum",
    "n_reference_patients",
    "reference_status",
    "unit",
    "test_key",
    "data_hash",
]
DIAGNOSTIC_COLUMNS = [
    "id",
    *GROUP_KEYS,
    "blood test name",
    "unit",
    "blood test value",
    "is_median_draw",
    "reason",
    "birth_date",
    "draw_date",
    "draw days since diagnosis",
    "matching_intervals",
    "draw_classification",
    "used_nearest_age",
    "selected_intervals",
]


class AgeBound(TypedDict):
    """One comparison with the patient's birth date plus a whole number of months."""

    months: int
    operator: str


class ReferenceLabel(TypedDict, total=False):
    """Optional display label retained when intervals come from the threshold CSV."""

    age_group: str


class ReferenceDefinition(ReferenceLabel):
    """Calculation fields from an exported interval; null age bounds mean unknown applicability."""

    age_bounds: list[AgeBound] | None
    lower: float | None
    upper: float | None
    lower_operator: str
    upper_operator: str


def run_calculation(directory: Path) -> Path:
    """Verify input, score patients' middle draws, then write aggregates and diagnostics."""
    config = REMOTE_SETTINGS
    measured, manifest = load_transfer(directory / config["measurements_csv"])
    references = load_reference_thresholds(directory / config["reference_thresholds_csv"], manifest)
    if not measured.empty:
        measured = attach_patient_dates(measured, config, directory)
    result = calculate_percentages(measured, manifest, references)
    baseline = manifest["context"]["windows"][0]["label"]
    diagnostics = reference_diagnostics(measured, baseline, references)
    outputs = [directory / config[key] for key in ("output_csv", "diagnostics_csv")]
    inputs = [directory / config[key] for key in ("measurements_csv", "reference_thresholds_csv", "patients_csv")]
    write_paths = [path.resolve() for output in outputs for path in (output, output.with_suffix(".csv.tmp"))]
    if len(set(write_paths)) != len(write_paths) or set(write_paths) & {
        path.resolve() for path in [*inputs, Path(__file__)]
    }:
        raise ValueError("Output CSVs and temporary files must not overlap or overwrite an input or calculator.")
    for frame, path in zip((result, diagnostics), outputs, strict=True):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False, float_format="%.17g")
        temporary.replace(path)
    print(f"Wrote reference diagnostics to {outputs[1].resolve()} ({len(diagnostics)} draw rows)")
    return outputs[0]


def attach_patient_dates(measured: pd.DataFrame, config: dict[str, Any], config_dir: Path) -> pd.DataFrame:
    """Attach exact dates to individual draws, preserving IDs and rejecting invalid dates."""
    columns = config["patient_columns"]
    patients = pd.read_csv(
        config_dir / config["patients_csv"],
        dtype=str,
        keep_default_na=False,
        sep=config.get("patients_separator", ","),
        encoding=config.get("patients_encoding", "utf-8-sig"),
        usecols=[columns["id"], columns["birth_date"], columns["diagnosis_date"]],
    ).rename(columns={value: key for key, value in columns.items()})
    patients = patients.loc[patients["id"].isin(measured["id"])].drop_duplicates()
    duplicates = patients.loc[patients["id"].duplicated(keep=False), "id"].unique().tolist()
    if duplicates:
        raise ValueError(f"Conflicting patient records for IDs: {duplicates}")
    for column in ("birth_date", "diagnosis_date"):
        patients[column] = pd.to_datetime(patients[column], format=config["date_formats"][column], errors="coerce")
    joined = measured.merge(patients, on="id", how="left", validate="many_to_one")
    invalid = joined[["birth_date", "diagnosis_date"]].isna().any(axis=1)
    invalid |= joined["birth_date"] > joined["diagnosis_date"]
    invalid |= joined["birth_date"].ne(joined["birth_date"].dt.normalize())
    invalid |= joined["diagnosis_date"].ne(joined["diagnosis_date"].dt.normalize())
    if invalid.any():
        raise ValueError(f"Missing or invalid patient dates for IDs: {joined.loc[invalid, 'id'].unique().tolist()}")
    days = joined["draw days since diagnosis"].astype("int64")
    joined["draw_date"] = joined["diagnosis_date"] + pd.to_timedelta(days, unit="D")
    invalid = joined["draw_date"].isna() | joined["draw_date"].lt(joined["birth_date"])
    if invalid.any():
        raise ValueError(f"Invalid blood-draw dates for IDs: {joined.loc[invalid, 'id'].unique().tolist()}")
    return joined


def calculate_percentages(
    measured: pd.DataFrame,
    manifest: dict[str, Any],
    references: dict[str, list[ReferenceDefinition]],
) -> pd.DataFrame:
    """Average patient scores over everyone tested at baseline or baseline-paired patients later."""
    baseline_window = manifest["context"]["windows"][0]["label"]
    groups = pd.DataFrame(manifest["groups"], columns=GROUP_COLUMNS)
    result = reference_group_percentages(measured, groups, baseline_window, references)
    result["data_hash"] = manifest["data_hash"]
    return result[RESULT_COLUMNS]


def reference_group_percentages(
    measured: pd.DataFrame,
    groups: pd.DataFrame,
    baseline_window: str,
    references: dict[str, list[ReferenceDefinition]],
) -> pd.DataFrame:
    """Share pairing, classification, denominator checks, and percentages between local and remote reports."""
    eligible = reference_population_measurements(measured, baseline_window)
    eligible["outside"] = float("nan")
    for key, rows in eligible.groupby("test_key", sort=False):
        # Remote draws use exact ages; a sole local interval is also the nearest fallback.
        eligible.loc[rows.index, "outside"] = classify_measurements(rows, references.get(str(key), []))
    counts = measured.groupby(GROUP_KEYS)["id"].nunique().rename("n_patients").reset_index()
    counts = counts.merge(
        aggregate_reference_patients(eligible, baseline_window), on=GROUP_KEYS, how="left", validate="one_to_one"
    )
    result = groups.merge(counts, on=GROUP_KEYS, how="left", suffixes=("_expected", ""))
    for column in ("n_patients", "n_pairs", "n_reference_patients"):
        result[column] = result[column].fillna(0).astype(int)
    result["outside_score_sum"] = result["outside_score_sum"].fillna(0.0).astype(float)
    for column in ("n_patients", "n_pairs"):
        if not result[column].eq(result[f"{column}_expected"]).all():
            raise ValueError(f"Measurement {column} counts do not match the manifest.")
    result["percentage outside reference area"], result["reference_status"] = reference_percentages_and_status(
        result, result["interval"].eq(baseline_window)
    )
    return result.drop(columns=["n_patients_expected", "n_pairs_expected"])


def reference_population_measurements(measured: pd.DataFrame, baseline_window: str) -> pd.DataFrame:
    """Include all baseline draws and later draws belonging to the same patient/site/test population."""
    keys = ["id", "recurrence site", "test_key"]
    baseline = measured["interval"].eq(baseline_window)
    patients = measured.loc[baseline, keys].drop_duplicates()
    return measured.merge(patients, on=keys, how="inner", validate="many_to_one")


def aggregate_reference_patients(measured: pd.DataFrame, baseline_window: str) -> pd.DataFrame:
    """Sum fractional patient scores; baseline patients count as tested, not as pairs."""
    patients = classify_reference_patients(measured, baseline_window)
    return patients.groupby(GROUP_KEYS, as_index=False, sort=False).agg(
        n_pairs=("is_pair", "sum"),
        outside_score_sum=("outside_score", "sum"),
        n_reference_patients=("is_known", "sum"),
    )


def median_draw_mask(measured: pd.DataFrame) -> pd.Series:
    """Select every draw tied with either middle raw value, including unequal tie counts."""
    values = measured.groupby([*GROUP_KEYS, "id"], sort=False)["blood test value"]
    lower = values.transform("quantile", q=0.5, interpolation="lower")
    upper = values.transform("quantile", q=0.5, interpolation="higher")
    return measured["blood test value"].eq(lower) | measured["blood test value"].eq(upper)


def classify_reference_patients(measured: pd.DataFrame, baseline_window: str) -> pd.DataFrame:
    """Average outside flags for median-selected draws, requiring coverage of each selected draw."""
    selected = measured.loc[median_draw_mask(measured)]
    patients = selected.groupby([*GROUP_KEYS, "id"], as_index=False, sort=False).agg(
        outside_score=("outside", "mean"),
        classified_results=("outside", "count"),
        total_results=("outside", "size"),
    )
    patients["is_known"] = patients["classified_results"].eq(patients["total_results"])
    patients["outside_score"] = patients["outside_score"].where(patients["is_known"])
    patients["is_pair"] = patients["interval"].ne(baseline_window)
    return patients


def reference_denominator(counts: pd.DataFrame, baseline: pd.Series) -> pd.Series:
    """Use everyone tested at baseline, and baseline-paired patients in each later window."""
    return counts["n_pairs"].where(~baseline, counts["n_patients"])


def reference_percentages_and_status(counts: pd.DataFrame, baseline: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Require complete classification of the full baseline or paired population for a percentage."""
    denominator = reference_denominator(counts, baseline)
    complete = counts["n_reference_patients"].eq(denominator)
    percentages = (100 * counts["outside_score_sum"] / denominator).where(complete)
    status = pd.Series("incomplete_reference", index=counts.index)
    status.loc[complete] = "ok"
    status.loc[counts["n_reference_patients"].eq(0)] = "unavailable_reference"
    status.loc[denominator.eq(0)] = "no_pairs"
    status.loc[counts["n_patients"].eq(0)] = "no_measurements"
    return percentages, status


def classify_measurements(rows: pd.DataFrame, ranges: list[ReferenceDefinition]) -> pd.Series:
    """Return 1 outside, 0 inside, or NaN unless exactly one reference interval is selected."""
    return measurement_classifications(rows, ranges)["outside"]


def measurement_classifications(rows: pd.DataFrame, ranges: list[ReferenceDefinition]) -> pd.DataFrame:
    """Classify actual results using their applicable or nearest age band."""
    selected = reference_matches(rows, ranges)
    matches = selected.sum(axis=1)
    outside = pd.Series(False, index=rows.index)
    fallback = pd.Series(False, index=rows.index)
    for index, interval in enumerate(ranges):
        applicable = selected[index]
        fallback |= applicable & ~age_applicability(rows, interval["age_bounds"])
        within = pd.Series(True, index=rows.index)
        for side in ("lower", "upper"):
            limit = interval[side]
            if limit is not None:
                if not math.isfinite(limit):
                    raise ValueError("Reference limits must be finite.")
                within &= compare(rows["blood test value"], limit, interval[f"{side}_operator"])
        outside |= applicable & ~within
    return pd.DataFrame(
        {
            "outside": outside.astype(float).where(matches.eq(1)),
            "matching_intervals": matches,
            "used_nearest_age": fallback,
        }
    )


def reference_matches(rows: pd.DataFrame, ranges: list[ReferenceDefinition]) -> pd.DataFrame:
    """Use exact applicability first, otherwise the nearest supported calendar-age boundary."""
    selected = pd.DataFrame(
        {i: age_applicability(rows, r["age_bounds"]) for i, r in enumerate(ranges)}, index=rows.index
    )
    distances = pd.DataFrame({i: age_distance(rows, r["age_bounds"]) for i, r in enumerate(ranges)}, index=rows.index)
    uncovered = ~selected.any(axis=1)
    nearest = distances.eq(distances.min(axis=1), axis=0) & distances.lt(math.inf)
    selected.loc[uncovered] = nearest.loc[uncovered]
    return selected


def age_distance(rows: pd.DataFrame, bounds: list[AgeBound] | None) -> pd.Series:
    """Distance in calendar days to the closest age boundary; unknown bounds cannot be extrapolated."""
    distance = pd.Series(math.inf if bounds is None else 0.0, index=rows.index)
    for bound in [] if bounds is None else bounds:
        boundary = age_boundary(rows, bound)
        outside = ~compare(rows["draw_date"], boundary, bound["operator"])
        difference = (rows["draw_date"] - boundary).abs().dt.days.where(outside, 0)
        distance = pd.concat([distance, difference], axis=1).max(axis=1)
    return distance


def reference_diagnostics(
    measured: pd.DataFrame,
    baseline_window: str,
    references: dict[str, list[ReferenceDefinition]],
) -> pd.DataFrame:
    """Explain unresolved patient scores, marking which raw draws contribute to the score."""
    eligible = reference_population_measurements(measured, baseline_window)
    failures = []
    for key, rows in eligible.groupby("test_key", sort=False):
        ranges = references.get(str(key), [])
        rows = rows.join(measurement_classifications(rows, ranges))
        rows["is_median_draw"] = median_draw_mask(rows)
        patients = classify_reference_patients(rows, baseline_window)
        unknown = patients.loc[~patients["is_known"], [*GROUP_KEYS, "id"]]
        rows = rows.merge(unknown, on=[*GROUP_KEYS, "id"], validate="many_to_one")
        if rows.empty:
            continue
        rows["reason"] = ""
        for condition, reason in (
            (rows["matching_intervals"].eq(0) & rows["is_median_draw"], "no_applicable_interval"),
            (rows["matching_intervals"].gt(1) & rows["is_median_draw"], "multiple_applicable_intervals"),
        ):
            rows.loc[condition, "reason"] += reason + ";"
        rows["reason"] = rows["reason"].str.rstrip(";")
        rows["reason"] = rows.groupby([*GROUP_KEYS, "id"])["reason"].transform(
            lambda reasons: ";".join(dict.fromkeys(reason for reason in reasons if reason))
        )
        rows["draw_classification"] = rows["outside"].map({0: "inside", 1: "outside"}).fillna("unclassified")
        applicable: dict[int, list[ReferenceDefinition]] = {index: [] for index in rows.index}
        selected = reference_matches(rows, ranges)
        for number, interval in enumerate(ranges):
            for index in rows.index[selected[number]]:
                applicable[index].append(interval)
        rows["selected_intervals"] = pd.Series(applicable).map(json.dumps)
        failures.append(rows[DIAGNOSTIC_COLUMNS])
    return pd.concat(failures, ignore_index=True) if failures else pd.DataFrame(columns=DIAGNOSTIC_COLUMNS)


def age_applicability(rows: pd.DataFrame, bounds: list[AgeBound] | None) -> pd.Series:
    """Compare draw dates with birthdays, including month-end/leap-day clipping."""
    applicable = pd.Series(bounds is not None, index=rows.index)
    if bounds is None:
        return applicable
    for bound in bounds:
        applicable &= compare(rows["draw_date"], age_boundary(rows, bound), bound["operator"])
    return applicable


def age_boundary(rows: pd.DataFrame, bound: AgeBound) -> pd.Series:
    months = bound["months"]
    if type(months) is not int or months < 0:
        raise ValueError("Reference age bounds must use nonnegative integer calendar months.")
    birthdays = {birth: birth + pd.DateOffset(months=months) for birth in rows["birth_date"].unique()}
    return rows["birth_date"].map(birthdays)


def compare(values: pd.Series, boundary: pd.Series | float, operator: str) -> pd.Series:
    if operator in {">=", "≥"}:
        return values >= boundary
    if operator == ">":
        return values > boundary
    if operator in {"<=", "≤"}:
        return values <= boundary
    if operator == "<":
        return values < boundary
    raise ValueError(f"Unsupported comparison operator: {operator!r}")


def load_transfer(path: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify CSV contents against the bundled manifest before parsing draw values."""
    measured = pd.read_csv(path, dtype=str, keep_default_na=False)
    if measured.columns.tolist() != MEASUREMENT_COLUMNS:
        raise ValueError("Unexpected measurement CSV columns; export a fresh package.")
    manifest = EXPORT_MANIFEST.copy()
    if manifest.get("format_version") != 7 or manifest_hash(manifest) != manifest.get("data_hash"):
        raise ValueError("Export settings hash/version mismatch; use the script provided with this export.")
    if csv_hash(measured) != manifest["measurements_sha256"]:
        raise ValueError("Measurement CSV hash mismatch; use the original exported CSV.")
    if calculator_hash(Path(__file__).read_bytes()) != manifest["calculator_sha256"]:
        raise ValueError("Calculator hash mismatch; use the script provided with this export.")

    if measured[["id", *GROUP_KEYS]].eq("").any().any():
        raise ValueError("Measurement IDs and group labels must be nonempty.")
    # Python's float parser round-trips the exported 17-digit results at strict limits.
    measured["blood test value"] = measured["blood test value"].map(float).astype(float)
    if not measured["blood test value"].map(math.isfinite).all():
        raise ValueError("Nonfinite blood test value.")
    measured["draw days since diagnosis"] = measured["draw days since diagnosis"].map(int).astype("int64")
    if measured.duplicated(["id", *GROUP_KEYS, "draw days since diagnosis"]).any():
        raise ValueError("Expected one draw per patient/site/test/window/day.")
    expected = pd.DataFrame(manifest["groups"], columns=GROUP_COLUMNS)
    coverage = measured.merge(expected[GROUP_KEYS], on=GROUP_KEYS, how="left", indicator=True, validate="many_to_one")
    if coverage["_merge"].ne("both").any():
        raise ValueError("Measurement groups do not match the manifest.")
    return measured, manifest


def load_reference_thresholds(path: Path, manifest: dict[str, Any]) -> dict[str, list[ReferenceDefinition]]:
    """Verify the separate threshold CSV before reading its numeric limits and calendar-age bounds."""
    thresholds = pd.read_csv(path, dtype=str, keep_default_na=False)
    if thresholds.columns.tolist() != REFERENCE_COLUMNS:
        raise ValueError("Unexpected reference threshold CSV columns; use the exported file.")
    if csv_hash(thresholds) != manifest["reference_thresholds_sha256"]:
        raise ValueError("Reference threshold CSV hash mismatch; use the original exported CSV.")
    references: dict[str, list[ReferenceDefinition]] = {}
    for row in thresholds.to_dict("records"):
        interval: ReferenceDefinition = {
            "age_group": row["age_group"],
            "age_bounds": json.loads(row["age_bounds"]),
            "lower": None if row["lower"] == "" else float(row["lower"]),
            "upper": None if row["upper"] == "" else float(row["upper"]),
            "lower_operator": row["lower_operator"],
            "upper_operator": row["upper_operator"],
        }
        references.setdefault(row["test_key"], []).append(interval)
    return references


def csv_hash(frame: pd.DataFrame) -> str:
    """Hash canonical CSV records, retaining numeric precision and normalizing quoting/newlines."""
    records = frame.to_csv(index=False, float_format="%.17g", lineterminator="\n")
    return sha256_bytes(records.encode("utf-8"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def calculator_hash(source: bytes) -> str:
    """Hash code separately from editable paths and the export manifest, which is verified by data_hash."""
    script = source.decode("utf-8")
    for section in ("REMOTE SETTINGS", "EXPORT MANIFEST"):
        script = replace_script_section(script, section, "")
    return sha256_bytes(script.encode("utf-8"))


def replace_script_section(source: str, section: str, content: str) -> str:
    """Share marked-section handling between script generation and code hashing."""
    before, start, rest = source.partition(f"# BEGIN {section}\n")
    _old, end, after = rest.partition(f"# END {section}\n")
    if not start or not end:
        raise ValueError(f"Calculator {section} markers are missing; cannot generate or hash the script.")
    return before + start + content + end + after


def manifest_hash(manifest: dict[str, Any]) -> str:
    """Hash the CSV digest, calculator digest, scope, and reference snapshot together."""
    payload = {key: value for key, value in manifest.items() if key != "data_hash"}
    return sha256_bytes(json.dumps(payload, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8"))


if __name__ == "__main__":
    print(f"Wrote aggregate percentages to {run_calculation(Path(__file__).resolve().parent).resolve()}")
