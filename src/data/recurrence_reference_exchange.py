from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Any

import pandas as pd

from src.data import PatientBloodTestDataset
from src.data.recurrence_blood_test_config import RecurrenceBloodTestConfig
from src.data.recurrence_blood_test_insights import recurrence_window_measurements
from src.data.recurrence_reference_import import REFERENCE_COLUMNS as REPORT_REFERENCE_COLUMNS
from src.data.recurrence_reference_import import import_reference_percentages
from src.data.reference_intervals import reference_age_bounds, reference_ranges_for_test
from src.data.report_return_lookup import find_report_return, return_directory
from src.script.calculate_recurrence_reference_percentages import (
    GROUP_COLUMNS,
    MEASUREMENT_COLUMNS,
    REFERENCE_COLUMNS,
    ReferenceDefinition,
    calculator_hash,
    csv_hash,
    manifest_hash,
    reference_group_percentages,
    replace_script_section,
)

SOURCE_DIR = Path(__file__).resolve().parents[1]
CALCULATOR_PATH = SOURCE_DIR / "script" / "calculate_recurrence_reference_percentages.py"
REMOTE_CONFIG_PATH = SOURCE_DIR / "config" / "recurrence_reference_remote.json"


def reference_threshold_export(config: RecurrenceBloodTestConfig, test_keys: tuple[str, ...]) -> pd.DataFrame:
    """One row per female reference interval; tests without usable intervals have no rows."""
    records = []
    for test_key in sorted(test_keys):
        for interval in reference_ranges_for_test(config.reference_ranges, test_key):
            records.append(
                {
                    "test_key": test_key,
                    **asdict(interval),
                    "age_bounds": json.dumps(reference_age_bounds(interval.age_group), sort_keys=True),
                }
            )
    records.sort(key=lambda record: json.dumps(record, sort_keys=True, ensure_ascii=False, allow_nan=False))
    return pd.DataFrame.from_records(records, columns=REFERENCE_COLUMNS)


def remote_reference_test_keys(config: RecurrenceBloodTestConfig, test_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Only tests with multiple usable female reference intervals need remote age selection."""
    return tuple(key for key in test_keys if len(reference_ranges_for_test(config.reference_ranges, key)) > 1)


def local_reference_definitions(
    config: RecurrenceBloodTestConfig, test_keys: tuple[str, ...],
) -> dict[str, list[ReferenceDefinition]]:
    """Read sole intervals directly from configuration, without the remote export."""
    references: dict[str, list[ReferenceDefinition]] = {}
    for key in test_keys:
        intervals = reference_ranges_for_test(config.reference_ranges, key)
        if len(intervals) != 1:
            continue
        interval = intervals[0]
        references[key] = [{
            "age_group": interval.age_group,
            "age_bounds": [],
            "lower": interval.lower,
            "upper": interval.upper,
            "lower_operator": interval.lower_operator,
            "upper_operator": interval.upper_operator,
        }]
    return references


def reference_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Express report groups in the calculator's schema, preserving their order."""
    return frame.rename(columns={"blood_test": "blood test", "site": "recurrence site", "window": "interval"})[
        GROUP_COLUMNS
    ]


def measurement_export(
    measurements: pd.DataFrame,
    frame: pd.DataFrame,
    config: RecurrenceBloodTestConfig,
) -> pd.DataFrame:
    """Export each observed draw with its value and day, preserving median-value ties."""
    export = measurements.rename(
        columns={
            "patient_id": "id",
            "site": "recurrence site",
            "result": "blood test value",
            "days_since_diagnosis": "draw days since diagnosis",
        }
    ).copy()
    export["interval"] = export["window_index"].map(
        {index: window.label for index, window in enumerate(config.windows)}
    )
    display = frame.drop_duplicates("test_key").set_index("test_key")
    export["blood test name"] = export["test_key"].map(display["blood_test"])
    export["unit"] = export["test_key"].map(display["unit"])
    return (
        export[MEASUREMENT_COLUMNS]
        .sort_values(
            ["id", "recurrence site", "test_key", "interval", "draw days since diagnosis"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def reference_manifest(
    draws: pd.DataFrame,
    thresholds: pd.DataFrame,
    frame: pd.DataFrame,
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> dict[str, Any]:
    """Bind returns to cohort membership and calculation inputs, excluding file metadata and model settings."""
    remote_keys = tuple(sorted(draws["test_key"].unique()))
    groups = reference_groups(frame.loc[frame["test_key"].isin(remote_keys)]).sort_values(
        ["test_key", "recurrence site", "interval"]
    )
    analysis_context = {
        "cohort_patient_ids": sorted({
            item.patient_id for items in dataset.prepared_data.items_by_split.values() for item in items
        }),
        "site_rules": [asdict(rule) for rule in config.site_rules],
        "other_label": config.other_label,
    }
    if config.include_overall:
        analysis_context["overall_label"] = config.overall_label
    manifest = {
        "format_version": 7,
        "analysis_sha256": manifest_hash(analysis_context),
        "measurements_sha256": csv_hash(draws),
        "reference_thresholds_sha256": csv_hash(thresholds),
        "calculator_sha256": calculator_hash(CALCULATOR_PATH.read_bytes()),
        "groups": groups.to_dict("records"),
        "context": {
            "remote_test_keys": list(remote_keys),
            "windows": [asdict(window) for window in config.windows],
        },
    }
    manifest["data_hash"] = manifest_hash(manifest)
    return manifest


def export_reference_package(
    frame: pd.DataFrame,
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
    *,
    draws: pd.DataFrame | None = None,
) -> tuple[Path, dict[str, Any]]:
    """Export remote draws and only their thresholds, plus a standalone calculator."""
    if draws is None:
        draws = measurement_export(recurrence_window_measurements(dataset, config), frame, config)
    remote_keys = remote_reference_test_keys(config, dataset.selected_test_keys)
    export = draws.loc[draws["test_key"].isin(remote_keys)].reset_index(drop=True)
    thresholds = reference_threshold_export(config, tuple(sorted(export["test_key"].unique())))
    manifest = reference_manifest(export, thresholds, frame, dataset, config)
    directory = config.output_dir / "remote_reference"
    directory.mkdir(parents=True, exist_ok=True)
    export.to_csv(directory / "measurements.csv", index=False, float_format="%.17g", lineterminator="\n")
    thresholds.to_csv(directory / "reference_thresholds.csv", index=False, float_format="%.17g", lineterminator="\n")
    settings = {
        "measurements_csv": "measurements.csv",
        "reference_thresholds_csv": "reference_thresholds.csv",
        "patients_csv": "patients.csv",
        "output_csv": "reference_percentages.csv",
        "diagnostics_csv": "reference_diagnostics.csv",
        **json.loads(REMOTE_CONFIG_PATH.read_text(encoding="utf-8")),
    }
    (directory / CALCULATOR_PATH.name).write_text(render_remote_calculator(settings, manifest), encoding="utf-8")
    return directory, manifest


def render_remote_calculator(settings: dict[str, Any], manifest: dict[str, Any]) -> str:
    """Embed editable file settings and the immutable calculation manifest in the standalone script."""
    source = CALCULATOR_PATH.read_text(encoding="utf-8")
    for section, name, values in (
        ("REMOTE SETTINGS", "REMOTE_SETTINGS", settings),
        ("EXPORT MANIFEST", "EXPORT_MANIFEST", manifest),
    ):
        content = f"{name}: dict[str, Any] = {pformat(values, width=110, sort_dicts=False)}\n"
        source = replace_script_section(source, section, content)
    return source


def prepare_reference_report(
    frame: pd.DataFrame,
    dataset: PatientBloodTestDataset,
    config: RecurrenceBloodTestConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Use the shared calculator locally for sole intervals, then import the remaining tests."""
    draws = measurement_export(recurrence_window_measurements(dataset, config), frame, config)
    directory, manifest = export_reference_package(frame, dataset, config, draws=draws)
    local_references = local_reference_definitions(config, dataset.selected_test_keys)
    counts = reference_group_percentages(
        draws, reference_groups(frame), config.windows[0].label, local_references
    ).rename(columns={"percentage outside reference area": "outside_reference_percent"})
    frame = frame.drop(columns="data_hash", errors="ignore").copy()
    for column in REPORT_REFERENCE_COLUMNS:
        frame[column] = counts[column].to_numpy()
    remote_keys = manifest["context"]["remote_test_keys"]
    remote_rows = frame["test_key"].isin(remote_keys)
    frame.loc[remote_rows & frame["reference_status"].eq("unavailable_reference"), "reference_status"] = (
        "pending_remote"
    )
    returned = None
    if remote_keys:
        returned = find_report_return(
            return_directory("reference_percentages"),
            "data_hash",
            manifest["data_hash"],
            load=lambda path: import_reference_percentages(frame.loc[remote_rows], path, manifest["data_hash"]),
            equivalent=lambda left, right: left.equals(right),
        )
    if returned is not None:
        for column in REPORT_REFERENCE_COLUMNS:
            frame.loc[remote_rows, column] = returned.value[column].to_numpy()
    return frame, {
        "package_dir": str(directory),
        "data_hash": manifest["data_hash"],
        "measurements_sha256": manifest["measurements_sha256"],
        "reference_thresholds_sha256": manifest["reference_thresholds_sha256"],
        "calculator_sha256": manifest["calculator_sha256"],
        "percentages_csv": None if returned is None else str(returned.paths[0]),
        "percentages_csvs": [] if returned is None else [str(path) for path in returned.paths],
        "remote_test_keys": list(remote_keys),
        "local_test_keys": list(local_references),
        "status": "verified" if returned is not None else "pending_remote" if remote_keys else "not_required",
    }
