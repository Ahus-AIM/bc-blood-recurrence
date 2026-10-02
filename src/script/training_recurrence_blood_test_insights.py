from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from src.data import PatientBloodTestDataset
from src.data.blood_test_display import load_test_name_translation
from src.data.measurement_selection import recurrence_report_measurements
from src.data.recurrence_blood_test_config import load_recurrence_blood_test_config
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.script.recurrence_blood_test_reporting import write_recurrence_blood_test_report

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "recurrence_blood_test_insights.yml"


def write_training_recurrence_blood_test_insights(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    *,
    plot_config_path: Path = CONFIG_PATH,
) -> dict[str, object]:
    """Write one report with shared study measurements, releasing report-only values afterwards."""
    with recurrence_report_measurements(dataset.prepared_data):
        return write_materialized_report(dataset, output_dir, plot_config_path=plot_config_path)


def write_materialized_report(
    dataset: PatientBloodTestDataset,
    output_dir: Path,
    *,
    plot_config_path: Path,
) -> dict[str, object]:
    """Report this cohort using shared hash-matched returns and local sole-interval percentages."""
    snapshot = output_dir / "training_config.yml"
    config = load_recurrence_blood_test_config(
        plot_config_path,
        source_config_path=snapshot if snapshot.is_file() else None,
        dataset_config=dataset.config,
    )
    config = replace(config, output_dir=output_dir)
    translation = load_test_name_translation(config.translation_path)
    frame = build_recurrence_blood_test_insights(dataset, config, translation)
    return write_recurrence_blood_test_report(frame, dataset, config)
