from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path

from src.data import PatientBloodTestDataset
from src.data.blood_test_display import load_test_name_translation
from src.data.measurement_selection import recurrence_report_measurements
from src.data.recurrence_blood_test_config import load_recurrence_blood_test_config
from src.data.recurrence_blood_test_insights import build_recurrence_blood_test_insights
from src.script.logging_config import configure_logging
from src.script.recurrence_blood_test_reporting import write_recurrence_blood_test_report

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config" / "recurrence_blood_test_insights.yml"
LOGGER = logging.getLogger(__name__)


def export_recurrence_report(config_path: Path) -> None:
    config = load_recurrence_blood_test_config(config_path)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(config.output_dir / "run.log")
    dataset = PatientBloodTestDataset(replace(config.dataset, build_patient_timeline_tensors=False))
    translation = load_test_name_translation(config.translation_path)
    with recurrence_report_measurements(dataset.prepared_data):
        frame = build_recurrence_blood_test_insights(dataset, config, translation)
        write_recurrence_blood_test_report(frame, dataset, config)
    LOGGER.info("Wrote %d rows to %s", len(frame), config.output_dir)
