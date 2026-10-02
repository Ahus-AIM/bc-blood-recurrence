from __future__ import annotations

import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.extraction.endpoints import capped_recurrence_records
from src.data import PatientBloodTestDataset
from src.data.benchmark_landmarks import BenchmarkLandmarks
from src.data.dataset import load_patient_data_sources
from src.data.dataset_config import PatientDatasetConfig
from src.data.landmark_event_counts import recurrence_target_item, has_first_recurrence_site
from src.data.recurrence_site_common import RecurrenceSiteRule, site_label
from src.data.recurrence_site_incidence import load_item_endpoint_site_rows, survival_durations
from test.test_patient_bloodtest import blood_row, config, endpoint_row


@pytest.mark.parametrize("code", ["C787", "C78.7", "c78.7", "C78 7"])
def test_recurrence_code_formats_agree_across_models_and_reports(tmp_path: Path, code: str) -> None:
    cfg = config(tmp_path, [blood_row("P0", 50)], [endpoint_row(
        "P0", diagnosis_year=2010, diagnosis_to_ddfs=120, ddfs_code=code,
        diagnosis_to_death=100, last_entry_days=200,
    )])
    cfg["endpoint_ddfs_code_filter_regex"] = r"C787\d*"
    sources = load_patient_data_sources(PatientDatasetConfig.from_mapping(cfg))
    dataset = PatientBloodTestDataset(cfg, data_sources=sources)
    item = dataset.items[0]
    canonical = PatientBloodTestDataset(replace(
        dataset.config, endpoint_ddfs_code_filter_regex=dataset.config.input_recurrence_code_filter_regex,
    ))
    projected = BenchmarkLandmarks(canonical).table(dataset.config)
    direct = dataset.to_landmark_table(bootstrap=False)
    reported = recurrence_target_item(
        item, re.compile(dataset.config.input_recurrence_code_filter_regex), re.compile(r"C787\d*"), dataset.config,
    )

    for table in (direct, projected):
        assert table.durations.tolist() == [50]
        assert table.events.tolist() == [True]
    assert item.input_recurrence_day == reported.input_recurrence_day == reported.endpoint_day == 100
    assert item.distant_recurrence_events[0].code == code
    assert sources.endpoint_patients["P0"].recurrence_events[0].code == code
    assert has_first_recurrence_site(item, re.compile(r"C7[89]\d*"), re.compile(r"C787\d*"))
    assert site_label(code, (RecurrenceSiteRule("Liver", r"C787\d*"),), "Other") == "Liver"
    rows = load_item_endpoint_site_rows(dataset.items, dataset.config)
    assert rows["code"].tolist() == [code]
    durations, events = survival_durations(rows, "recurrence")
    np.testing.assert_equal(durations, [0])
    np.testing.assert_equal(events, [True])


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("early_code", ["C787", "c78.7"])
def test_endpoint_extraction_keeps_earliest_normalized_code_and_original_row(reverse: bool, early_code: str) -> None:
    rows = [
        {"ID": "P0", "Datatype": "Diagnoser", "Kode": "C787", "Dager_siden_BC_diagnose": 80, "Tekst": "later"},
        {"ID": "P0", "Datatype": "Diagnoser", "Kode": early_code, "Dager_siden_BC_diagnose": 40, "Tekst": None},
        {"ID": "P0", "Datatype": "Diagnoser", "Kode": "C795", "Dager_siden_BC_diagnose": 90, "Tekst": "bone"},
    ]
    dips = pd.DataFrame(rows[::-1] if reverse else rows)
    original = dips.copy(deep=True)
    deaths = pd.DataFrame({"ID": ["P0"], "days_from_diagnosis_to_death": [70]})
    result = capped_recurrence_records(dips, deaths).sort_values("Dager_siden_BC_diagnose")

    assert result["Dager_siden_BC_diagnose"].tolist() == [40, 70]
    assert result["Kode"].tolist() == [early_code, "C795"]
    assert result["Tekst"].tolist() == [None, "bone"]
    pd.testing.assert_frame_equal(dips, original)
