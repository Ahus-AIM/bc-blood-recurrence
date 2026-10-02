"""Migration boundaries: fixed paths, original extraction quirks and synthetic-data isolation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.extraction.blood import extract_blood_tests
from src.paths import CONFIG, OUTPUT, PROCESSED, apply_fixed_paths
from src.script import generate_synthetic_data, run_demo


def test_original_blood_cleaning_keeps_order_fallback_and_nonnumeric_rows() -> None:
    source = pd.DataFrame({
        "ID": ["A"] * 6 + ["excluded"],
        "Dager_siden_BC_diagnose": [40, 32, 50, 60, 70, 80, 90],
        "Tekst": ["KOS_FLAB_B-Hemoglobin (Hb)", "B-HbA1C", "U-Test", "P-Test", "P-Missing", "P-Other", "P-Test"],
        "Svar1": ["< 12,3", "6,5", "1", "not numeric", "not numeric", "4", "7"],
        "Svar2": ["99", None, None, "> 3,2", None, None, None],
        "Prøvesvar_ENHET": ["G/DL", "%", "U/L", "10*9/L", "U/L", "U/L", "U/L"],
        # The original notebook does not impose an additional Datatype filter here.
        "Datatype": ["Blodprøver"] * 5 + ["Prosedyrer", "Blodprøver"],
    })
    original = source.copy(deep=True)
    result = extract_blood_tests(source, pd.DataFrame({"id": ["A"]})).reset_index(drop=True)
    assert result["days_since_diagnosis"].tolist() == [40, 32, 60, 70, 80]
    assert result["test_name"].tolist() == ["B-Hemoglobin", "B-HbA1c", "P-Test", "P-Missing", "P-Other"]
    assert result["unit"].tolist() == ["g/dl", "mmol/mol", "x10e9/l", "u/l", "u/l"]
    assert result["result"].iloc[0] == 12.3
    assert result["result"].iloc[1] == pytest.approx(47.529514647)
    assert result["result"].iloc[2] == 3.2
    assert pd.isna(result["result"].iloc[3])
    assert result["result"].iloc[4] == 4
    pd.testing.assert_frame_equal(source, original)


def test_configuration_rejects_path_overrides() -> None:
    with pytest.raises(ValueError, match="Paths are fixed"):
        apply_fixed_paths(CONFIG / "ebm_survival.yml", {"dataset": {"bloodtest_csv_path": "/elsewhere.csv"}})


def test_relocated_saved_run_retains_nulls_and_scientific_settings() -> None:
    raw = {
        "dataset": {"endpoint_csv_path": "/original/bc_endpoints.csv", "bc_diagnosis_days_csv_path": None},
        "output": {"normal_ranges_csv_path": None},
        "ebm": {"max_rounds": 50_000},
    }
    apply_fixed_paths(OUTPUT / "benchmark/All/6m/__all_blood_tests__/training_config.yml", raw)
    assert raw["dataset"]["endpoint_csv_path"] == str(PROCESSED / "bc_endpoints.csv")
    assert raw["dataset"]["bc_diagnosis_days_csv_path"] is None
    assert raw["output"]["normal_ranges_csv_path"] is None
    assert raw["ebm"] == {"max_rounds": 50_000}


def test_synthetic_generation_refuses_existing_data_before_writing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("RAW", "PROCESSED", "OUTPUT"):
        directory = tmp_path / name.lower()
        directory.mkdir()
        monkeypatch.setattr(generate_synthetic_data, name, directory)
    source = tmp_path / "processed/do_not_overwrite.csv"
    source.write_bytes(b"existing data\n")
    with pytest.raises(FileExistsError):
        generate_synthetic_data.main()
    assert source.read_bytes() == b"existing data\n"
    assert list((tmp_path / "raw").iterdir()) == []
    assert list((tmp_path / "output").iterdir()) == []


def test_synthetic_marker_cannot_authorize_modified_inputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(run_demo, "RAW", tmp_path)
    (tmp_path / "SYNTHETIC.json").write_text(json.dumps({"synthetic": True, "patients": 72, "seed": 42}))
    (tmp_path / "dips.csv").write_text("changed source\n")
    with pytest.raises(ValueError, match="Synthetic input changed"):
        run_demo.validate_synthetic_inputs()
