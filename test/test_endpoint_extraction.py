from __future__ import annotations

import logging

import pandas as pd
import pytest

from src.data import endpoint_extraction


def source_rows() -> pd.DataFrame:
    return pd.DataFrame({
        "ID": ["kept", "undated", "undated", "mixed", "mixed", "negative", "unselected"],
        "Dager_siden_BC_diagnose": ["0 days", None, None, None, "20 days", "-1 days", "30 days"],
        "Datatype": ["Diagnoser"] * 4 + ["Død", "Diagnoser", "Diagnoser"],
        "Kode": ["C50"] * 7,
        "Tekst": ["Example"] * 7,
        "BC_diagnose_år": [2010] * 7,
    })


def test_source_logging_explains_losses_without_changing_retained_records(caplog: pytest.LogCaptureFixture) -> None:
    source = source_rows()
    original = source.copy(deep=True)
    selected = pd.Series(["kept", "undated", "mixed", "absent", "negative", "kept", None])
    with caplog.at_level(logging.INFO, logger=endpoint_extraction.__name__):
        retained = endpoint_extraction.prepare_endpoint_source(source, selected)

    expected = source.iloc[[0, 4, 5]].reset_index(drop=True)
    expected["Dager_siden_BC_diagnose"] = pd.Series([0, 20, -1], dtype="Int64")
    expected.loc[1, ["Kode", "Tekst"]] = "Død"
    pd.testing.assert_frame_equal(retained, expected)
    pd.testing.assert_frame_equal(source, original)
    assert "to_keep selection: 7 rows; 5 unique patients; 1 repeated IDs; 1 missing-ID rows" in caplog.messages
    assert "selected patients absent from raw DIPS: 1" in caplog.messages
    assert "selected patients with no dated DIPS records: 1" in caplog.messages
    assert "reason=absent from raw DIPS; patient_id='absent'; raw_records=0; dated_records=0" in caplog.messages
    assert "reason=no dated DIPS records; patient_id='undated'; raw_records=2; dated_records=0" in caplog.messages
    assert "selected DIPS records: 6; missing-date records removed: 3; retained records: 3" in caplog.messages
    assert "source reconciliation: 5 selected = 1 absent + 1 without dates + 3 retained patients" in caplog.messages


def test_invalid_source_dates_still_raise() -> None:
    source = source_rows()
    source.loc[source["ID"] == "unselected", "Dager_siden_BC_diagnose"] = "invalid"

    with pytest.raises(ValueError):
        endpoint_extraction.prepare_endpoint_source(source, pd.Series(["kept"]))


@pytest.mark.parametrize("selected", [pd.Series([], dtype=str), pd.Series(["kept"])])
def test_source_logging_reports_zero_losses(selected: pd.Series, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=endpoint_extraction.__name__):
        retained = endpoint_extraction.prepare_endpoint_source(source_rows(), selected)

    assert retained["ID"].tolist() == selected.tolist()
    assert "selected patients absent from raw DIPS: 0" in caplog.messages
    assert "selected patients with no dated DIPS records: 0" in caplog.messages


def test_export_logging_identifies_missing_unexpected_and_duplicate_patients(caplog: pytest.LogCaptureFixture) -> None:
    source = pd.DataFrame({"ID": ["kept", "lost", "lost"]})
    endpoints = pd.DataFrame({"id": ["kept", "kept", "unexpected"]})

    with caplog.at_level(logging.INFO, logger=endpoint_extraction.__name__):
        endpoint_extraction.log_endpoint_export(source, endpoints)

    assert "endpoint export: 3 rows; 2 unique patients; 1 repeated IDs; 0 missing-ID rows" in caplog.messages
    assert "patients lost between dated DIPS and endpoint export: 1; IDs=['lost']" in caplog.messages
    assert "unexpected endpoint patients: 1; IDs=['unexpected']" in caplog.messages
    assert "duplicate endpoint patient IDs: ['kept']" in caplog.messages


@pytest.mark.parametrize(("selected_id", "retained_count"), [(1.0, 1), ("1", 0)])
def test_source_diagnostics_preserve_existing_id_matching(
    selected_id: str | float, retained_count: int, caplog: pytest.LogCaptureFixture,
) -> None:
    source = source_rows().iloc[[0]].copy()
    source["ID"] = 1
    selected = pd.Series([selected_id])
    with caplog.at_level(logging.INFO, logger=endpoint_extraction.__name__):
        retained = endpoint_extraction.prepare_endpoint_source(source, selected)

    assert len(retained) == retained_count
    assert f"ID dtype=int64; to_keep ID dtype={selected.dtype}" in caplog.text


