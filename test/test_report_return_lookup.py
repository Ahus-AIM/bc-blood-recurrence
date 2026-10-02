from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.data.cohort_age_exchange import AGE_FOLDS, find_age_summary
from src.data.report_return_lookup import return_directory


def write_ages(path: Path, *, cohort_hash: str = "expected", median: float = 60) -> None:
    pd.DataFrame([(fold, median, 50, 70, cohort_hash) for fold in AGE_FOLDS],
                 columns=["fold", "median", "lower_q", "upper_q", "hash"]).to_csv(path, index=False)


def test_lookup_uses_contents_and_accepts_equivalent_copies(caplog: pytest.LogCaptureFixture) -> None:
    directory = return_directory("ages")
    write_ages(directory / "arbitrary.csv")
    write_ages(directory / "unrelated.csv", cohort_hash="old")
    equivalent = pd.read_csv(directory / "arbitrary.csv", dtype=str)
    equivalent["median"] = "60.0"
    equivalent.iloc[::-1, ::-1].to_csv(directory / "duplicate.csv", index=False, encoding="utf-8-sig")
    (directory / "broken.csv").write_text("unrecognisable\ncontent")
    (directory / "binary.csv").write_bytes(b"\xff\xfe")
    (directory / "empty.csv").write_text("")
    (directory / "no_hash.csv").write_text("hash\n\"\"\n")

    returned = find_age_summary("expected")

    assert returned is not None
    assert returned.paths == (directory / "arbitrary.csv", directory / "duplicate.csv")
    assert returned.value["age_years"]["overall"]["median"] == 60
    assert "Skipping report return" in caplog.text
    assert "Skipping unreadable report return" in caplog.text


@pytest.mark.parametrize("problem", ["conflicting", "missing_quartile", "mixed_hash", "malformed", "broken_quoted_row"])
def test_any_bad_matching_file_fails_even_alongside_a_valid_one(problem: str) -> None:
    directory = return_directory("ages")
    write_ages(directory / "a_valid.csv")
    bad = directory / "z_bad.csv"
    write_ages(bad, median=61 if problem == "conflicting" else 60)
    if problem == "malformed":
        bad.write_text("fold,median,lower_q,upper_q,hash\ntrain,60,50,70,expected,extra\n")
    elif problem == "broken_quoted_row":
        bad.write_text("fold,median,lower_q,upper_q,hash\ntrain,60,50,70,expected\n\"unterminated")
    elif problem in {"mixed_hash", "missing_quartile"}:
        rows = pd.read_csv(bad, dtype=str)
        rows.loc[0, "hash" if problem == "mixed_hash" else "median"] = "old" if problem == "mixed_hash" else ""
        rows.to_csv(bad, index=False)

    with pytest.raises(ValueError):
        find_age_summary("expected")


@pytest.mark.parametrize("state", ["absent_directory", "empty_directory", "unrelated", "unreadable", "nested_only"])
def test_no_match_warns_and_returns_unavailable(state: str, caplog: pytest.LogCaptureFixture) -> None:
    directory = return_directory("ages")
    if state == "absent_directory":
        directory.rmdir()
    elif state == "unrelated":
        write_ages(directory / "old.csv", cohort_hash="old")
    elif state == "unreadable":
        (directory / "bad.csv").write_bytes(b"\xff")
    elif state == "nested_only":
        (directory / "nested").mkdir()
        write_ages(directory / "nested/ages.csv")

    assert find_age_summary("expected") is None
    assert "No matching report return" in caplog.text
