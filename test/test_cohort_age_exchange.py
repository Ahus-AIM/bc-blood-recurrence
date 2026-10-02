from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from src.data.cohort_age_exchange import (
    AGE_FOLDS,
    SUMMARY_COLUMNS,
    build_age_membership,
    load_age_summaries,
    load_cohort_membership,
    membership_hash,
)


@pytest.fixture
def membership() -> pd.DataFrame:
    return pd.DataFrame([
        ("001", "included", "train"),
        ("002", "included", "validation"),
        ("003", "included", "test"),
        ("NA", "excluded", ""),
    ], columns=["id", "population", "split"])


@pytest.fixture
def returned() -> pd.DataFrame:
    return pd.DataFrame([(fold, "60", "50", "70", "cohort-hash") for fold in AGE_FOLDS], columns=SUMMARY_COLUMNS)


def test_membership_expands_groups_and_preserves_literal_patient_ids(membership: pd.DataFrame, tmp_path: Path) -> None:
    cohort_hash = membership_hash(membership)
    exported = build_age_membership(membership, cohort_hash)
    assert exported.groupby("id")["fold"].agg(set).to_dict() == {
        "001": {"train", "overall", "included"},
        "002": {"validation", "overall", "included"},
        "003": {"test", "overall", "included"},
        "NA": {"excluded"},
    }
    path = tmp_path / "membership.csv"
    exported.to_csv(path, index=False)
    restored = load_cohort_membership(path, cohort_hash)
    pd.testing.assert_frame_equal(restored, membership)
    assert set(exported["hash"]) == {cohort_hash}


def test_hash_preserves_existing_snapshots_and_covers_the_complete_partition(membership: pd.DataFrame) -> None:
    serialized = json.dumps({"version": 1, "patients": membership.to_dict("records")},
                            sort_keys=True, separators=(",", ":")).encode("utf-8")
    cohort_hash = membership_hash(membership)
    assert cohort_hash == hashlib.sha256(serialized).hexdigest()
    assert membership_hash(membership.iloc[::-1]) == cohort_hash
    changed_fold = membership.copy()
    changed_fold.loc[0, "split"] = "test"
    assert membership_hash(changed_fold) != cohort_hash
    changed_id = membership.copy()
    changed_id.loc[0, "id"] = "1"
    assert membership_hash(changed_id) != cohort_hash
    assert membership_hash(membership.iloc[:-1]) != cohort_hash


@pytest.mark.parametrize("mutation", ["duplicate", "missing_alias", "overlap", "hash", "unknown_fold"])
def test_modified_membership_is_rejected(membership: pd.DataFrame, tmp_path: Path, mutation: str) -> None:
    cohort_hash = membership_hash(membership)
    exported = build_age_membership(membership, cohort_hash)
    if mutation == "duplicate":
        exported = pd.concat([exported, exported.iloc[:1]], ignore_index=True)
    elif mutation == "missing_alias":
        exported = exported[~((exported["id"] == "001") & (exported["fold"] == "overall"))]
    elif mutation == "overlap":
        exported.loc[len(exported)] = ["001", "excluded", cohort_hash]
    elif mutation == "hash":
        exported.loc[0, "hash"] = "another-cohort"
    else:
        exported.loc[0, "fold"] = "all"
    path = tmp_path / "membership.csv"
    exported.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_cohort_membership(path, cohort_hash)


@pytest.mark.parametrize("bad_value", ["", " ", "NA", "NaN", "inf", "-inf", "None", "True", "-1"])
@pytest.mark.parametrize("column", ["median", "lower_q", "upper_q"])
def test_every_summary_value_must_be_finite_nonnegative_numeric(
    returned: pd.DataFrame, tmp_path: Path, bad_value: str, column: str,
) -> None:
    returned.loc[0, column] = bad_value
    path = tmp_path / "ages.csv"
    returned.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_age_summaries(path, "cohort-hash")


@pytest.mark.parametrize("mutation", [
    "missing_fold", "duplicate_fold", "extra_fold", "unknown_fold", "wrong_hash", "blank_hash",
    "all_blank", "unordered_lower", "unordered_upper", "pooled_disagreement", "missing_column", "extra_n",
])
def test_incomplete_or_inconsistent_returns_are_rejected(returned: pd.DataFrame, tmp_path: Path, mutation: str) -> None:
    if mutation == "missing_fold":
        returned = returned.iloc[:-1]
    elif mutation == "duplicate_fold":
        returned.loc[0, "fold"] = "validation"
    elif mutation == "extra_fold":
        returned = pd.concat([returned, returned.iloc[:1]], ignore_index=True)
    elif mutation == "unknown_fold":
        returned.loc[0, "fold"] = "val"
    elif mutation in {"wrong_hash", "blank_hash"}:
        returned.loc[0, "hash"] = "another cohort" if mutation == "wrong_hash" else ""
    elif mutation == "all_blank":
        returned.loc[0, ["median", "lower_q", "upper_q"]] = ""
    elif mutation == "unordered_lower":
        returned.loc[0, "lower_q"] = "61"
    elif mutation == "unordered_upper":
        returned.loc[0, "upper_q"] = "59"
    elif mutation == "pooled_disagreement":
        returned.loc[returned["fold"] == "included", "median"] = "60.01"
    elif mutation == "missing_column":
        returned = returned.drop(columns="hash")
    else:
        returned["n"] = "1"
    path = tmp_path / "ages.csv"
    returned.to_csv(path, index=False)
    with pytest.raises(ValueError):
        load_age_summaries(path, "cohort-hash")


@pytest.mark.parametrize("contents", [
    "fold,median,lower_q,upper_q,hash,hash\ntrain,60,50,70,cohort-hash,cohort-hash\n",
    "fold,median,lower_q,upper_q,hash\ntrain,60,50,70\n",
    "fold,median,lower_q,upper_q,hash\ntrain,60,50,70,cohort-hash,extra\n",
])
def test_malformed_csv_fails(tmp_path: Path, contents: str) -> None:
    path = tmp_path / "ages.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_age_summaries(path, "cohort-hash")


def test_valid_return_preserves_precision_accepts_zero_and_is_order_independent(
    returned: pd.DataFrame, tmp_path: Path,
) -> None:
    returned.loc[0, ["median", "lower_q", "upper_q"]] = "0"
    returned.loc[returned["fold"].isin(["overall", "included"]), "median"] = "60.12345"
    path = tmp_path / "ages.csv"
    returned.iloc[::-1, ::-1].to_csv(path, index=False, encoding="utf-8-sig")
    summaries = load_age_summaries(path, "cohort-hash")
    assert summaries["age_years"]["train"] == {"median": 0, "q1": 0, "q3": 0}
    assert summaries["age_years"]["overall"]["median"] == 60.12345
    assert summaries["age_years"]["overall"] == summaries["age_years"]["included"]
    assert "counts" not in summaries
