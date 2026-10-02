from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from src.data import PatientBloodTestDataset
from src.data.dataset_config import PatientDatasetConfig
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.dataset_statistics_export import dataset_statistics_payload
from src.script.plot.dataset_statistics import CONFIG_PATH, load_config, validate_statistics_source
from src.script.plot.dataset_statistics_config import DatasetStatisticsSourceConfig, load_dataset_section
from src.script.single_bloodtest_benchmark_config import load_config as load_benchmark_config
from test.test_patient_bloodtest import blood_row, config, endpoint_row


def test_statistics_default_uses_effective_benchmark_dataset() -> None:
    source_path = load_config(CONFIG_PATH).source.dataset_config_path
    assert source_path == CONFIG_PATH.parent / "ebm_single_bloodtest_benchmark.yml"
    benchmark, _ = load_benchmark_config(source_path)

    dataset = load_dataset_section(source_path)

    assert dataset == benchmark["dataset"]
    assert dataset["min_blood_test_types_per_day"] == 13
    assert dataset["include_blood_test_change"] is False


def test_benchmark_override_updates_cohort_counts_and_rejects_old_statistics(tmp_path: Path) -> None:
    tests = [f"A{index}" for index in range(13)]
    base_dataset = {
        **config(
            tmp_path,
            [blood_row("one_test", 40, test_name=tests[0], result=1.0, unit="u")]
            + [blood_row("full_panel", 40, test_name=name, result=1.0, unit="u") for name in tests],
            [endpoint_row(patient_id, last_entry_days=100) for patient_id in ("one_test", "full_panel")],
            split_by_patient={"one_test": "train", "full_panel": "train"},
        ),
        "selected_blood_tests": [f"{name}|||u" for name in tests],
        "min_blood_test_types_per_day": 13,
        "min_landmark_day": 31,
        "min_diagnosis_year": 2004,
    }
    base_path = tmp_path / "base.yml"
    base_path.write_text(yaml.safe_dump({"dataset": base_dataset}), encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.yml"
    benchmark_path.write_text(
        yaml.safe_dump({
            "source": {"base_config_path": "base.yml"},
            "dataset": {"min_blood_test_types_per_day": 1, "min_diagnosis_year": None},
            "benchmark": {},
        }),
        encoding="utf-8",
    )
    source = DatasetStatisticsSourceConfig.from_mapping(
        {"dataset_config_path": "benchmark.yml"}, config_dir=tmp_path,
    )
    assert source.endpoint_csv_path == Path(base_dataset["endpoint_csv_path"])
    effective = load_dataset_section(benchmark_path)
    assert effective["min_diagnosis_year"] is None
    old_dataset = PatientBloodTestDataset(base_dataset)
    new_dataset = PatientBloodTestDataset(PatientDatasetConfig.from_mapping(effective, config_dir=tmp_path))
    old_summary = build_dataset_statistics_summary(old_dataset.config, prepared_dataset=old_dataset)
    new_summary = build_dataset_statistics_summary(new_dataset.config, prepared_dataset=new_dataset)

    assert old_summary.stage_ids[-1] == {"full_panel"}
    assert new_summary.stage_ids[-1] == {"one_test", "full_panel"}
    assert new_summary.inclusion_box_counts[-1] == "N = 2\nblood tests = 14"
    assert new_summary.blood_test_counts == (2, *(1 for _ in range(12)))
    assert all("different blood tests" not in label for label in new_summary.inclusion_box_labels)
    assert all("different blood tests" not in label for label in new_summary.exclusion_box_labels)
    assert all("different blood tests" not in label for label in new_summary.series_labels)
    payload = dataset_statistics_payload(new_summary, dataset_config=effective)
    validate_statistics_source(payload, effective, tmp_path)
    old_payload = dataset_statistics_payload(old_summary, dataset_config=base_dataset)
    with pytest.raises(ValueError, match="min_blood_test_types_per_day"):
        validate_statistics_source(old_payload, effective, tmp_path)
