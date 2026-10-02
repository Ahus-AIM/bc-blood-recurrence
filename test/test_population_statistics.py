from __future__ import annotations

from pathlib import Path

import pytest

from src.data import PatientBloodTestDataset
from src.data.dataset_statistics import build_dataset_statistics_summary
from src.data.population_statistics import (
    PopulationStatisticsConfig,
    build_population_statistics_table,
    population_statistics_latex,
    write_population_statistics_latex,
)
from test.test_patient_bloodtest import blood_row, config, endpoint_row


def test_population_statistics_latex_counts_first_day_multi_site_recurrence(
    tmp_path: Path,
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    table_config = PopulationStatisticsConfig.from_mapping(
        {
            "table_label": "tab:population_statistics_test",
            "site_labels": ["Bone", "Brain", "Liver", "Lung", "Pleura"],
            "site_regexes": [r"C795\d*", r"C793\d*", r"C787\d*", r"C780\d*", r"C782\d*"],
        }
    )

    table = build_population_statistics_table(dataset, table_config)
    dataset_statistics = build_dataset_statistics_summary(
        dataset.config,
        prepared_dataset=dataset,
    )
    latex = population_statistics_latex(table, table_config)

    assert tuple(row.label for row in table.rows[:3]) == ("Patients", "Landmark days", "Blood tests")
    assert r"\label{tab:population_statistics_test}" in latex
    assert "Patients & 1 & 1 & 1 & 3 \\\\" in latex
    assert "Landmark days & 2 & 1 & 1 & 4 \\\\" in latex
    assert dataset_statistics.split_box_counts == (
        "2016 - 2016\nN = 1\nblood tests = 3",
        "2021 - 2021\nN = 1\nblood tests = 1",
        "2022 - 2022\nN = 1\nblood tests = 1",
    )
    assert "Blood tests & 3 & 1 & 1 & 5 \\\\" in latex
    assert (
        "Median observation time, years & 0.3 (0.3--0.3) & 0.5 (0.5--0.5) "
        "& 0.2 (0.2--0.2) & 0.3 (0.2--0.4) \\\\"
        in latex
    )
    assert "Recurrences & 2 & 0 & 1 & 3 \\\\" in latex
    assert r"\quad Bone & 1 & 0 & 0 & 1 \\" in latex
    assert r"\quad Brain & 1 & 0 & 0 & 1 \\" in latex
    assert r"\quad Liver & 0 & 0 & 0 & 0 \\" in latex
    assert r"\quad Lung & 0 & 0 & 0 & 0 \\" in latex
    assert r"\quad Pleura & 0 & 0 & 0 & 0 \\" in latex
    assert r"\quad Other & 0 & 0 & 1 & 1 \\" in latex


def test_write_population_statistics_latex_writes_configured_output(
    tmp_path: Path,
) -> None:
    dataset = population_statistics_dataset(tmp_path)
    table_config = PopulationStatisticsConfig.from_mapping({"output_latex_path": "tables/population.tex"})

    metadata = write_population_statistics_latex(dataset, table_config, tmp_path)

    assert metadata == {"path": "tables/population.tex"}
    assert (tmp_path / "tables" / "population.tex").is_file()


def test_population_statistics_rejects_mismatched_site_config() -> None:
    with pytest.raises(ValueError, match="site_labels.*site_regexes"):
        PopulationStatisticsConfig.from_mapping(
            {
                "site_labels": ["Bone"],
                "site_regexes": [r"C795\d*", r"C793\d*"],
            }
        )


def population_statistics_dataset(tmp_path: Path) -> PatientBloodTestDataset:
    cfg = {
        **config(
            tmp_path,
            [
                blood_row("train_multi_site", 40, test_name="A", unit="u"),
                blood_row("train_multi_site", 40, test_name="B", unit="u"),
                blood_row("train_multi_site", 50, test_name="A", unit="u"),
                blood_row("validation_censored", 40, test_name="A", unit="u"),
                blood_row("test_other", 40, test_name="A", unit="u"),
            ],
            [
                endpoint_row(
                    "train_multi_site",
                    diagnosis_year=2016,
                    diagnosis_to_ddfs=[100, 100, 120],
                    ddfs_code=["C795", "C793", "C780"],
                    ddfs_description=["bone", "brain", "lung"],
                    last_entry_days=400,
                ),
                endpoint_row(
                    "validation_censored",
                    diagnosis_year=2021,
                    diagnosis_to_death=200,
                    last_entry_days=500,
                ),
                endpoint_row(
                    "test_other",
                    diagnosis_year=2022,
                    diagnosis_to_ddfs=60,
                    ddfs_code="C790",
                    ddfs_description="other metastasis",
                    last_entry_days=300,
                ),
            ],
            split_by_patient={
                "train_multi_site": "train",
                "validation_censored": "val",
                "test_other": "test",
            },
        ),
        "selected_blood_tests": ["A|||u", "B|||u"],
        "min_landmark_day": 30,
    }
    return PatientBloodTestDataset(cfg)
