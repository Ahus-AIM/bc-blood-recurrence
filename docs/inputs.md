# Input contract

All paths below are relative to this repository. Filenames and directories are
fixed. CSV files use a header and comma separator, with quoting where necessary.
Preserve original identifiers, row order, precision and missing values: these can
affect source selection, imputation, model fitting and hashes.

| File | Required columns | Purpose |
| --- | --- | --- |
| `data/raw/dips.csv` | `ID`, `Datatype`, `Kode`, `Tekst`, `Dager_siden_BC_diagnose`, `BC_diagnose_år`, `Svar1`, `Svar2`, `Prøvesvar_ENHET` | Original clinical export |
| `data/raw/to_keep.csv` | `ID`, `to_keep` | Original selected source population; `to_keep` converts through integer to boolean |
| `data/raw/patients.csv` | `id`, `birth date`, `day of first bc diagnosis` | Exact dates for age-specific reference classification |
| `data/processed/bc_diagnosis_patient_age.csv` | `id`, `age` | Model ages at diagnosis, rounded down to the lower bound of five-year bands |

`Dager_siden_BC_diagnose` must have the original timedelta representation (for
example, `31 days`), not bare integer offsets interpreted as nanoseconds.
`Datatype` retains the original values, including `Diagnoser`, `Prosedyrer` and
`Død`. Blood-test names and units are normalized by the original notebook rules.
Both result columns retain the original decimal-comma and inequality-prefix
handling, HbA1c conversion, and preference for numeric `Svar1` over `Svar2`.
Date columns in `patients.csv` use `%Y-%m-%d`; separator, encoding, column names
and date formats are analysis input settings in `src/config/recurrence_reference_remote.json`.

Extraction writes these five files to `data/processed/`:

| File | Contents |
| --- | --- |
| `bc_endpoints.csv` | Diagnosis year, first recurrence records by code, death and last observation |
| `blood_test.csv` | Patient, day, test name, result and unit |
| `diagnosis_procedure.csv` | Patient, day, diagnosis/procedure, code and description |
| `bc_split.csv` | Patient and train/val/test assignment |
| `bc_diagnosis_days.csv` | Dated C50 diagnoses from the original full export |

The extraction log is `output/extraction.log`. As in the original notebooks,
endpoint selection and the C50-day export have different selection scopes.
Extraction preserves the CSV reload boundaries between notebook stages.

## External age summaries

The original clinical age-generation code is pending. Supply its model-age CSV
above; do not estimate precise report ages from the five-year-banded model ages.
Training exports the cohort membership at:

```
output/benchmark/All/6m/__all_blood_tests__/cohort_age_export/cohort_membership.csv
```

It contains `id,fold,hash`. Use this partition with the original external age
workflow, then place its returned CSV in `data/processed/ages/`. The returned
columns are `fold,median,lower_q,upper_q,hash`, with one row for each of `train`,
`validation`, `test`, `overall`, `included` and `excluded`. Every row must carry
the exported cohort hash. A missing matching return produces unavailable age
cells, so those reports do not reproduce the manuscript. The synthetic demo's
illustrative age routine is restricted to byte-verified invented inputs.

## Reference-range percentages

Training exports the original standalone reference calculator, measurements and
thresholds under `output/benchmark/All/6m/__all_blood_tests__/remote_reference/`.
`calculate_reference_returns` runs that calculator using `data/raw/patients.csv`
and places its hash-matched CSV in `data/processed/reference_percentages/`.
The original calculator source and data-hash semantics are preserved. Its
diagnostics remain beside the exported calculator. A return with a different
data hash will not be substituted.

Regenerate reports after supplying both external returns. If input data or cohort
settings change, obtain matching new returns. Patient-level exports, logs, model
inputs, risk scores, fitted models and reference files stay in ignored directories.
