# Reproducing the manuscript

The acceptance reference is the figures and tables present in the manuscript on
2026-10-02. `src/config/manuscript_reference.json` freezes their hashes without
distributing clinical outputs. It uses the current manuscript SVGs and inline
table bodies, rather than an older generated paper bundle. There are 18 figures
and six tables; the reference-limit table is static study metadata.

## Environment and saved run

The source manifest identifies InterpretML development commit
`ec1c18d962c3deab1ab8af374e8eb69e0d708969` for clinical fitting. Its package metadata
reports 0.7.8, but that version string alone does not identify the native binary.
The repository's lockfile preserves the locally available PyPI environment for
tests and the synthetic demo. It is not an assertion that the clinical build is
identical. Reuse the original clinical environment and native binary, or recover
its build and dependency record before refitting. `capture_environment` records
installed versions, native-library hashes, Python/platform details and thread
settings to `output/environment.json`. Run it in both environments for comparison.

To start from the saved benchmark, copy its full contents into
`output/benchmark/`, keeping the original site/horizon/model hierarchy, configs,
models, metadata, bootstrap artifacts and aggregate results. Supply the original
processed inputs and matching external returns. Saved input paths are relocated
to this checkout; scientific settings and explicit nulls are preserved. Report
generation uses saved fitted models and does not fit missing competing-event
models by default.

The original manifest records changes after some saved manuscript outputs were
produced: post-recurrence survival follow-up, patient-based test bootstrap sizes,
and washout event/censoring counts. Therefore identical source alone cannot
establish agreement with the current manuscript. A discrepancy must be resolved
against the actual saved run; it must not be hidden by changing the expected
values or silently selecting older calculations.

`src/config/regenerate.yml` selects the manuscript reports by default. It rebuilds
cohort/trajectory statistics, cohort counts, full benchmark tables and the five
all-test model explanations. It retains saved benchmark evaluation results. To
reevaluate an existing run, select `cindex`, `washout` and `brier` tasks, and set
`models: null` to include every predictor set, then run `src.script.regenerate`.
This is a scientific recalculation and may change the manuscript numbers. The
original regeneration modules and tests are retained.

## Comparison

Run `uv run --no-sync python -m src.script.verify_reproduction` after rendering.
It writes `output/verification.json` and exits nonzero on a manuscript mismatch.
Synthetic data are expected to differ. The comparison checks:

- SVG structure, plotted values, geometry, styling and text, ignoring metadata,
  XML indentation and generated element IDs. A change of Matplotlib or fonts can
  still cause a mismatch. This is a strict structural comparison, not a perceptual
  pixel comparison.
- Complete LaTeX tabular bodies, including labels and significance commands,
  ignoring whitespace. Captions and external document macros are outside this
  comparison. The final performance-table exporter uses the manuscript's labels,
  column widths and row ordering without changing numerical results.
- Optional byte-for-byte extraction references: place the five original notebook
  exports under `reference/processed/` with their existing filenames.
- Optional byte-for-byte benchmark CSV references: put original model-input,
  risk-score, evaluation and table CSVs under `reference/benchmark/`, preserving
  paths relative to the saved benchmark. Every supplied CSV is compared. Missing
  references are explicitly reported; the comparator does not inspect model
  pickle internals or claim unprovided artifacts were checked.

`output/paper/presentation_manifest.json` hashes final styled artifacts. The
upstream bundle manifest describes files before those presentation steps. PNGs
are exporter previews; the manuscript reference and final styling use SVGs.

## Validation performed during extraction of this repository

The original notebook cells and converted extraction scripts produced byte-identical
versions of all five CSVs on identical synthetic raw inputs. The synthetic
end-to-end workflow completed 130 recurrence models and their competing models,
all 390 split-level result records, and the 18 figures and six tables. This checks
execution and migration consistency, not agreement with clinical manuscript data.

Retained regression tests cover cohort selection, imputation, weighting,
bootstrap sampling, model artifacts, recurrence statistics and reporting. Four
expected failures document native Cox tied-event/weighting limitations. Three
strict expected failures document publication-layout assertions that also failed
in the unchanged original source; changing figure geometry to fix them would
alter the reference output. Three checks requiring original clinical SVG assets
skip unless those assets are supplied under `reference/paper/figure/`.

Clinical manuscript equality, recovery of the original fitting environment, and
integration of the original clinical age routine remain pending.

Local validation on 2026-10-02: the full suite reported 1,131 passed, three
skipped and seven expected failures. After adding the performance-table divider
case, all eight focused artifact-comparison/presentation tests passed. Ruff
passed, and all six generated synthetic LaTeX tables compiled together. The
manuscript comparator correctly rejected the synthetic run: only the fixed
reference-limit table matched (1 of 24 artifacts).
