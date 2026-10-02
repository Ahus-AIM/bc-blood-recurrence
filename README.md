# Breast cancer recurrence from blood tests

Code for extracting study data, fitting the Cox explainable boosting machine (EBM)
benchmark, and generating the manuscript and supplementary figures and tables.
The numerical analysis modules are retained from the study repository. Notebook
extraction has been moved into scripts with the same transformations and ordering.

Clinical data, fitted clinical models, and the manuscript are not distributed.
Exact agreement with the manuscript remains to be validated on the clinical-data
machine. The original clinical age-calculation code will be added when supplied;
its required outputs are described in [docs/inputs.md](docs/inputs.md).

## Try the complete workflow with synthetic data

Run from this repository using Python 3.13 and `uv`:

```bash
uv sync --locked
uv run python -m src.script.generate_synthetic_data
uv run python -m src.script.run_demo
```

The generator refuses to overwrite populated input or output directories. The demo
fits all 130 recurrence models (24 individual tests, a clinical baseline, and all
tests for each of five endpoints), plus the competing-event models. It uses fewer
boosting rounds and bootstrap replicates than the study. It takes several minutes
and writes clearly marked synthetic outputs, which cannot reproduce the paper's
clinical findings. Demonstration settings are in `src/config/demo.yml` and
`src/config/synthetic.yml`.

The final SVG figures are in `output/paper/figure/` and LaTeX/CSV tables are in
`output/paper/table/`. There are 18 figures and six tables. SVG files receive the
manuscript finishing steps; PNG files are the original exporter previews.

## Run with the study inputs

Use a separate clean checkout from the demo, with the original clinical environment
and inputs. See [reproducibility](docs/reproducibility.md) before fitting: the locked
demo dependencies do not establish equivalence to the development InterpretML build
used for the manuscript.

```bash
uv run --no-sync python -m src.script.capture_environment
uv run --no-sync python -m src.script.extract_data
uv run --no-sync python -m src.script.train_single_bloodtest_benchmark
uv run --no-sync python -m src.script.calculate_reference_returns
# Supply the matching external age-summary CSV described in docs/inputs.md.
uv run --no-sync python -m src.script.render_paper
uv run --no-sync python -m src.script.verify_reproduction
```

`--no-sync` preserves the prepared clinical environment. It does not install it.
Training resumes completed compatible benchmark cells by default. A saved benchmark
can instead be placed in `output/benchmark/` for report regeneration.

Paths are fixed, relative to the repository, and independent of the working
directory: source inputs in `data/raw/`, extracted data and returned summaries in
`data/processed/`, and generated artifacts in `output/`. These directories and
`reference/` are ignored by Git. Scripts take no command-line arguments. Analysis
and presentation settings live in `src/config/`; paths are defined in `src/paths.py`
and `src/fixed_config_paths.py`.

## Check the code

```bash
uv run pytest
uv run ruff check src test
```

[manifest.txt](manifest.txt) records the retained study assumptions, known numerical
limitations, and validation scope. [docs/source_provenance.json](docs/source_provenance.json)
records copied file hashes and migration changes.

## License

[Creative Commons Attribution-ShareAlike 4.0 International](LICENSE), using the
unchanged license text from [Ahus-AIM/Open-ECG-Digitizer](https://github.com/Ahus-AIM/Open-ECG-Digitizer/blob/main/LICENSE).
