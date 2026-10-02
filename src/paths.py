"""Repository-relative locations shared by every public entry point."""

from pathlib import Path

from src.fixed_config_paths import CONFIG_PATHS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "src/config"
RAW = ROOT / "data/raw"
PROCESSED = ROOT / "data/processed"
OUTPUT = ROOT / "output"
BENCHMARK = OUTPUT / "benchmark"
PAPER = OUTPUT / "paper"
REFERENCE = ROOT / "reference"


def apply_fixed_paths(path: Path, config: dict[str, dict[str, object]]) -> None:
    """Supply fixed locations without changing calculation settings or explicit nulls in saved runs."""
    if path.parent.resolve() == CONFIG:
        for dotted, value in CONFIG_PATHS.get(path.name, {}).items():
            *sections, key = dotted.split(".")
            section = config
            for name in sections:
                section = section.setdefault(name, {})
            if key in section:
                raise ValueError(f"Paths are fixed: remove {dotted} from {path.name}.")
            section[key] = value
    elif path.name == "training_config.yml" and path.absolute().is_relative_to(OUTPUT):
        relocate_saved_paths(config)


def relocate_saved_paths(config: dict[str, dict[str, object]]) -> None:
    """Relocate known inputs in a copied run; retain all model, cohort and evaluation parameters."""
    dataset = config.get("dataset", {})
    for key, value in dataset.items():
        if key.endswith("_csv_path") and value is not None:
            dataset[key] = str(PROCESSED / Path(str(value)).name)
    output = config.get("output", {})
    for key in ("normal_ranges_csv_path", "bloodtest_translation_csv_path"):
        if key in output and output[key] is not None:
            output[key] = str(CONFIG / Path(str(output[key])).name)
