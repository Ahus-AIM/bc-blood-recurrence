"""Recompute manuscript figures/tables from saved models, then apply the publication styling."""

from __future__ import annotations

import json
import shutil

from src.paths import OUTPUT, PAPER
from src.regeneration.runner import run_regeneration
from src.script import (
    compact_paper_ebm_rows,
    style_paper_ebm_legends,
    style_paper_figures,
    style_paper_reference_colors,
    style_paper_tables,
    style_paper_trajectory_legends,
)
from src.script.supplementary_tables import write_supplementary_tables
from src.script.survival_benchmark_artifacts import file_sha256


def main() -> None:
    run_regeneration()
    # These are the existing manuscript finishing steps; calculation outputs stay in the benchmark.
    style_paper_figures.main()
    compact_paper_ebm_rows.main()
    style_paper_ebm_legends.main()
    style_paper_reference_colors.main()
    style_paper_trajectory_legends.main()
    write_supplementary_tables()
    style_paper_tables.main()
    write_presentation_manifest()


def write_presentation_manifest() -> None:
    """Hash the finished publication files after every presentation step succeeds."""
    if (OUTPUT / "SYNTHETIC.json").is_file():
        shutil.copy2(OUTPUT / "SYNTHETIC.json", PAPER / "SYNTHETIC.json")
    manifest = {
        "stage": "after manuscript presentation styling",
        "synthetic": (PAPER / "SYNTHETIC.json").is_file(),
        "files": {
            str(path.relative_to(PAPER)): file_sha256(path)
            for directory in (PAPER / "figure", PAPER / "table")
            for path in sorted(directory.glob("*"))
            if path.is_file()
        },
    }
    (PAPER / "presentation_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
