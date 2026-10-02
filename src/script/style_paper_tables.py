"""Apply the current manuscript's performance-table presentation to generated numerical rows."""

from __future__ import annotations

import json
import re

from src.paths import CONFIG, PAPER


def result_rows(content: str) -> list[str]:
    """Read original exporter rows, retaining every number and significance command verbatim."""
    tabular = re.search(r"\\begin\{tabular\}.*?\\end\{tabular\}", content, flags=re.DOTALL)
    if tabular is None:
        raise ValueError("Missing exported performance table.")
    parts = tabular[0].split(r"\hline")
    if len(parts) not in (4, 6):
        raise ValueError("Unexpected performance-table structure.")
    rows = [line.strip() for part in parts[2:-1] for line in part.splitlines() if line.strip()]
    baseline = [row for row in rows if row.startswith("Clinical baseline &")]
    if len(rows) != 26 or len(baseline) != 1:
        raise ValueError("Manuscript presentation requires 24 individual tests, all tests and a clinical baseline.")
    rows.remove(baseline[0])
    all_tests = next(index for index, row in enumerate(rows) if row.startswith("All blood tests &"))
    rows.insert(all_tests, "\\hline\n\\hline")
    # Match the manuscript's final row; this moves the complete row without changing its values.
    rows.append(baseline[0].replace("Clinical baseline", r"Diagnosis age \& time since diagnosis (baseline)", 1))
    return rows


def main() -> None:
    captions = json.loads((CONFIG / "performance_table_captions.json").read_text())
    template = (CONFIG / "performance_table.tex").read_text()
    for stem, settings in captions.items():
        path = PAPER / "table" / f"{stem}.tex"
        content = template.replace("%% CAPTION", settings["caption"])
        content = content.replace("%% LABEL", r"\label{tab:" + settings["label"] + "}")
        content = content.replace("%% RESULT_ROWS", "\n".join(result_rows(path.read_text())))
        path.write_text(content)


if __name__ == "__main__":
    main()
