"""Comparison checks must reject changed values rather than normalize them away."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.reproduction import compare_files, normalized_svg, normalized_tabular
from src.script.style_paper_tables import result_rows

FIGURE = '''<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink">
<metadata>2026-10-02</metadata>
<defs><path id="random_id" d="M 0 0 L 2 3"/></defs>
<use xlink:href="#random_id" style="clip-path:url(#random_id)"/>
<text>Value: 0.45<tspan> A</tspan> <tspan>B</tspan></text>
</svg>'''


def test_svg_comparison_ignores_generated_ids_and_metadata() -> None:
    equivalent = FIGURE.replace("random_id", "other_id").replace("2026-10-02", "2026-11-03").replace("\n", "\n  ")
    assert normalized_svg(FIGURE) == normalized_svg(equivalent)


@pytest.mark.parametrize(("before", "after"), [("0.45", "0.46"), ("L 2 3", "L 2 4"), ("> <tspan>B", "><tspan>B")])
def test_svg_comparison_retains_numbers_geometry_and_text_spacing(before: str, after: str) -> None:
    assert normalized_svg(FIGURE) != normalized_svg(FIGURE.replace(before, after))


def test_tabular_comparison_preserves_significance() -> None:
    table = r"\begin{tabular}{lr}A & \textbf{0.45} \\\end{tabular}"
    assert normalized_tabular(table) == normalized_tabular(table.replace(" & ", "\n&\n"))
    assert normalized_tabular(table) != normalized_tabular(table.replace(r"\textbf{0.45}", "0.45"))
    with pytest.raises(ValueError):
        normalized_tabular("not a table")


def test_csv_comparison_is_strict_about_order_and_precision(tmp_path: Path) -> None:
    reference, actual = tmp_path / "reference.csv", tmp_path / "actual.csv"
    reference.write_text("id,value\nA,0.50\nB,0.60\n")
    assert compare_files(reference, actual)["status"] == "missing_output"
    actual.write_bytes(reference.read_bytes())
    assert compare_files(reference, actual)["status"] == "match"
    actual.write_text("id,value\nB,0.60\nA,0.50\n")
    assert compare_files(reference, actual)["status"] == "different"
    actual.write_text("id,value\nA,0.5\nB,0.60\n")
    assert compare_files(reference, actual)["status"] == "different"


@pytest.mark.parametrize("divider", ["", "\\hline\n\\hline\n"])
def test_presentation_moves_intact_baseline_after_all_tests(divider: str) -> None:
    baseline = r"Clinical baseline & 0.451 [0.400, 0.501] \\"
    individual = [f"Test {index} & \\textbf{{0.{index:03d}}} \\\\" for index in range(24)]
    all_tests = r"All blood tests & \underline{0.789} \\"
    content = "\\begin{tabular}{lr}\n\\hline\nModel & Value\\\\\n\\hline\n"
    content += "\n".join([baseline, *individual]) + "\n" + divider + all_tests + "\n\\hline\n\\end{tabular}"
    rows = result_rows(content)
    assert rows[:24] == individual
    assert rows[-2] == all_tests
    assert rows[-1] == r"Diagnosis age \& time since diagnosis (baseline) & 0.451 [0.400, 0.501] \\"
