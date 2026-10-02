from __future__ import annotations

from math import isfinite
from pathlib import Path


def recurrence_endpoint_label(label: str) -> str:
    """Display the pooled recurrence endpoint as Any without renaming stored keys."""
    return "Any" if label.casefold() in {"all", "any"} else label


def p_value_text(value: float, *, latex: bool = False, missing: str = "N/A") -> str:
    """Show small positive p-values to two significant digits; retain a bound for computed zeros."""
    if not isfinite(value):
        return missing
    if value == 0:
        return r"$<0.001$" if latex else "<0.001"
    if value >= 0.001:
        return f"{value:.3f}"
    coefficient, exponent = f"{value:.1e}".split("e")
    power = str(int(exponent))
    if latex:
        return rf"${coefficient} \times 10^{{{power}}}$"
    superscript = power.translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{coefficient} × 10{superscript}"


def percentage_text(value: float) -> str:
    """Format a percentage already on the 0--100 scale, or N/A when unavailable."""
    return f"{value:.1f}%" if isfinite(value) else "N/A"


def count_percentage_text(count: int, total: int) -> str:
    """Format an exact count and its percentage of a nonnegative total."""
    percentage = 0.0 if total == 0 else count * 100.0 / total
    percentage_text = "<0.1%" if 0.0 < percentage < 0.1 else f"{percentage:.1f}%"
    return f"{count:,} ({percentage_text})"


def report_output_path(output_dir: Path, value: object) -> Path:
    """Resolve a configured report path relative to its output directory."""
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else output_dir / path


def display_output_path(path: Path, output_dir: Path) -> str:
    """Prefer a portable output-directory-relative artifact path."""
    try:
        return str(path.relative_to(output_dir))
    except ValueError:
        return str(path)


def latex_escape(value: str) -> str:
    """Escape LaTeX-reserved characters in display text."""
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
        "≤": r"\(\leq\)",
        "≥": r"\(\geq\)",
        "<": r"\(<\)",
        ">": r"\(>\)",
    }
    return "".join(replacements.get(character, character) for character in value)
