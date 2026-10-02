from __future__ import annotations

import re


def compact_icd_code(code: str) -> str:
    """Normalize an ICD code for matching without changing the source spelling."""
    return "".join(character for character in code.upper() if character.isalnum())


def matches_icd_code(code: str, code_regex: re.Pattern[str]) -> bool:
    """Match a configured prefix pattern against a compact uppercase code."""
    return code_regex.match(compact_icd_code(code)) is not None
