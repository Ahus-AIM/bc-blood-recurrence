from __future__ import annotations


def failure_details(error: Exception) -> dict[str, str]:
    """Return stable metadata fields for a failed benchmark model run."""
    return {
        "error_type": type(error).__name__,
        "failure_reason": str(error),
    }
