"""Shared defaults for EBM explanation plots."""

DEFAULT_COMBINED_TERM_HAZARD_RATIO_Y_LIMITS = (0.3, 4.0)
IMPORTANCE_TITLE = "EBM feature importance"
FEATURE_EFFECTS_TITLE = "EBM feature effects on relative hazard"
TOP_FEATURES_TITLE = "Top EBM feature effects"
HAZARD_RATIO_LABEL = "Hazard ratio"


def recurrence_figure_title(title: str, endpoint_name: str | None) -> str:
    """Append the model endpoint, using Any recurrence for the pooled target."""
    if endpoint_name is None:
        return title
    endpoint = endpoint_name.replace("_", " ").strip().lower()
    if endpoint in {"all", "any", "distant recurrence", "all recurrence", "any recurrence"}:
        endpoint = "any recurrence"
    elif not endpoint.endswith("recurrence"):
        endpoint += " recurrence"
    return f"{title} — {endpoint.capitalize()}"
