from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

MICE_METHOD = "mice"
LOCF_METHOD = "locf"
MEDIAN_FALLBACK_METHOD = "median"
ENABLED_METHODS = (LOCF_METHOD, MICE_METHOD)
LOCF_FALLBACK_METHODS = (MEDIAN_FALLBACK_METHOD, MICE_METHOD)


@dataclass(frozen=True)
class ImputationConfig:
    """Model-feature imputation settings.

    `method` selects plain MICE or LOCF. LOCF carries raw blood-test values from
    prior valid landmark days up to `locf_max_age_days`, then
    `locf_fallback_method` fills values that remain missing. With blood-test
    changes enabled, current/past values and enabled context are completed
    jointly before subtraction; past values never receive LOCF. MICE controls
    are used by plain MICE and by LOCF with MICE fallback.
    """

    enabled: bool
    method: str
    max_iter: int
    random_state: int
    tol: float = 0.001
    locf_max_age_days: int | None = None
    locf_fallback_method: str = MEDIAN_FALLBACK_METHOD

    @classmethod
    def from_mapping(cls, config: Mapping[str, object] | None) -> ImputationConfig:
        if config is None:
            return cls(enabled=False, method=MICE_METHOD, max_iter=10, random_state=42)
        imputation_config = cls(
            enabled=bool(config.get("enabled", False)),
            method=str(config.get("method", MICE_METHOD)).strip().lower(),
            max_iter=int(config.get("max_iter", 10)),
            random_state=int(config.get("random_state", 42)),
            tol=float(config.get("tol", 0.001)),
            locf_max_age_days=optional_positive_int(config.get("locf_max_age_days")),
            locf_fallback_method=str(config.get("locf_fallback_method", MEDIAN_FALLBACK_METHOD))
            .strip()
            .lower(),
        )
        validate_imputation_config(imputation_config)
        return imputation_config


def optional_positive_int(value: object) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    parsed = int(value)
    if parsed < 1:
        raise ValueError("`imputation.locf_max_age_days` must be a positive integer or null.")
    return parsed


def validate_imputation_config(config: ImputationConfig) -> None:
    if config.enabled and config.method not in ENABLED_METHODS:
        allowed = ", ".join(ENABLED_METHODS)
        raise ValueError(f"`imputation.method` must be one of: {allowed}.")
    if config.locf_fallback_method not in LOCF_FALLBACK_METHODS:
        allowed = ", ".join(LOCF_FALLBACK_METHODS)
        raise ValueError(f"`imputation.locf_fallback_method` must be one of: {allowed}.")
    uses_mice = config.method == MICE_METHOD or (
        config.enabled and config.method == LOCF_METHOD and config.locf_fallback_method == MICE_METHOD
    )
    if uses_mice and config.max_iter < 1:
        raise ValueError("`imputation.max_iter` must be at least 1 when MICE is used.")
    if uses_mice and config.tol <= 0.0:
        raise ValueError("`imputation.tol` must be positive when MICE is used.")
