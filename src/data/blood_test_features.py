from __future__ import annotations

BLOOD_TEST_CHANGE_SUFFIX = " change"


def blood_test_feature_names(
    selected_test_keys: tuple[str, ...],
    include_change: bool,
) -> tuple[str, ...]:
    """Order observed values before optional signed changes for the same tests."""
    if not include_change:
        return selected_test_keys
    changes = tuple(f"{test_key}{BLOOD_TEST_CHANGE_SUFFIX}" for test_key in selected_test_keys)
    return (*selected_test_keys, *changes)
