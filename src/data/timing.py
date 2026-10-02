from __future__ import annotations

import logging
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import wraps
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
R = TypeVar("R")
StepLabel = str | Callable[..., str]


@contextmanager
def timed_step(logger: logging.Logger, label: str) -> Iterator[None]:
    """Log elapsed wall-clock time for one named processing step."""
    start = time.perf_counter()
    logger.info("start %s", label)
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start
        logger.info("finished %s in %.3f s", label, elapsed_seconds)


def timed(logger: logging.Logger, label: StepLabel) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorate a function so its complete call is logged as one timed step."""

    def decorator(function: Callable[P, R]) -> Callable[P, R]:
        @wraps(function)
        def wrapped(*args: P.args, **kwargs: P.kwargs) -> R:
            step_label = label(*args, **kwargs) if callable(label) else label
            with timed_step(logger, step_label):
                return function(*args, **kwargs)

        return wrapped

    return decorator
