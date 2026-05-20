"""Memory management for serial model loading.

Two diffusion models must never coexist in memory. All model lifecycle is
centralized here; hand-rolled `del` in business code is forbidden (CLAUDE.md §5.2).
"""

from __future__ import annotations

import gc
from contextlib import contextmanager
from typing import Any, Callable, Generator, TypeVar

import torch

T = TypeVar("T")


@contextmanager
def with_isolated_model(factory: Callable[[], T]) -> Generator[T, None, None]:
    """Load a model, yield it, then deterministically free all memory.

    Usage::

        with with_isolated_model(lambda: load_sdedit_pipe(cfg)) as pipe:
            result = pipe(prompt=...)
        # pipe is deleted and GPU/MPS cache is cleared before this line.

    Args:
        factory: zero-argument callable that constructs and returns the model.
                 Called exactly once, inside the context.

    Yields:
        The object returned by *factory*.
    """
    model: T | None = factory()
    try:
        yield model
    finally:
        # Drop our reference before GC so the object can be collected.
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()
