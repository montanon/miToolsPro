"""Top level package for :mod:`mitoolspro`.

This module exposes commonly used subpackages without importing them
eagerly. Importing :mod:`mitoolspro` will no longer trigger imports of
all submodules which can be expensive. Instead, submodules are loaded
on first access using :func:`importlib.import_module`.
"""

from __future__ import annotations

import importlib
from types import ModuleType
from typing import Iterable, List

__all__: List[str] = [
    "clustering",
    "databases",
    "files",
    "llms",
    "nlp",
    "pandas_utils",
    "plotting",
    "regressions",
    "scraping",
    "utils",
]


def __getattr__(name: str) -> ModuleType:
    """Lazily import submodules listed in ``__all__``."""

    if name in __all__:
        return importlib.import_module(f"{__name__}.{name}")
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")


def __dir__() -> Iterable[str]:
    return sorted(list(globals().keys()) + __all__)
