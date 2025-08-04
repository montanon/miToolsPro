from __future__ import annotations

import importlib
from importlib.metadata import PackageNotFoundError, version
from types import ModuleType
from typing import Iterable, List

try:
    __version__ = version("mitoolspro")
except PackageNotFoundError:
    __version__ = "0.0.0+dev"

__all__: List[str] = [
    "__version__",
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
