from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .app import main

__all__ = ["main"]


def __getattr__(name: str) -> Any:
    if name != "main":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module("v2a_inspect.ui.app"), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
