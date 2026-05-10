from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .state import InspectOptions, InspectState

__all__ = ["InspectOptions", "InspectState"]

_EXPORT_MAP = {
    "InspectOptions": ("v2a_inspect.workflows.state", "InspectOptions"),
    "InspectState": ("v2a_inspect.workflows.state", "InspectState"),
}


def __getattr__(name: str) -> Any:
    target = _EXPORT_MAP.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute_name = target
    value = getattr(import_module(module_name), attribute_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
