from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .inspect_graph import (
        InspectRuntime,
        build_initial_inspect_state,
        build_inspect_graph,
        build_state_from_scene_analysis,
    )
    from .state import InspectOptions, InspectState

__all__ = [
    "InspectRuntime",
    "InspectOptions",
    "InspectState",
    "build_initial_inspect_state",
    "build_inspect_graph",
    "build_state_from_scene_analysis",
]

_EXPORT_MAP = {
    "InspectRuntime": ("v2a_inspect.workflows.inspect_graph", "InspectRuntime"),
    "build_initial_inspect_state": (
        "v2a_inspect.workflows.inspect_graph",
        "build_initial_inspect_state",
    ),
    "build_inspect_graph": (
        "v2a_inspect.workflows.inspect_graph",
        "build_inspect_graph",
    ),
    "build_state_from_scene_analysis": (
        "v2a_inspect.workflows.inspect_graph",
        "build_state_from_scene_analysis",
    ),
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
