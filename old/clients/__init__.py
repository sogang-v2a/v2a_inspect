from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import run_server_inspect, run_server_inspect_raw

__all__ = ["run_server_inspect", "run_server_inspect_raw"]

_EXPORT_MAP = {
    "run_server_inspect": ("v2a_inspect.clients.server", "run_server_inspect"),
    "run_server_inspect_raw": ("v2a_inspect.clients.server", "run_server_inspect_raw"),
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
