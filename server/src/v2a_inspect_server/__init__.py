from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .bootstrap import WeightsArtifact, WeightsBootstrapper, WeightsManifest
    from .embeddings import EmbeddingClient, LabelClient
    from .runtime import ToolingRuntime, build_tooling_runtime
    from .sam3 import Sam3Client

__all__ = [
    "EmbeddingClient",
    "LabelClient",
    "Sam3Client",
    "ToolingRuntime",
    "WeightsArtifact",
    "WeightsBootstrapper",
    "WeightsManifest",
    "build_tooling_runtime",
]

_EXPORT_MAP = {
    "EmbeddingClient": ("v2a_inspect_server.embeddings", "EmbeddingClient"),
    "LabelClient": ("v2a_inspect_server.embeddings", "LabelClient"),
    "Sam3Client": ("v2a_inspect_server.sam3", "Sam3Client"),
    "ToolingRuntime": ("v2a_inspect_server.runtime", "ToolingRuntime"),
    "WeightsArtifact": ("v2a_inspect_server.bootstrap", "WeightsArtifact"),
    "WeightsBootstrapper": (
        "v2a_inspect_server.bootstrap",
        "WeightsBootstrapper",
    ),
    "WeightsManifest": ("v2a_inspect_server.bootstrap", "WeightsManifest"),
    "build_tooling_runtime": (
        "v2a_inspect_server.runtime",
        "build_tooling_runtime",
    ),
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
