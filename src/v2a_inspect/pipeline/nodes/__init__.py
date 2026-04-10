from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .analyze import analyze_scenes
    from .assemble import assemble_grouped_analysis
    from .extract import extract_raw_tracks
    from .group import group_tracks
    from .select_model import select_models
    from .upload import upload_video
    from .verify import verify_groups

__all__ = [
    "upload_video",
    "analyze_scenes",
    "extract_raw_tracks",
    "group_tracks",
    "verify_groups",
    "select_models",
    "assemble_grouped_analysis",
]

_EXPORT_MAP = {
    "upload_video": ("v2a_inspect.pipeline.nodes.upload", "upload_video"),
    "analyze_scenes": ("v2a_inspect.pipeline.nodes.analyze", "analyze_scenes"),
    "extract_raw_tracks": ("v2a_inspect.pipeline.nodes.extract", "extract_raw_tracks"),
    "group_tracks": ("v2a_inspect.pipeline.nodes.group", "group_tracks"),
    "verify_groups": ("v2a_inspect.pipeline.nodes.verify", "verify_groups"),
    "select_models": ("v2a_inspect.pipeline.nodes.select_model", "select_models"),
    "assemble_grouped_analysis": (
        "v2a_inspect.pipeline.nodes.assemble",
        "assemble_grouped_analysis",
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
