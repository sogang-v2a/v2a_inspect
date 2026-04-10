from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .server import run_server_inspect
    from v2a_inspect.constants import DEFAULT_GEMINI_MODEL
    from .video import (
        build_inline_video_content_block,
        build_uploaded_video_content_block,
        encode_file_base64,
        guess_mime_type,
        state_name,
        upload_file,
        upload_video,
        wait_for_file_active,
    )

__all__ = [
    "DEFAULT_GEMINI_MODEL",
    "build_inline_video_content_block",
    "build_uploaded_video_content_block",
    "encode_file_base64",
    "guess_mime_type",
    "state_name",
    "upload_file",
    "upload_video",
    "wait_for_file_active",
    "run_server_inspect",
]

_EXPORT_MAP = {
    "DEFAULT_GEMINI_MODEL": ("v2a_inspect.constants", "DEFAULT_GEMINI_MODEL"),
    "build_inline_video_content_block": (
        "v2a_inspect.clients.video",
        "build_inline_video_content_block",
    ),
    "build_uploaded_video_content_block": (
        "v2a_inspect.clients.video",
        "build_uploaded_video_content_block",
    ),
    "encode_file_base64": ("v2a_inspect.clients.video", "encode_file_base64"),
    "guess_mime_type": ("v2a_inspect.clients.video", "guess_mime_type"),
    "state_name": ("v2a_inspect.clients.video", "state_name"),
    "upload_file": ("v2a_inspect.clients.video", "upload_file"),
    "upload_video": ("v2a_inspect.clients.video", "upload_video"),
    "wait_for_file_active": ("v2a_inspect.clients.video", "wait_for_file_active"),
    "run_server_inspect": ("v2a_inspect.clients.server", "run_server_inspect"),
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
