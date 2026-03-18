from __future__ import annotations

import base64
import mimetypes
import time
from pathlib import Path
from typing import Any

import google.genai as genai

DEFAULT_GEMINI_MODEL = "gemini-3-pro-preview"
DEFAULT_POLL_INTERVAL_SECONDS = 2.0


def state_name(state: object) -> str:
    """Normalize SDK file states across string and enum-like variants."""

    if hasattr(state, "name"):
        return str(getattr(state, "name"))
    return str(state)


def upload_file(client: genai.Client, file_path: str) -> Any:
    """Upload a file to Gemini."""

    return client.files.upload(file=file_path)


def wait_for_file_active(
    client: genai.Client,
    file_name: str,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: int = 300,
) -> Any:
    """Poll an uploaded Gemini file until it becomes active."""

    waited_seconds = 0.0
    file_obj = client.files.get(name=file_name)

    while state_name(file_obj.state) == "PROCESSING":
        if waited_seconds >= max_wait_seconds:
            raise TimeoutError(
                f"Gemini file processing timed out after {max_wait_seconds}s."
            )

        time.sleep(poll_interval_seconds)
        waited_seconds += poll_interval_seconds
        file_obj = client.files.get(name=file_name)

    final_state = state_name(file_obj.state)
    if final_state != "ACTIVE":
        raise RuntimeError(f"Gemini file ended in unexpected state: {final_state}.")

    return file_obj


def upload_video(
    client: genai.Client,
    video_path: str,
    *,
    poll_interval_seconds: float = DEFAULT_POLL_INTERVAL_SECONDS,
    max_wait_seconds: int = 300,
) -> Any:
    """Upload a video and wait for Gemini processing to complete."""

    file_obj = upload_file(client, video_path)
    initial_state = state_name(file_obj.state)

    if initial_state == "ACTIVE":
        return file_obj
    if initial_state != "PROCESSING":
        raise RuntimeError(f"Gemini file ended in unexpected state: {initial_state}.")

    return wait_for_file_active(
        client,
        file_obj.name,
        poll_interval_seconds=poll_interval_seconds,
        max_wait_seconds=max_wait_seconds,
    )


def encode_file_base64(file_path: str) -> str:
    """Encode a local file as an ASCII base64 string."""

    return base64.b64encode(Path(file_path).read_bytes()).decode("ascii")


def guess_mime_type(file_path: str, *, fallback: str) -> str:
    """Guess a MIME type for the given file path."""

    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type or fallback


def build_inline_video_content_block(
    video_path: str,
    *,
    mime_type: str | None = None,
) -> dict[str, str]:
    """Build an inline LangChain video content block from a local file."""

    resolved_mime_type = mime_type or guess_mime_type(video_path, fallback="video/mp4")
    return {
        "type": "video",
        "base64": encode_file_base64(video_path),
        "mime_type": resolved_mime_type,
    }


def build_uploaded_video_content_block(
    file_obj: Any,
    *,
    fps: float,
) -> dict[str, Any]:
    """Build an uploaded-video block for ChatGoogleGenerativeAI."""

    return {
        "type": "media",
        "file_uri": file_obj.uri,
        "mime_type": getattr(file_obj, "mime_type", None) or "video/mp4",
        "video_metadata": {"fps": fps},
    }
