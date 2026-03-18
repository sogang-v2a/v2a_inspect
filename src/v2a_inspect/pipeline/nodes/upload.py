from __future__ import annotations

from pathlib import Path

import google.genai as genai

from v2a_inspect.clients import upload_video as upload_gemini_video
from v2a_inspect.workflows.state import InspectState

from ._shared import append_state_message


def upload_video(
    state: InspectState,
    *,
    genai_client: genai.Client,
) -> dict[str, object]:
    """Upload the input video to Gemini and wait until it is ready."""

    video_path = state.get("video_path")
    if not video_path:
        raise ValueError("upload_video requires 'video_path' in state.")

    options = state.get("options")
    if options is None:
        raise ValueError("upload_video requires 'options' in state.")

    gemini_file = upload_gemini_video(
        genai_client,
        video_path,
        poll_interval_seconds=options.poll_interval_seconds,
        max_wait_seconds=options.upload_timeout_seconds,
    )
    message = f"Uploaded {Path(video_path).name} to Gemini."
    return {
        "gemini_file": gemini_file,
        "progress_messages": append_state_message(state, "progress_messages", message),
    }
