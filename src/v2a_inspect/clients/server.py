from __future__ import annotations

import base64
import json
from pathlib import Path
from urllib import request

from v2a_inspect.pipeline.response_models import GroupedAnalysis, VideoSceneAnalysis
from v2a_inspect.workflows import InspectOptions, InspectState


def run_server_inspect(
    *,
    server_base_url: str,
    video_path: str,
    options: InspectOptions,
) -> InspectState:
    payload = _build_request_payload(video_path=video_path, options=options)
    request_obj = request.Request(
        url=f"{server_base_url.rstrip('/')}/analyze",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
    )
    with request.urlopen(
        request_obj, timeout=options.video_timeout_ms / 1000
    ) as response:
        body = response.read().decode("utf-8")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise TypeError("Server analysis response must be a JSON object.")

    scene_analysis_payload = decoded.get("scene_analysis")
    grouped_analysis_payload = decoded.get("grouped_analysis")
    if not isinstance(scene_analysis_payload, dict):
        raise ValueError("Server analysis response is missing scene_analysis.")
    if not isinstance(grouped_analysis_payload, dict):
        raise ValueError("Server analysis response is missing grouped_analysis.")

    scene_analysis = VideoSceneAnalysis.model_validate(scene_analysis_payload)
    grouped_analysis = GroupedAnalysis.model_validate(grouped_analysis_payload)
    warnings = list(decoded.get("warnings", []))
    progress_messages = list(decoded.get("progress_messages", []))
    state: InspectState = {
        "scene_analysis": scene_analysis,
        "grouped_analysis": grouped_analysis,
        "warnings": [str(item) for item in warnings],
        "progress_messages": [str(item) for item in progress_messages],
    }
    return state


def _build_request_payload(
    *,
    video_path: str,
    options: InspectOptions,
) -> dict[str, object]:
    payload = {
        "video_path": video_path,
        "options": options.model_dump(mode="json"),
        "video_filename": Path(video_path).name,
        "video_base64": base64.b64encode(Path(video_path).read_bytes()).decode("ascii"),
    }
    return payload
