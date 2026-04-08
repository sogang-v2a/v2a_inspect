from __future__ import annotations

import json
from pathlib import Path
from urllib import parse, request

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.pipeline.response_models import GroupedAnalysis, VideoSceneAnalysis
from v2a_inspect.workflows import InspectOptions, InspectState

CLIENT_USER_AGENT = "v2a-inspect-client/1.0"


def run_server_inspect(
    *,
    server_base_url: str,
    video_path: str,
    options: InspectOptions,
) -> InspectState:
    remote_video_path = _upload_video(
        server_base_url=server_base_url,
        video_path=video_path,
        timeout_seconds=options.video_timeout_ms / 1000,
    )
    payload = _build_request_payload(
        video_path=video_path,
        remote_video_path=remote_video_path,
        options=options,
    )
    request_obj = request.Request(
        url=f"{server_base_url.rstrip('/')}/analyze",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": CLIENT_USER_AGENT,
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
    bundle_payload = decoded.get("multitrack_bundle")
    warnings = list(decoded.get("warnings", []))
    progress_messages = list(decoded.get("progress_messages", []))
    state: InspectState = {
        "scene_analysis": scene_analysis,
        "grouped_analysis": grouped_analysis,
        "warnings": [str(item) for item in warnings],
        "progress_messages": [str(item) for item in progress_messages],
    }
    if isinstance(bundle_payload, dict):
        state["multitrack_bundle"] = MultitrackDescriptionBundle.model_validate(
            bundle_payload
        )
    return state


def _upload_video(
    *,
    server_base_url: str,
    video_path: str,
    timeout_seconds: float,
) -> str:
    path = Path(video_path)
    upload_url = (
        f"{server_base_url.rstrip('/')}/upload?"
        + parse.urlencode({"filename": path.name})
    )
    request_obj = request.Request(
        url=upload_url,
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(path.stat().st_size),
            "X-Filename": path.name,
            "User-Agent": CLIENT_USER_AGENT,
        },
        data=path.read_bytes(),
        method="POST",
    )
    with request.urlopen(request_obj, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    payload = json.loads(body)
    if not isinstance(payload, dict) or not isinstance(payload.get("video_path"), str):
        raise ValueError("Server upload response is missing video_path.")
    return payload["video_path"]


def _build_request_payload(
    *,
    video_path: str,
    remote_video_path: str,
    options: InspectOptions,
) -> dict[str, object]:
    return {
        "video_path": remote_video_path,
        "video_filename": Path(video_path).name,
        "options": options.model_dump(mode="json"),
    }
