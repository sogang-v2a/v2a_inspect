from __future__ import annotations

import json
from pathlib import Path
from urllib import parse, request

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.workflows import InspectOptions, InspectState

CLIENT_USER_AGENT = "v2a-inspect-client/1.0"


def run_server_inspect(
    *,
    server_base_url: str,
    video_path: str,
    options: InspectOptions,
) -> InspectState:
    decoded = run_server_inspect_raw(
        server_base_url=server_base_url,
        video_path=video_path,
        options=options,
    )
    bundle_payload = decoded.get("multitrack_bundle")
    if not isinstance(bundle_payload, dict):
        raise ValueError("Server analysis response is missing multitrack_bundle.")
    warnings = list(decoded.get("warnings", []))
    progress_messages = list(decoded.get("progress_messages", []))
    state: InspectState = {
        "multitrack_bundle": MultitrackDescriptionBundle.model_validate(bundle_payload),
        "warnings": [str(item) for item in warnings],
        "progress_messages": [str(item) for item in progress_messages],
    }
    return state


def run_server_inspect_raw(
    *,
    server_base_url: str,
    video_path: str,
    options: InspectOptions,
) -> dict[str, object]:
    remote_video_path = _upload_video(
        server_base_url=server_base_url,
        video_path=video_path,
        timeout_seconds=float(options.remote_timeout_seconds),
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
    with request.urlopen(request_obj, timeout=float(options.remote_timeout_seconds)) as response:
        body = response.read().decode("utf-8")
    decoded = json.loads(body)
    if not isinstance(decoded, dict):
        raise TypeError("Server analysis response must be a JSON object.")
    return decoded


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
