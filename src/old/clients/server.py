from __future__ import annotations

from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.local_pipeline import run_local_inspect_raw
from v2a_inspect.workflows import InspectOptions, InspectState


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
        raise ValueError("Local inspect response is missing multitrack_bundle.")
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
    return run_local_inspect_raw(
        video_path=video_path,
        options=options.model_copy(update={"server_base_url": server_base_url}),
    )
