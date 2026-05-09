from __future__ import annotations

from typing import Callable

from v2a_inspect.clients import run_server_inspect
from v2a_inspect.contracts import MultitrackDescriptionBundle
from v2a_inspect.settings_views import get_client_runtime_settings
from v2a_inspect.workflows import InspectOptions, InspectState

ProgressCallback = Callable[[str], None]


def run_inspect(
    video_path: str,
    *,
    options: InspectOptions | None = None,
    progress_callback: ProgressCallback | None = None,
    warning_callback: ProgressCallback | None = None,
    initial_state_overrides: dict[str, object] | None = None,
    **_: object,
) -> InspectState:
    """Run the supported silent-video inspect workflow through the server."""

    resolved_options = options or InspectOptions()
    if resolved_options.runtime_mode != "nvidia_docker":
        raise ValueError(
            "Only the nvidia_docker tool-first server runtime is supported."
        )
    if initial_state_overrides:
        raise ValueError("initial_state_overrides is no longer supported.")
    if progress_callback is not None:
        progress_callback("Submitting silent-video analysis request to server...")
    state = run_server_inspect(
        server_base_url=resolved_options.server_base_url
        or get_client_runtime_settings().server_base_url,
        video_path=video_path,
        options=resolved_options,
    )
    if warning_callback is not None:
        for warning in state.get("warnings", []):
            warning_callback(str(warning))
    return state


def get_multitrack_bundle(state: InspectState) -> MultitrackDescriptionBundle:
    bundle = state.get("multitrack_bundle")
    if bundle is None:
        raise ValueError("Inspect workflow did not produce 'multitrack_bundle'.")
    return bundle
