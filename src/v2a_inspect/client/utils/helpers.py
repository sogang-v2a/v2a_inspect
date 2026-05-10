from __future__ import annotations

def validate_video_path(path: str) -> str:
    """Validate that a video file exists and has an acceptable extension."""
    import os
    if not os.path.exists(path):
        raise FileNotFoundError(f"Video file not found: {path}")
    
    ext = os.path.splitext(path)[1].lower()
    if ext not in [".mp4", ".mov", ".avi", ".mkv"]:
        raise ValueError(f"Unsupported video format: {ext}. Supported: .mp4, .mov, .avi, .mkv")
    
    return path

# Note: HTTP error handling is done in BaseClient._request, but we keep this for completeness if needed elsewhere.
def handle_http_error(status_code: int, response_text: str) -> None:
    """Raise a ClientError with details from an HTTP response."""
    from .base import ClientError
    raise ClientError(f"HTTP {status_code} error: {response_text}")