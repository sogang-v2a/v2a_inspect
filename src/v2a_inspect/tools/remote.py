from __future__ import annotations

import json
from typing import Any
from urllib import request


def build_bearer_headers(api_key: str | None = None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def post_json(
    url: str,
    payload: dict[str, Any],
    *,
    api_key: str | None = None,
    timeout_seconds: int = 120,
    extra_headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    headers = build_bearer_headers(api_key=api_key)
    if extra_headers:
        headers.update(extra_headers)

    request_obj = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with request.urlopen(request_obj, timeout=timeout_seconds) as response:
        body = response.read().decode("utf-8")
    if not body.strip():
        return {}
    decoded = json.loads(body)
    if isinstance(decoded, dict):
        return decoded
    raise TypeError("Remote endpoint did not return a JSON object.")
