from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from v2a_inspect_server.remote import build_bearer_headers, post_json


class _FakeResponse:
    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload

    def read(self) -> bytes:
        return json.dumps(self.payload).encode("utf-8")

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


class RemoteToolsTests(unittest.TestCase):
    def test_build_bearer_headers(self) -> None:
        headers = build_bearer_headers("secret")
        self.assertEqual(headers["Authorization"], "Bearer secret")

    @patch("v2a_inspect_server.remote.request.urlopen")
    def test_post_json_round_trips_payload(self, mock_urlopen) -> None:
        mock_urlopen.return_value = _FakeResponse({"output": {"ok": True}})
        payload = post_json(
            "https://example.com/run", {"hello": "world"}, api_key="secret"
        )
        self.assertEqual(payload, {"output": {"ok": True}})


if __name__ == "__main__":
    unittest.main()
