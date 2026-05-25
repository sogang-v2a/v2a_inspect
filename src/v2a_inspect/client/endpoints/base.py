import httpx
from typing import Optional, Dict, Any
from ..config.settings import settings


class ClientError(Exception):
    """Custom exception for client errors."""

    pass


class BaseClient:
    """Base client for interacting with the inference server."""

    def __init__(self, base_url: Optional[str] = None, timeout: Optional[float] = None):
        self.base_url = base_url or settings.server_url
        self.timeout = timeout or settings.timeout
        self._client: Optional[httpx.AsyncClient] = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    async def _request(
        self,
        method: str,
        endpoint: str,
        *,
        params: Optional[Dict[str, Any]] = None,
        json: Optional[Dict[str, Any]] = None,
        files: Optional[Dict[str, Any]] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Make an HTTP request and handle errors."""
        if not self._client:
            raise RuntimeError("Client must be used as an async context manager")

        url = f"{self.base_url}{endpoint}"
        try:
            response = await self._client.request(
                method,
                url,
                params=params,
                json=json,
                files=files,
                data=data,
            )
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as exc:
            # Try to parse error details from response
            try:
                error_detail = exc.response.json()
            except Exception:
                error_detail = exc.response.text
            raise ClientError(
                f"HTTP {exc.response.status_code} error: {error_detail}"
            ) from exc
        except httpx.RequestError as exc:
            raise ClientError(f"Request failed: {exc}") from exc
