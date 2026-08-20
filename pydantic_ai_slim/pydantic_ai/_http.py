"""Shared HTTP client types and helpers for the HTTPX2 clients Pydantic AI creates and owns."""

from __future__ import annotations

import warnings
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, TypeAlias

# Import httpcore2 eagerly: httpx2 defers it to first client construction, which performs blocking
# I/O if that happens inside the event loop.
import httpcore2  # noqa: F401  # pyright: ignore[reportUnusedImport]
import httpx2

from ._warnings import PydanticAIDeprecationWarning

__all__ = (
    'DEFAULT_HTTP_TIMEOUT',
    'AsyncHTTPClient',
    'LegacyHttpxAsyncClient',
    'create_async_httpx2_client',
    'legacy_httpx',
    'warn_if_legacy_httpx_client',
)

DEFAULT_HTTP_TIMEOUT: int = 600
"""Default HTTP timeout in seconds for API requests.

This matches the default timeout used by OpenAI's Python client.
See https://github.com/openai/openai-python/blob/v1.54.4/src/openai/_constants.py#L9
"""

try:
    import httpx as legacy_httpx
except ImportError:
    legacy_httpx = None

if TYPE_CHECKING:
    import httpx

    AsyncHTTPClient: TypeAlias = httpx.AsyncClient | httpx2.AsyncClient
elif legacy_httpx is not None:
    AsyncHTTPClient = legacy_httpx.AsyncClient | httpx2.AsyncClient
else:
    AsyncHTTPClient = httpx2.AsyncClient


def create_async_httpx2_client(*, timeout: int = DEFAULT_HTTP_TIMEOUT, connect: int = 5) -> httpx2.AsyncClient:
    """Create an `httpx2.AsyncClient` with Pydantic AI's default timeouts and user agent.

    Each call creates a new client instance. When used via a [`Provider`][pydantic_ai.providers.Provider],
    the client's lifecycle is managed automatically — it will be closed when the provider (or agent) exits.
    """
    from .models import get_user_agent

    return httpx2.AsyncClient(
        timeout=httpx2.Timeout(timeout=timeout, connect=connect),
        headers={'User-Agent': get_user_agent()},
    )


class _LegacyResponseStream(httpx2.AsyncByteStream):
    """Streams a legacy `httpx.Response` body as an `httpx2` byte stream."""

    def __init__(self, response: httpx.Response) -> None:
        self._response = response

    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._response.aiter_raw():
            yield chunk

    async def aclose(self) -> None:
        await self._response.aclose()


# TODO(v3): remove, along with the legacy `httpx.AsyncClient` support it exists for.
class LegacyHttpxAsyncClient(httpx2.AsyncClient):
    """An `httpx2.AsyncClient` that performs every request through a caller-owned legacy `httpx.AsyncClient`.

    SDKs built on `httpx2` accept only `httpx2.AsyncClient` instances, so a legacy client cannot be passed to
    them directly. It also cannot be converted: its proxies, TLS context, transport and connection pool are
    private state. This facade satisfies the SDK's type check and delegates the actual I/O to the legacy
    client, so everything the caller configured on it — proxies, mounts, certificates, limits, auth, event
    hooks, default headers — keeps applying. The legacy client stays the caller's to close.
    """

    def __init__(self, legacy_client: httpx.AsyncClient) -> None:
        timeout = legacy_client.timeout
        # The SDK reads `timeout` off the client it is given to decide whether a custom timeout is in effect.
        super().__init__(
            timeout=httpx2.Timeout(connect=timeout.connect, read=timeout.read, write=timeout.write, pool=timeout.pool)
        )
        self.legacy_client = legacy_client

    async def send(
        self,
        request: httpx2.Request,
        *,
        stream: bool = False,
        auth: Any = httpx2.USE_CLIENT_DEFAULT,
        follow_redirects: Any = httpx2.USE_CLIENT_DEFAULT,
    ) -> httpx2.Response:
        assert legacy_httpx is not None  # a legacy client exists, so legacy httpx is importable
        timeout = request.extensions.get('timeout')
        legacy_request = self.legacy_client.build_request(
            request.method,
            str(request.url),
            headers=request.headers.raw,
            # Bodies cross as bytes: legacy httpx does not recognise httpx2 stream types.
            content=await request.aread(),
            extensions={'timeout': timeout} if timeout is not None else None,
        )
        legacy_response = await self.legacy_client.send(legacy_request, stream=True)
        try:
            # A transport may hand back an already-buffered body (`MockTransport` does); stream otherwise.
            body: dict[str, Any] = {'content': legacy_response.content}
        except legacy_httpx.ResponseNotRead:
            body = {'stream': _LegacyResponseStream(legacy_response)}
        response = httpx2.Response(
            legacy_response.status_code,
            headers=legacy_response.headers.raw,
            request=request,
            extensions={
                key: value
                for key, value in legacy_response.extensions.items()
                if key in ('http_version', 'reason_phrase')
            },
            **body,
        )
        if not stream:
            await response.aread()
            await response.aclose()
        return response

    async def aclose(self) -> None:
        # Only the idle `httpx2` pool is ours; the legacy client belongs to the caller.
        await super().aclose()


# TODO(v3): remove, along with the legacy `httpx.AsyncClient` support it warns about.
def warn_if_legacy_httpx_client(http_client: object, *, consumer: str, stacklevel: int) -> None:
    """Warn when a caller-owned HTTP client is a legacy `httpx.AsyncClient` rather than an `httpx2.AsyncClient`.

    Does nothing when legacy `httpx` isn't installed, since no client can then be an instance of it.

    Args:
        http_client: The client the caller was handed; only legacy `httpx.AsyncClient` instances warn.
        consumer: Name of the surface accepting the client, interpolated into the warning message.
        stacklevel: The stacklevel the caller would pass to its own `warnings.warn` call — this helper
            adds 1 to account for its own frame. Callers pick the value that lands the warning on the
            user's provider-constructor call site.
    """
    if legacy_httpx is None:
        return

    if isinstance(http_client, legacy_httpx.AsyncClient):
        warnings.warn(
            f'`httpx.AsyncClient` support for {consumer} is deprecated and will be removed in v3; '
            'use `httpx2.AsyncClient` instead.',
            PydanticAIDeprecationWarning,
            stacklevel=stacklevel + 1,
        )
