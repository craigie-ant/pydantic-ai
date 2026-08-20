"""`LegacyHttpxAsyncClient` lets an `httpx2`-based SDK run on a caller-owned legacy `httpx.AsyncClient`.

Not VCR tests: the point is that the legacy client's own transport serves the requests, so a legacy
`MockTransport` stands in for it and the assertions follow the bytes across the bridge in both directions.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import httpx2
import pytest

from pydantic_ai._http import LegacyHttpxAsyncClient

from .conftest import try_import

with try_import() as imports_successful:
    import anthropic

pytestmark = [
    pytest.mark.skipif(not imports_successful(), reason='anthropic not installed'),
    pytest.mark.anyio,
]

_MESSAGE: dict[str, Any] = {
    'id': 'msg',
    'type': 'message',
    'role': 'assistant',
    'model': 'claude-sonnet-4-5',
    'content': [{'type': 'text', 'text': 'hello'}],
    'stop_reason': 'end_turn',
    'stop_sequence': None,
    'usage': {'input_tokens': 1, 'output_tokens': 1},
}

_STREAM_EVENTS: list[tuple[str, dict[str, Any]]] = [
    ('message_start', {'type': 'message_start', 'message': {**_MESSAGE, 'content': [], 'stop_reason': None}}),
    ('content_block_start', {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}}),
    ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'hel'}}),
    ('content_block_delta', {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': 'lo'}}),
    ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
    (
        'message_delta',
        {'type': 'message_delta', 'delta': {'stop_reason': 'end_turn', 'stop_sequence': None}, 'usage': {'output_tokens': 2}},
    ),
    ('message_stop', {'type': 'message_stop'}),
]


class _ChunkedStream(httpx.AsyncByteStream):
    """A genuinely streamed legacy body, so the bridge's streaming path (not the buffered one) is exercised."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


def _sdk_client(handler: Any, **kwargs: Any) -> tuple[anthropic.AsyncAnthropic, httpx.AsyncClient]:
    legacy_client = httpx.AsyncClient(transport=httpx.MockTransport(handler), **kwargs)
    return anthropic.AsyncAnthropic(api_key='k', max_retries=0, http_client=LegacyHttpxAsyncClient(legacy_client)), legacy_client


async def test_bridge_routes_requests_through_the_legacy_client():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_MESSAGE)

    client, legacy_client = _sdk_client(handler, headers={'x-from-legacy-client': 'yes'})
    message = await client.messages.create(
        model='claude-sonnet-4-5',
        max_tokens=5,
        messages=[{'role': 'user', 'content': 'hi'}],
        extra_body={'temperature': 0.2},
    )

    assert message.content[0].type == 'text' and message.content[0].text == 'hello'
    (request,) = seen
    assert request.headers['x-api-key'] == 'k'
    assert request.headers['x-from-legacy-client'] == 'yes', 'the legacy client default headers apply'
    assert json.loads(request.content)['temperature'] == 0.2
    await client.close()
    assert not legacy_client.is_closed, 'the legacy client is caller-owned'


async def test_bridge_streams_a_legacy_response():
    def handler(request: httpx.Request) -> httpx.Response:
        chunks = [f'event: {event}\ndata: {json.dumps(data)}\n\n'.encode() for event, data in _STREAM_EVENTS]
        return httpx.Response(200, headers={'content-type': 'text/event-stream'}, stream=_ChunkedStream(chunks))

    client, _ = _sdk_client(handler)
    async with client.messages.stream(
        model='claude-sonnet-4-5', max_tokens=5, messages=[{'role': 'user', 'content': 'hi'}]
    ) as stream:
        text = ''.join([chunk async for chunk in stream.text_stream])
        final = await stream.get_final_message()

    assert text == 'hello'
    assert final.stop_reason == 'end_turn'


async def test_bridge_surfaces_legacy_errors_as_httpx2_responses():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            429,
            headers={'retry-after': '3'},
            json={'type': 'error', 'error': {'type': 'rate_limit_error', 'message': 'slow down'}},
        )

    client, _ = _sdk_client(handler)
    with pytest.raises(anthropic.RateLimitError) as exc_info:
        await client.messages.create(
            model='claude-sonnet-4-5', max_tokens=5, messages=[{'role': 'user', 'content': 'hi'}]
        )

    assert exc_info.value.status_code == 429
    assert isinstance(exc_info.value.response, httpx2.Response)
    assert exc_info.value.response.headers['retry-after'] == '3'


async def test_bridge_forwards_the_legacy_timeout_and_per_request_timeouts():
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=_MESSAGE)

    client, legacy_client = _sdk_client(handler, timeout=httpx.Timeout(7, connect=3))
    facade = client._client  # pyright: ignore[reportPrivateUsage]
    assert isinstance(facade, httpx2.AsyncClient)
    assert (facade.timeout.connect, facade.timeout.read) == (3, 7), 'the SDK sees the legacy client timeout'

    await client.messages.create(
        model='claude-sonnet-4-5', max_tokens=5, messages=[{'role': 'user', 'content': 'hi'}], timeout=11
    )
    assert seen[0].extensions['timeout'] == httpx2.Timeout(11).as_dict()
