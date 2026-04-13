"""
Unit tests for formatshield.backends.groq_backend.GroqBackend.

Covers construction, property values, and generate/stream behaviour
using a monkeypatched AsyncGroq client — no real API calls are made.
"""

from __future__ import annotations

import json
import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.backends.groq_backend import GroqBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str | None) -> MagicMock:
    """Return a minimal mock chat completion response."""
    choice = MagicMock()
    choice.message.content = content
    resp = MagicMock()
    resp.choices = [choice]
    return resp


def _make_stream_chunk(delta_content: str | None) -> MagicMock:
    """Return a minimal mock streaming chunk."""
    delta = MagicMock()
    delta.content = delta_content
    choice = MagicMock()
    choice.delta = delta
    chunk = MagicMock()
    chunk.choices = [choice]
    return chunk


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_groq_backend_name() -> None:
    """GroqBackend.name must be 'groq'."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        assert GroqBackend().name == "groq"


def test_groq_backend_requires_api_key() -> None:
    """GroqBackend raises ValueError when no API key is available."""
    with patch.dict("os.environ", {}, clear=True):
        os.environ.pop("GROQ_API_KEY", None)
        with pytest.raises(ValueError, match="GROQ_API_KEY"):
            GroqBackend(api_key=None)


def test_groq_backend_strips_prefix() -> None:
    """'groq/' prefix is stripped from the model name."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend(model="groq/llama-3.1-70b-versatile")
        assert backend.model == "llama-3.1-70b-versatile"


def test_groq_backend_plain_model_name_unchanged() -> None:
    """Model name without 'groq/' prefix is kept as-is."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend(model="llama-3.1-70b-versatile")
        assert backend.model == "llama-3.1-70b-versatile"


def test_groq_supports_kv_cache_reuse_false() -> None:
    """GroqBackend.supports_kv_cache_reuse must be False."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        assert GroqBackend().supports_kv_cache_reuse is False


def test_groq_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline must be a float between 0 and 1."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        baseline = GroqBackend().accuracy_loss_baseline
        assert baseline is not None
        assert 0.0 < baseline < 1.0


def test_groq_accuracy_loss_baseline_value() -> None:
    """accuracy_loss_baseline must be 0.18."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        assert GroqBackend().accuracy_loss_baseline == 0.18


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_generate_returns_string() -> None:
    """generate() returns the model's text content as a string."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response("Hello from Groq!")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("Say hello")
    assert result == "Hello from Groq!"


@pytest.mark.asyncio
async def test_groq_generate_json_mode_sets_response_format() -> None:
    """generate(constraints='json') activates json_object response_format."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response('{"answer": 42}')
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    result = await backend.generate("test", constraints="json")
    assert json.loads(result) == {"answer": 42}

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_groq_generate_with_schema_embeds_system_prompt() -> None:
    """generate(schema=...) without json constraint embeds schema in system prompt."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response('{"name": "Alice"}')
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract the name", schema=schema)

    messages = create_mock.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "json" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_groq_generate_schema_with_json_constraint_no_system_prompt() -> None:
    """generate(schema=..., constraints='json') skips schema system prompt."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response('{"name": "Bob"}')
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract name", schema=schema, constraints="json")

    messages = create_mock.call_args.kwargs["messages"]
    # With constraints="json", schema is NOT embedded in a system message
    roles = [m["role"] for m in messages]
    assert roles == ["user"]


@pytest.mark.asyncio
async def test_groq_generate_none_content_returns_empty_string() -> None:
    """generate() returns '' when the model returns None content."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response(None)  # type: ignore[arg-type]
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("test")
    assert result == ""


@pytest.mark.asyncio
async def test_groq_generate_kv_cache_prefix_ignored() -> None:
    """kv_cache_prefix is accepted but ignored (Groq has no prefix caching)."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    mock_resp = _make_chat_response("ok")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    result = await backend.generate("test", kv_cache_prefix="system prompt prefix")
    assert result == "ok"

    # kv_cache_prefix must NOT appear as a system message
    messages = create_mock.call_args.kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert all("system prompt prefix" not in m["content"] for m in system_messages)


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_groq_stream_yields_output_and_complete_events() -> None:
    """stream() yields output events and a final complete event."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    chunks = [
        _make_stream_chunk("Hello"),
        _make_stream_chunk(" world"),
        _make_stream_chunk(None),
    ]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=stream_ctx)

    stream = await backend.stream("hello")
    events = [e async for e in stream]

    output_events = [e for e in events if e.type == "output"]
    complete_events = [e for e in events if e.type == "complete"]

    assert len(output_events) == 2
    assert len(complete_events) == 1
    assert complete_events[0].content == "Hello world"


@pytest.mark.asyncio
async def test_groq_stream_events_have_backend_name() -> None:
    """All stream events must carry backend='groq'."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    chunks = [_make_stream_chunk("hi"), _make_stream_chunk(None)]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=stream_ctx)

    stream = await backend.stream("hi")
    events = [e async for e in stream]
    for e in events:
        assert e.backend == "groq"


@pytest.mark.asyncio
async def test_groq_stream_json_mode_sets_response_format() -> None:
    """stream(constraints='json') activates json_object response_format."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    chunks = [_make_stream_chunk('{"x": 1}'), _make_stream_chunk(None)]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=stream_ctx)
    backend._client.chat.completions.create = create_mock

    stream = await backend.stream("test", constraints="json")
    _ = [e async for e in stream]

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_groq_stream_complete_event_has_full_text() -> None:
    """The final complete event's content contains all accumulated tokens."""
    with patch.dict("os.environ", {"GROQ_API_KEY": "gsk-test"}):
        backend = GroqBackend()

    chunks = [
        _make_stream_chunk("foo"),
        _make_stream_chunk("bar"),
        _make_stream_chunk("baz"),
    ]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=stream_ctx)

    stream = await backend.stream("test")
    events = [e async for e in stream]

    complete_events = [e for e in events if e.type == "complete"]
    assert complete_events[0].content == "foobarbaz"
