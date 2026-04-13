"""
Unit tests for formatshield.backends.openai_backend.OpenAIBackend.

Covers construction, property values, and generate/stream behaviour
using a monkeypatched AsyncOpenAI client — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.backends.openai_backend import OpenAIBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_chat_response(content: str) -> MagicMock:
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


def test_openai_backend_name() -> None:
    """OpenAIBackend.name must be 'openai'."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        assert OpenAIBackend().name == "openai"


def test_openai_backend_requires_api_key() -> None:
    """OpenAIBackend raises ValueError when no API key is available."""
    with patch.dict("os.environ", {}, clear=True):
        # Ensure OPENAI_API_KEY is not set
        import os

        os.environ.pop("OPENAI_API_KEY", None)
        with pytest.raises(ValueError, match="OPENAI_API_KEY"):
            OpenAIBackend(api_key=None)


def test_openai_backend_strips_prefix() -> None:
    """'openai/' prefix is stripped from the model name."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend(model="openai/gpt-4o-mini")
        assert backend.model == "gpt-4o-mini"


def test_openai_supports_kv_cache_reuse_false() -> None:
    """OpenAIBackend.supports_kv_cache_reuse must be False."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        assert OpenAIBackend().supports_kv_cache_reuse is False


def test_openai_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline must be a float between 0 and 1."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        baseline = OpenAIBackend().accuracy_loss_baseline
        assert baseline is not None
        assert 0.0 < baseline < 1.0


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_generate_returns_string() -> None:
    """generate() returns the model's text content as a string."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    mock_resp = _make_chat_response("Hello, world!")
    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("Say hello")
    assert result == "Hello, world!"


@pytest.mark.asyncio
async def test_openai_generate_json_mode_sets_response_format() -> None:
    """generate(constraints='json') activates json_object response_format."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    mock_resp = _make_chat_response('{"answer": 42}')
    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    result = await backend.generate("test", constraints="json")
    assert json.loads(result) == {"answer": 42}

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("response_format") == {"type": "json_object"}


@pytest.mark.asyncio
async def test_openai_generate_with_schema_embeds_system_prompt() -> None:
    """generate(schema=...) without json constraint embeds schema in system prompt."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    mock_resp = _make_chat_response('{"name": "Alice"}')
    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract the name", schema=schema)

    messages = create_mock.call_args.kwargs["messages"]
    # First message should be the system prompt with schema
    assert messages[0]["role"] == "system"
    assert "json" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_openai_generate_none_content_returns_empty_string() -> None:
    """generate() returns '' when the model returns None content."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    mock_resp = _make_chat_response(None)  # type: ignore[arg-type]
    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("test")
    assert result == ""


@pytest.mark.asyncio
async def test_openai_generate_kv_cache_prefix_ignored() -> None:
    """kv_cache_prefix parameter is accepted but ignored (no KV cache support)."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    mock_resp = _make_chat_response("ok")
    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("test", kv_cache_prefix="system prompt prefix")
    assert result == "ok"


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_openai_stream_yields_output_and_complete_events() -> None:
    """stream() yields output events and a final complete event."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    chunks = [_make_stream_chunk("Hello"), _make_stream_chunk(" world"), _make_stream_chunk(None)]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client.chat = MagicMock()
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
async def test_openai_stream_events_have_backend_name() -> None:
    """All stream events must carry backend='openai'."""
    with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
        backend = OpenAIBackend()

    chunks = [_make_stream_chunk("hi"), _make_stream_chunk(None)]

    async def _async_iter():
        for c in chunks:
            yield c

    stream_ctx = MagicMock()
    stream_ctx.__aenter__ = AsyncMock(return_value=_async_iter())
    stream_ctx.__aexit__ = AsyncMock(return_value=False)

    backend._client.chat = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=stream_ctx)

    stream = await backend.stream("hi")
    events = [e async for e in stream]
    for e in events:
        assert e.backend == "openai"
