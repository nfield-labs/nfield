"""
Unit tests for formatshield.backends.anthropic_backend.AnthropicBackend.

Covers construction, property values, and generate/stream behaviour
using a monkeypatched AsyncAnthropic client — no real API calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.backends.anthropic_backend import AnthropicBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_messages_response(text: str) -> MagicMock:
    """Return a minimal mock Messages response."""
    content_block = MagicMock()
    content_block.text = text
    resp = MagicMock()
    resp.content = [content_block]
    return resp


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_anthropic_backend_name() -> None:
    """AnthropicBackend.name must be 'anthropic'."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        assert AnthropicBackend().name == "anthropic"


def test_anthropic_backend_requires_api_key() -> None:
    """AnthropicBackend raises ValueError when no API key is available."""
    import os

    with patch.dict("os.environ", {}, clear=True):
        os.environ.pop("ANTHROPIC_API_KEY", None)
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            AnthropicBackend(api_key=None)


def test_anthropic_backend_strips_prefix() -> None:
    """'anthropic/' prefix is stripped from the model name."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend(model="anthropic/claude-3-5-haiku-20241022")
        assert backend.model == "claude-3-5-haiku-20241022"


def test_anthropic_supports_kv_cache_reuse_false() -> None:
    """AnthropicBackend.supports_kv_cache_reuse must be False."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        assert AnthropicBackend().supports_kv_cache_reuse is False


def test_anthropic_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline must be a float between 0 and 1."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        baseline = AnthropicBackend().accuracy_loss_baseline
        assert baseline is not None
        assert 0.0 < baseline < 1.0


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_anthropic_generate_returns_string() -> None:
    """generate() returns the model's text content as a string."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response("Hello from Claude!")
    backend._client.messages = MagicMock()
    backend._client.messages.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("Say hello")
    assert result == "Hello from Claude!"


@pytest.mark.asyncio
async def test_anthropic_generate_json_mode_injects_system_prompt() -> None:
    """generate(constraints='json') injects a JSON system prompt."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response('{"answer": 42}')
    backend._client.messages = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.messages.create = create_mock

    result = await backend.generate("test", constraints="json")
    assert json.loads(result) == {"answer": 42}

    call_kwargs = create_mock.call_args.kwargs
    assert "system" in call_kwargs
    assert "json" in call_kwargs["system"].lower()


@pytest.mark.asyncio
async def test_anthropic_generate_with_schema_injects_system_prompt() -> None:
    """generate(schema=...) injects a system prompt referencing the schema."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response('{"name": "Alice"}')
    backend._client.messages = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.messages.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract the name", schema=schema)

    call_kwargs = create_mock.call_args.kwargs
    assert "system" in call_kwargs
    assert "json" in call_kwargs["system"].lower()


@pytest.mark.asyncio
async def test_anthropic_generate_no_schema_no_system_prompt() -> None:
    """generate() without schema or constraints sends no system prompt."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response("free text response")
    backend._client.messages = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.messages.create = create_mock

    await backend.generate("What is 2+2?")

    call_kwargs = create_mock.call_args.kwargs
    assert "system" not in call_kwargs


@pytest.mark.asyncio
async def test_anthropic_generate_kv_cache_prefix_ignored() -> None:
    """kv_cache_prefix is accepted but ignored (no KV cache support)."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response("ok")
    backend._client.messages = MagicMock()
    backend._client.messages.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("test", kv_cache_prefix="prefix")
    assert result == "ok"


@pytest.mark.asyncio
async def test_anthropic_generate_max_tokens_set() -> None:
    """generate() always sends max_tokens in the request."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    mock_resp = _make_messages_response("response")
    backend._client.messages = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.messages.create = create_mock

    await backend.generate("test")

    call_kwargs = create_mock.call_args.kwargs
    assert "max_tokens" in call_kwargs
    assert call_kwargs["max_tokens"] > 0


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


def _make_stream_ctx(text_chunks: list[str]) -> MagicMock:
    """Return a mock that matches Anthropic's messages.stream() context manager.

    The Anthropic streaming API exposes a `text_stream` async generator on the
    stream object, not on the context manager's ``__aenter__`` return value.
    """

    async def _text_stream():
        for chunk in text_chunks:
            yield chunk

    stream_obj = MagicMock()
    stream_obj.text_stream = _text_stream()

    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=stream_obj)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return ctx


@pytest.mark.asyncio
async def test_anthropic_stream_yields_output_and_complete_events() -> None:
    """stream() yields output events and a final complete event."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    stream_ctx = _make_stream_ctx(["Hello ", "Claude"])
    backend._client.messages = MagicMock()
    backend._client.messages.stream = MagicMock(return_value=stream_ctx)

    stream = await backend.stream("hello")
    events = [e async for e in stream]

    output_events = [e for e in events if e.type == "output"]
    complete_events = [e for e in events if e.type == "complete"]

    assert len(output_events) == 2
    assert len(complete_events) == 1
    assert complete_events[0].content == "Hello Claude"


@pytest.mark.asyncio
async def test_anthropic_stream_events_have_backend_name() -> None:
    """All stream events must carry backend='anthropic'."""
    with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-ant-test"}):
        backend = AnthropicBackend()

    stream_ctx = _make_stream_ctx(["hi"])
    backend._client.messages = MagicMock()
    backend._client.messages.stream = MagicMock(return_value=stream_ctx)

    stream = await backend.stream("hi")
    events = [e async for e in stream]
    for e in events:
        assert e.backend == "anthropic"
