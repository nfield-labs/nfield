"""
Unit tests for formatshield.backends.ollama_backend.OllamaBackend.

Covers construction, property values, and generate/stream behaviour
using a monkeypatched ollama.AsyncClient — no real server calls are made.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.backends.ollama_backend import OllamaBackend

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ollama_response(content: str) -> MagicMock:
    """Return a minimal mock ollama chat response."""
    message = MagicMock()
    message.content = content
    resp = MagicMock()
    resp.message = message
    return resp


def _make_ollama_stream_chunk(content: str | None) -> MagicMock:
    """Return a minimal mock ollama streaming chunk."""
    message = MagicMock()
    message.content = content
    chunk = MagicMock()
    chunk.message = message
    return chunk


# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_ollama_backend_name() -> None:
    """OllamaBackend.name must be 'ollama'."""
    assert OllamaBackend().name == "ollama"


def test_ollama_backend_no_api_key_required() -> None:
    """OllamaBackend does not require an API key — construction always succeeds."""
    # No env vars, no key argument: should not raise
    backend = OllamaBackend()
    assert backend is not None


def test_ollama_backend_default_host() -> None:
    """OllamaBackend defaults to localhost:11434."""
    backend = OllamaBackend()
    assert "localhost" in backend.host
    assert "11434" in backend.host


def test_ollama_backend_custom_base_url() -> None:
    """OllamaBackend accepts a custom host/base_url."""
    backend = OllamaBackend(host="http://myserver:11434")
    assert backend.host == "http://myserver:11434"


def test_ollama_backend_strips_prefix() -> None:
    """'ollama/' prefix is stripped from the model name."""
    backend = OllamaBackend(model="ollama/llama3.1:70b")
    assert backend.model == "llama3.1:70b"


def test_ollama_backend_plain_model_name_unchanged() -> None:
    """Model name without 'ollama/' prefix is kept as-is."""
    backend = OllamaBackend(model="llama3.1:70b")
    assert backend.model == "llama3.1:70b"


def test_ollama_supports_kv_cache_reuse_false() -> None:
    """OllamaBackend.supports_kv_cache_reuse must be False."""
    assert OllamaBackend().supports_kv_cache_reuse is False


def test_ollama_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline must be a float between 0 and 1."""
    baseline = OllamaBackend().accuracy_loss_baseline
    assert baseline is not None
    assert 0.0 < baseline < 1.0


def test_ollama_accuracy_loss_baseline_value() -> None:
    """accuracy_loss_baseline must be 0.22."""
    assert OllamaBackend().accuracy_loss_baseline == 0.22


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_generate_returns_string() -> None:
    """generate() returns the model's text content as a string."""
    backend = OllamaBackend()

    mock_resp = _make_ollama_response("Hello from Ollama!")

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        result = await backend.generate("Say hello")

    assert result == "Hello from Ollama!"


@pytest.mark.asyncio
async def test_ollama_generate_json_mode_activated_by_schema() -> None:
    """generate(schema=...) activates JSON mode (format='json')."""
    backend = OllamaBackend()

    mock_resp = _make_ollama_response('{"name": "Alice"}')

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_client.chat = create_mock
        mock_cls.return_value = mock_client

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        result = await backend.generate("Extract name", schema=schema)

    assert result == '{"name": "Alice"}'
    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("format") == "json"


@pytest.mark.asyncio
async def test_ollama_generate_with_schema_embeds_system_prompt() -> None:
    """generate(schema=...) includes schema in a system message."""
    backend = OllamaBackend()

    mock_resp = _make_ollama_response('{"name": "Alice"}')

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_client.chat = create_mock
        mock_cls.return_value = mock_client

        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        await backend.generate("Extract name", schema=schema)

    messages = create_mock.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "json" in messages[0]["content"].lower()


@pytest.mark.asyncio
async def test_ollama_generate_none_content_returns_empty_string() -> None:
    """generate() returns '' when the model returns None/empty content."""
    backend = OllamaBackend()

    mock_resp = _make_ollama_response("")

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        result = await backend.generate("test")

    assert result == ""


@pytest.mark.asyncio
async def test_ollama_generate_kv_cache_prefix_ignored() -> None:
    """kv_cache_prefix is accepted but ignored (Ollama has no prefix caching)."""
    backend = OllamaBackend()

    mock_resp = _make_ollama_response("ok")

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        create_mock = AsyncMock(return_value=mock_resp)
        mock_client.chat = create_mock
        mock_cls.return_value = mock_client

        result = await backend.generate("test", kv_cache_prefix="prefix text")

    assert result == "ok"
    messages = create_mock.call_args.kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert all("prefix text" not in m["content"] for m in system_messages)


@pytest.mark.asyncio
async def test_ollama_generate_response_error_not_found_raises_runtime() -> None:
    """ResponseError with 'not found' message raises RuntimeError immediately."""
    from ollama import ResponseError

    backend = OllamaBackend(model="missing-model")

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(side_effect=ResponseError("model 'missing-model' not found"))
        mock_cls.return_value = mock_client

        with pytest.raises(RuntimeError, match="missing-model"):
            await backend.generate("test")


@pytest.mark.asyncio
async def test_ollama_generate_uses_host_for_client() -> None:
    """AsyncClient is constructed with the configured host."""
    backend = OllamaBackend(host="http://custom-host:11434")

    mock_resp = _make_ollama_response("result")

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=mock_resp)
        mock_cls.return_value = mock_client

        await backend.generate("test")

    mock_cls.assert_called_once_with(host="http://custom-host:11434")


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ollama_stream_yields_output_and_complete_events() -> None:
    """stream() yields output events and a final complete event."""
    backend = OllamaBackend()

    chunks = [
        _make_ollama_stream_chunk("Hello"),
        _make_ollama_stream_chunk(" world"),
        _make_ollama_stream_chunk(None),
    ]

    async def _async_iter():
        for c in chunks:
            yield c

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_async_iter())
        mock_cls.return_value = mock_client

        stream = await backend.stream("hello")
        events = [e async for e in stream]

    output_events = [e for e in events if e.type == "output"]
    complete_events = [e for e in events if e.type == "complete"]

    assert len(output_events) == 2
    assert len(complete_events) == 1
    assert complete_events[0].content == "Hello world"


@pytest.mark.asyncio
async def test_ollama_stream_events_have_backend_name() -> None:
    """All stream events must carry backend='ollama'."""
    backend = OllamaBackend()

    chunks = [_make_ollama_stream_chunk("hi"), _make_ollama_stream_chunk(None)]

    async def _async_iter():
        for c in chunks:
            yield c

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_async_iter())
        mock_cls.return_value = mock_client

        stream = await backend.stream("hi")
        events = [e async for e in stream]

    for e in events:
        assert e.backend == "ollama"


@pytest.mark.asyncio
async def test_ollama_stream_json_mode_activated_by_schema() -> None:
    """stream(schema=...) activates JSON mode (format='json') in the request."""
    backend = OllamaBackend()

    chunks = [_make_ollama_stream_chunk('{"x":1}')]

    async def _async_iter():
        for c in chunks:
            yield c

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        create_mock = AsyncMock(return_value=_async_iter())
        mock_client.chat = create_mock
        mock_cls.return_value = mock_client

        schema = {"type": "object"}
        stream = await backend.stream("test", schema=schema)
        _ = [e async for e in stream]

    call_kwargs = create_mock.call_args.kwargs
    assert call_kwargs.get("format") == "json"


@pytest.mark.asyncio
async def test_ollama_stream_complete_event_has_full_text() -> None:
    """The final complete event's content contains all accumulated tokens."""
    backend = OllamaBackend()

    chunks = [
        _make_ollama_stream_chunk("foo"),
        _make_ollama_stream_chunk("bar"),
        _make_ollama_stream_chunk("baz"),
    ]

    async def _async_iter():
        for c in chunks:
            yield c

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_async_iter())
        mock_cls.return_value = mock_client

        stream = await backend.stream("test")
        events = [e async for e in stream]

    complete_events = [e for e in events if e.type == "complete"]
    assert complete_events[0].content == "foobarbaz"


@pytest.mark.asyncio
async def test_ollama_stream_response_error_raises_runtime() -> None:
    """stream() wraps ResponseError in a RuntimeError."""
    from ollama import ResponseError

    backend = OllamaBackend(model="missing-model")

    async def _bad_iter():
        raise ResponseError("model 'missing-model' not found")
        yield  # make it a generator

    with patch("formatshield.backends.ollama_backend.AsyncClient") as mock_cls:
        mock_client = MagicMock()
        mock_client.chat = AsyncMock(return_value=_bad_iter())
        mock_cls.return_value = mock_client

        stream = await backend.stream("test")
        with pytest.raises(RuntimeError, match="missing-model"):
            _ = [e async for e in stream]
