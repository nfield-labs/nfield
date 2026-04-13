"""
Unit tests for formatshield.backends.vllm_backend.VLLMBackend.

Covers construction, property values, and generate/stream behaviour
using a monkeypatched AsyncOpenAI client — no real server calls are made.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from formatshield.backends.vllm_backend import VLLMBackend

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


def test_vllm_backend_name() -> None:
    """VLLMBackend.name must be 'vllm'."""
    assert VLLMBackend().name == "vllm"


def test_vllm_backend_no_api_key_required() -> None:
    """VLLMBackend does not require a real API key — construction always succeeds."""
    backend = VLLMBackend()
    assert backend is not None


def test_vllm_backend_default_base_url() -> None:
    """VLLMBackend defaults to localhost:8000/v1."""
    backend = VLLMBackend()
    assert "localhost" in backend.base_url
    assert "8000" in backend.base_url


def test_vllm_backend_custom_base_url() -> None:
    """VLLMBackend accepts a custom base_url."""
    backend = VLLMBackend(base_url="http://myserver:8000/v1")
    assert backend.base_url == "http://myserver:8000/v1"


def test_vllm_backend_strips_prefix() -> None:
    """'vllm/' prefix is stripped from the model name."""
    backend = VLLMBackend(model="vllm/meta-llama/Llama-3-70b-Instruct")
    assert backend.model == "meta-llama/Llama-3-70b-Instruct"


def test_vllm_backend_plain_model_name_unchanged() -> None:
    """Model name without 'vllm/' prefix is kept as-is."""
    backend = VLLMBackend(model="meta-llama/Llama-3-70b-Instruct")
    assert backend.model == "meta-llama/Llama-3-70b-Instruct"


def test_vllm_supports_kv_cache_reuse_true() -> None:
    """VLLMBackend.supports_kv_cache_reuse must be True."""
    assert VLLMBackend().supports_kv_cache_reuse is True


def test_vllm_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline must be a float between 0 and 1."""
    baseline = VLLMBackend().accuracy_loss_baseline
    assert baseline is not None
    assert 0.0 < baseline < 1.0


def test_vllm_accuracy_loss_baseline_value() -> None:
    """accuracy_loss_baseline must be 0.23."""
    assert VLLMBackend().accuracy_loss_baseline == 0.23


# ---------------------------------------------------------------------------
# generate()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_generate_returns_string() -> None:
    """generate() returns the model's text content as a string."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response("Hello from vLLM!")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("Say hello")
    assert result == "Hello from vLLM!"


@pytest.mark.asyncio
async def test_vllm_generate_json_mode_sets_response_format() -> None:
    """generate(constraints='json') activates json_object response_format."""
    backend = VLLMBackend()

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
async def test_vllm_generate_with_schema_embeds_system_prompt() -> None:
    """generate(schema=...) without json constraint embeds schema in system prompt."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response('{"name": "Alice"}')
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract the name", schema=schema)

    messages = create_mock.call_args.kwargs["messages"]
    system_messages = [m for m in messages if m["role"] == "system"]
    assert len(system_messages) >= 1
    assert any("json" in m["content"].lower() for m in system_messages)


@pytest.mark.asyncio
async def test_vllm_generate_schema_with_json_constraint_no_schema_prompt() -> None:
    """generate(schema=..., constraints='json') skips the schema system message."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response('{"name": "Bob"}')
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    await backend.generate("Extract name", schema=schema, constraints="json")

    messages = create_mock.call_args.kwargs["messages"]
    roles = [m["role"] for m in messages]
    assert roles == ["user"]


@pytest.mark.asyncio
async def test_vllm_generate_none_content_returns_empty_string() -> None:
    """generate() returns '' when the model returns None content."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response(None)  # type: ignore[arg-type]
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    backend._client.chat.completions.create = AsyncMock(return_value=mock_resp)

    result = await backend.generate("test")
    assert result == ""


@pytest.mark.asyncio
async def test_vllm_generate_kv_cache_prefix_used() -> None:
    """kv_cache_prefix IS used — injected as first system message for prefix caching."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response("ok")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    result = await backend.generate("test prompt", kv_cache_prefix="shared system context")
    assert result == "ok"

    messages = create_mock.call_args.kwargs["messages"]
    # The first message must be the kv_cache_prefix system message
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "shared system context"


@pytest.mark.asyncio
async def test_vllm_generate_kv_cache_prefix_is_first_message() -> None:
    """kv_cache_prefix system message comes before any schema system message."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response("result")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    await backend.generate(
        "extract x",
        schema=schema,
        kv_cache_prefix="my prefix",
    )

    messages = create_mock.call_args.kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == "my prefix"
    # Second message is the schema system prompt
    assert messages[1]["role"] == "system"
    assert "json" in messages[1]["content"].lower()
    # Last message is the user prompt
    assert messages[-1]["role"] == "user"


@pytest.mark.asyncio
async def test_vllm_generate_no_kv_cache_prefix_no_extra_system_message() -> None:
    """Without kv_cache_prefix, no extra system message is prepended."""
    backend = VLLMBackend()

    mock_resp = _make_chat_response("ok")
    backend._client = MagicMock()
    backend._client.chat.completions = MagicMock()
    create_mock = AsyncMock(return_value=mock_resp)
    backend._client.chat.completions.create = create_mock

    await backend.generate("hello")

    messages = create_mock.call_args.kwargs["messages"]
    assert messages[0]["role"] == "user"


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_vllm_stream_yields_output_and_complete_events() -> None:
    """stream() yields output events and a final complete event."""
    backend = VLLMBackend()

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
async def test_vllm_stream_events_have_backend_name() -> None:
    """All stream events must carry backend='vllm'."""
    backend = VLLMBackend()

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
        assert e.backend == "vllm"


@pytest.mark.asyncio
async def test_vllm_stream_json_mode_sets_response_format() -> None:
    """stream(constraints='json') activates json_object response_format."""
    backend = VLLMBackend()

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
async def test_vllm_stream_complete_event_has_full_text() -> None:
    """The final complete event's content contains all accumulated tokens."""
    backend = VLLMBackend()

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


@pytest.mark.asyncio
async def test_vllm_stream_kv_cache_prefix_not_forwarded() -> None:
    """stream() does not forward kv_cache_prefix (only generate() uses it)."""
    backend = VLLMBackend()

    chunks = [_make_stream_chunk("response")]

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

    # stream() has no kv_cache_prefix parameter; internally uses None
    stream = await backend.stream("test prompt")
    _ = [e async for e in stream]

    messages = create_mock.call_args.kwargs["messages"]
    # Without a prefix, first message should be the user prompt
    assert messages[0]["role"] == "user"
