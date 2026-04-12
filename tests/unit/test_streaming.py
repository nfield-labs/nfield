"""Unit tests for StreamingEngine."""

from __future__ import annotations

import json

import pytest

from formatshield.scorer.features import StreamEvent
from formatshield.streaming.engine import StreamingEngine


async def _make_stream(events: list[StreamEvent]):
    """Helper: create async generator from list."""
    for e in events:
        yield e


def test_streaming_engine_instantiates() -> None:
    engine = StreamingEngine()
    assert engine is not None


@pytest.mark.asyncio
async def test_stream_filters_thinking_by_default() -> None:
    engine = StreamingEngine(expose_thinking=False)
    events = [
        StreamEvent(type="thinking", content="step 1"),
        StreamEvent(type="output", token="hello"),
        StreamEvent(type="complete", content="hello"),
    ]

    collected = []
    async for e in await engine.stream(_make_stream(events)):
        collected.append(e)

    types = [e.type for e in collected]
    assert "thinking" not in types
    assert "output" in types or "complete" in types


@pytest.mark.asyncio
async def test_stream_with_thinking_yields_all_events() -> None:
    engine = StreamingEngine(expose_thinking=True)
    events = [
        StreamEvent(type="thinking", content="step 1"),
        StreamEvent(type="output", token="hello"),
        StreamEvent(type="complete", content="hello"),
    ]

    collected = []
    async for e in await engine.stream_with_thinking(_make_stream(events)):
        collected.append(e)

    types = [e.type for e in collected]
    assert "thinking" in types
    assert "output" in types


@pytest.mark.asyncio
async def test_collect_returns_thinking_and_output() -> None:
    engine = StreamingEngine()
    events = [
        StreamEvent(type="thinking", content="my reasoning"),
        StreamEvent(type="output", token="result"),
        StreamEvent(type="complete", content="result"),
    ]

    thinking, output = await engine.collect(_make_stream(events))
    assert "reasoning" in thinking
    assert output  # non-empty


def test_to_sse_format_is_valid() -> None:
    engine = StreamingEngine()
    event = StreamEvent(type="output", token="hello", backend="groq")
    sse = engine.to_sse(event)

    assert sse.startswith("data: ")
    assert sse.endswith("\n\n")
    # Parse the JSON payload
    payload = json.loads(sse[len("data: ") :].strip())
    assert payload["type"] == "output"


@pytest.mark.asyncio
async def test_from_text_creates_stream() -> None:
    engine = StreamingEngine()
    tokens = []
    async for event in await engine.from_text("hello world", backend="test"):
        tokens.append(event)

    assert len(tokens) >= 1
    types = {e.type for e in tokens}
    assert "complete" in types


@pytest.mark.asyncio
async def test_streaming_produces_events_in_order() -> None:
    """Events must come in order: thinking* then output* then complete."""
    engine = StreamingEngine(expose_thinking=True)
    events = [
        StreamEvent(type="thinking", content="think 1"),
        StreamEvent(type="thinking", content="think 2"),
        StreamEvent(type="output", token="tok1"),
        StreamEvent(type="output", token="tok2"),
        StreamEvent(type="complete", content="tok1tok2"),
    ]

    collected = []
    async for e in await engine.stream_with_thinking(_make_stream(events)):
        collected.append(e.type)

    # thinking events before output events
    last_thinking = max((i for i, t in enumerate(collected) if t == "thinking"), default=-1)
    first_output = min(
        (i for i, t in enumerate(collected) if t == "output"), default=len(collected)
    )
    assert last_thinking < first_output

    # complete is last
    assert collected[-1] == "complete"


@pytest.mark.asyncio
async def test_complete_event_has_json() -> None:
    engine = StreamingEngine()
    payload = {"answer": 42}
    events = [
        StreamEvent(type="output", token='{"answer": 42}'),
        StreamEvent(type="complete", json=payload, content='{"answer": 42}'),
    ]

    collected = []
    async for e in await engine.stream(_make_stream(events)):
        collected.append(e)

    complete_events = [e for e in collected if e.type == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0].json == payload
