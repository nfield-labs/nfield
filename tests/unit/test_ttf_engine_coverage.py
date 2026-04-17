"""
Targeted coverage tests for formatshield.ttf.engine.

Covers the following previously-uncovered lines:
  199     : stream() → return self._stream_impl(prompt, schema)
  233-236 : _stream_impl Pass 1 except block (backend.stream raises)
  268-276 : _stream_impl Pass 2 except block (backend.stream raises on 2nd call)
  364     : _validate_or_fallback() success path (validation passes)
  388-390 : _validate_or_fallback() generate_direct raises in fallback
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from formatshield.scorer.features import StreamEvent
from formatshield.ttf.engine import TTFEngine

# ---------------------------------------------------------------------------
# Helpers — backends that raise at controlled points
# ---------------------------------------------------------------------------


class _RaisingStreamBackend:
    """Backend whose stream() always raises RuntimeError."""

    name: str = "raising_stream"

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return None

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        return json.dumps({"result": "ok"})

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        return self._stream_gen()

    async def _stream_gen(self) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        raise RuntimeError("simulated Pass 1 stream failure")
        yield StreamEvent(type="output", token="", backend=self.name, latency_ms=0.0)


class _Pass2RaisingStreamBackend:
    """Backend whose stream() works the first time but raises on the second call."""

    name: str = "pass2_raiser"

    def __init__(self) -> None:
        self._call_count = 0

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return None

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        return json.dumps({"result": "ok"})

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        self._call_count += 1
        if self._call_count == 1:
            return self._good_stream()
        return self._bad_stream()

    async def _good_stream(self) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="output", token="thinking...", backend=self.name, latency_ms=10.0)
        yield StreamEvent(
            type="complete", content="thinking done", backend=self.name, latency_ms=20.0
        )

    async def _bad_stream(self) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
        raise RuntimeError("simulated Pass 2 stream failure")
        yield StreamEvent(type="output", token="", backend=self.name, latency_ms=0.0)


# ---------------------------------------------------------------------------
# Line 199: stream() public API (not _stream_impl directly)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_public_api_yields_events(mock_backend) -> None:
    """Calling stream() (not _stream_impl) must yield StreamEvent instances (line 199)."""
    engine = TTFEngine(backend=mock_backend)
    events: list[StreamEvent] = []
    async for event in engine.stream("Test prompt via public stream()"):
        events.append(event)
    assert len(events) > 0
    assert all(isinstance(e, StreamEvent) for e in events)


@pytest.mark.asyncio
async def test_stream_public_api_yields_complete_event(mock_backend) -> None:
    """stream() must emit a 'complete' event (exercises line 199 and _stream_impl)."""
    engine = TTFEngine(backend=mock_backend)
    types_seen = [e.type async for e in engine.stream("Question?")]
    assert "complete" in types_seen


# ---------------------------------------------------------------------------
# Lines 233-236: _stream_impl Pass 1 except block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_impl_pass1_failure_emits_thinking_error_event() -> None:
    """When Pass 1 stream raises, _stream_impl emits an error thinking event (233-236)."""
    backend = _RaisingStreamBackend()
    engine = TTFEngine(backend=backend)
    events: list[StreamEvent] = []
    async for event in engine._stream_impl("Any prompt", schema=None):
        events.append(event)

    # Should have at least one thinking event with the error message
    thinking_events = [e for e in events if e.type == "thinking"]
    assert len(thinking_events) >= 1
    assert any("Pass 1 failed" in (e.content or "") for e in thinking_events)


@pytest.mark.asyncio
async def test_stream_impl_pass1_failure_still_emits_complete() -> None:
    """Even when Pass 1 fails, _stream_impl must emit a complete event."""
    backend = _RaisingStreamBackend()
    engine = TTFEngine(backend=backend)
    types_seen = [e.type async for e in engine._stream_impl("Prompt", schema=None)]
    assert "complete" in types_seen


# ---------------------------------------------------------------------------
# Lines 268-276: _stream_impl Pass 2 except block
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_impl_pass2_failure_emits_complete_with_null_json() -> None:
    """When Pass 2 stream raises, _stream_impl emits a complete event with json=None (268-276)."""
    backend = _Pass2RaisingStreamBackend()
    engine = TTFEngine(backend=backend)
    events: list[StreamEvent] = []
    async for event in engine._stream_impl("Some prompt", schema=None):
        events.append(event)

    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) >= 1
    # Pass 2 failed → complete event with json=None
    assert complete_events[-1].json is None


@pytest.mark.asyncio
async def test_stream_impl_pass2_failure_returns_early() -> None:
    """After Pass 2 failure the generator returns (no further events after complete)."""
    backend = _Pass2RaisingStreamBackend()
    engine = TTFEngine(backend=backend)
    types_seen = [e.type async for e in engine._stream_impl("Q?", schema=None)]
    # complete must be present and must be the last event
    assert "complete" in types_seen
    assert types_seen[-1] == "complete"


# ---------------------------------------------------------------------------
# Line 364: _validate_or_fallback() success path
# ---------------------------------------------------------------------------


class _MatchingModel(BaseModel):
    """Pydantic model that matches MockBackend's constrained output."""

    result: str
    confidence: float


@pytest.mark.asyncio
async def test_validate_or_fallback_success_path(mock_backend) -> None:
    """When validation succeeds, generate() returns the output without triggering fallback."""
    engine = TTFEngine(backend=mock_backend, ttf_fallback=True)
    # MockBackend returns {"result": "mock_answer", "confidence": 0.95} for constrained output
    _thinking, output = await engine.generate(
        prompt="Return matching JSON",
        schema_model=_MatchingModel,
    )
    # Validation succeeded — output is the valid JSON string
    assert isinstance(output, str)
    parsed = json.loads(output)
    assert parsed["result"] == "mock_answer"
    assert parsed["confidence"] == pytest.approx(0.95)


# ---------------------------------------------------------------------------
# Lines 388-390: generate_direct raises inside _validate_or_fallback fallback
# ---------------------------------------------------------------------------


class _NeverValidModel(BaseModel):
    """Pydantic model that will never match mock backend output."""

    impossible_field_xyz_abc: int
    another_required_field: str


@pytest.mark.asyncio
async def test_validate_or_fallback_generate_direct_raises(mock_backend) -> None:
    """
    When validation fails AND generate_direct raises inside fallback,
    _validate_or_fallback must catch the error and return the original output (388-390).
    """
    engine = TTFEngine(backend=mock_backend, ttf_fallback=True)

    with patch.object(engine, "generate_direct", new_callable=AsyncMock) as mock_direct:
        mock_direct.side_effect = RuntimeError("generate_direct blew up")

        thinking, output = await engine.generate(
            prompt="Produce unmatched JSON",
            schema_model=_NeverValidModel,
        )

    assert isinstance(thinking, str)
    assert isinstance(output, str)
    # When generate_direct raises, the original (invalid) JSON output is returned
    assert output is not None
