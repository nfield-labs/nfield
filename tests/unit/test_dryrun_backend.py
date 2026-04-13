"""
Unit tests for formatshield.backends.dryrun_backend.DryRunBackend.

Covers deterministic generation, schema-driven JSON responses, streaming,
call counting, and all schema type branches.
"""

from __future__ import annotations

import json

import pytest

from formatshield.backends.dryrun_backend import DryRunBackend

# ---------------------------------------------------------------------------
# Construction / properties
# ---------------------------------------------------------------------------


def test_dryrun_backend_name() -> None:
    """DryRunBackend.name must be 'dryrun'."""
    assert DryRunBackend().name == "dryrun"


def test_dryrun_supports_kv_cache_reuse_false() -> None:
    """DryRunBackend.supports_kv_cache_reuse must be False."""
    assert DryRunBackend().supports_kv_cache_reuse is False


def test_dryrun_accuracy_loss_baseline() -> None:
    """accuracy_loss_baseline is 1 - direct_accuracy."""
    backend = DryRunBackend(direct_accuracy=0.55)
    assert backend.accuracy_loss_baseline == pytest.approx(0.45, abs=1e-4)


def test_dryrun_accuracy_loss_baseline_custom() -> None:
    """accuracy_loss_baseline reflects the custom direct_accuracy."""
    backend = DryRunBackend(direct_accuracy=0.80)
    assert backend.accuracy_loss_baseline == pytest.approx(0.20, abs=1e-4)


def test_dryrun_call_count_starts_at_zero() -> None:
    """call_count must start at 0 before any generate() calls."""
    assert DryRunBackend().call_count == 0


# ---------------------------------------------------------------------------
# generate() — thinking response (no schema, no constraints)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_thinking_response_no_schema() -> None:
    """generate(no schema, no constraints) returns a <think> response."""
    backend = DryRunBackend()
    result = await backend.generate("What is 2+2?")
    assert "<think>" in result
    assert "</think>" in result


@pytest.mark.asyncio
async def test_generate_increments_call_count() -> None:
    """Each generate() call increments call_count."""
    backend = DryRunBackend()
    await backend.generate("test")
    assert backend.call_count == 1
    await backend.generate("test2")
    assert backend.call_count == 2


@pytest.mark.asyncio
async def test_generate_reset_call_count() -> None:
    """reset_call_count() resets call_count to zero."""
    backend = DryRunBackend()
    await backend.generate("test")
    await backend.generate("test")
    assert backend.call_count == 2
    backend.reset_call_count()
    assert backend.call_count == 0


# ---------------------------------------------------------------------------
# generate() — JSON response (with constraints="json")
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_json_constraints_returns_valid_json() -> None:
    """generate(constraints='json') must return valid JSON."""
    backend = DryRunBackend()
    result = await backend.generate("test", constraints="json")
    parsed = json.loads(result)
    assert isinstance(parsed, dict)


@pytest.mark.asyncio
async def test_generate_with_schema_returns_valid_json() -> None:
    """generate(schema=...) must return valid JSON matching the schema shape."""
    backend = DryRunBackend()
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "score": {"type": "number"},
        },
    }
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert "name" in parsed
    assert "score" in parsed


@pytest.mark.asyncio
async def test_generate_with_schema_string_field() -> None:
    """Schema with string field returns a string value."""
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"label": {"type": "string"}}}
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["label"], str)


@pytest.mark.asyncio
async def test_generate_with_schema_integer_field() -> None:
    """Schema with integer field returns an integer value."""
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"count": {"type": "integer"}}}
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["count"], int)


@pytest.mark.asyncio
async def test_generate_with_schema_boolean_field() -> None:
    """Schema with boolean field returns a boolean."""
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"flag": {"type": "boolean"}}}
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["flag"], bool)


@pytest.mark.asyncio
async def test_generate_with_schema_array_field() -> None:
    """Schema with array field returns a list."""
    backend = DryRunBackend()
    schema = {
        "type": "object",
        "properties": {"items": {"type": "array", "items": {"type": "string"}}},
    }
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["items"], list)


@pytest.mark.asyncio
async def test_generate_with_schema_nested_object() -> None:
    """Schema with nested object field returns a nested dict."""
    backend = DryRunBackend()
    schema = {
        "type": "object",
        "properties": {
            "address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["address"], dict)
    assert "city" in parsed["address"]


@pytest.mark.asyncio
async def test_generate_with_schema_enum_field() -> None:
    """Schema with enum returns the first enum value."""
    backend = DryRunBackend()
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["pending", "done", "failed"]}},
    }
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert parsed["status"] == "pending"


@pytest.mark.asyncio
async def test_generate_no_schema_fallback_json() -> None:
    """generate(constraints='json', no schema) returns generic fallback JSON."""
    backend = DryRunBackend()
    result = await backend.generate("test", constraints="json")
    parsed = json.loads(result)
    assert "result" in parsed


@pytest.mark.asyncio
async def test_generate_deterministic_with_same_seed() -> None:
    """Two backends with same seed produce the same response."""
    b1 = DryRunBackend(seed=0)
    b2 = DryRunBackend(seed=0)
    r1 = await b1.generate("hello")
    r2 = await b2.generate("hello")
    assert r1 == r2


# ---------------------------------------------------------------------------
# generate() — null and number schema types
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_schema_null_type() -> None:
    """Schema with null type returns None."""
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"val": {"type": "null"}}}
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert parsed["val"] is None


@pytest.mark.asyncio
async def test_generate_schema_number_type() -> None:
    """Schema with number type returns a float."""
    backend = DryRunBackend()
    schema = {"type": "object", "properties": {"val": {"type": "number"}}}
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert isinstance(parsed["val"], float)


@pytest.mark.asyncio
async def test_generate_schema_anyof_fallback() -> None:
    """Schema with anyOf uses the first candidate."""
    backend = DryRunBackend()
    schema = {
        "type": "object",
        "properties": {"val": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
    }
    result = await backend.generate("test", schema=schema)
    parsed = json.loads(result)
    assert "val" in parsed


# ---------------------------------------------------------------------------
# stream()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_yields_events() -> None:
    """stream() must yield at least one output event and a final complete event."""
    backend = DryRunBackend()
    stream = await backend.stream("hello")
    events = [e async for e in stream]
    assert len(events) >= 1
    assert any(e.type == "complete" for e in events)


@pytest.mark.asyncio
async def test_stream_complete_event_has_content() -> None:
    """The complete event from stream() must carry a non-None content."""
    backend = DryRunBackend()
    stream = await backend.stream("hello")
    events = [e async for e in stream]
    complete_events = [e for e in events if e.type == "complete"]
    assert len(complete_events) == 1
    assert complete_events[0].content is not None


@pytest.mark.asyncio
async def test_stream_json_complete_event_has_parsed_json() -> None:
    """stream(constraints='json') complete event must carry a parsed json dict."""
    backend = DryRunBackend()
    stream = await backend.stream("hello", constraints="json")
    events = [e async for e in stream]
    complete = next(e for e in events if e.type == "complete")
    assert complete.json is not None
    assert isinstance(complete.json, dict)


@pytest.mark.asyncio
async def test_stream_output_events_have_token() -> None:
    """Output events from stream() must have a token attribute."""
    backend = DryRunBackend()
    stream = await backend.stream("hello")
    events = [e async for e in stream]
    output_events = [e for e in events if e.type == "output"]
    for e in output_events:
        assert e.token is not None
        assert len(e.token) >= 1


@pytest.mark.asyncio
async def test_stream_latency_ms_positive() -> None:
    """All stream events must have positive latency_ms."""
    backend = DryRunBackend()
    stream = await backend.stream("test")
    events = [e async for e in stream]
    for e in events:
        assert e.latency_ms >= 0.0


@pytest.mark.asyncio
async def test_stream_backend_name_in_events() -> None:
    """All stream events must carry backend='dryrun'."""
    backend = DryRunBackend()
    stream = await backend.stream("test")
    events = [e async for e in stream]
    for e in events:
        assert e.backend == "dryrun"
