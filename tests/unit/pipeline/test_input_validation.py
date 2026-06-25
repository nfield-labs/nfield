"""Input validation: schema preflight wiring, prior-value retry, unknown-line metric."""

from __future__ import annotations

import asyncio

import pytest

from nfield.assembly._blackboard import Blackboard
from nfield.config import ExtractionConfig
from nfield.engine._async import AsyncNField, _require_document_matches_mode
from nfield.exceptions import SchemaError
from nfield.pipeline.s5b_recover import _failure_reason


class _NeverProvider:
    """Provider that must never be called (preflight should reject before any call)."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):  # pragma: no cover
        raise AssertionError("preflight must reject before any provider call")

    async def count_tokens(self, text):  # pragma: no cover
        raise AssertionError("preflight must reject before any provider call")


def _engine(monkeypatch, config: ExtractionConfig) -> AsyncNField:
    monkeypatch.setattr("nfield.engine._async.from_model", lambda *a, **k: _NeverProvider())
    return AsyncNField("mock/model", config=config)


# ---------------------------------------------------------------------------
# Schema preflight wired into the engine (rejects before any API call)
# ---------------------------------------------------------------------------


def test_engine_rejects_contradictory_schema_before_calling(monkeypatch) -> None:
    engine = _engine(monkeypatch, ExtractionConfig())
    bad_schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 100, "maximum": 10}},
    }
    with pytest.raises(SchemaError, match="minimum"):
        asyncio.run(engine.extract("any document", bad_schema))


def test_non_string_document_is_rejected_with_clear_message(monkeypatch) -> None:
    engine = _engine(monkeypatch, ExtractionConfig())
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    for bad in (None, 123, b"bytes", ["text"]):
        with pytest.raises(TypeError, match="document must be text"):
            asyncio.run(engine.extract(bad, schema))  # type: ignore[arg-type]


def test_empty_document_rejected_in_document_mode(monkeypatch) -> None:
    # Document mode needs text; an empty document is a usage error that points to
    # closed_book, not a silent empty result.
    engine = _engine(monkeypatch, ExtractionConfig())
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    with pytest.raises(ValueError, match="no document to extract from"):
        asyncio.run(engine.extract("", schema))


def test_document_rejected_in_closed_book_mode(monkeypatch) -> None:
    # Closed-book ignores the document; passing one is a usage error, not silent.
    engine = _engine(monkeypatch, ExtractionConfig(closed_book=True))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    with pytest.raises(ValueError, match="closed_book=True"):
        asyncio.run(engine.extract("some real document text", schema))


def test_empty_document_allowed_in_closed_book_mode(monkeypatch) -> None:
    # An empty document is the closed-book signal: it must pass the gates and reach the
    # (never-called) provider stage, not raise at the boundary.
    engine = _engine(monkeypatch, ExtractionConfig(closed_book=True))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    with pytest.raises(AssertionError):  # the stub provider asserts it is reached
        asyncio.run(engine.extract("", schema))


def test_whitespace_only_document_counts_as_empty(monkeypatch) -> None:
    # A whitespace-only document carries no evidence: it is rejected in document mode and
    # accepted as the no-document signal in closed-book mode.
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    doc_engine = _engine(monkeypatch, ExtractionConfig())
    with pytest.raises(ValueError, match="no document to extract from"):
        asyncio.run(doc_engine.extract("   \n\t ", schema))
    cb_engine = _engine(monkeypatch, ExtractionConfig(closed_book=True))
    with pytest.raises(AssertionError):  # passes the gates, reaches the stub provider
        asyncio.run(cb_engine.extract("   \n\t ", schema))


@pytest.mark.parametrize("closed_book", [False, True])
def test_non_string_document_is_type_error_in_both_modes(monkeypatch, closed_book) -> None:
    # The type gate runs before the mode gate (the mode gate calls document.strip(), valid
    # only on a str), so a non-string is a TypeError regardless of closed_book.
    engine = _engine(monkeypatch, ExtractionConfig(closed_book=closed_book))
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    for bad in (None, 123, b"bytes", ["text"]):
        with pytest.raises(TypeError, match="document must be text"):
            asyncio.run(engine.extract(bad, schema))  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# The mode gate in isolation: every (document, closed_book) branch, no engine
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("document", "closed_book"),
    [
        ("", True),  # no document is the closed-book signal
        ("   \n\t ", True),  # whitespace collapses to no document
        ("real text", False),  # text is what document mode needs
    ],
)
def test_mode_gate_accepts_valid_pairings(document, closed_book) -> None:
    # A matching pairing passes silently (returns None, raises nothing).
    assert _require_document_matches_mode(document, closed_book) is None


def test_mode_gate_rejects_document_in_closed_book() -> None:
    for document in ("real text", "x", "  word  "):
        with pytest.raises(ValueError, match="closed_book=True"):
            _require_document_matches_mode(document, closed_book=True)


def test_mode_gate_rejects_empty_in_document_mode() -> None:
    for document in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="no document to extract from"):
            _require_document_matches_mode(document, closed_book=False)


def test_engine_preflight_can_be_disabled(monkeypatch) -> None:
    # With validate_schema=False the preflight is skipped; the contradictory schema then
    # reaches the (never-called) provider stage, proving the gate is what rejects.
    engine = _engine(monkeypatch, ExtractionConfig(validate_schema=False))
    bad_schema = {
        "type": "object",
        "properties": {"age": {"type": "integer", "minimum": 100, "maximum": 10}},
    }
    # No SchemaError now — it fails later (the stub provider asserts), not at preflight.
    with pytest.raises(AssertionError):
        asyncio.run(engine.extract("any document", bad_schema))


# ---------------------------------------------------------------------------
# Carry the prior failed value into the retry reason
# ---------------------------------------------------------------------------


def test_failure_reason_includes_prior_value() -> None:
    bb = Blackboard(["age"])
    bb.write("age", "thirty")  # the value the model returned
    bb.mark_failed("age", "expected integer, got str 'thirty'")
    reason = _failure_reason(bb, "age")
    assert "thirty" in reason
    assert "failed validation" in reason


def test_failure_reason_without_value_is_plain() -> None:
    bb = Blackboard(["age"])
    bb.mark_failed("age", "some error")  # no value ever written
    reason = _failure_reason(bb, "age")
    assert "previously returned" not in reason
    assert "failed validation" in reason


def test_missing_field_reason_unchanged() -> None:
    bb = Blackboard(["age"])  # EMPTY, never extracted
    reason = _failure_reason(bb, "age")
    assert "did not find this field" in reason


def test_transient_reason_is_neutral_no_prior_value() -> None:
    # A call-failed field's request never reached the model, so the reason must not claim
    # a prior output or a validation failure — it asks for a fresh extraction.
    bb = Blackboard(["age"])
    bb.mark_failed("age", "provider error: 429 rate limit", transient=True)
    reason = _failure_reason(bb, "age", transient=True)
    assert "did not complete" in reason
    assert "previously returned" not in reason
    assert "failed validation" not in reason


# ---------------------------------------------------------------------------
# An uncastable value is surfaced into the retry reason (not silently dropped)
# ---------------------------------------------------------------------------


def test_cast_failure_is_marked_failed_with_raw_value() -> None:
    # "age = abc" cannot be cast to integer; parse_sfep drops it. Stage 4 must instead
    # mark the field FAILED carrying the raw text, so recovery can show it to the model.
    from nfield.pipeline._state import PipelineState
    from nfield.pipeline.s4_extract import _mark_cast_failures
    from nfield.schema._types import Field

    f = Field("age", "integer", {}, "", {})
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [f]
    state.field_by_path = {"age": f}
    state.blackboard = Blackboard(["age"])
    state.blackboard.mark_pending("age")

    _mark_cast_failures("age = abc", [f], extracted={}, state=state)

    reason = _failure_reason(state.blackboard, "age")
    assert "abc" in reason
    assert "integer" in reason


def test_cast_failure_not_marked_when_value_also_parses() -> None:
    # If the same field also produced a castable value (it is in `extracted`), the good
    # value must win — the field is not clobbered to FAILED.
    from nfield.assembly._blackboard import FieldState
    from nfield.pipeline._state import PipelineState
    from nfield.pipeline.s4_extract import _mark_cast_failures
    from nfield.schema._types import Field

    f = Field("age", "integer", {}, "", {})
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [f]
    state.field_by_path = {"age": f}
    state.blackboard = Blackboard(["age"])
    state.blackboard.write("age", 30)

    _mark_cast_failures("age = abc", [f], extracted={"age": 30}, state=state)

    assert state.blackboard.get_state("age") == FieldState.FILLED
    assert state.blackboard.get_value("age") == 30


# ---------------------------------------------------------------------------
# Unknown-output-line metric surfaces in Metadata via Stage 6
# ---------------------------------------------------------------------------


def test_unknown_lines_flow_to_metadata() -> None:
    from nfield.pipeline._state import PipelineState
    from nfield.pipeline.s6_assemble import run_stage_6
    from nfield.schema._types import Field

    f = Field("name", "string", {}, "", {})
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [f]
    bb = Blackboard(["name"])
    bb.write("name", "Alice")
    state.blackboard = bb
    state.unknown_lines = 3  # as if Stage 4 saw 3 out-of-schema lines

    result = run_stage_6(state)
    assert result.metadata.unknown_output_lines == 3


def test_unknown_lines_default_zero() -> None:
    from nfield.pipeline._state import PipelineState
    from nfield.pipeline.s6_assemble import run_stage_6
    from nfield.schema._types import Field

    f = Field("name", "string", {}, "", {})
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [f]
    bb = Blackboard(["name"])
    bb.write("name", "Alice")
    state.blackboard = bb

    result = run_stage_6(state)
    assert result.metadata.unknown_output_lines == 0
