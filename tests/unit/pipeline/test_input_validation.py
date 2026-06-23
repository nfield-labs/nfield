"""Input validation: schema preflight wiring, prior-value retry, unknown-line metric."""

from __future__ import annotations

import asyncio

import pytest

from formatshield.assembly._blackboard import Blackboard
from formatshield.config import ExtractionConfig
from formatshield.engine._async import AsyncFormatShield
from formatshield.exceptions import SchemaError
from formatshield.pipeline.s5b_recover import _failure_reason


class _NeverProvider:
    """Provider that must never be called (preflight should reject before any call)."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):  # pragma: no cover
        raise AssertionError("preflight must reject before any provider call")

    async def count_tokens(self, text):  # pragma: no cover
        raise AssertionError("preflight must reject before any provider call")


def _engine(monkeypatch, config: ExtractionConfig) -> AsyncFormatShield:
    monkeypatch.setattr("formatshield.engine._async.from_model", lambda *a, **k: _NeverProvider())
    return AsyncFormatShield("mock/model", config=config)


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
    from formatshield.pipeline._state import PipelineState
    from formatshield.pipeline.s4_extract import _mark_cast_failures
    from formatshield.schema._types import Field

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
    from formatshield.assembly._blackboard import FieldState
    from formatshield.pipeline._state import PipelineState
    from formatshield.pipeline.s4_extract import _mark_cast_failures
    from formatshield.schema._types import Field

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
    from formatshield.pipeline._state import PipelineState
    from formatshield.pipeline.s6_assemble import run_stage_6
    from formatshield.schema._types import Field

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
    from formatshield.pipeline._state import PipelineState
    from formatshield.pipeline.s6_assemble import run_stage_6
    from formatshield.schema._types import Field

    f = Field("name", "string", {}, "", {})
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [f]
    bb = Blackboard(["name"])
    bb.write("name", "Alice")
    state.blackboard = bb

    result = run_stage_6(state)
    assert result.metadata.unknown_output_lines == 0
