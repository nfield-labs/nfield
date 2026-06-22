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
