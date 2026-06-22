"""Stage 5 grounding gate + Stage 6 hallucination metric (wiring tests)."""

from __future__ import annotations

import asyncio

from formatshield.assembly._blackboard import Blackboard, FieldState
from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s5_validate import run_stage_5
from formatshield.pipeline.s6_assemble import run_stage_6
from formatshield.schema._types import CapacityLeaf, Field


class _UnusedProvider:
    """Stage 5 makes no API calls; this satisfies the signature only."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):  # pragma: no cover - never called
        raise AssertionError("Stage 5 must not call the provider")

    async def count_tokens(self, text):  # pragma: no cover - never called
        raise AssertionError("Stage 5 must not call the provider")


_EXCERPT = "Acme Corp was founded in 1947."


def _grounding_state(*, ground_values: bool) -> tuple[PipelineState, Blackboard]:
    """Build a settled state: one grounded value and one hallucinated value."""
    company = Field("company", "string", {}, "", {})
    year = Field("year", "integer", {}, "", {})
    bb = Blackboard(["company", "year"])
    bb.write("company", "Globex Inc")  # NOT in the excerpt — a hallucination
    bb.write("year", 1947)  # present in the excerpt — grounded
    leaf = CapacityLeaf(fields=[company, year], document_excerpt=_EXCERPT, leaf_id=1)
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [company, year]
    state.field_by_path = {"company": company, "year": year}
    state.leaves = [leaf]
    state.blackboard = bb
    state.ground_values = ground_values
    state.grounding_min_score = 0.5
    return state, bb


def test_grounding_gate_marks_unsupported_value_failed() -> None:
    state, bb = _grounding_state(ground_values=True)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))

    # The hallucinated value is rejected; the grounded one survives.
    assert bb.get_state("company") == FieldState.FAILED
    assert bb.get_state("year") == FieldState.FILLED
    assert bb.get_filled() == {"year": 1947}
    # Scores recorded for both groundable fields.
    assert state.grounding_scores["company"] == 0.0
    assert state.grounding_scores["year"] == 1.0


def test_grounding_disabled_is_do_no_harm() -> None:
    state, bb = _grounding_state(ground_values=False)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig()))

    # Without grounding, the unsupported value is accepted (type-valid) and no scores.
    assert bb.get_state("company") == FieldState.FILLED
    assert state.grounding_scores == {}


def test_stage6_reports_hallucination_rate() -> None:
    state, _ = _grounding_state(ground_values=True)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))
    result = run_stage_6(state)

    meta = result.metadata
    assert meta.fields_grounded == 1
    assert meta.fields_ungrounded == 1
    assert meta.hallucination_rate == 0.5


def test_stage6_metric_is_none_without_grounding() -> None:
    state, _ = _grounding_state(ground_values=False)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig()))
    result = run_stage_6(state)

    assert result.metadata.hallucination_rate is None
    assert result.metadata.fields_grounded == 0
    assert result.metadata.fields_ungrounded == 0
