"""Stage 5 grounding label + Stage 6 support metric (non-destructive wiring)."""

from __future__ import annotations

import asyncio

from nfield.assembly._blackboard import Blackboard, FieldState
from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s5_validate import run_stage_5
from nfield.pipeline.s6_assemble import run_stage_6
from nfield.schema._types import CapacityLeaf, Field, Segment
from nfield.validation._grounding import GroundingStatus


class _UnusedProvider:
    """Stage 5 makes no API calls; this satisfies the signature only."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):  # pragma: no cover - never called
        raise AssertionError("Stage 5 must not call the provider")


_EXCERPT = "Acme Corp was founded in 1947. Figures are reported in $."


def _grounding_state(*, ground_values: bool) -> tuple[PipelineState, Blackboard]:
    """Build a settled state: one supported value and one unsupported value."""
    company = Field("company", "string", {}, "", {})
    year = Field("year", "integer", {}, "", {})
    bb = Blackboard(["company", "year"])
    bb.write("company", "Globex Inc")  # NOT in the excerpt - unsupported
    bb.write("year", 1947)  # present in the excerpt - supported
    leaf = CapacityLeaf(fields=[company, year], document_excerpt=_EXCERPT, leaf_id=1)
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [company, year]
    state.field_by_path = {"company": company, "year": year}
    state.leaves = [leaf]
    state.blackboard = bb
    state.ground_values = ground_values
    state.grounding_min_score = 0.5
    return state, bb


def test_grounding_labels_without_dropping() -> None:
    state, bb = _grounding_state(ground_values=True)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))

    # Non-destructive: the unsupported value is labelled, never dropped.
    assert bb.get_state("company") == FieldState.FILLED
    assert bb.get_state("year") == FieldState.FILLED
    assert bb.get_filled() == {"company": "Globex Inc", "year": 1947}
    assert state.grounding_results["company"].status is GroundingStatus.NONE
    assert state.grounding_results["company"].score == 0.0
    assert state.grounding_results["year"].status is GroundingStatus.EXACT
    assert state.grounding_results["year"].score == 1.0


def test_non_verbatim_value_is_kept() -> None:
    # A correct value the document renders differently ("USD" -> "$") must survive.
    unit = Field("unit", "string", {}, "", {})
    bb = Blackboard(["unit"])
    bb.write("unit", "USD")
    leaf = CapacityLeaf(fields=[unit], document_excerpt=_EXCERPT, leaf_id=1)
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.leaves = [leaf]
    state.blackboard = bb
    state.ground_values = True
    state.grounding_min_score = 0.5
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))

    assert bb.get_filled() == {"unit": "USD"}  # kept
    # "$" alias grounds it verbatim, so it is not counted as unsupported.
    assert state.grounding_results["unit"].status is GroundingStatus.EXACT


def test_enum_value_is_schema_derived() -> None:
    # An enum choice comes from the schema, not the prose; it is exempt from search.
    kind = Field("kind", "enum", {}, "", {})
    bb = Blackboard(["kind"])
    bb.write("kind", "business_segment")
    leaf = CapacityLeaf(fields=[kind], document_excerpt=_EXCERPT, leaf_id=1)
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.leaves = [leaf]
    state.blackboard = bb
    state.ground_values = True
    state.grounding_min_score = 0.5
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))

    assert bb.get_filled() == {"kind": "business_segment"}
    assert state.grounding_results["kind"].status is GroundingStatus.SCHEMA_DERIVED
    # Schema-derived values are excluded from the support metric.
    result = run_stage_6(state)
    assert result.metadata.hallucination_rate is None


def test_grounding_disabled_is_do_no_harm() -> None:
    state, bb = _grounding_state(ground_values=False)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig()))

    # Without grounding, nothing is labelled and every value is kept.
    assert bb.get_state("company") == FieldState.FILLED
    assert state.grounding_results == {}


def test_stage6_reports_support_rate() -> None:
    state, _ = _grounding_state(ground_values=True)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig(ground_values=True)))
    result = run_stage_6(state)

    meta = result.metadata
    assert meta.fields_grounded == 1  # year
    assert meta.fields_ungrounded == 1  # company
    assert meta.hallucination_rate == 0.5


def test_stage6_metric_is_none_without_grounding() -> None:
    state, _ = _grounding_state(ground_values=False)
    asyncio.run(run_stage_5(state, _UnusedProvider(), ExtractionConfig()))
    result = run_stage_6(state)

    assert result.metadata.hallucination_rate is None
    assert result.metadata.fields_grounded == 0
    assert result.metadata.fields_ungrounded == 0


def test_stage6_attaches_provenance_when_requested() -> None:
    company = Field("company", "string", {}, "", {})
    year = Field("year", "integer", {}, "", {})
    bb = Blackboard(["company", "year"])
    bb.write("company", "Globex Inc")  # not in the document
    bb.write("year", 1947)  # verbatim in the document
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state.fields = [company, year]
    state.field_by_path = {"company": company, "year": year}
    state.segments = [
        Segment(text=_EXCERPT, start=0, end=len(_EXCERPT), segment_type="unstructured")
    ]
    state.blackboard = bb
    state.include_provenance = True

    result = run_stage_6(state)

    # Only the verbatim value gets a span; offsets index the document exactly.
    assert result.provenance is not None
    assert "company" not in result.provenance
    start, end = result.provenance["year"]
    assert _EXCERPT[start:end] == "1947"


def test_stage6_provenance_absent_by_default() -> None:
    state, _ = _grounding_state(ground_values=False)
    result = run_stage_6(state)
    assert result.provenance is None
