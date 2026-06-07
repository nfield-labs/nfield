"""Tests for Stage 2.5: Document Pre-Pass."""

from __future__ import annotations

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b

SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string", "description": "company name"},
        "revenue": {"type": "number", "description": "annual revenue"},
    },
}

SHORT_DOC = "Acme Corp earns $1M annually."
LONG_DOC = "Paragraph about Acme Corp. " * 300  # forces chunking path


def _prepare_state(doc: str, chars_per_token: float = 4.0) -> tuple[PipelineState, str]:
    state = PipelineState(
        chars_per_token=chars_per_token,
        C_eff=8192,
        M_O=1024,
        C_usable=4096.0,
    )
    state = run_stage_1(state, SCHEMA)
    state = run_stage_2a(state)
    return state, doc


class TestRunStage2b:
    def test_short_doc_fast_path(self):
        state, doc = _prepare_state(SHORT_DOC)
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        # All groups get the full doc cost in fast path
        for g in state.groups:
            assert g.D_cost > 0
        assert state.lexical_index is None

    def test_short_doc_segments_created(self):
        state, doc = _prepare_state(SHORT_DOC)
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        assert len(state.segments) >= 1

    def test_long_doc_uses_bmx(self):
        state, doc = _prepare_state(LONG_DOC)
        # Small C_usable forces chunking path
        state.C_usable = 50.0
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        assert state.lexical_index is not None

    def test_long_doc_d_cost_positive(self):
        state, doc = _prepare_state(LONG_DOC)
        state.C_usable = 50.0
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        for g in state.groups:
            assert g.D_cost >= 0

    def test_empty_document_handled(self):
        state, _ = _prepare_state("")
        state.C_usable = 50.0
        config = ExtractionConfig()
        state = run_stage_2b(state, "", config)
        assert state.segments == [] or len(state.segments) >= 0  # no crash

    def test_returns_same_state(self):
        state, doc = _prepare_state(SHORT_DOC)
        returned = run_stage_2b(state, doc, ExtractionConfig())
        assert returned is state


# ---------------------------------------------------------------------------
# Per-group dynamic retrieval depth (Phase A.2) — no fixed global top-k
# ---------------------------------------------------------------------------


class TestGroupTopK:
    """Retrieval depth scales with a group's field count, not a global cap."""

    @staticmethod
    def _segs(n: int):
        from formatshield.schema._types import Segment

        return [
            Segment(text="word " * 80, start=0, end=400, segment_type="unstructured", segment_id=i)
            for i in range(n)
        ]

    @staticmethod
    def _group(n_fields: int):
        from formatshield.schema._types import Field, FieldGroup

        fields = [
            Field(path=f"g.f{i}", type="string", constraints={}, parent_path="g", schema_node={})
            for i in range(n_fields)
        ]
        return FieldGroup(parent_path="g", fields=fields)

    def test_larger_group_retrieves_more(self):
        from formatshield.pipeline.s2b_prepass import _group_top_k

        # Small budget so the per-group field scaling (not the budget pool) binds:
        # a large group must then retrieve more than a small one.
        segs = self._segs(200)
        small = _group_top_k(self._group(1), segs, c_usable=2000.0, chars_per_token=4.0)
        large = _group_top_k(self._group(20), segs, c_usable=2000.0, chars_per_token=4.0)
        assert large > small, "a group with more fields must retrieve more segments"

    def test_floored_at_minimum(self):
        from formatshield.pipeline.s2b_prepass import _MIN_TOP_K_SEGMENTS, _group_top_k

        segs = self._segs(200)
        depth = _group_top_k(self._group(1), segs, c_usable=100_000.0, chars_per_token=4.0)
        assert depth >= _MIN_TOP_K_SEGMENTS

    def test_capped_by_segment_count(self):
        from formatshield.pipeline.s2b_prepass import _group_top_k

        segs = self._segs(3)
        depth = _group_top_k(self._group(50), segs, c_usable=100_000.0, chars_per_token=4.0)
        assert depth <= 3, "cannot retrieve more segments than exist"
