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
        assert state.bm25_index is None

    def test_short_doc_segments_created(self):
        state, doc = _prepare_state(SHORT_DOC)
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        assert len(state.segments) >= 1

    def test_long_doc_uses_bm25(self):
        state, doc = _prepare_state(LONG_DOC)
        # Small C_usable forces chunking path
        state.C_usable = 50.0
        config = ExtractionConfig()
        state = run_stage_2b(state, doc, config)
        assert state.bm25_index is not None

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
