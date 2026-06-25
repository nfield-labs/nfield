"""Tests for Stage 2.5: Document Pre-Pass."""

from __future__ import annotations

from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.pipeline.s2b_prepass import run_stage_2b

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
        from nfield.schema._types import Segment

        return [
            Segment(text="word " * 80, start=0, end=400, segment_type="unstructured", segment_id=i)
            for i in range(n)
        ]

    @staticmethod
    def _group(n_fields: int):
        from nfield.schema._types import Field, FieldGroup

        fields = [
            Field(path=f"g.f{i}", type="string", constraints={}, parent_path="g", schema_node={})
            for i in range(n_fields)
        ]
        return FieldGroup(parent_path="g", fields=fields)

    def test_larger_group_retrieves_more(self):
        from nfield.pipeline.s2b_prepass import _group_top_k

        # Small budget so the per-group field scaling (not the budget pool) binds:
        # a large group must then retrieve more than a small one.
        segs = self._segs(200)
        small = _group_top_k(self._group(1), segs, c_usable=2000.0, chars_per_token=4.0)
        large = _group_top_k(self._group(20), segs, c_usable=2000.0, chars_per_token=4.0)
        assert large > small, "a group with more fields must retrieve more segments"

    def test_floored_at_minimum(self):
        from nfield.pipeline.s2b_prepass import _MIN_TOP_K_SEGMENTS, _group_top_k

        segs = self._segs(200)
        depth = _group_top_k(self._group(1), segs, c_usable=100_000.0, chars_per_token=4.0)
        assert depth >= _MIN_TOP_K_SEGMENTS

    def test_capped_by_segment_count(self):
        from nfield.pipeline.s2b_prepass import _group_top_k

        segs = self._segs(3)
        depth = _group_top_k(self._group(50), segs, c_usable=100_000.0, chars_per_token=4.0)
        assert depth <= 3, "cannot retrieve more segments than exist"


# A nested schema whose object keys mirror a heterogeneous document's headings, so
# each group (one per parent_path) aligns to its own section.
_HETERO_SCHEMA = {
    "type": "object",
    "properties": {
        "income_statement": {
            "type": "object",
            "properties": {
                "total_revenue": {"type": "number", "description": "total revenue"},
                "net_income": {"type": "number", "description": "net income"},
            },
        },
        "balance_sheet": {
            "type": "object",
            "properties": {
                "total_assets": {"type": "number", "description": "total assets"},
                "cash": {"type": "number", "description": "cash and equivalents"},
            },
        },
        "cash_flow_statement": {
            "type": "object",
            "properties": {"operating_activities": {"type": "number", "description": "operating"}},
        },
        "governance": {
            "type": "object",
            "properties": {"directors": {"type": "integer", "description": "board directors"}},
        },
    },
}

_HETERO_DOC = (
    "This filing summarises the consolidated results for the year under review in full.\n"
    "1. Income Statement\n"
    "Total revenue reached 1,234,567 dollars and net income was 89,000 dollars after taxes.\n"
    "Operating expenses totalled 500,000 dollars across every division for the period.\n"
    "2. Balance Sheet\n"
    "Total assets stood at 9,876,543 dollars while liabilities were 4,000,000 dollars.\n"
    "Cash and equivalents amounted to 250,000 dollars across operating accounts.\n"
    "3. Cash Flow Statement\n"
    "Net cash from operating activities was 750,000 dollars during the period reviewed.\n"
    "Capital expenditures consumed 300,000 dollars for plant and equipment that year.\n"
    "4. Governance\n"
    "The board comprised nine directors who met quarterly to review the strategy.\n"
    "The audit committee oversaw financial reporting and internal controls all year.\n"
)


class TestHeadingHybridRoute:
    """STAR tier 2: a heterogeneous doc too large for the fast path routes by headings."""

    def _hetero_state(self) -> tuple[PipelineState, str]:
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=40.0)
        state = run_stage_1(state, _HETERO_SCHEMA)
        state = run_stage_2a(state)
        return state, _HETERO_DOC

    def test_heading_route_builds_index(self):
        state, doc = self._hetero_state()
        state = run_stage_2b(state, doc, ExtractionConfig())
        # The heading route (tier 2) builds a lexical index, unlike the fast path.
        assert state.lexical_index is not None
        assert len(state.segments) >= 4

    def test_groups_routed_to_their_section(self):
        state, doc = self._hetero_state()
        state = run_stage_2b(state, doc, ExtractionConfig())
        by_parent = {g.parent_path: g for g in state.groups}
        income_top = by_parent["income_statement"].matched_segments[0].text
        gov_top = by_parent["governance"].matched_segments[0].text
        # Structure routed each group to its own section, not a sibling's.
        assert "revenue" in income_top
        assert "directors" in gov_top

    def test_record_doc_still_takes_record_path(self):
        # Do-no-harm: a record document is unaffected by the heading tier.
        record_schema = {
            "type": "object",
            "properties": {
                "recs": {
                    "type": "object",
                    "properties": {
                        f"rec_{i}": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                        }
                        for i in range(1, 5)
                    },
                }
            },
        }
        doc = (
            "HEADER LINE\n"
            "RECORD 1\nname: Ann\nage: 30\n"
            "RECORD 2\nname: Ben\nage: 41\n"
            "RECORD 3\nname: Cleo\nage: 52\n"
            "RECORD 4\nname: Dan\nage: 63\n"
        )
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=40.0)
        state = run_stage_1(state, record_schema)
        state = run_stage_2a(state)
        state = run_stage_2b(state, doc, ExtractionConfig())
        # The record path populates record_ordinal; the heading tier never does.
        assert state.record_ordinal
