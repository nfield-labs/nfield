"""Tests for dependency value injection and cascade invalidation.

Covers, deterministically and without network:
- the [Resolved dependency values] prompt block,
- per-leaf injection_cost (cross-leaf deps cost tokens, intra-leaf deps don't),
- Stage 4 gathering of resolved upstream deps from the blackboard,
- cascade_invalidate flagging downstream dependents.
"""

from __future__ import annotations

from nfield.assembly._blackboard import Blackboard, FieldState
from nfield.config import ExtractionConfig
from nfield.extraction._papt import TemplateType
from nfield.extraction._prompt import build_extraction_prompt
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s2c_packing import _injection_cost, run_stage_2c
from nfield.pipeline.s4_extract import _resolved_dependencies
from nfield.pipeline.s5b_recover import run_recovery_pass
from nfield.schema._types import CapacityLeaf, Field, FieldGroup, Segment
from nfield.validation._retry import cascade_invalidate


class _RecoverProvider:
    """Mock provider whose retry call recovers the upstream field ``up``."""

    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/recover"

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        return "up = recovered"


def _field(path: str, *, tau: float = 5.0) -> Field:
    return Field(
        path=path, type="string", constraints={}, parent_path="", schema_node={}
    ).with_tau(tau=tau, var_tau=1.0)


class TestDependencyPromptBlock:
    def test_block_present_and_formatted(self):
        msgs = build_extraction_prompt(
            [_field("tax")],
            "doc",
            TemplateType.STANDARD,
            dependency_values={"total": 100.0, "paid": True, "note": None},
        )
        user = msgs[1]["content"]
        assert "[Resolved dependency values" in user
        assert "total = 100.0" in user
        assert "paid = true" in user  # bool rendered SFEP-style
        assert "note = NULL" in user
        # SFEP contract still intact in the system message.
        assert "OUTPUT FORMAT" in msgs[0]["content"]

    def test_no_block_when_empty(self):
        base = build_extraction_prompt([_field("tax")], "doc", TemplateType.STANDARD)
        with_empty = build_extraction_prompt(
            [_field("tax")], "doc", TemplateType.STANDARD, dependency_values=None
        )
        assert "Resolved dependency values" not in base[1]["content"]
        assert base[1]["content"] == with_empty[1]["content"]


class TestInjectionCost:
    def test_cross_leaf_dependency_costs_tokens(self):
        leaf_fields = [_field("invoice.tax")]
        field_by_path = {"invoice.tax": leaf_fields[0], "invoice.total": _field("invoice.total")}
        dep_dag = {"invoice.tax": {"invoice.total"}}  # total lives outside the leaf
        cost = _injection_cost(leaf_fields, dep_dag, field_by_path, 4.0)
        assert cost > 0

    def test_intra_leaf_dependency_is_free(self):
        a, b = _field("a"), _field("b")
        field_by_path = {"a": a, "b": b}
        dep_dag = {"b": {"a"}}  # both a and b are in the same leaf
        assert _injection_cost([a, b], dep_dag, field_by_path, 4.0) == 0

    def test_no_dependencies_is_free(self):
        a = _field("a")
        assert _injection_cost([a], {}, {"a": a}, 4.0) == 0


class TestInjectionThroughPacking:
    """End-to-end (Stage 2C): a cross-leaf dependency inflates leaf overhead."""

    def _two_leaf_state(self) -> PipelineState:
        # Two heavy fields that cannot share one call (output ceiling), with a
        # cross-leaf dependency b -> a, so b's leaf must inject a's value.
        a = _field("a", tau=1000.0)
        b = _field("b", tau=1000.0)
        state = PipelineState(
            fields=[a, b],
            field_by_path={"a": a, "b": b},
            dep_dag={"b": {"a"}},
            chars_per_token=4.0,
            C_eff=40_000,
            M_O=2_000,
            C_usable=20_000.0,
        )
        state.groups = [
            FieldGroup(parent_path="", fields=[a]),
            FieldGroup(parent_path="", fields=[b]),
        ]
        return state

    def test_injection_increases_total_overhead(self):
        off = run_stage_2c(self._two_leaf_state(), ExtractionConfig(inject_dependencies=False))
        on = run_stage_2c(self._two_leaf_state(), ExtractionConfig(inject_dependencies=True))
        # Same split (2 leaves) but the dependent leaf reserves injection tokens.
        assert len(off.leaves) == len(on.leaves) == 2
        assert sum(leaf.overhead for leaf in on.leaves) > sum(leaf.overhead for leaf in off.leaves)


class TestResolvedDependencies:
    def test_gathers_cross_leaf_filled_deps(self):
        tax = _field("tax")
        state = PipelineState(
            fields=[_field("total"), tax],
            field_by_path={"total": _field("total"), "tax": tax},
            dep_dag={"tax": {"total"}},
            inject_dependencies=True,
        )
        state.blackboard = Blackboard(["total", "tax"])
        state.blackboard.write("total", 100.0)  # upstream already extracted
        leaf = CapacityLeaf(fields=[tax], groups=[], leaf_id=1)
        resolved = _resolved_dependencies(leaf, state)
        assert resolved == {"total": 100.0}

    def test_disabled_returns_none(self):
        tax = _field("tax")
        state = PipelineState(dep_dag={"tax": {"total"}}, inject_dependencies=False)
        state.blackboard = Blackboard(["total", "tax"])
        state.blackboard.write("total", 100.0)
        leaf = CapacityLeaf(fields=[tax], groups=[], leaf_id=1)
        assert _resolved_dependencies(leaf, state) is None


class TestCascadeInvalidate:
    def test_flags_filled_dependents(self):
        bb = Blackboard(["total", "tax", "grand_total"])
        bb.write("total", 100.0)
        bb.write("tax", 9.0)
        bb.write("grand_total", 109.0)
        # tax depends on total; grand_total depends on tax → both cascade.
        dep_dag = {"tax": {"total"}, "grand_total": {"tax"}}
        invalidated = cascade_invalidate(bb, dep_dag, {"total"})
        assert invalidated == ["grand_total", "tax"]
        assert bb.get_state("tax") == FieldState.NEEDS_REVALIDATION
        assert bb.get_state("grand_total") == FieldState.NEEDS_REVALIDATION
        assert bb.get_state("total") == FieldState.FILLED  # the changed field stays

    def test_no_dependents_no_change(self):
        bb = Blackboard(["a", "b"])
        bb.write("a", 1)
        bb.write("b", 2)
        assert cascade_invalidate(bb, {}, {"a"}) == []
        assert bb.get_state("b") == FieldState.FILLED


class TestCascadeThroughRecovery:
    """recovery: re-extracting an upstream field cascades revalidation to dependents."""

    def _state(self) -> PipelineState:
        up, dep = _field("up"), _field("dep")
        seg = Segment(
            text="up is recovered", start=0, end=15, segment_type="unstructured", segment_id=0
        )
        state = PipelineState(
            fields=[up, dep],
            field_by_path={"up": up, "dep": dep},
            dep_dag={"dep": {"up"}},  # dep depends on up
            chars_per_token=4.0,
            C_eff=8192,
            M_O=2048,
            C_usable=4000.0,
        )
        state.groups = [
            FieldGroup(
                parent_path="", fields=[up, dep], matched_segments=[seg], segment_scores=[1.0]
            )
        ]
        state.segments = [seg]
        state.blackboard = Blackboard(["up", "dep"])
        state.blackboard.write("dep", "dval")  # dep already extracted in Stage 4
        # 'up' stays EMPTY → recovery re-extracts and recovers it.
        return state

    async def test_recovered_upstream_invalidates_dependent(self):
        state = self._state()
        cfg = ExtractionConfig(
            max_retry_rounds=1, inject_dependencies=True, cascade_dependency_invalidation=True
        )
        await run_recovery_pass(state, _RecoverProvider(), cfg)
        assert state.blackboard.get_state("up") == FieldState.FILLED
        assert state.blackboard.get_state("dep") == FieldState.NEEDS_REVALIDATION

    async def test_cascade_without_injection_is_noop(self):
        state = self._state()
        cfg = ExtractionConfig(
            max_retry_rounds=1, inject_dependencies=False, cascade_dependency_invalidation=True
        )
        await run_recovery_pass(state, _RecoverProvider(), cfg)
        assert state.blackboard.get_state("up") == FieldState.FILLED
        # dep keeps its independently-extracted value (no injection → not stale).
        assert state.blackboard.get_state("dep") == FieldState.FILLED
