"""Tests for Stage 2C: Capacity Packing."""

from __future__ import annotations

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import (
    compute_execution_order,
    compute_K_min,
    fits,
    run_stage_2c,
    tarjan_scc,
)
from formatshield.schema._types import CapacityLeaf, Field


def _make_field(path: str, tau: float = 5.0, var_tau: float = 0.5) -> Field:
    f = Field(path=path, type="string", constraints={}, parent_path="", schema_node={})
    return f.with_tau(tau=tau, var_tau=var_tau)


def _prepare_state(schema: dict, doc: str = "test doc") -> PipelineState:
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, doc, ExtractionConfig())
    return state


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
}


class TestComputeKMin:
    def test_single_field(self):
        fields = [_make_field("x", tau=10.0)]
        k = compute_K_min(fields, safe_output=100.0)
        assert k == 1

    def test_many_small_fields(self):
        fields = [_make_field(str(i), tau=1.0) for i in range(50)]
        # sum_tau = 50, safe_output = 100 → ceil(50/100) = 1
        k = compute_K_min(fields, safe_output=100.0)
        assert k == 1

    def test_large_single_field_forces_own_leaf(self):
        fields = [_make_field("big", tau=60.0)]
        # tau > 0.5 * safe_output=100, so at least 1 large field
        k = compute_K_min(fields, safe_output=100.0)
        assert k >= 1

    def test_zero_safe_output_returns_field_count(self):
        fields = [_make_field("x"), _make_field("y")]
        k = compute_K_min(fields, safe_output=0.0)
        assert k == len(fields)

    def test_k_min_at_least_1(self):
        k = compute_K_min([], safe_output=100.0)
        assert k >= 1


class TestFits:
    def test_fits_within_budget(self):
        f = _make_field("x", tau=5.0)
        assert fits([f], D_cost=100, overhead=50, C_usable=500.0, output_ceiling=200.0)

    def test_output_constraint_exceeded(self):
        f = _make_field("x", tau=300.0)
        assert not fits([f], D_cost=0, overhead=0, C_usable=10000.0, output_ceiling=200.0)

    def test_context_constraint_exceeded(self):
        f = _make_field("x", tau=5.0)
        assert not fits([f], D_cost=1000, overhead=50, C_usable=500.0, output_ceiling=200.0)

    def test_empty_fields_fits(self):
        assert fits([], D_cost=0, overhead=50, C_usable=500.0, output_ceiling=200.0)

    def test_exact_budget(self):
        # overhead=100, D_cost=100, output=100 → total=300 == C_usable=300
        f = _make_field("x", tau=100.0)
        assert fits([f], D_cost=100, overhead=100, C_usable=300.0, output_ceiling=200.0)


class TestTarjanSCC:
    def test_no_cycles(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": set()}
        sccs = tarjan_scc(graph)
        # Each node is its own SCC (no cycles)
        assert all(len(scc) == 1 for scc in sccs)

    def test_simple_cycle(self):
        graph = {"a": {"b"}, "b": {"a"}}
        sccs = tarjan_scc(graph)
        cycle_sccs = [s for s in sccs if len(s) > 1]
        assert len(cycle_sccs) == 1
        assert set(cycle_sccs[0]) == {"a", "b"}

    def test_three_node_cycle(self):
        graph = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        sccs = tarjan_scc(graph)
        cycle_sccs = [s for s in sccs if len(s) > 1]
        assert len(cycle_sccs) == 1

    def test_disconnected_graph(self):
        graph = {"a": set(), "b": set(), "c": set()}
        sccs = tarjan_scc(graph)
        assert len(sccs) == 3

    def test_empty_graph(self):
        sccs = tarjan_scc({})
        assert sccs == []


class TestComputeExecutionOrder:
    def test_single_leaf_single_round(self):
        leaf = CapacityLeaf(leaf_id=0)
        leaf.fields = [_make_field("x")]
        order = compute_execution_order([leaf], {})
        assert len(order) == 1
        assert order[0] == [leaf]

    def test_no_deps_all_in_one_round(self):
        leaves = [CapacityLeaf(leaf_id=i) for i in range(3)]
        for i, leaf in enumerate(leaves):
            leaf.fields = [_make_field(f"f{i}")]
        order = compute_execution_order(leaves, {})
        assert len(order) == 1
        assert len(order[0]) == 3

    def test_chain_deps_separate_rounds(self):
        # leaf0 has field "a", leaf1 has field "b" that depends on "a"
        leaf0 = CapacityLeaf(leaf_id=0)
        leaf0.fields = [_make_field("a")]
        leaf1 = CapacityLeaf(leaf_id=1)
        leaf1.fields = [_make_field("b")]
        # b depends on a → leaf1 depends on leaf0
        dep_dag = {"b": {"a"}}
        order = compute_execution_order([leaf0, leaf1], dep_dag)
        assert len(order) == 2
        assert leaf0 in order[0]
        assert leaf1 in order[1]

    def test_empty_leaves(self):
        order = compute_execution_order([], {})
        assert order == [[]]

    def test_two_leaf_cycle_condensed_into_one_round(self):
        """Mutually dependent leaves form one SCC → executed in the same round.

        Verifies tarjan_scc is wired into compute_execution_order (review H2):
        a -> b and b -> a is a cycle, so the two leaves must share a round
        rather than being dumped into a defensive fallback.
        """
        leaf0 = CapacityLeaf(leaf_id=0)
        leaf0.fields = [_make_field("a")]
        leaf1 = CapacityLeaf(leaf_id=1)
        leaf1.fields = [_make_field("b")]
        dep_dag = {"a": {"b"}, "b": {"a"}}  # cycle across leaves
        order = compute_execution_order([leaf0, leaf1], dep_dag)
        assert len(order) == 1
        assert leaf0 in order[0]
        assert leaf1 in order[0]

    def test_three_leaf_cycle_single_round(self):
        """A 3-leaf dependency cycle collapses to one round."""
        leaves = [CapacityLeaf(leaf_id=i) for i in range(3)]
        for i, leaf in enumerate(leaves):
            leaf.fields = [_make_field("abc"[i])]
        dep_dag = {"a": {"b"}, "b": {"c"}, "c": {"a"}}
        order = compute_execution_order(leaves, dep_dag)
        assert len(order) == 1
        assert len(order[0]) == 3


class TestRunStage2c:
    def test_leaves_created(self):
        state = _prepare_state(SIMPLE_SCHEMA)
        state = run_stage_2c(state, ExtractionConfig())
        assert len(state.leaves) >= 1

    def test_k_min_set(self):
        state = _prepare_state(SIMPLE_SCHEMA)
        state = run_stage_2c(state, ExtractionConfig())
        assert state.K_min >= 1

    def test_execution_order_covers_all_leaves(self):
        state = _prepare_state(SIMPLE_SCHEMA)
        state = run_stage_2c(state, ExtractionConfig())
        all_in_order = [leaf for round_ in state.execution_order for leaf in round_]
        assert len(all_in_order) == len(state.leaves)

    def test_all_fields_in_leaves(self):
        state = _prepare_state(SIMPLE_SCHEMA)
        state = run_stage_2c(state, ExtractionConfig())
        leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
        schema_paths = {f.path for f in state.fields}
        assert leaf_paths == schema_paths

    def test_leaf_safe_output_is_per_leaf_reservation_not_ceiling(self):
        """safe_output reserves only what the leaf needs, not M_O/half-context.

        Review H1: a model with a huge M_O but small C_usable must still
        produce a small per-leaf reservation (Στ + margin), leaving most of
        C_usable for the document excerpt.
        """
        state = _prepare_state(SIMPLE_SCHEMA)
        state.M_O = 131_072  # huge output ceiling
        state.C_usable = 4096.0
        state = run_stage_2c(state, ExtractionConfig())
        for leaf in state.leaves:
            # Reservation is tiny vs both M_O and C_usable for a 2-field leaf
            assert leaf.safe_output < 500, (
                f"safe_output {leaf.safe_output} is not a per-leaf reservation"
            )
            # B_excerpt keeps the large majority of C_usable for the document
            b_excerpt = state.C_usable - leaf.overhead - leaf.safe_output
            assert b_excerpt > 0.7 * state.C_usable

    def test_k_min_uses_output_ceiling_not_capped_value(self):
        """K_min for a tiny schema under a huge M_O is 1 (volume bound)."""
        state = _prepare_state(SIMPLE_SCHEMA)
        state.M_O = 131_072
        state.C_usable = 4096.0
        state = run_stage_2c(state, ExtractionConfig())
        # sum(tau) for 2 fields << ceiling → K_min == 1
        assert state.K_min == 1


def _synthetic_schema(n: int) -> dict:
    """Build a flat schema of *n* string fields (one parent group)."""
    return {
        "type": "object",
        "properties": {
            f"f{i:03d}": {"type": "string", "description": f"field {i}"} for i in range(n)
        },
    }


class TestAggressiveScale:
    """Hundreds of fields: packing, splitting, and ordering stay correct (no API)."""

    def test_250_fields_pack_split_and_order(self):
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=8192, C_usable=4096.0)
        state = run_stage_1(state, _synthetic_schema(250))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "short document", ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())

        leaf_paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert len(leaf_paths) == 250  # every field placed
        assert len(set(leaf_paths)) == 250  # exactly once
        assert len(state.leaves) > 1  # a flat 250-field group must split
        in_order = [leaf for r in state.execution_order for leaf in r]
        assert len(in_order) == len(state.leaves)
        assert state.K_min >= 1

    def test_400_fields_still_partitions_cleanly(self):
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=8192, C_usable=4096.0)
        state = run_stage_1(state, _synthetic_schema(400))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "short document", ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        leaf_paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert sorted(leaf_paths) == sorted(f.path for f in state.fields)

    def test_500_fields_pack_split_and_order(self):
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=8192, C_usable=4096.0)
        state = run_stage_1(state, _synthetic_schema(500))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "short document", ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        leaf_paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert len(leaf_paths) == 500
        assert len(set(leaf_paths)) == 500
        assert len(state.leaves) > 1
        in_order = [leaf for r in state.execution_order for leaf in r]
        assert len(in_order) == len(state.leaves)

    def test_1000_fields_partition_and_order(self):
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=8192, C_usable=4096.0)
        state = run_stage_1(state, _synthetic_schema(1000))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "short document", ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        leaf_paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert sorted(leaf_paths) == sorted(f.path for f in state.fields)
        in_order = [leaf for r in state.execution_order for leaf in r]
        assert len(in_order) == len(state.leaves)

    def test_tarjan_deep_chain_is_recursion_safe(self):
        """A 5000-node dependency chain — recursion would blow the stack limit.

        The recursive Tarjan would raise RecursionError well before 5000; the
        iterative implementation handles it. This guards huge dependency graphs.
        """
        n = 5000
        graph: dict[str, set[str]] = {str(i): {str(i + 1)} for i in range(n - 1)}
        graph[str(n - 1)] = set()
        sccs = tarjan_scc(graph)
        assert len(sccs) == n  # acyclic chain → n singleton components

    def test_tarjan_deep_cycle_single_component(self):
        """A 3000-node cycle collapses to one SCC without recursion errors."""
        n = 3000
        graph: dict[str, set[str]] = {str(i): {str((i + 1) % n)} for i in range(n)}
        sccs = tarjan_scc(graph)
        big = [c for c in sccs if len(c) > 1]
        assert len(big) == 1
        assert len(big[0]) == n
