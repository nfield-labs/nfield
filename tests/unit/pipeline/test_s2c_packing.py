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


_CPT = 4.0  # chars-per-token for these unit tests (short paths → ~1 path token)


class TestComputeKMin:
    def test_single_field(self):
        fields = [_make_field("x", tau=10.0)]
        # output(f) = path(1) + tau(10) + line_overhead(8) = 19; 19/100 → 1
        k = compute_K_min(fields, safe_output=100.0, chars_per_token=_CPT)
        assert k == 1

    def test_many_small_fields(self):
        fields = [_make_field(str(i), tau=1.0) for i in range(50)]
        # each line ~10 output tokens (path + value + line overhead); 50*10=500;
        # ceil(500/100) = 5 — the echoed path + line overhead are real output cost.
        k = compute_K_min(fields, safe_output=100.0, chars_per_token=_CPT)
        assert k == 5

    def test_large_single_field_forces_own_leaf(self):
        fields = [_make_field("big", tau=60.0)]
        # output > 0.5 * safe_output=100, so at least 1 large field
        k = compute_K_min(fields, safe_output=100.0, chars_per_token=_CPT)
        assert k >= 1

    def test_zero_safe_output_returns_field_count(self):
        fields = [_make_field("x"), _make_field("y")]
        k = compute_K_min(fields, safe_output=0.0, chars_per_token=_CPT)
        assert k == len(fields)

    def test_k_min_at_least_1(self):
        k = compute_K_min([], safe_output=100.0, chars_per_token=_CPT)
        assert k >= 1


class TestFits:
    def test_fits_within_budget(self):
        f = _make_field("x", tau=5.0)
        assert fits(
            [f],
            D_cost=100,
            overhead=50,
            C_usable=500.0,
            output_ceiling=200.0,
            chars_per_token=_CPT,
        )

    def test_output_constraint_exceeded(self):
        f = _make_field("x", tau=300.0)
        assert not fits(
            [f], D_cost=0, overhead=0, C_usable=10000.0, output_ceiling=200.0, chars_per_token=_CPT
        )

    def test_context_constraint_exceeded(self):
        # The document is shared + trimmed, so a large D_cost alone does NOT make
        # a leaf infeasible. Context fails only when overhead + output leave no
        # room for even a minimal excerpt. Here overhead (480) + output (~14)
        # leaves < MIN_EXCERPT (256) of the 500 budget → does not fit.
        f = _make_field("x", tau=5.0)
        assert not fits(
            [f],
            D_cost=1000,
            overhead=480,
            C_usable=500.0,
            output_ceiling=200.0,
            chars_per_token=_CPT,
        )

    def test_large_doc_pool_does_not_block_packing(self):
        # A huge retrieval pool (D_cost) must NOT block a leaf — Stage 3 trims it.
        f = _make_field("x", tau=5.0)
        assert fits(
            [f],
            D_cost=100_000,  # enormous pool
            overhead=50,
            C_usable=5000.0,
            output_ceiling=200.0,
            chars_per_token=_CPT,
        )

    def test_empty_fields_fits(self):
        assert fits(
            [], D_cost=0, overhead=50, C_usable=500.0, output_ceiling=200.0, chars_per_token=_CPT
        )

    def test_exact_budget(self):
        # output(f) = path(1) + tau(100) + line_overhead(8) = 109;
        # doc_needed = min(D_cost=100, MIN_EXCERPT=256) = 100;
        # overhead(100) + 100 + 109 = 309 == C_usable → fits (boundary).
        f = _make_field("x", tau=100.0)
        assert fits(
            [f],
            D_cost=100,
            overhead=100,
            C_usable=309.0,
            output_ceiling=200.0,
            chars_per_token=_CPT,
        )


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

    def test_output_is_decoupled_from_input_budget(self):
        """Output generates into the window headroom, not the input budget.

        Decoupling: the excerpt keeps the FULL input budget (C_usable - overhead,
        output not subtracted), and the call's output cap (safe_output) is bounded
        by the headroom (C_eff - C_usable), never by the input budget — so a
        verbose answer never steals excerpt space, yet prompt + output still fits
        the full window. safe_output is right-sized to the leaf's predicted output,
        so a tiny schema reserves far less than the headroom (rate-limit friendly).
        """
        state = _prepare_state(SIMPLE_SCHEMA)
        state.C_eff = 8192
        state.C_usable = 4096.0  # the 50% input ceiling
        state.M_O = 131_072  # huge model output limit — headroom must bind it
        state = run_stage_2c(state, ExtractionConfig())
        headroom = state.C_eff - state.C_usable
        for leaf in state.leaves:
            # Right-sized, headroom-bounded output reservation (> 0, never the
            # input budget): a tiny leaf reserves only what it will emit.
            assert 0 < leaf.safe_output <= headroom
            # The excerpt keeps the full input budget; output is not subtracted.
            b_excerpt = state.C_usable - leaf.overhead
            assert b_excerpt > 0.7 * state.C_usable
            # Decoupling invariant: prompt + output never exceeds the full window.
            assert leaf.overhead + b_excerpt + leaf.safe_output <= state.C_eff

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


def _record_schema(n_records: int, fields_per_record: int) -> dict:
    """Schema with ``n_records`` identical-shape siblings under ``recs``."""
    rec_shape = {
        "type": "object",
        "properties": {f"f{j:02d}": {"type": "string"} for j in range(fields_per_record)},
    }
    return {
        "type": "object",
        "properties": {
            "recs": {
                "type": "object",
                "properties": {f"rec_{i + 1}": rec_shape for i in range(n_records)},
            }
        },
    }


def _record_document(n_records: int, fields_per_record: int) -> str:
    blocks = ["HEADER LINE"]
    for i in range(1, n_records + 1):
        lines = [f"RECORD {i}"]
        lines += [f"f{j:02d}: value-{i}-{j}" for j in range(fields_per_record)]
        blocks.append("\n".join(lines))
    return "\n".join(blocks) + "\n"


class TestRecordAwarePacking:
    """When the document has a record structure, packing is record-local."""

    def _state(self, n_records: int, fpr: int):
        doc = _record_document(n_records, fpr)
        state = PipelineState(chars_per_token=4.0, C_eff=131_000, M_O=10_000, C_usable=65_500.0)
        state = run_stage_1(state, _record_schema(n_records, fpr))
        state = run_stage_2a(state)
        state = run_stage_2b(state, doc, ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        return state

    def test_record_structure_detected(self):
        state = self._state(6, 10)
        assert state.record_ordinal  # populated → record-aware path taken
        assert len(state.record_block_tokens) == 6

    def test_every_field_placed_exactly_once(self):
        state = self._state(6, 10)
        paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert len(paths) == len(set(paths)) == 6 * 10

    def test_leaves_are_record_contiguous(self):
        # Next-Fit by record order → each leaf's records form a CONTIGUOUS run
        # (consecutive ordinals), never a scatter of distant records. The run's
        # length depends on record size; the no-scatter property is what matters.
        state = self._state(6, 10)
        ro = state.record_ordinal
        for leaf in state.leaves:
            ords = sorted({ro[f.path] for f in leaf.fields if f.path in ro})
            if ords:
                assert ords == list(range(ords[0], ords[-1] + 1))  # contiguous, no gaps

    def test_k_stays_at_reliability_floor(self):
        state = self._state(6, 10)
        # 60 fields, cap 50 → 2 leaves minimum; record-local packing stays near it.
        assert state.K_min <= len(state.leaves) <= state.K_min + 2

    def test_record_bigger_than_budget_does_not_explode_k(self):
        # A record whose block far exceeds the tiny budget must still pack by the
        # field cap, not degenerate to one field per leaf (the over_budget edge).
        doc = _record_document(4, 20)
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=2000, C_usable=20.0)
        state = run_stage_1(state, _record_schema(4, 20))
        state = run_stage_2a(state)
        state = run_stage_2b(state, doc, ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        # 80 fields ÷ cap 50 ≈ 2 reliability leaves; never ~80 one-field leaves.
        assert len(state.leaves) <= state.K_min + 4
        paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert len(paths) == len(set(paths)) == 80  # still every field, once

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


class TestSmallDocSharedDocumentCost:
    """The shared small-doc is counted once per leaf, not once per group."""

    def test_multi_group_small_doc_packs_to_k_min(self):
        # 20 records x 10 string fields = 200 fields in 20 groups, on a small
        # shared document. Each group's D_cost is the same full doc; the packer
        # must count it once per leaf (max), not sum it per group — otherwise the
        # phantom-inflated document cost forces far more leaves than K_min.
        props = {
            f"rec_{r:02d}": {
                "type": "object",
                "properties": {f"f_{i:02d}": {"type": "string"} for i in range(10)},
            }
            for r in range(20)
        }
        schema = {"type": "object", "properties": props}
        doc = "\n".join(f"rec {r} f {i} = v{r}_{i}" for r in range(20) for i in range(10))

        state = PipelineState(chars_per_token=4.0, C_eff=50_000, M_O=10_000, C_usable=25_000.0)
        state = run_stage_1(state, schema)
        state = run_stage_2a(state)
        state = run_stage_2b(state, doc, ExtractionConfig())
        # Sanity: this is the small-doc fast path (no per-group segments).
        assert all(not g.matched_segments for g in state.groups)
        # Raise the reliability budget so the field cap does not bind here — this
        # test isolates the shared-document cost behaviour (the field cap is its
        # own test, TestMaxFieldsPerCall).
        state = run_stage_2c(state, ExtractionConfig(max_fields_per_call=1000))

        # With the shared doc counted once, packing reaches the theoretical
        # minimum number of leaves (no phantom-document inflation).
        assert len(state.leaves) == state.K_min


class TestSafeOutputCappedAtMO:
    """A leaf's max_tokens reservation never exceeds the model's M_O."""

    def test_large_leaf_safe_output_capped_at_m_o(self):
        # One group of 200 heavy fields on a generous context but a modest M_O:
        # the raw reservation (Stau + line overhead + margin) would exceed M_O,
        # so it must be capped at M_O (never request more than the model allows).
        props = {f"f_{i:03d}": {"type": "string"} for i in range(200)}
        schema = {"type": "object", "properties": props}
        m_o = 4000
        state = PipelineState(chars_per_token=4.0, C_eff=200_000, M_O=m_o, C_usable=100_000.0)
        state = run_stage_1(state, schema)
        state = run_stage_2a(state)
        state = run_stage_2b(state, "small doc", ExtractionConfig())
        state = run_stage_2c(state, ExtractionConfig())
        assert state.leaves
        for leaf in state.leaves:
            assert leaf.safe_output <= m_o, f"safe_output {leaf.safe_output} exceeds M_O {m_o}"


# ---------------------------------------------------------------------------
# Evidence-aware split (Set-Union Bin Packing) — Phase A.2
# ---------------------------------------------------------------------------
from formatshield.pipeline.s2c_packing import _coverage_fits  # noqa: E402
from formatshield.schema._types import FieldGroup, Segment  # noqa: E402


def _field_in(path: str, parent: str, tau: float = 5.0) -> Field:
    f = Field(path=path, type="string", constraints={}, parent_path=parent, schema_node={})
    return f.with_tau(tau=tau, var_tau=0.5)


def _seg(seg_id: int, n_chars: int) -> Segment:
    return Segment(
        text="x" * n_chars, start=0, end=n_chars, segment_type="unstructured", segment_id=seg_id
    )


def _group_with(parent: str, seg_ids: list[int], n_chars: int) -> FieldGroup:
    segs = [_seg(i, n_chars) for i in seg_ids]
    return FieldGroup(
        parent_path=parent,
        fields=[_field_in(f"{parent}.f", parent)],
        matched_segments=segs,
        segment_scores=[1.0] * len(segs),
        D_cost=0,
    )


class TestCoverageFits:
    def test_fits_when_union_small(self):
        g = _group_with("a", [0], 400)
        assert _coverage_fits(
            [g], g.fields, overhead=50, c_usable=1000.0, output_ceiling=500.0, chars_per_token=4.0
        )

    def test_rejects_when_union_exceeds_budget(self):
        g = _group_with("a", [0], 8000)
        assert not _coverage_fits(
            [g], g.fields, overhead=50, c_usable=1000.0, output_ceiling=500.0, chars_per_token=4.0
        )


class TestEvidenceAwareSplit:
    @staticmethod
    def _state_with_groups(groups: list[FieldGroup], c_usable: float) -> PipelineState:
        fields = [f for g in groups for f in g.fields]
        state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=c_usable)
        state.fields = fields
        state.field_by_path = {f.path: f for f in fields}
        state.groups = groups
        state.dep_dag = {}
        return state

    def test_disjoint_evidence_forces_split(self):
        groups = [_group_with(p, [i], 2000) for i, p in enumerate(("a", "b", "c"))]
        state = self._state_with_groups(groups, c_usable=1300.0)
        run_stage_2c(state, ExtractionConfig())
        assert len(state.leaves) >= 2

    def test_shared_evidence_packs_together(self):
        groups = [_group_with(p, [0], 2000) for p in ("a", "b", "c")]
        state = self._state_with_groups(groups, c_usable=1300.0)
        run_stage_2c(state, ExtractionConfig())
        assert len(state.leaves) == 1

    def test_field_level_coverage_forces_split(self):
        # One group, four typed fields, each needing a DIFFERENT large segment.
        # Per-group coverage (one segment) fits the budget, but per-field coverage
        # (four segments) does not — so the leaf must split rather than let Stage 3
        # trim a field's only evidence. Every field must still be placed.
        segs = [_seg(i, 2000) for i in range(4)]
        fields = [_field_in(f"a.f{i}", "a") for i in range(4)]
        g = FieldGroup(
            parent_path="a",
            fields=fields,
            matched_segments=segs,
            segment_scores=[4.0, 3.0, 2.0, 1.0],
            field_best_segment={f"a.f{i}": i for i in range(4)},
        )
        state = self._state_with_groups([g], c_usable=1400.0)
        run_stage_2c(state, ExtractionConfig())
        assert len(state.leaves) >= 2
        placed = {f.path for leaf in state.leaves for f in leaf.fields}
        assert placed == {f.path for f in fields}


# ---------------------------------------------------------------------------
# Field-count reliability cap (max_fields_per_call)
# ---------------------------------------------------------------------------
class TestMaxFieldsPerCall:
    """No leaf exceeds the field cap, even when the token budget would allow it."""

    @staticmethod
    def _wide_schema(n: int) -> dict:
        return {"type": "object", "properties": {f"f{i}": {"type": "string"} for i in range(n)}}

    def test_cap_forces_multiple_leaves_on_huge_budget(self):
        # Huge context + output → token budget alone would pack all 200 in one leaf.
        state = PipelineState(
            chars_per_token=4.0, C_eff=1_000_000, M_O=1_000_000, C_usable=500_000.0
        )
        state = run_stage_1(state, self._wide_schema(200))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "tiny doc", ExtractionConfig())
        run_stage_2c(state, ExtractionConfig(max_fields_per_call=50))
        assert len(state.leaves) >= 4, "200 fields / cap 50 → at least 4 leaves"
        for leaf in state.leaves:
            assert len(leaf.fields) <= 50, f"leaf has {len(leaf.fields)} fields > cap 50"

    def test_every_field_still_placed(self):
        state = PipelineState(
            chars_per_token=4.0, C_eff=1_000_000, M_O=1_000_000, C_usable=500_000.0
        )
        state = run_stage_1(state, self._wide_schema(130))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "tiny doc", ExtractionConfig())
        run_stage_2c(state, ExtractionConfig(max_fields_per_call=40))
        placed = {f.path for leaf in state.leaves for f in leaf.fields}
        assert placed == {f.path for f in state.fields}

    def test_k_min_respects_cap(self):
        state = PipelineState(
            chars_per_token=4.0, C_eff=1_000_000, M_O=1_000_000, C_usable=500_000.0
        )
        state = run_stage_1(state, self._wide_schema(100))
        state = run_stage_2a(state)
        state = run_stage_2b(state, "tiny doc", ExtractionConfig())
        run_stage_2c(state, ExtractionConfig(max_fields_per_call=25))
        assert state.K_min >= 4  # 100 fields, difficulty-weighted load / budget 25

    def test_harder_fields_yield_smaller_leaves(self):
        # Same field count + budget: a harder schema must use >= as many leaves as
        # an easy one (difficulty-weighted budget, not a raw count).
        def _typed(n: int, ftype: str, desc: str) -> dict:
            return {
                "type": "object",
                "properties": {f"f{i}": {"type": ftype, "description": desc} for i in range(n)},
            }

        def _leaves_for(schema: dict) -> int:
            st = PipelineState(
                chars_per_token=4.0, C_eff=1_000_000, M_O=1_000_000, C_usable=500_000.0
            )
            st = run_stage_1(st, schema)
            st = run_stage_2a(st)
            st = run_stage_2b(st, "tiny doc", ExtractionConfig())
            run_stage_2c(st, ExtractionConfig(max_fields_per_call=40))
            return len(st.leaves)

        easy = _leaves_for(_typed(120, "boolean", ""))
        hard = _leaves_for(_typed(120, "string", "a long nuanced free-text clinical note"))
        assert hard >= easy, "harder fields pack fewer per leaf -> more leaves"
