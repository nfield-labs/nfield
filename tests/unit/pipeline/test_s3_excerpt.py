"""Tests for Stage 3 excerpt finalisation - field-level coverage (CFCS)."""

from __future__ import annotations

from nfield.pipeline._state import PipelineState
from nfield.pipeline.s3_excerpt import _coverage_segment_ids, run_stage_3
from nfield.schema._types import CapacityLeaf, Field, FieldGroup, Segment


def _seg(text: str, sid: int) -> Segment:
    return Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=sid)


def _field(path: str) -> Field:
    return Field(path=path, type="string", constraints={}, parent_path="g", schema_node={})


class TestCoverageSegmentIds:
    def test_unions_each_fields_best_segment(self) -> None:
        g = FieldGroup(
            parent_path="g",
            fields=[_field("g.a"), _field("g.b")],
            matched_segments=[_seg("x", 0), _seg("y", 1), _seg("z", 2)],
            segment_scores=[3.0, 2.0, 1.0],
            field_best_segment={"g.a": 0, "g.b": 2},
        )
        leaf = CapacityLeaf(fields=list(g.fields), groups=[g], leaf_id=1)
        assert _coverage_segment_ids(leaf) == {0, 2}

    def test_split_group_leaf_only_covers_its_own_fields(self) -> None:
        # A wide group split across leaves attaches the whole group object to each
        # leaf. Coverage must be scoped to THIS leaf's fields (a, b) - never the
        # sibling leaf's fields (c, d) - or it would crowd out its own evidence.
        g = FieldGroup(
            parent_path="g",
            fields=[_field("g.a"), _field("g.b")],
            matched_segments=[_seg("w", 0), _seg("x", 1), _seg("y", 2), _seg("z", 3)],
            segment_scores=[1.0, 1.0, 1.0, 1.0],
            field_best_segment={"g.a": 0, "g.b": 1, "g.c": 2, "g.d": 3},
        )
        leaf = CapacityLeaf(fields=list(g.fields), groups=[g], leaf_id=1)
        assert _coverage_segment_ids(leaf) == {0, 1}

    def test_falls_back_to_group_best_without_field_map(self) -> None:
        g = FieldGroup(
            parent_path="g",
            fields=[_field("g.a")],
            matched_segments=[_seg("x", 0), _seg("y", 1)],
            segment_scores=[1.0, 5.0],  # segment 1 is the group's best
            field_best_segment={},
        )
        leaf = CapacityLeaf(fields=list(g.fields), groups=[g], leaf_id=1)
        assert _coverage_segment_ids(leaf) == {1}


class TestFieldLevelCoverage:
    def test_typed_fields_chunk_survives_a_tight_budget(self) -> None:
        # Three 40-char segments; budget fits two. s0 is the group's best (kept by
        # the base CFCS); s2 holds a typed field's low-scored evidence. Per-field
        # coverage must keep s2 over the mid-scored-but-redundant s1.
        s0, s1, s2 = _seg("A" * 40, 0), _seg("B" * 40, 1), _seg("C" * 40, 2)
        g = FieldGroup(
            parent_path="g",
            fields=[_field("g.s"), _field("g.t")],
            matched_segments=[s0, s1, s2],
            segment_scores=[3.0, 2.0, 1.0],
            field_best_segment={"g.t": 2},  # typed field t's evidence is the low-score s2
        )
        leaf = CapacityLeaf(fields=list(g.fields), groups=[g], leaf_id=1)
        state = PipelineState(chars_per_token=1.0, C_eff=1000, M_O=0, C_usable=100.0)
        state.segments = [s0, s1, s2]
        state.leaves = [leaf]

        run_stage_3(state)
        excerpt = leaf.document_excerpt
        assert "C" * 40 in excerpt  # typed field's evidence kept
        assert "A" * 40 in excerpt  # group base kept
        assert "B" * 40 not in excerpt  # mid-scored redundant segment dropped
