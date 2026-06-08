"""Tests for the shared coverage-set logic (Stage 2C packing + Stage 3 excerpt)."""

from __future__ import annotations

from formatshield.pipeline._coverage import coverage_segment_ids, coverage_tokens
from formatshield.schema._types import Field, FieldGroup, Segment


def _seg(seg_id: int, n_chars: int) -> Segment:
    return Segment(
        text="x" * n_chars, start=0, end=n_chars, segment_type="unstructured", segment_id=seg_id
    )


def _field(path: str, parent: str = "") -> Field:
    return Field(path=path, type="string", constraints={}, parent_path=parent, schema_node={})


def _group(
    parent: str,
    field_paths: list[str],
    seg_ids: list[int],
    *,
    n_chars: int = 400,
    scores: list[float] | None = None,
    field_best: dict[str, int] | None = None,
) -> FieldGroup:
    segs = [_seg(i, n_chars) for i in seg_ids]
    return FieldGroup(
        parent_path=parent,
        fields=[_field(p, parent) for p in field_paths],
        matched_segments=segs,
        segment_scores=scores if scores is not None else [1.0] * len(segs),
        field_best_segment=field_best or {},
    )


class TestCoverageSegmentIds:
    def test_per_group_best_union(self) -> None:
        g1 = _group("a", ["a.f"], [0])
        g2 = _group("b", ["b.f"], [1])
        assert coverage_segment_ids([g1, g2], {"a.f", "b.f"}) == {0, 1}

    def test_shared_segment_deduped(self) -> None:
        g1 = _group("a", ["a.f"], [0])
        g2 = _group("b", ["b.f"], [0])
        assert coverage_segment_ids([g1, g2], {"a.f", "b.f"}) == {0}

    def test_typed_field_adds_its_segment(self) -> None:
        # group best = seg 0 (score 9); typed field a.t's best = seg 1 → both kept.
        g = _group("a", ["a.f", "a.t"], [0, 1], scores=[9.0, 0.1], field_best={"a.t": 1})
        assert coverage_segment_ids([g], {"a.f", "a.t"}) == {0, 1}

    def test_leaf_scoping_excludes_sibling_field(self) -> None:
        # Whole group attached, but leaf only extracts a.f → a.t's segment excluded.
        g = _group("a", ["a.f", "a.t"], [0, 1], scores=[9.0, 0.1], field_best={"a.t": 1})
        assert coverage_segment_ids([g], {"a.f"}) == {0}

    def test_group_best_skipped_when_no_field_in_leaf(self) -> None:
        g = _group("a", ["a.f"], [0])
        assert coverage_segment_ids([g], {"other"}) == set()


class TestCoverageTokens:
    def test_disjoint_segments_sum(self) -> None:
        g1 = _group("a", ["a.f"], [0], n_chars=400)
        g2 = _group("b", ["b.f"], [1], n_chars=400)
        assert coverage_tokens([g1, g2], {"a.f", "b.f"}, 4.0) == 200

    def test_shared_segment_counted_once(self) -> None:
        g1 = _group("a", ["a.f"], [0], n_chars=400)
        g2 = _group("b", ["b.f"], [0], n_chars=400)
        assert coverage_tokens([g1, g2], {"a.f", "b.f"}, 4.0) == 100

    def test_no_segments_is_zero(self) -> None:
        g = FieldGroup(parent_path="a", fields=[_field("a.f", "a")])
        assert coverage_tokens([g], {"a.f"}, 4.0) == 0

    def test_best_segment_is_highest_score(self) -> None:
        g = FieldGroup(
            parent_path="a",
            fields=[_field("a.f", "a")],
            matched_segments=[_seg(0, 4000), _seg(1, 400)],
            segment_scores=[0.1, 9.0],
        )
        assert coverage_tokens([g], {"a.f"}, 4.0) == 100

    def test_typed_field_increases_cost(self) -> None:
        # Without the typed field only seg 0 (100 tok); with it, seg 1 adds 100 more.
        g = _group("a", ["a.f", "a.t"], [0, 1], scores=[9.0, 0.1], field_best={"a.t": 1})
        assert coverage_tokens([g], {"a.f"}, 4.0) == 100
        assert coverage_tokens([g], {"a.f", "a.t"}, 4.0) == 200
