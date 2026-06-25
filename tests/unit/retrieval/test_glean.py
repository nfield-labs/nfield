"""Unit tests for the GLEAN schema-typed fusion retriever."""

from __future__ import annotations

from nfield.retrieval._bmx import bmx_rescore
from nfield.retrieval._glean import (
    build_glean_index,
    field_best_segments,
    glean_rescore,
)
from nfield.schema._types import Field, Segment


def _seg(text: str, sid: int) -> Segment:
    return Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=sid)


def _field(path: str, ftype: str, *, constraints: dict | None = None, **node: object) -> Field:
    schema_node = {"type": ftype, **node}
    return Field(
        path=path,
        type=ftype,
        constraints=constraints or {},
        parent_path="",
        schema_node=schema_node,
    )


class TestBuildGleanIndex:
    def test_pairs_both_indices(self) -> None:
        idx = build_glean_index([_seg("revenue 4591", 0)])
        assert idx.lexical.n == 1
        assert len(idx.morphology.segments) == 1


class TestGleanRescore:
    def test_lmc_lifts_the_segment_holding_the_value(self) -> None:
        # s0 has the label next to a number; s1 just repeats the label word.
        segs = [
            _seg("the enrollment total was 4591 participants", 0),
            _seg("enrollment enrollment enrollment criteria here", 1),
        ]
        idx = build_glean_index(segs)
        f = _field("enrollmentCount", "integer", description="enrollment total")
        ranked = glean_rescore(idx, [f], "enrollment total", top_k=2)
        # GLEAN puts the segment that actually contains the number first.
        assert ranked[0][0].segment_id == 0

    def test_enum_hit_recovers_evidence_pure_lexical_misses(self) -> None:
        segs = [
            _seg("the study used a standard protocol", 0),
            _seg("results were grouped by phase 3 outcomes", 1),
        ]
        idx = build_glean_index(segs)
        enum = ["phase 1", "phase 2", "phase 3"]
        f = _field("stage", "string", constraints={"enum": enum}, enum=enum)
        # The path term "stage" appears in neither segment → BMX finds nothing.
        assert bmx_rescore(idx.lexical, "stage", top_k=2) == []
        ranked = glean_rescore(idx, [f], "stage", top_k=2)
        assert ranked[0][0].segment_id == 1

    def test_degrades_to_bmx_for_plain_strings(self) -> None:
        segs = [
            _seg("alpha beta gamma", 0),
            _seg("beta beta delta", 1),
            _seg("gamma delta epsilon", 2),
        ]
        idx = build_glean_index(segs)
        f = _field("name", "string")  # no type/enum/format → no morphology
        glean_order = [s.segment_id for s, _ in glean_rescore(idx, [f], "beta gamma", top_k=3)]
        bmx_order = [s.segment_id for s, _ in bmx_rescore(idx.lexical, "beta gamma", top_k=3)]
        assert glean_order == bmx_order

    def test_top_k_zero_returns_empty(self) -> None:
        idx = build_glean_index([_seg("revenue 4591", 0)])
        f = _field("n", "integer")
        assert glean_rescore(idx, [f], "revenue", top_k=0) == []

    def test_empty_index_returns_empty(self) -> None:
        idx = build_glean_index([])
        f = _field("n", "integer")
        assert glean_rescore(idx, [f], "revenue", top_k=5) == []


class TestFieldBestSegments:
    def test_typed_field_maps_to_its_evidence(self) -> None:
        # Typed field is located; the plain-string sibling is left to per-group
        # coverage (gated out to avoid fragmenting the budget).
        segs = [
            _seg("study enrollment was 4591 participants", 0),
            _seg("the eligibility criteria are listed below", 1),
        ]
        idx = build_glean_index(segs)
        count = _field("count", "integer", description="enrollment")
        crit = _field("criteria", "string", description="eligibility criteria")
        best = field_best_segments(idx, [count, crit], segs)
        assert best["count"] == 0
        assert "criteria" not in best

    def test_plain_string_field_is_omitted(self) -> None:
        segs = [_seg("alpha beta gamma", 0)]
        idx = build_glean_index(segs)
        plain = _field("name", "string", description="alpha beta")  # has label match
        assert field_best_segments(idx, [plain], segs) == {}

    def test_empty_candidates_returns_empty(self) -> None:
        idx = build_glean_index([_seg("anything", 0)])
        f = _field("x", "integer")
        assert field_best_segments(idx, [f], []) == {}


class TestDateProximity:
    def test_date_near_label_outranks_date_with_no_label(self) -> None:
        # s0 has the label terms next to a date; s1 has a date but no label term,
        # so only s0 gets a co-location boost.
        segs = [
            _seg("the trial start was 2020-04-13 as recorded", 0),
            _seg("published on 1999-01-01 in the journal", 1),
        ]
        idx = build_glean_index(segs)
        f = _field("startDate", "string", format="date", description="trial start")
        ranked = glean_rescore(idx, [f], "trial start", top_k=2)
        assert ranked[0][0].segment_id == 0
