"""Unit tests for assembly._quality - quality scoring."""

from __future__ import annotations

import pytest

from nfield.assembly._blackboard import Blackboard
from nfield.assembly._quality import QualityReport, compute_quality_score
from nfield.schema._types import Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_field(path: str, ftype: str = "string") -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints={},
        parent_path="",
        schema_node={},
    )


def make_blackboard_filled(paths: list[str], fill: list[str]) -> Blackboard:
    bb = Blackboard(paths)
    for p in fill:
        bb.write(p, f"value_for_{p}")
    return bb


# ---------------------------------------------------------------------------
# QualityReport dataclass
# ---------------------------------------------------------------------------


class TestQualityReport:
    def test_is_immutable(self):
        report = QualityReport(
            quality_score=1.0,
            confidence_level="HIGH",
            per_field_confidence={"a": 1.0},
            optimality_gap=0.0,
            fields_extracted=1,
            fields_missing=0,
            fields_conflicted=0,
            fields_needs_revalidation=0,
        )
        with pytest.raises(Exception):
            report.quality_score = 0.5  # type: ignore[misc]

    def test_fields_accessible(self):
        report = QualityReport(
            quality_score=0.9,
            confidence_level="MEDIUM",
            per_field_confidence={},
            optimality_gap=0.1,
            fields_extracted=9,
            fields_missing=1,
            fields_conflicted=0,
            fields_needs_revalidation=0,
        )
        assert report.quality_score == pytest.approx(0.9)
        assert report.confidence_level == "MEDIUM"
        assert report.fields_extracted == 9


# ---------------------------------------------------------------------------
# compute_quality_score - quality_score (fill rate)
# ---------------------------------------------------------------------------


class TestComputeQualityScore:
    def test_all_fields_filled_quality_is_1(self):
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=3, K_min=3)
        assert report.quality_score == pytest.approx(1.0)

    def test_no_fields_filled_quality_is_0(self):
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.quality_score == pytest.approx(0.0)

    def test_partial_fill_quality(self):
        paths = ["a", "b", "c", "d"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, ["a", "b"])  # 2/4 filled
        report = compute_quality_score(bb, fields, K=4, K_min=2)
        assert report.quality_score == pytest.approx(0.5)

    def test_empty_fields_list_quality_is_0(self):
        bb = Blackboard([])
        report = compute_quality_score(bb, [], K=0, K_min=0)
        assert report.quality_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_quality_score - confidence_level
# ---------------------------------------------------------------------------


class TestConfidenceLevel:
    def test_all_filled_no_conflicts_is_high(self):
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=2, K_min=2)
        assert report.confidence_level == "HIGH"

    def test_conflicts_prevent_high(self):
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.write("a", "first")
        bb.write("a", "second")  # conflict
        bb.write("b", "ok")
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.confidence_level != "HIGH"

    def test_high_fill_rate_is_medium(self):
        paths = [f"f{i}" for i in range(10)]
        fields = [make_field(p) for p in paths]
        # 9/10 = 90% fill → MEDIUM (not all filled)
        bb = make_blackboard_filled(paths, paths[:9])
        report = compute_quality_score(bb, fields, K=5, K_min=3)
        assert report.confidence_level in ("MEDIUM", "HIGH")

    def test_low_fill_rate_is_low(self):
        paths = [f"f{i}" for i in range(10)]
        fields = [make_field(p) for p in paths]
        # 5/10 = 50% fill → LOW
        bb = make_blackboard_filled(paths, paths[:5])
        report = compute_quality_score(bb, fields, K=5, K_min=3)
        assert report.confidence_level == "LOW"


# ---------------------------------------------------------------------------
# compute_quality_score - optimality_gap
# ---------------------------------------------------------------------------


class TestOptimalityGap:
    def test_optimal_k_equals_k_min_gap_is_zero(self):
        paths = ["a"]
        fields = [make_field("a")]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=3, K_min=3)
        assert report.optimality_gap == pytest.approx(0.0)

    def test_double_k_gap_is_half(self):
        paths = ["a"]
        fields = [make_field("a")]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=6, K_min=3)
        assert report.optimality_gap == pytest.approx(0.5)

    def test_gap_always_in_zero_to_one(self):
        paths = ["a"]
        fields = [make_field("a")]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=10, K_min=1)
        assert 0.0 <= report.optimality_gap <= 1.0

    def test_zero_k_gap_is_zero(self):
        paths = ["a"]
        fields = [make_field("a")]
        bb = Blackboard(paths)
        report = compute_quality_score(bb, fields, K=0, K_min=0)
        assert report.optimality_gap == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# compute_quality_score - per_field_confidence
# ---------------------------------------------------------------------------


class TestPerFieldConfidence:
    def test_filled_field_confidence_is_one(self):
        paths = ["x"]
        fields = [make_field("x")]
        bb = make_blackboard_filled(paths, paths)
        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.per_field_confidence["x"] == pytest.approx(1.0)

    def test_missing_field_confidence_is_zero(self):
        paths = ["x"]
        fields = [make_field("x")]
        bb = Blackboard(paths)
        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.per_field_confidence["x"] == pytest.approx(0.0)

    def test_needs_revalidation_confidence_is_half(self):
        paths = ["x"]
        fields = [make_field("x")]
        bb = Blackboard(paths)
        bb.mark_needs_revalidation("x")
        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.per_field_confidence["x"] == pytest.approx(0.5)

    def test_all_fields_have_confidence_entry(self):
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, ["a"])
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert set(report.per_field_confidence.keys()) == {"a", "b", "c"}


# ---------------------------------------------------------------------------
# compute_quality_score - field counts
# ---------------------------------------------------------------------------


class TestFieldCounts:
    def test_fields_extracted_count(self):
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, ["a", "b"])
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.fields_extracted == 2

    def test_fields_missing_count(self):
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = make_blackboard_filled(paths, ["a"])
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.fields_missing == 2

    def test_fields_conflicted_count(self):
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.write("a", 1)
        bb.write("a", 2)  # conflict
        bb.write("b", "ok")
        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.fields_conflicted == 1

    def test_fields_needs_revalidation_count(self):
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.mark_needs_revalidation("a")
        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.fields_needs_revalidation == 1


class TestNoneNotCountedAsExtracted:
    def test_none_counts_as_missing_not_extracted(self):
        f_real = Field("name", "string", {}, "", {})
        f_absent = Field("nickname", "string", {}, "", {})
        bb = Blackboard(["name", "nickname"])
        bb.write("name", "Alice")
        bb.write_raw("nickname", None)  # confirmed absent
        report = compute_quality_score(bb, [f_real, f_absent], K=1, K_min=1)
        assert report.fields_extracted == 1, "only the real value is extracted"
        assert report.fields_missing == 1, "the None confirmed-absent field is missing"
        assert report.quality_score == 0.5


# ---------------------------------------------------------------------------
# Narrowed except: a field path absent from the blackboard scores 0.0
# ---------------------------------------------------------------------------
class TestUnknownFieldConfidence:
    def test_field_not_on_blackboard_scores_zero(self):
        # Blackboard knows only "a"; "b" is unregistered → get_state raises
        # AssemblyError, which is caught and scored 0.0 (not a masked crash).
        bb = Blackboard(["a"])
        bb.write("a", "x")
        f_a = Field(path="a", type="string", constraints={}, parent_path="", schema_node={})
        f_b = Field(path="b", type="string", constraints={}, parent_path="", schema_node={})
        report = compute_quality_score(bb, [f_a, f_b], K=1, K_min=1)
        assert report.per_field_confidence["a"] == 1.0
        assert report.per_field_confidence["b"] == 0.0
