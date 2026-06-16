"""Edge-case tests for assembly._quality.

Covers cases the main suite does not:
- per_field_confidence is mutable inside a frozen dataclass
- fields_failed is not tracked in QualityReport
- _determine_confidence_level with fields in CONFLICT state
- FAILED fields count toward quality_score as missing but go unreported
"""

from __future__ import annotations

import pytest

from formatshield.assembly._blackboard import Blackboard
from formatshield.assembly._quality import QualityReport, compute_quality_score
from formatshield.schema._types import Field


def make_field(path: str) -> Field:
    return Field(path=path, type="string", constraints={}, parent_path="", schema_node={})


def make_all_filled(paths: list[str]) -> Blackboard:
    bb = Blackboard(paths)
    for p in paths:
        bb.write(p, f"val_{p}")
    return bb


# ---------------------------------------------------------------------------
# BUG L1: QualityReport.per_field_confidence dict is mutable despite frozen=True
# ---------------------------------------------------------------------------


class TestQualityReportMutability:
    def test_quality_report_frozen_for_scalar_fields(self):
        """Scalar fields on frozen dataclass must be immutable."""
        report = QualityReport(
            quality_score=1.0,
            confidence_level="HIGH",
            per_field_confidence={},
            optimality_gap=0.0,
            fields_extracted=1,
            fields_missing=0,
            fields_conflicted=0,
            fields_needs_revalidation=0,
        )
        with pytest.raises(Exception):
            report.quality_score = 0.5  # type: ignore[misc]

    def test_per_field_confidence_dict_is_mutable(self):
        """BUG L1: per_field_confidence dict inside frozen dataclass IS mutable."""
        report = QualityReport(
            quality_score=1.0,
            confidence_level="HIGH",
            per_field_confidence={"name": 1.0},
            optimality_gap=0.0,
            fields_extracted=1,
            fields_missing=0,
            fields_conflicted=0,
            fields_needs_revalidation=0,
        )
        # This SHOULD raise AttributeError or TypeError, but it doesn't
        # because frozen only prevents rebinding, not mutation of contained objects
        report.per_field_confidence["name"] = 0.0  # This succeeds!
        # Document: frozen dataclass does NOT make dicts immutable
        assert report.per_field_confidence["name"] == 0.0  # Mutation succeeded (bug)


# ---------------------------------------------------------------------------
# FAILED fields not tracked in QualityReport
# ---------------------------------------------------------------------------


class TestFailedFieldsNotTracked:
    def test_failed_fields_counted_as_missing(self):
        """FAILED fields lower quality_score but are not separately reported."""
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.write("a", "value")
        bb.mark_failed("b", "parse error")
        # c is EMPTY

        report = compute_quality_score(bb, fields, K=1, K_min=1)

        # a=FILLED → extracted. b=FAILED and c=EMPTY → both missing.
        assert report.fields_extracted == 1
        # quality_score = 1/3
        assert report.quality_score == pytest.approx(1 / 3)
        # fields_missing is the remainder (total - extracted - conflicted -
        # needs_reval), so a FAILED field now correctly counts as missing too.
        assert report.fields_missing == 2  # b (FAILED) + c (EMPTY)

    def test_failed_fields_reduce_quality_score(self):
        """FAILED fields do reduce quality_score (they're not in FILLED)."""
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.mark_failed("a", "error")
        bb.mark_failed("b", "error")

        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.quality_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Confidence level with fields in various states
# ---------------------------------------------------------------------------


class TestConfidenceLevelEdgeCases:
    def test_all_filled_with_conflict_is_not_high(self):
        """If even one field is in CONFLICT, confidence cannot be HIGH."""
        paths = ["a", "b", "c"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.write("a", "val1")
        bb.write("b", "val")
        bb.write("c", "first")
        bb.write("c", "second")  # conflict

        report = compute_quality_score(bb, fields, K=2, K_min=1)
        assert report.confidence_level != "HIGH"

    def test_80_percent_filled_no_conflicts_is_medium(self):
        """80% fill rate with no conflicts → MEDIUM."""
        paths = [f"f{i}" for i in range(10)]
        fields = [make_field(p) for p in paths]
        bb2 = Blackboard(paths)
        for p in paths[:8]:
            bb2.write(p, f"v_{p}")

        report = compute_quality_score(bb2, fields, K=3, K_min=2)
        assert report.confidence_level == "MEDIUM"

    def test_zero_fields_schema_returns_high_confidence(self):
        """Empty schema → no fields → HIGH confidence (nothing to fail)."""
        bb = Blackboard([])
        report = compute_quality_score(bb, [], K=0, K_min=0)
        assert report.confidence_level == "HIGH"

    def test_needs_revalidation_confidence_not_counted_as_fill(self):
        """NEEDS_REVALIDATION fields lower quality score (they're not FILLED)."""
        paths = ["a", "b"]
        fields = [make_field(p) for p in paths]
        bb = Blackboard(paths)
        bb.write("a", "good")
        bb.mark_needs_revalidation("b")

        report = compute_quality_score(bb, fields, K=1, K_min=1)
        assert report.fields_extracted == 1  # Only "a" is FILLED
        assert report.quality_score == pytest.approx(0.5)
        assert report.per_field_confidence["b"] == pytest.approx(0.5)  # NEEDS_REVALIDATION


# ---------------------------------------------------------------------------
# Optimality gap formula correctness
# ---------------------------------------------------------------------------


class TestOptimalityGapFormula:
    def test_k_less_than_k_min_clamped_to_zero(self):
        """If K < K_min (should never happen), gap is clamped to 0."""
        bb = Blackboard(["a"])
        bb.write("a", "v")
        report = compute_quality_score(bb, [make_field("a")], K=1, K_min=5)
        # (1 - 5) / 1 = -4 → clamped to 0
        assert report.optimality_gap == pytest.approx(0.0)

    def test_k_equals_k_min_is_optimal(self):
        """K == K_min → gap = 0 (optimal)."""
        bb = Blackboard(["a"])
        bb.write("a", "v")
        report = compute_quality_score(bb, [make_field("a")], K=3, K_min=3)
        assert report.optimality_gap == pytest.approx(0.0)

    def test_k_100_k_min_1_near_1(self):
        """K=100, K_min=1 → gap = 99/100 = 0.99."""
        bb = Blackboard(["a"])
        bb.write("a", "v")
        report = compute_quality_score(bb, [make_field("a")], K=100, K_min=1)
        assert report.optimality_gap == pytest.approx(0.99)
