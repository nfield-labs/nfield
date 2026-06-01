"""Tests for retrieval._bm25 — BM25 keyword retrieval."""

from __future__ import annotations

import pytest

from formatshield.retrieval._bm25 import (
    BM25Index,
    bm25_rescore,
    bm25_rescore_single,
    build_bm25_index,
)
from formatshield.schema._types import SEGMENT_TYPE_UNSTRUCTURED, Segment


@pytest.fixture
def sample_segments() -> list[Segment]:
    """Create sample segments for testing."""
    return [
        Segment(
            text="The patient was admitted with fever and cough.",
            start=0,
            end=46,
            segment_type=SEGMENT_TYPE_UNSTRUCTURED,
            segment_id=0,
        ),
        Segment(
            text="Treatment included antibiotics and rest.",
            start=46,
            end=85,
            segment_type=SEGMENT_TYPE_UNSTRUCTURED,
            segment_id=1,
        ),
        Segment(
            text="The invoice total was $500. Payment is due in 30 days.",
            start=85,
            end=140,
            segment_type=SEGMENT_TYPE_UNSTRUCTURED,
            segment_id=2,
        ),
    ]


class TestBuildBM25Index:
    """Tests for index construction."""

    def test_build_index_from_segments(self, sample_segments: list[Segment]) -> None:
        """Index builds successfully from segments."""
        index = build_bm25_index(sample_segments)
        assert isinstance(index, BM25Index)
        assert len(index.segments) == len(sample_segments)

    def test_index_segments_preserved(self, sample_segments: list[Segment]) -> None:
        """Segments in index match input order."""
        index = build_bm25_index(sample_segments)
        assert index.segments == sample_segments

    def test_empty_segment_list_builds(self) -> None:
        """Empty segment list builds without error."""
        index = build_bm25_index([])
        assert len(index.segments) == 0


class TestBM25Rescore:
    """Tests for BM25 rescoring."""

    def test_rescore_single_returns_all_scores(self, sample_segments: list[Segment]) -> None:
        """Single rescore returns scores for all segments."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "fever")
        assert len(scores) == len(sample_segments)
        assert all(isinstance(s, float) for s in scores)

    def test_fever_query_ranks_medical_segment_highest(
        self, sample_segments: list[Segment]
    ) -> None:
        """'fever' query ranks medical segment highest."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "fever")
        # First segment contains 'fever', should have highest score
        assert scores[0] > scores[2]

    def test_invoice_query_ranks_financial_segment_highest(
        self, sample_segments: list[Segment]
    ) -> None:
        """'invoice' query ranks financial segment highest."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "invoice")
        # Third segment contains 'invoice', should rank high
        assert scores[2] > scores[0]

    def test_empty_query_returns_zero_scores(self, sample_segments: list[Segment]) -> None:
        """Empty query produces all-zero scores."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "")
        assert all(s == 0.0 for s in scores)

    def test_bm25_rescore_returns_top_k(self, sample_segments: list[Segment]) -> None:
        """Rescore with top_k returns at most k results."""
        index = build_bm25_index(sample_segments)
        results = bm25_rescore(index, "payment", top_k=2)
        assert len(results) <= 2
        assert all(isinstance(pair, tuple) and len(pair) == 2 for pair in results)

    def test_results_sorted_by_score_descending(self, sample_segments: list[Segment]) -> None:
        """Results are sorted by score, highest first."""
        index = build_bm25_index(sample_segments)
        results = bm25_rescore(index, "patient", top_k=10)
        if len(results) > 1:
            scores = [score for _, score in results]
            assert scores == sorted(scores, reverse=True)

    def test_empty_query_returns_empty_results(self, sample_segments: list[Segment]) -> None:
        """Empty query returns empty results list."""
        index = build_bm25_index(sample_segments)
        results = bm25_rescore(index, "", top_k=5)
        assert results == []

    def test_top_k_zero_returns_empty(self, sample_segments: list[Segment]) -> None:
        """top_k=0 returns empty results."""
        index = build_bm25_index(sample_segments)
        results = bm25_rescore(index, "fever", top_k=0)
        assert results == []

    def test_nonexistent_term_returns_low_scores(
        self, sample_segments: list[Segment]
    ) -> None:
        """Querying non-existent term returns low/zero scores."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "xyzuniqueneverappears")
        assert all(s == 0.0 for s in scores)
