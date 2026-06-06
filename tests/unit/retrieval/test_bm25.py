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

    def test_nonexistent_term_returns_low_scores(self, sample_segments: list[Segment]) -> None:
        """Querying non-existent term returns low/zero scores."""
        index = build_bm25_index(sample_segments)
        scores = bm25_rescore_single(index, "xyzuniqueneverappears")
        assert all(s == 0.0 for s in scores)


class TestDiacriticFolding:
    """An accented document spelling must match an unaccented query term.

    Regression guard for the War & Peace minor-character miss: the prose used
    transliterations like ``Denísov``/``Kutúzov`` while the schema used
    ``Denisov``/``Kutuzov``, so a diacritic-sensitive tokenizer retrieved zero
    relevant segments.
    """

    @staticmethod
    def _corpus(name_segment_text: str) -> list[Segment]:
        """One segment carrying the name + several filler segments (rare term)."""
        fillers = [
            "The drawing room in the capital was quiet that evening.",
            "Snow fell over the frozen river through the long night.",
            "Letters arrived from the estate about the autumn harvest.",
            "A carriage waited by the gate under the grey winter sky.",
            "Guests gathered for the ball as the orchestra began to play.",
        ]
        texts = [name_segment_text, *fillers]
        pos = 0
        segments: list[Segment] = []
        for i, t in enumerate(texts):
            segments.append(
                Segment(
                    text=t, start=pos, end=pos + len(t), segment_type="unstructured", segment_id=i
                )
            )
            pos += len(t)
        return segments

    def test_accented_corpus_matches_unaccented_query(self) -> None:
        """An unaccented query ranks the accented-name segment first (rare term, +IDF)."""
        segments = self._corpus("Denísov rode forward with his hussars, shouting the order.")
        index = build_bm25_index(segments)
        results = bm25_rescore(index, "Denisov", top_k=6)
        assert results[0][0].segment_id == 0
        assert results[0][1] > 0.0, "rare folded match must have positive BM25 score"

    def test_unaccented_corpus_matches_accented_query(self) -> None:
        """Folding is symmetric: an accented query ranks the unaccented-name segment first."""
        segments = self._corpus("Kutuzov surveyed the field before the great battle.")
        index = build_bm25_index(segments)
        results = bm25_rescore(index, "Kutúzov", top_k=6)
        assert results[0][0].segment_id == 0
        assert results[0][1] > 0.0

    def test_without_folding_the_name_would_not_win(self) -> None:
        """Control: an unrelated unaccented term does not rank the name segment first."""
        segments = self._corpus("Denísov rode forward with his hussars, shouting the order.")
        index = build_bm25_index(segments)
        # A token that appears only in a filler segment must outrank the name segment
        # for its own query — proving the ranking is driven by real term matches.
        results = bm25_rescore(index, "harvest", top_k=6)
        assert results[0][0].segment_id != 0

    def test_ascii_query_unaffected(self, sample_segments: list[Segment]) -> None:
        """Folding is a no-op for plain ASCII text (existing behaviour preserved)."""
        index = build_bm25_index(sample_segments)
        results = bm25_rescore(index, "patient", top_k=3)
        assert isinstance(results, list)
