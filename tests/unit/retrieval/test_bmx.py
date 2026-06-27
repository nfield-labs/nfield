"""Tests for retrieval._bmx - entropy-weighted lexical retrieval (BMX)."""

from __future__ import annotations

from nfield.retrieval._bmx import bmx_rescore, build_bmx_index
from nfield.schema._types import Segment


def _seg(seg_id: int, text: str) -> Segment:
    return Segment(
        text=text,
        start=seg_id,
        end=seg_id + len(text),
        segment_type="unstructured",
        segment_id=seg_id,
    )


_CORPUS = [
    _seg(0, "net sales total revenue rose this fiscal year"),
    _seg(1, "the weather in the city was sunny all week"),
    _seg(2, "operating expenses and cost of revenue declined"),
    _seg(3, "snow fell over the frozen river at night"),
    _seg(4, "shareholders approved the annual dividend payout"),
]


class TestBuildIndex:
    def test_index_basics(self):
        idx = build_bmx_index(_CORPUS)
        assert idx.n == 5
        assert idx.avgdl > 0
        assert "revenue" in idx.postings

    def test_postings_record_term_frequency(self):
        idx = build_bmx_index([_seg(0, "revenue revenue revenue")])
        assert idx.postings["revenue"] == [(0, 3)]

    def test_empty_corpus(self):
        idx = build_bmx_index([])
        assert idx.n == 0
        assert bmx_rescore(idx, "revenue", top_k=3) == []


class TestRescore:
    def test_relevant_doc_ranks_first(self):
        idx = build_bmx_index(_CORPUS)
        results = bmx_rescore(idx, "revenue", top_k=5)
        assert results, "a real query term must return results"
        assert results[0][0].segment_id in (0, 2)  # the two revenue docs

    def test_irrelevant_doc_not_top(self):
        idx = build_bmx_index(_CORPUS)
        top = bmx_rescore(idx, "revenue sales", top_k=1)[0][0].segment_id
        assert top == 0  # "net sales total revenue" is the best match

    def test_results_sorted_descending(self):
        idx = build_bmx_index(_CORPUS)
        scores = [s for _, s in bmx_rescore(idx, "the revenue river", top_k=5)]
        assert scores == sorted(scores, reverse=True)

    def test_top_k_truncates(self):
        idx = build_bmx_index(_CORPUS)
        assert len(bmx_rescore(idx, "the", top_k=2)) <= 2

    def test_empty_query_returns_empty(self):
        idx = build_bmx_index(_CORPUS)
        assert bmx_rescore(idx, "", top_k=5) == []

    def test_top_k_zero_returns_empty(self):
        idx = build_bmx_index(_CORPUS)
        assert bmx_rescore(idx, "revenue", top_k=0) == []

    def test_unknown_term_returns_empty(self):
        idx = build_bmx_index(_CORPUS)
        assert bmx_rescore(idx, "zzzznonexistentterm", top_k=5) == []


class TestDiacriticFoldingReuse:
    """BMX reuses the diacritic-folding tokenizer, so accents still match."""

    def test_accented_corpus_matches_unaccented_query(self):
        corpus = [
            _seg(0, "Kutuzov surveyed the field before the great battle began"),
            _seg(1, "letters arrived from the estate about the harvest season"),
            _seg(2, "guests gathered for the ball as the orchestra played on"),
            _seg(3, "a carriage waited by the gate under a grey winter sky"),
            _seg(4, "Kutuzov gave the order and the army advanced at dawn"),
        ]
        # 'Kutúzov' (accented) query must still hit the 'Kutuzov' docs.
        idx = build_bmx_index(corpus)
        top = bmx_rescore(idx, "Kutúzov", top_k=1)[0][0].segment_id
        assert top in (0, 4)
