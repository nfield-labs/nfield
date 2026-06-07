"""BMX — entropy-weighted, semantic-enhanced lexical retrieval (Stage 2.5).

A drop-in successor to BM25 (Li et al., "BMX: Entropy-weighted Similarity and
Semantic-enhanced Lexical Search", arXiv:2408.06643). Same inverted-index cost as
BM25 — tokenize + count, **no per-document embedding** — but it weights each query
term by the entropy of its frequency distribution and adds a query-overlap bonus.
On BEIR it beats BM25 on 11/15 datasets and beats a 7B embedding model on the
long-context LoCo benchmark, while staying purely lexical. That is exactly what
FormatShield's re-index-per-document constraint needs: a better lexical core that
costs no more to build than BM25.

Reuses the diacritic-folding tokenizer (see ``_tokenize.tokenize``) so accented
spellings still match their plain forms.

Scoring (arXiv:2408.06643, sec.3), all symbols in ASCII:

    score(D,Q) = sum_i  IDF(q_i) * F(q_i,D)*(alpha+1) / (F(q_i,D) + alpha*|D|/avgdl + alpha*Ebar)
               + beta * E(q_i) * S(Q,D)

    IDF(q_i) = log((n - df + 0.5)/(df + 0.5) + 1)
    E(q_i)   = Eraw(q_i) / max_j Eraw(q_j),  Eraw(q_i) = -sum p*log p,  p = sigmoid(tf)
    Ebar     = mean_i E(q_i)
    S(Q,D)   = |Q intersect D| / m
    alpha = max(min(1.5, avgdl/100), 0.5),  beta = 1/log(1 + n)

Query augmentation (the optional LLM-driven semantic layer in the paper) is not
implemented here -- the entropy-weighted lexical core is the no-API part.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from formatshield.retrieval._tokenize import tokenize

if TYPE_CHECKING:
    from formatshield.schema._types import Segment

__all__ = ["BMXIndex", "bmx_rescore", "build_bmx_index"]

# Hyperparameter bounds from the paper's English defaults; alpha is data-adaptive
# (avgdl/100, clamped) and beta decays with corpus size -- both computed at query
# time, never hard-coded magic numbers.
_ALPHA_MIN: float = 0.5
_ALPHA_MAX: float = 1.5
_ALPHA_AVGDL_DIVISOR: float = 100.0


@dataclass
class BMXIndex:
    """Inverted index plus the length statistics BMX scoring needs.

    Attributes:
        segments: The indexed segments, positionally aligned with doc indices.
        postings: ``term -> [(doc_index, term_frequency), ...]``.
        doc_len: Token length of each segment, by doc index.
        avgdl: Average document (segment) length in tokens.
        n: Number of segments.
    """

    segments: list[Segment]
    postings: dict[str, list[tuple[int, int]]]
    doc_len: list[int]
    avgdl: float
    n: int


def build_bmx_index(segments: list[Segment]) -> BMXIndex:
    """Build a BMX inverted index over *segments* (tokenize + count, no embedding).

    Args:
        segments: Document segments to index.

    Returns:
        A :class:`BMXIndex` ready for :func:`bmx_rescore`.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> idx = build_bmx_index([
        ...     Segment(text="net sales rose", start=0, end=14, segment_type="unstructured", segment_id=0),
        ... ])
        >>> idx.n
        1
    """
    postings: dict[str, list[tuple[int, int]]] = {}
    doc_len: list[int] = []
    for i, seg in enumerate(segments):
        tokens = tokenize(seg.text)
        doc_len.append(len(tokens))
        counts: dict[str, int] = {}
        for tok in tokens:
            counts[tok] = counts.get(tok, 0) + 1
        for tok, tf in counts.items():
            postings.setdefault(tok, []).append((i, tf))
    n = len(segments)
    avgdl = (sum(doc_len) / n) if n else 0.0
    return BMXIndex(segments=segments, postings=postings, doc_len=doc_len, avgdl=avgdl, n=n)


def _term_entropy(term_frequencies: list[int]) -> float:
    """Raw entropy of a term: ``-sum p*log p`` with ``p = sigmoid(tf)``.

    Measures how spread-out a term's per-document frequencies are; terms with a
    more informative distribution get a higher weight (arXiv:2408.06643, sec.3).

    Args:
        term_frequencies: The term's frequency in each document that contains it.

    Returns:
        The raw (un-normalised) entropy weight.
    """
    entropy = 0.0
    for tf in term_frequencies:
        p = 1.0 / (1.0 + math.exp(-tf))
        entropy -= p * math.log(p)
    return entropy


def bmx_rescore(
    index: BMXIndex,
    query: str,
    *,
    top_k: int,
) -> list[tuple[Segment, float]]:
    """Score segments against *query* with BMX and return the top-k.

    Args:
        index: A :class:`BMXIndex` from :func:`build_bmx_index`.
        query: The group's retrieval query.
        top_k: Maximum number of results.

    Returns:
        ``(segment, score)`` pairs sorted by score descending; empty when the query
        is blank, the index is empty, ``top_k <= 0``, or no query term is indexed.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> segs = [
        ...     Segment(text="net sales total revenue", start=0, end=23, segment_type="unstructured", segment_id=0),
        ...     Segment(text="the weather was sunny", start=23, end=44, segment_type="unstructured", segment_id=1),
        ... ]
        >>> idx = build_bmx_index(segs)
        >>> bmx_rescore(idx, "revenue", top_k=1)[0][0].segment_id
        0
    """
    if not query.strip() or index.n == 0 or top_k <= 0:
        return []

    # Unique query terms (order-preserving) that actually occur in the corpus.
    query_terms = [t for t in dict.fromkeys(tokenize(query)) if t in index.postings]
    if not query_terms:
        return []

    m = len(query_terms)
    avgdl = index.avgdl or 1.0
    alpha = max(min(_ALPHA_MAX, avgdl / _ALPHA_AVGDL_DIVISOR), _ALPHA_MIN)
    beta = 1.0 / math.log(1.0 + index.n) if index.n > 1 else 0.0

    # Normalised entropy weight E(q_i) and the mean entropy Ē.
    raw_entropy = {t: _term_entropy([tf for _, tf in index.postings[t]]) for t in query_terms}
    max_entropy = max(raw_entropy.values()) or 1.0
    entropy_weight = {t: raw_entropy[t] / max_entropy for t in query_terms}
    sum_entropy = sum(entropy_weight.values())
    mean_entropy = sum_entropy / m

    scores: dict[int, float] = {}
    overlap: dict[int, int] = {}
    for term in query_terms:
        postings = index.postings[term]
        df = len(postings)
        idf = math.log((index.n - df + 0.5) / (df + 0.5) + 1.0)
        for doc, tf in postings:
            denom = tf + alpha * (index.doc_len[doc] / avgdl) + alpha * mean_entropy
            scores[doc] = scores.get(doc, 0.0) + idf * tf * (alpha + 1.0) / denom
            overlap[doc] = overlap.get(doc, 0) + 1

    # Semantic overlap bonus: β · S(Q,D) · Σ_i E(q_i), with S(Q,D) = |Q∩D|/m.
    for doc, hits in overlap.items():
        scores[doc] += beta * (hits / m) * sum_entropy

    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(index.segments[doc], score) for doc, score in ranked]
