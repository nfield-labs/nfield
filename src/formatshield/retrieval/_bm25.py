"""BM25 keyword-based retrieval for Stage 2.5 pre-pass.

Implements BM25 ranking of document segments against field-based queries.
Used for group-level document scoring in the pre-pass to estimate D_cost(g).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING

from rank_bm25 import BM25Okapi

if TYPE_CHECKING:
    from formatshield.schema._types import Segment

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOP_K: int = 5
# Split on runs of non-word characters.
_TOKENIZATION_PATTERN: str = r"\W+"
# Unicode category for accent marks; dropped so "Denísov" matches "Denisov".
_COMBINING_MARK_CATEGORY: str = "Mn"


# ---------------------------------------------------------------------------
# BM25Index wrapper
# ---------------------------------------------------------------------------


@dataclass
class BM25Index:
    """Wrapper around rank-bm25's BM25Okapi.

    Stores the underlying BM25 model and the segment list for retrieval.

    Attributes:
        model: BM25Okapi instance from rank-bm25, or None if index is empty.
        segments: List of Segment objects indexed by the model.
    """

    model: BM25Okapi | None
    segments: list[Segment]


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_bm25_index(segments: list[Segment]) -> BM25Index:
    """Build a BM25 index from a list of document segments.

    Tokenizes each segment text and creates a BM25Okapi model.

    Args:
        segments: List of Segment objects to index.

    Returns:
        BM25Index containing the model and segment references.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> seg1 = Segment(text="apple banana", start=0, end=12, segment_type="unstructured")
        >>> seg2 = Segment(text="orange grape", start=12, end=24, segment_type="unstructured")
        >>> index = build_bm25_index([seg1, seg2])
        >>> len(index.segments)
        2
    """
    # Tokenize all segments
    tokenized_corpus = [_tokenize(seg.text) for seg in segments]

    # Build BM25 model: None for empty corpus (BM25Okapi cannot handle empty corpora)
    if not tokenized_corpus:
        return BM25Index(model=None, segments=segments)

    # Build BM25 model
    bm25_model = BM25Okapi(tokenized_corpus)

    return BM25Index(model=bm25_model, segments=segments)


# ---------------------------------------------------------------------------
# Retrieval functions
# ---------------------------------------------------------------------------


def bm25_rescore_single(index: BM25Index, query: str) -> list[float]:
    """Score all segments in the index against a single query.

    Args:
        index: BM25Index to query.
        query: Query string (can be multiple terms or empty).

    Returns:
        List of BM25 scores parallel to index.segments. Higher scores
        indicate better relevance. Returns all zeros if index.model is None
        (empty corpus) or query is empty.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> index = build_bm25_index([
        ...     Segment(text="apple pie", start=0, end=9, segment_type="unstructured"),
        ...     Segment(text="orange juice", start=9, end=21, segment_type="unstructured"),
        ...     Segment(text="banana bread", start=21, end=33, segment_type="unstructured"),
        ... ])
        >>> scores = bm25_rescore_single(index, "apple")
        >>> bool(scores[0] > scores[1])  # apple ranks higher in the first segment
        True
        >>> bm25_rescore_single(index, "")  # empty query
        [0.0, 0.0, 0.0]
    """
    if not query.strip() or index.model is None:
        # Empty query or empty index: all zeros
        return [0.0] * len(index.segments)

    tokenized_query = _tokenize(query)
    scores = index.model.get_scores(tokenized_query)
    return list(scores)


def bm25_rescore(
    index: BM25Index,
    query: str,
    *,
    top_k: int = _DEFAULT_TOP_K,
) -> list[tuple[Segment, float]]:
    """Score segments and return top-k ranked results.

    Args:
        index: BM25Index to query.
        query: Query string (can be empty).
        top_k: Number of top results to return.

    Returns:
        List of (Segment, score) tuples sorted by score descending.
        Returns up to top_k results. Empty list if query is empty or
        index has no segments.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> segments = [
        ...     Segment(text="apple pie recipe", start=0, end=16, segment_type="unstructured"),
        ...     Segment(text="orange marmalade", start=16, end=32, segment_type="unstructured"),
        ...     Segment(text="apple tart design", start=32, end=49, segment_type="unstructured"),
        ... ]
        >>> index = build_bm25_index(segments)
        >>> results = bm25_rescore(index, "apple", top_k=2)
        >>> len(results) <= 2
        True

        Edge cases:
        >>> bm25_rescore(index, "", top_k=2)  # empty query
        []
        >>> empty_index = build_bm25_index([])
        >>> bm25_rescore(empty_index, "apple")  # empty corpus
        []
    """
    if not query.strip():
        return []

    scores = bm25_rescore_single(index, query)

    # Pair segments with scores and sort descending
    ranked = sorted(
        zip(index.segments, scores, strict=False),
        key=lambda x: x[1],
        reverse=True,
    )

    return ranked[:top_k]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _fold_diacritics(text: str) -> str:
    """Strip accents so an accented spelling matches its plain form.

    Args:
        text: Text to fold.

    Returns:
        The text with accent marks removed; unchanged for plain ASCII.

    Example:
        >>> _fold_diacritics("Denísov")
        'Denisov'
        >>> _fold_diacritics("café résumé")
        'cafe resume'
    """
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(ch for ch in decomposed if unicodedata.category(ch) != _COMBINING_MARK_CATEGORY)


def _tokenize(text: str) -> list[str]:
    """Split text into lowercase, accent-folded words.

    Folding both the document and the query lets "Denísov" and "Denisov" match.

    Args:
        text: Text to tokenize.

    Returns:
        Lowercase, accent-folded tokens (non-empty only).

    Example:
        >>> _tokenize("Hello, World!")
        ['hello', 'world']
        >>> _tokenize("Denísov rode past Kutúzov")
        ['denisov', 'rode', 'past', 'kutuzov']
        >>> _tokenize("")
        []
    """
    folded = _fold_diacritics(text.lower())
    tokens = re.split(_TOKENIZATION_PATTERN, folded)
    return [t for t in tokens if t]


# ---------------------------------------------------------------------------
# Post-MVP stubs
# ---------------------------------------------------------------------------


def bocs_knapsack_selection(
    _segments: list[Segment],
    _budget_tokens: int,
) -> list[Segment]:
    """Select segment subset via knapsack optimization (post-MVP).

    Args:
        _segments: Candidate segments.
        _budget_tokens: Token budget constraint.

    Returns:
        Optimal subset of segments (post-MVP feature).

    Raises:
        NotImplementedError: This is a post-MVP feature.
    """
    raise NotImplementedError(
        "BOCS knapsack selection is a post-MVP feature. MVP uses simple top-k truncation."
    )
