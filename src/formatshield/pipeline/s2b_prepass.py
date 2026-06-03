"""Stage 2.5: Document Pre-Pass (DDF).

Zero API calls. Chunks the document, builds a BM25 index, and scores each
FieldGroup against the index to estimate D_cost(g) — the token cost of
the document segments needed for that group.

Key invariant: D_cost(g) >= D_cost(any_subset_of_g). This makes fits()
conservative — never over-promises on context.

Small-doc fast path: if the entire document fits in C_usable, skip chunking
and assign D_cost = total document tokens to every group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from formatshield.retrieval._bm25 import bm25_rescore, build_bm25_index
from formatshield.retrieval._chunker import chunk_document

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.schema._types import FieldGroup, Segment

__all__ = ["run_stage_2b"]

# Top-scoring document segments kept per field group after BM25 ranking
# (Robertson & Zaragoza, "The Probabilistic Relevance Framework: BM25 and
# Beyond", 2009). Five balances recall against context budget.
_DEFAULT_TOP_K_SEGMENTS: int = 5
# Words taken from each field's description to enrich its group's retrieval
# query (field names alone are often too sparse for BM25 term matching).
_GROUP_QUERY_MAX_DESC_WORDS: int = 5


def run_stage_2b(
    state: PipelineState,
    document: str,
    config: ExtractionConfig,
) -> PipelineState:
    """Chunk document, score segments per group, estimate D_cost(g).

    Populates:
    - ``state.segments`` — all document segments
    - ``state.bm25_index`` — BM25 index over segments (None for small docs)
    - ``group.matched_segments`` — top-k segments for each group
    - ``group.segment_scores`` — BM25 scores parallel to matched_segments
    - ``group.D_cost`` — token cost estimate for group's segments

    Args:
        state: Pipeline state from Stage 2A (must have ``state.groups``).
        document: Raw document text.
        config: Extraction configuration (uses ``context_utilization_ratio``).

    Returns:
        Updated ``PipelineState``.

    Example:
        >>> # Short doc: D_cost equals total doc tokens for all groups.
        True
    """
    total_doc_tokens = _estimate_tokens(document, state.chars_per_token)

    # Small-doc fast path: entire document fits in the usable context window.
    # Skip BM25 entirely — every group gets the full document.
    if total_doc_tokens <= state.C_usable:
        for g in state.groups:
            g.D_cost = total_doc_tokens
        state.bm25_index = None
        # Create a single segment covering the full document
        from formatshield.schema._types import Segment

        state.segments = [
            Segment(
                text=document,
                start=0,
                end=len(document),
                segment_type="unstructured",
                segment_id=0,
            )
        ]
        return state

    # Full pre-pass: chunk, index, score per group
    segments = chunk_document(document)
    state.segments = segments

    if not segments:
        for g in state.groups:
            g.D_cost = 0
        return state

    bm25_index = build_bm25_index(segments)
    state.bm25_index = bm25_index

    for g in state.groups:
        query = _build_group_query(g)
        ranked = bm25_rescore(bm25_index, query, top_k=_DEFAULT_TOP_K_SEGMENTS)
        g.matched_segments = [seg for seg, _ in ranked]
        g.segment_scores = [score for _, score in ranked]
        g.D_cost = _compute_dcost(g.matched_segments, state.chars_per_token)

    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _build_group_query(group: FieldGroup) -> str:
    """Build a keyword query string from field names and schema descriptions.

    Uses field path tokens + first ``_GROUP_QUERY_MAX_DESC_WORDS`` words from
    each field's description (if present in schema_node).

    Args:
        group: FieldGroup to build query for.

    Returns:
        Space-separated keyword query string.
    """
    tokens: list[str] = []
    for f in group.fields:
        # Field path fragments (dots → spaces)
        tokens.extend(f.path.replace(".", " ").split())
        # First few words of description
        desc: str = f.schema_node.get("description", "")
        desc_words = desc.split()[:_GROUP_QUERY_MAX_DESC_WORDS]
        tokens.extend(desc_words)
    return " ".join(tokens)


def _estimate_tokens(text: str, chars_per_token: float) -> int:
    """Estimate token count from character count.

    Args:
        text: Text to estimate.
        chars_per_token: Characters per token ratio from Stage 0.

    Returns:
        Estimated token count (minimum 1).
    """
    if chars_per_token <= 0:
        return len(text)
    return max(1, math.ceil(len(text) / chars_per_token))


def _compute_dcost(segments: list[Segment], chars_per_token: float) -> int:
    """Compute total token cost for a list of segments.

    Args:
        segments: Segments to cost.
        chars_per_token: Characters per token ratio.

    Returns:
        Total token estimate for all segments combined.
    """
    total_chars = sum(len(s.text) for s in segments)
    return max(1, math.ceil(total_chars / max(chars_per_token, 1.0)))
