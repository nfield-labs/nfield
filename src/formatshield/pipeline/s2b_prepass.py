"""Stage 2.5: Document Pre-Pass (DDF).

Zero API calls. Chunks the document, builds a BMX index, and scores each
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

from formatshield.retrieval._bmx import bmx_rescore, build_bmx_index
from formatshield.retrieval._chunker import chunk_document

if TYPE_CHECKING:
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.schema._types import FieldGroup, Segment

__all__ = ["run_stage_2b"]

# Floor on segments kept per group after BMX ranking, so even a single-field
# group retrieves a few candidates (Robertson & Zaragoza, "The Probabilistic
# Relevance Framework: BM25 and Beyond", 2009). Depth is computed PER GROUP from
# its field count (see _group_top_k) — never a fixed global top-k, which would
# under-serve large groups and over-serve small ones.
_MIN_TOP_K_SEGMENTS: int = 5
# Candidate segments retrieved per field in a group. A group with more fields can
# need evidence from more places, so its retrieval depth scales with field count.
_SEGMENTS_PER_FIELD: int = 3
# Words taken from each field's description to enrich its group's retrieval
# query (field names alone are often too sparse for lexical term matching).
_GROUP_QUERY_MAX_DESC_WORDS: int = 5
# English-average characters per token; used only if the calibrated ratio is
# missing when sizing the dynamic retrieval depth.
_FALLBACK_CHARS_PER_TOKEN: float = 4.0


def run_stage_2b(
    state: PipelineState,
    document: str,
    config: ExtractionConfig,
) -> PipelineState:
    """Chunk document, score segments per group, estimate D_cost(g).

    Populates:
    - ``state.segments`` — all document segments
    - ``state.lexical_index`` — BMX index over segments (None for small docs)
    - ``group.matched_segments`` — top-k segments for each group
    - ``group.segment_scores`` — BMX scores parallel to matched_segments
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
    # Skip retrieval entirely — every group gets the full document.
    if total_doc_tokens <= state.C_usable:
        for g in state.groups:
            g.D_cost = total_doc_tokens
        state.lexical_index = None
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

    # BMX (entropy-weighted lexical) — a drop-in BM25 successor at the same
    # inverted-index cost, no per-document embedding (arXiv:2408.06643).
    lexical_index = build_bmx_index(segments)
    state.lexical_index = lexical_index

    # Retrieve a per-group number of top-ranked segments — depth scales with the
    # group's field count, not a global cap — then Stage 3 trims each leaf's pooled
    # segments to its own B_excerpt.
    for g in state.groups:
        query = _build_group_query(g)
        g_top_k = _group_top_k(g, segments, state.C_usable, state.chars_per_token)
        ranked = bmx_rescore(lexical_index, query, top_k=g_top_k)
        g.matched_segments = [seg for seg, _ in ranked]
        g.segment_scores = [score for _, score in ranked]
        g.D_cost = _compute_dcost(g.matched_segments, state.chars_per_token)

    return state


def _group_top_k(
    group: FieldGroup,
    segments: list[Segment],
    c_usable: float,
    chars_per_token: float,
) -> int:
    """Retrieval depth for ONE group, scaled to its field count.

    A group with more fields can need evidence from more places, so it retrieves
    more candidates (``field_count * _SEGMENTS_PER_FIELD``); a single-field group
    retrieves at least the budget-fill baseline. Never below the baseline pool the
    usable budget can hold, then capped by the segments that actually exist.

    Args:
        group: The group whose retrieval depth to size.
        segments: All document segments from chunking.
        c_usable: Usable context budget in tokens.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        Per-group retrieval depth in ``[_MIN_TOP_K_SEGMENTS, len(segments)]``.
    """
    if not segments:
        return _MIN_TOP_K_SEGMENTS
    cpt = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    avg_seg_tokens = max(1.0, (sum(len(s.text) for s in segments) / len(segments)) / cpt)
    budget_pool = math.ceil(c_usable / avg_seg_tokens)
    want = max(1, len(group.fields)) * _SEGMENTS_PER_FIELD
    depth = max(_MIN_TOP_K_SEGMENTS, budget_pool, want)
    return min(len(segments), depth)


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
