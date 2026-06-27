"""Stage 2.5: Document Pre-Pass.

Zero API calls. Chunks the document, builds a BMX index, and scores each
FieldGroup against the index to estimate D_cost(g) - the token cost of
the document segments needed for that group.

Key invariant: D_cost(g) >= D_cost(any_subset_of_g). This makes fits()
conservative - never over-promises on context.

Small-doc fast path: if the entire document fits in C_usable, skip chunking
and assign D_cost = total document tokens to every group.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from nfield.pipeline._structure import (
    RecordSegments,
    SectionStructure,
    align_path_to_section,
    detect_section_structure,
    group_record_ordinal,
    record_segments,
)
from nfield.retrieval._chunker import chunk_document
from nfield.retrieval._glean import build_glean_index, field_best_segments, glean_rescore

if TYPE_CHECKING:
    from nfield.config import ExtractionConfig
    from nfield.pipeline._state import PipelineState
    from nfield.schema._types import FieldGroup, Segment

__all__ = ["run_stage_2b"]

# Floor on segments kept per group after BMX ranking, so even a single-field
# group retrieves a few candidates (Robertson & Zaragoza, "The Probabilistic
# Relevance Framework: BM25 and Beyond", 2009). Depth is computed PER GROUP from
# its field count (see _group_top_k) - never a fixed global top-k, which would
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
# The heading route is taken only when at least this fraction of groups align to a
# section. Below it the schema does not mirror the document's headings, so the doc
# descends to the paragraph tier - structural routing is used only when it is the
# better mechanism (DeepRead arXiv:2602.05014).
_MIN_ALIGN_COVERAGE: float = 0.5


def run_stage_2b(
    state: PipelineState,
    document: str,
    config: ExtractionConfig,
) -> PipelineState:
    """Chunk document, score segments per group, estimate D_cost(g).

    Populates:
    - ``state.segments`` - all document segments
    - ``state.lexical_index`` - BMX index over segments (None for small docs)
    - ``group.matched_segments`` - top-k segments for each group
    - ``group.segment_scores`` - BMX scores parallel to matched_segments
    - ``group.D_cost`` - token cost estimate for group's segments

    Args:
        state: Pipeline state from Stage 2A (must have ``state.groups``).
        document: Raw document text.
        config: Extraction configuration (uses ``context_utilization_ratio``).

    Returns:
        Updated ``PipelineState``.

    """
    # Hybrid tier 1 - record axis. Structure routes each group to its record block;
    # GLEAN orders the chunks within it (wrapper induction TWIX arXiv:2501.06659).
    # Non-record docs descend to the heading tier.
    record = record_segments(state.fields, document, state.chars_per_token, state.C_usable)
    if record is not None:
        return _run_record_hybrid(state, record)

    total_doc_tokens = _estimate_tokens(document, state.chars_per_token)

    # Small-doc fast path: entire document fits in the usable context window.
    # Skip retrieval entirely - every group gets the full document.
    if total_doc_tokens <= state.C_usable:
        for g in state.groups:
            g.D_cost = total_doc_tokens
        state.lexical_index = None
        # Create a single segment covering the full document
        from nfield.schema._types import Segment

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

    # Hybrid tier 2 - heading hierarchy. A non-record doc may still be organised by
    # headings; route each group to the section its path names (UniHDSA arXiv:2503.15893;
    # locate-then-read DeepRead arXiv:2602.05014). Used only when the schema aligns to
    # the headings; otherwise the document descends to the paragraph tier below.
    structure = detect_section_structure(document, state.chars_per_token, state.C_usable)
    if structure is not None and _run_heading_hybrid(state, structure):
        return state

    # Hybrid tier 3 - paragraph base case. No record/heading structure: chunk the whole
    # document and let GLEAN order the segments per group. Retrieval is the within-tier
    # scorer here, not a separate fallback.
    segments = chunk_document(document)
    state.segments = segments

    if not segments:
        for g in state.groups:
            g.D_cost = 0
        return state

    # GLEAN keeps a BMX index internally; expose it so Stage 5 re-retrieval reuses
    # the same lexical index.
    glean_index = build_glean_index(segments)
    state.lexical_index = glean_index.lexical

    # Per-group retrieval depth scales with the group's field count (not a global
    # cap); Stage 3 then trims each leaf's pooled segments to its own B_excerpt.
    for g in state.groups:
        query = _build_group_query(g)
        g_top_k = _group_top_k(g, segments, state.C_usable, state.chars_per_token)
        ranked = glean_rescore(glean_index, g.fields, query, top_k=g_top_k)
        _apply_ranking(g, ranked, state.chars_per_token)
        g.field_best_segment = field_best_segments(glean_index, g.fields, g.matched_segments)

    return state


def _run_record_hybrid(state: PipelineState, record: RecordSegments) -> PipelineState:
    """Parent-child retrieval for a record document.

    Each record's block is chunked into child segments; a group is restricted to its
    own record's children (plus the shared header) and GLEAN-ranks within them. A
    small block keeps all its children (Stage 3 fits it whole = no starvation); a big
    block keeps the top field-relevant children. The wrong record is structurally
    unreachable, so identical records never confuse retrieval.

    Args:
        state: Pipeline state from Stage 2A.
        record: The document's parent-child structure from ``record_segments``.

    Returns:
        Updated ``PipelineState``.
    """
    state.record_ordinal = record.field_ordinal
    state.record_block_tokens = record.block_tokens
    state.segments = record.segments
    state.record_block_segments = record.by_record
    state.record_header_segments = record.header_segments

    glean_index = build_glean_index(record.segments)
    state.lexical_index = glean_index.lexical

    for g in state.groups:
        rec = group_record_ordinal([f.path for f in g.fields], record.field_ordinal)
        # Structure decides INCLUSION: the group's own record block (+ shared header)
        # are always candidates. Retrieval only decides ORDER, so a block GLEAN scores
        # zero (snake_case field names vs spaced prose) is still kept - Stage 3 keeps
        # all that fit, trimming an oversized block to its top-ranked children.
        candidates = [*record.by_record.get(rec, []), *record.header_segments]
        query = _build_group_query(g)
        scored = glean_rescore(glean_index, g.fields, query, top_k=len(record.segments))
        score_by_id = {seg.segment_id: score for seg, score in scored}
        ranked = sorted(candidates, key=lambda s: score_by_id.get(s.segment_id, 0.0), reverse=True)
        _apply_ranking(
            g, [(s, score_by_id.get(s.segment_id, 0.0)) for s in ranked], state.chars_per_token
        )
        g.field_best_segment = field_best_segments(glean_index, g.fields, g.matched_segments)

    return state


def _run_heading_hybrid(state: PipelineState, structure: SectionStructure) -> bool:
    """Route groups to their heading sections; the heterogeneous twin of the record path.

    Each group is aligned to the section its field paths name (``align_path_to_section``);
    the section's chunks (plus the shared preamble) are the candidates GLEAN orders
    within. Mutates state only when the schema aligns to the headings - at least
    ``_MIN_ALIGN_COVERAGE`` of groups must match a section - so a document whose schema
    does not mirror its headings is left untouched for the paragraph tier (do-no-harm).

    Args:
        state: Pipeline state from Stage 2A (must have ``state.groups``).
        structure: Detected heading structure from ``detect_section_structure``.

    Returns:
        ``True`` when the heading route was applied; ``False`` to leave state unchanged.
    """
    if not state.groups:
        return False
    alignments = [
        align_path_to_section([f.path for f in g.fields], structure.sections) for g in state.groups
    ]
    aligned = sum(1 for index, _ in alignments if index >= 0)
    if aligned / len(state.groups) < _MIN_ALIGN_COVERAGE:
        return False

    glean_index = build_glean_index(structure.segments)
    state.segments = structure.segments
    state.lexical_index = glean_index.lexical
    for g, (index, _score) in zip(state.groups, alignments, strict=True):
        # Structure decides inclusion: the aligned section + shared preamble are the
        # candidates; an unaligned group falls back to the whole document within this
        # tier (never starved). Retrieval only decides ORDER.
        candidates = (
            [*structure.by_section.get(index, []), *structure.preamble_segments]
            if index >= 0
            else list(structure.segments)
        )
        query = _build_group_query(g)
        scored = glean_rescore(glean_index, g.fields, query, top_k=len(structure.segments))
        score_by_id = {seg.segment_id: score for seg, score in scored}
        ranked = sorted(candidates, key=lambda s: score_by_id.get(s.segment_id, 0.0), reverse=True)
        _apply_ranking(
            g, [(s, score_by_id.get(s.segment_id, 0.0)) for s in ranked], state.chars_per_token
        )
        g.field_best_segment = field_best_segments(glean_index, g.fields, g.matched_segments)
    return True


def _apply_ranking(
    group: FieldGroup,
    ranked: list[tuple[Segment, float]],
    chars_per_token: float,
) -> None:
    """Store a group's ranked segments, scores, and document-cost estimate.

    Args:
        group: The group to populate (mutated in place).
        ranked: ``(segment, score)`` pairs from the retriever, best first.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).
    """
    group.matched_segments = [seg for seg, _ in ranked]
    group.segment_scores = [score for _, score in ranked]
    group.D_cost = _compute_dcost(group.matched_segments, chars_per_token)


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
