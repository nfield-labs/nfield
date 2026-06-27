"""Targeted re-retrieval for failed-field recovery (GSGRF).

When a field fails because its evidence was trimmed out of its leaf's excerpt
(a retrieval miss, not a model miss), re-asking the model with the *same* text
cannot recover it. This module re-queries the full segment set with ONLY the
failed fields' own terms and builds a fresh, focused excerpt, so the retry sees
different, field-relevant document text - closing Stage 5 flaw B.

It reuses the same BMX index already built in Stage 2.5; no new index, no API
call. Domain-agnostic: the query is the failed fields' paths + descriptions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nfield.retrieval._bmx import bmx_rescore

if TYPE_CHECKING:
    from nfield.retrieval._bmx import BMXIndex
    from nfield.schema._types import Field, Segment

__all__ = ["build_field_query", "record_block_excerpt", "targeted_excerpt"]

# Words taken from each field's description to enrich its retrieval query - a few
# more than the group-query pre-pass uses, since a targeted retry wants the field's
# full meaning to relocate evidence the first pass missed.
_MAX_DESC_WORDS: int = 8
_EXCERPT_SEPARATOR: str = "\n\n"
_FALLBACK_CHARS_PER_TOKEN: float = 4.0


def build_field_query(fields: list[Field]) -> str:
    """Build a keyword query from the failed fields' paths and descriptions.

    Args:
        fields: The fields to re-retrieve evidence for.

    Returns:
        Space-separated keyword query string.
    """
    tokens: list[str] = []
    for f in fields:
        tokens.extend(f.path.replace(".", " ").split())
        desc = f.schema_node.get("description", "") if isinstance(f.schema_node, dict) else ""
        if isinstance(desc, str) and desc:
            tokens.extend(desc.split()[:_MAX_DESC_WORDS])
    return " ".join(tokens)


def targeted_excerpt(
    fields: list[Field],
    bmx_index: BMXIndex | None,
    segments: list[Segment],
    *,
    budget_tokens: float,
    chars_per_token: float,
) -> str:
    """Build a fresh excerpt focused on *fields*, drawn from the WHOLE document.

    Re-ranks every segment against the failed fields' query and fills the budget
    with the top matches, ordered by document position. Differs from the leaf's
    original excerpt, which was pooled per group then trimmed - so evidence the
    first pass dropped can resurface here.

    Args:
        fields: Failed fields to retrieve evidence for.
        bmx_index: The BMX index from Stage 2.5 (``None`` on the small-doc path,
            where the whole document was already in context - re-retrieval is a
            no-op and the caller should keep the original excerpt).
        segments: All document segments from Stage 2.5.
        budget_tokens: Token budget for the excerpt (same B_excerpt the leaf used).
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        A fresh document excerpt, or ``""`` when re-retrieval is not applicable.
    """
    if bmx_index is None or not segments or not fields:
        return ""
    cpt = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    budget_chars = int(max(0.0, budget_tokens) * cpt)
    query = build_field_query(fields)
    if not query.strip():
        return ""
    ranked = bmx_rescore(bmx_index, query, top_k=len(segments))
    if not ranked:
        return ""
    selected: list[Segment] = []
    used_chars = 0
    for seg, _ in ranked:
        seg_len = len(seg.text)
        if budget_chars > 0 and used_chars + seg_len > budget_chars:
            continue
        selected.append(seg)
        used_chars += seg_len
    if not selected:
        selected = [ranked[0][0]]
    selected.sort(key=lambda s: s.start)
    return _EXCERPT_SEPARATOR.join(s.text for s in selected)


def record_block_excerpt(
    fields: list[Field],
    record_ordinal: dict[str, int],
    header_segments: list[Segment],
    block_segments: dict[int, list[Segment]],
    *,
    budget_tokens: float,
    chars_per_token: float,
) -> str | None:
    """Build a record-local excerpt from *fields*' own record blocks.

    Gathers the shared header plus each field's record block (in document order),
    capped at *budget_tokens*. Used by the record document path, where structure
    routes each field to its own block; identical sibling records are kept apart by
    ordinal rather than by lexical similarity.

    Args:
        fields: The fields whose record blocks supply the evidence.
        record_ordinal: ``field path -> record index`` from the record pre-pass.
        header_segments: The shared header segments common to every record.
        block_segments: ``record index -> that record's segments``.
        budget_tokens: Token budget for the excerpt.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        The record-local excerpt, or ``None`` when no record block applies.
    """
    ordinals = sorted({record_ordinal[f.path] for f in fields if f.path in record_ordinal})
    segments: list[Segment] = list(header_segments)
    for ordinal in ordinals:
        segments.extend(block_segments.get(ordinal, []))
    if not segments:
        return None
    segments.sort(key=lambda s: s.start)
    cpt = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    budget_chars = int(max(0.0, budget_tokens) * cpt)
    kept: list[str] = []
    used = 0
    for seg in segments:
        if kept and budget_chars > 0 and used + len(seg.text) > budget_chars:
            continue
        kept.append(seg.text)
        used += len(seg.text)
    return _EXCERPT_SEPARATOR.join(kept) or None
