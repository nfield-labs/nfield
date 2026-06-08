"""Stage 3: Excerpt Finalization.

Zero API calls. For each CapacityLeaf, collects the matched segments from
all its groups, deduplicates by segment_id, trims to the excerpt budget
B_excerpt(L), reorders remaining segments by document position, and sets
leaf.document_excerpt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from formatshield.pipeline._coverage import coverage_segment_ids

if TYPE_CHECKING:
    from formatshield.pipeline._state import PipelineState
    from formatshield.schema._types import CapacityLeaf, Segment

__all__ = ["run_stage_3"]

_EXCERPT_SEPARATOR: str = "\n\n"


def run_stage_3(state: PipelineState) -> PipelineState:
    """Finalise document_excerpt for every leaf.

    For each leaf:
    1. Collect all matched_segments from its groups
    2. Deduplicate by segment_id (same segment may appear in multiple groups)
    3. Compute B_excerpt = C_usable - overhead - safe_output
    4. Sort by relevance score, trim lowest-scoring segments to fit budget
    5. Reorder remaining by document position (preserve reading order)
    6. Set leaf.document_excerpt

    Small-doc fast path: if a group has no matched_segments (the full doc was
    used in Stage 2.5), the leaf gets the full first segment as its excerpt.

    Args:
        state: Pipeline state from Stage 2C (must have ``state.leaves``).

    Returns:
        Updated ``PipelineState`` with ``leaf.document_excerpt`` set.

    Example:
        >>> callable(run_stage_3)
        True
    """
    for leaf in state.leaves:
        leaf.document_excerpt = _finalize_excerpt(leaf, state)
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _finalize_excerpt(leaf: CapacityLeaf, state: PipelineState) -> str:
    """Compute the trimmed, ordered document excerpt for a leaf.

    Args:
        leaf: The leaf to finalize.
        state: Full pipeline state (for C_usable and chars_per_token).

    Returns:
        Document excerpt string for this leaf.
    """
    # Collect and deduplicate all matched segments from the leaf's groups.
    seen_ids: set[int] = set()
    seg_score: list[tuple[Segment, float]] = []
    for g in leaf.groups:
        for seg, score in zip(g.matched_segments, g.segment_scores, strict=False):
            if seg.segment_id not in seen_ids:
                seen_ids.add(seg.segment_id)
                seg_score.append((seg, score))

    # Small-doc fast path: no matched segments → use full document (first segment)
    if not seg_score and state.segments:
        return state.segments[0].text
    if not seg_score:
        return ""

    # Excerpt budget: B_excerpt = C_usable - overhead - safe_output
    b_excerpt = max(0.0, state.C_usable - leaf.overhead - leaf.safe_output)
    budget_chars = int(b_excerpt * max(state.chars_per_token, 1.0))

    # --- Coverage-first selection (CFCS) -------------------------------------
    # Coverage: guarantee each FIELD its best supporting segment FIRST, so a
    # field's only supporting chunk is never crowded out by globally-higher but
    # redundant chunks (mixed-type groups would otherwise drop a field's evidence).
    # A segment covering several fields is counted once. Fill: spend the remaining
    # budget on the next best segments.
    cover_ids = _coverage_segment_ids(leaf)
    covering = [(s, sc) for (s, sc) in seg_score if s.segment_id in cover_ids]
    rest = [(s, sc) for (s, sc) in seg_score if s.segment_id not in cover_ids]
    covering.sort(key=lambda x: x[1], reverse=True)
    rest.sort(key=lambda x: x[1], reverse=True)

    selected: list[Segment] = []
    used_chars = 0
    for seg, _ in covering + rest:
        seg_len = len(seg.text)
        if budget_chars > 0 and used_chars + seg_len > budget_chars:
            continue
        selected.append(seg)
        used_chars += seg_len

    if not selected:
        # Always include at least the single best segment, even if over budget.
        best = max(seg_score, key=lambda x: x[1])[0]
        selected = [best]

    # Reorder by document position for coherent reading order.
    selected.sort(key=lambda s: s.start)
    return _EXCERPT_SEPARATOR.join(s.text for s in selected)


def _coverage_segment_ids(leaf: CapacityLeaf) -> set[int]:
    """Segment ids that must stay in the excerpt to cover the leaf's fields.

    Delegates to the shared coverage definition (see :mod:`_coverage`) so Stage 3's
    excerpt and Stage 2C's split decision use the identical must-have set: each
    group's best segment plus each typed field's own best segment, scoped to the
    leaf's fields.

    Args:
        leaf: The leaf whose coverage segments to collect.

    Returns:
        Set of ``segment_id`` values forming the coverage set.
    """
    return coverage_segment_ids(leaf.groups, {f.path for f in leaf.fields})
