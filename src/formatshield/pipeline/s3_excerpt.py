"""Stage 3: Excerpt Finalization.

Zero API calls. For each CapacityLeaf, collects the matched segments from
all its groups, deduplicates by segment_id, trims to the excerpt budget
B_excerpt(L), reorders remaining segments by document position, and sets
leaf.document_excerpt.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

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
        >>> # After run_stage_3, every leaf has a non-empty document_excerpt.
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
    # Collect and deduplicate segments from all groups in this leaf
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

    # Compute excerpt budget: B_excerpt = C_usable - overhead - safe_output
    b_excerpt = max(0.0, state.C_usable - leaf.overhead - leaf.safe_output)
    budget_chars = int(b_excerpt * max(state.chars_per_token, 1.0))

    # Sort by relevance descending, trim to budget
    seg_score.sort(key=lambda x: x[1], reverse=True)
    selected: list[Segment] = []
    used_chars = 0
    for seg, _ in seg_score:
        seg_len = len(seg.text)
        if budget_chars > 0 and used_chars + seg_len > budget_chars:
            continue
        selected.append(seg)
        used_chars += seg_len

    if not selected and seg_score:
        # Always include at least the best segment even if over budget
        selected = [seg_score[0][0]]

    # Reorder by document position for coherent reading order
    selected.sort(key=lambda s: s.start)

    return _EXCERPT_SEPARATOR.join(s.text for s in selected)
