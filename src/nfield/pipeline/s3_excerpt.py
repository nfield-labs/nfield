"""Stage 3: Excerpt Finalization.

Zero API calls. For each CapacityLeaf, collects the matched segments from
all its groups, deduplicates by segment_id, trims to the excerpt budget
B_excerpt(L), reorders remaining segments by document position, and sets
leaf.document_excerpt.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING

from nfield.pipeline._coverage import coverage_segment_ids
from nfield.pipeline.s2c_packing import safe_excerpt_chars

if TYPE_CHECKING:
    from nfield.pipeline._state import PipelineState
    from nfield.schema._types import CapacityLeaf, Segment

__all__ = ["run_stage_3"]

_EXCERPT_SEPARATOR: str = "\n\n"
_SEPARATOR_LEN: int = len(_EXCERPT_SEPARATOR)


def run_stage_3(state: PipelineState) -> PipelineState:
    """Finalise document_excerpt for every leaf.

    For each leaf:
    1. Collect all matched_segments from its groups
    2. Deduplicate by segment_id (same segment may appear in multiple groups)
    3. Compute B_excerpt = C_usable - overhead (output uses the window headroom)
    4. Sort by relevance score, trim lowest-scoring segments to fit budget
    5. Reorder remaining by document position (preserve reading order)
    6. Set leaf.document_excerpt

    Small-doc fast path: if a group has no matched_segments (the full doc was
    used in Stage 2.5), the leaf gets the full first segment as its excerpt.

    Args:
        state: Pipeline state from Stage 2C (must have ``state.leaves``).

    Returns:
        Updated ``PipelineState`` with ``leaf.document_excerpt`` set.
    """
    # Later leaves prefer unseen segments, so the excerpt union maximises coverage.
    used_ids: set[int] = set()
    for leaf in state.leaves:
        leaf.document_excerpt = _finalize_excerpt(leaf, state, used_ids)
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _finalize_excerpt(leaf: CapacityLeaf, state: PipelineState, used_ids: set[int]) -> str:
    """Compute the trimmed, ordered document excerpt for a leaf.

    Args:
        leaf: The leaf to finalize.
        state: Full pipeline state (for C_usable and chars_per_token).
        used_ids: Segment ids already excerpted by earlier leaves; updated in place.
            Spare-budget fill prefers segments not in this set (union coverage).

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
        leaf.excerpt_segment_ids = {s.segment_id for s in state.segments}
        return state.segments[0].text
    if not seg_score:
        return ""

    # Excerpt budget: B_excerpt = C_usable - overhead, then capped so input +
    # output stays within the real window even when the tokenizer emits more
    # tokens than the chars-per-token heuristic predicts (safe_excerpt_chars).
    b_excerpt = max(0.0, state.C_usable - leaf.overhead)
    cpt = max(state.chars_per_token, 1.0)
    safe_chars = safe_excerpt_chars(state.C_eff, leaf.overhead, leaf.safe_output, cpt)
    budget_chars = min(int(b_excerpt * cpt), safe_chars)

    # When the whole document fits within the safe cap, ship all of it.
    doc_chars = sum(len(s.text) for s in state.segments) + _SEPARATOR_LEN * len(state.segments)
    if budget_chars < doc_chars <= safe_chars:
        budget_chars = doc_chars

    # --- Coverage-first selection -------------------------------------
    # Coverage: guarantee each FIELD its best supporting segment FIRST, so a
    # field's only supporting chunk is never crowded out by globally-higher but
    # redundant chunks (mixed-type groups would otherwise drop a field's evidence).
    # A segment covering several fields is counted once. Fill: spend the remaining
    # budget on the next best segments.
    cover_ids = _coverage_segment_ids(leaf)
    covering = [(s, sc) for (s, sc) in seg_score if s.segment_id in cover_ids]
    rest = [(s, sc) for (s, sc) in seg_score if s.segment_id not in cover_ids]
    covering.sort(key=lambda x: x[1], reverse=True)
    # Unseen-first, then score: parallel leaves complement instead of duplicate.
    rest.sort(key=lambda x: (x[0].segment_id in used_ids, -x[1]))

    selected: list[Segment] = []
    used_chars = 0
    for seg, _ in covering + rest:
        seg_len = len(seg.text)
        if budget_chars > 0 and used_chars + seg_len > budget_chars:
            continue
        selected.append(seg)
        used_chars += seg_len

    # Ship the contiguous run of small neighbours around each small match.
    if budget_chars > 0 and used_chars < budget_chars:
        used_chars = _expand_small_runs(selected, state.segments, budget_chars, used_chars)

    # Spend leftover budget on unmatched segments at the largest uncovered gaps.
    if budget_chars > 0 and used_chars < budget_chars:
        selected_ids = {s.segment_id for s in selected}
        candidates = [
            seg
            for seg in state.segments
            if seg.segment_id not in seen_ids and seg.segment_id not in selected_ids
        ]
        used_chars = _stratified_fill(selected, candidates, budget_chars, used_chars)

    if not selected:
        # Always include at least the single best segment, even if over budget.
        best = max(seg_score, key=lambda x: x[1])[0]
        selected = [best]

    used_ids.update(s.segment_id for s in selected)
    leaf.excerpt_segment_ids = {s.segment_id for s in selected}
    # Reorder by document position for coherent reading order.
    selected.sort(key=lambda s: s.start)
    return _EXCERPT_SEPARATOR.join(s.text for s in selected)


# Chunks this small are sharded table rows or list items, not paragraphs.
_SMALL_SEGMENT_CHARS: int = 200
# Cap on the characters one small match may pull in around itself.
_RUN_EXPANSION_CHARS: int = 2_000


def _expand_small_runs(
    selected: list[Segment],
    all_segments: list[Segment],
    budget_chars: int,
    used_chars: int,
) -> int:
    """Add the contiguous run of small neighbours around each selected small segment.

    Walks left and right from every small selected segment over adjacent (by
    ``segment_id``) segments that are themselves small, adding each until the run cap
    or the leaf budget stops it. Strictly additive.

    Args:
        selected: Segments already in the excerpt; appended to in place.
        all_segments: Every document segment (any order).
        budget_chars: The leaf's total excerpt budget.
        used_chars: Characters already spent.

    Returns:
        Updated ``used_chars`` after expansion.
    """
    by_id = {s.segment_id: s for s in all_segments}
    in_excerpt = {s.segment_id for s in selected}
    for seed in [s for s in selected if len(s.text) <= _SMALL_SEGMENT_CHARS]:
        run_chars = 0
        for direction in (-1, 1):
            neighbor_id = seed.segment_id + direction
            while run_chars < _RUN_EXPANSION_CHARS and used_chars < budget_chars:
                seg = by_id.get(neighbor_id)
                if seg is None or len(seg.text) > _SMALL_SEGMENT_CHARS:
                    break
                if seg.segment_id not in in_excerpt:
                    if used_chars + len(seg.text) > budget_chars:
                        break
                    selected.append(seg)
                    in_excerpt.add(seg.segment_id)
                    used_chars += len(seg.text)
                    run_chars += len(seg.text)
                neighbor_id += direction
    return used_chars


def _stratified_fill(
    selected: list[Segment],
    candidates: list[Segment],
    budget_chars: int,
    used_chars: int,
) -> int:
    """Fill spare budget with candidates at the centre of the largest uncovered gap.

    Greedy uniform-spread selection: each pick takes the candidate closest to the
    midpoint of the widest document interval not yet represented in *selected*,
    then splits that interval. Front, middle, and end of the document are reached
    in proportion to how uncovered they are, so no region is systematically
    starved by an ordering bias. O(M·N) picks over N candidates - fine for the
    hundreds of segments a large document produces.

    Args:
        selected: Segments already in the excerpt; appended to in place.
        candidates: Eligible unmatched segments (not yet in any pool for this leaf).
        budget_chars: The leaf's total excerpt budget.
        used_chars: Characters already spent.

    Returns:
        Updated ``used_chars`` after filling.
    """
    pool = sorted(candidates, key=lambda s: s.start)
    while pool and used_chars < budget_chars:
        covered = sorted(s.start for s in selected)
        # Midpoint of the widest gap between covered positions, within the pool span.
        points = [pool[0].start, *covered, pool[-1].start]
        points.sort()
        widest = max(itertools.pairwise(points), key=lambda ab: ab[1] - ab[0], default=None)
        if widest is None:
            break
        target = (widest[0] + widest[1]) // 2
        pick = min(pool, key=lambda s: abs(s.start - target))
        pool.remove(pick)
        if used_chars + len(pick.text) > budget_chars:
            continue  # too large for the remaining budget; try other candidates
        selected.append(pick)
        used_chars += len(pick.text)
    return used_chars


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
