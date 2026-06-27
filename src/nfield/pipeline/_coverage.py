"""Shared coverage-set logic for Stage 2C packing and Stage 3 excerpt.

A leaf's coverage set is its must-have evidence: each group's single best segment
plus each typed field's own best segment (Set-Union Bin Packing - groups in one
leaf share the document, so the cost is the deduplicated union; Nemhauser-Wolsey-
Fisher 1978). Stage 2C costs this set to decide when to split a leaf; Stage 3 builds
the excerpt from it. Defining it once keeps the two stages consistent.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

    from nfield.schema._types import FieldGroup

__all__ = ["coverage_segment_ids", "coverage_tokens"]

# English-average characters per token; used only if the calibrated ratio is missing.
_FALLBACK_CHARS_PER_TOKEN: float = 4.0


def coverage_segment_ids(
    groups: Iterable[FieldGroup],
    leaf_field_paths: frozenset[str] | set[str],
) -> set[int]:
    """Segment ids that must stay in a leaf's excerpt to cover its fields.

    The union of (a) each group's single best matched segment, for groups that
    contribute a field to this leaf, and (b) each typed field's own best segment
    (``field_best_segment`` from Stage 2.5), restricted to the leaf's fields.
    Deduplicated by construction.

    Args:
        groups: The groups in (or being packed into) the leaf.
        leaf_field_paths: Paths of the fields actually extracted in this leaf -
            scopes coverage so a split leaf never reserves a sibling's evidence.

    Returns:
        The set of ``segment_id`` values forming the leaf's coverage set.
    """
    ids: set[int] = set()
    for group in groups:
        if group.matched_segments and any(f.path in leaf_field_paths for f in group.fields):
            best = max(
                zip(group.matched_segments, group.segment_scores, strict=False),
                key=lambda pair: pair[1],
            )
            ids.add(best[0].segment_id)
        for path, seg_id in group.field_best_segment.items():
            if path in leaf_field_paths:
                ids.add(seg_id)
    return ids


def coverage_tokens(
    groups: Iterable[FieldGroup],
    leaf_field_paths: frozenset[str] | set[str],
    chars_per_token: float,
) -> int:
    """Token cost of a leaf's coverage set (the deduplicated must-have evidence).

    Args:
        groups: The groups in (or being packed into) the leaf.
        leaf_field_paths: Paths of the fields extracted in this leaf.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        Token cost of the coverage segments; ``0`` when there is no coverage
        (small-doc fast path or no retrieval hit).
    """
    group_list = list(groups)
    ids = coverage_segment_ids(group_list, leaf_field_paths)
    if not ids:
        return 0
    seg_by_id: dict[int, int] = {}
    for group in group_list:
        for seg in group.matched_segments:
            if seg.segment_id not in seg_by_id:
                seg_by_id[seg.segment_id] = len(seg.text)
    total_chars = sum(seg_by_id[i] for i in ids if i in seg_by_id)
    if total_chars == 0:
        return 0
    ratio = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    return max(1, math.ceil(total_chars / ratio))
