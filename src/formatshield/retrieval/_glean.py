"""GLEAN — schema-typed fusion retriever (Stage 2.5).

Ranks document segments for a group of typed fields by combining two signals:

- Lexical: the BMX score (:mod:`_bmx`).
- Morphological (:mod:`_morphology`): whether a segment carries the surface form a
  field's value takes — a literal enum value, a ``format``/``pattern`` match, or a
  field's label positioned near a token of its expected type (LMC,
  lexical-morphological co-location, scored ``exp(-gap/sigma)``).

Morphological evidence is weighted per field by type-distinctiveness ``tau_d`` and
added to the lexical Reciprocal Rank Fusion score (Cormack et al., SIGIR 2009) as a
bounded re-ranking boost, so a typed segment can be lifted but a lexically strong
segment is never demoted below its lexical rank. When a group has no typed evidence
the boost is zero and the ranking equals BMX.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from formatshield.retrieval._bmx import bmx_scores, build_bmx_index
from formatshield.retrieval._morphology import (
    build_morphology_index,
    field_type_classes,
    nearest_gap,
)
from formatshield.retrieval._tokenize import fold_diacritics, tokenize

if TYPE_CHECKING:
    from formatshield.retrieval._bmx import BMXIndex
    from formatshield.retrieval._morphology import MorphologyIndex, MorphologySegment
    from formatshield.schema._types import Field, Segment

__all__ = ["GleanIndex", "build_glean_index", "glean_rescore"]

# RRF damping (Cormack et al., SIGIR 2009 default).
_RRF_K: float = 60.0
# Max morph boost = one top-rank step, so a decisive typed hit reaches ~lexical
# rank-1 and weaker LMC nudges proportionally, never an equal co-vote.
_MORPH_BOOST: float = 1.0 / (_RRF_K + 1.0)
# Token gap at which label↔value co-location decays to ~1/e.
_LMC_SIGMA: float = 8.0
# Description words added to a field's label terms for LMC.
_FIELD_QUERY_MAX_DESC_WORDS: int = 6

# Type-distinctiveness tau_d: how reliably a value's shape identifies its evidence.
_TAU_ENUM: float = 1.0
_TAU_FORMAT: float = 0.8
_TAU_NUMBER: float = 0.6
_TAU_BOOLEAN: float = 0.5
_TAU_STRING: float = 0.1
_TAU_NONE: float = 0.0


@dataclass(frozen=True, slots=True)
class GleanIndex:
    """Paired lexical and morphological indices over one segment list.

    Attributes:
        segments: The indexed segments (positionally aligned across both indices).
        lexical: BMX inverted index for the lexical signal.
        morphology: Per-segment typed-surface-form features.
    """

    segments: list[Segment]
    lexical: BMXIndex
    morphology: MorphologyIndex


@dataclass(frozen=True, slots=True)
class _FieldProbe:
    """Pre-computed, segment-independent morphological artifacts for one field.

    Built once per :func:`glean_rescore` call and reused across every segment.

    Attributes:
        label_terms: Folded name/description tokens used for LMC label positions.
        type_classes: Expected value shapes (e.g. ``("number",)``).
        enum_folded: Folded enum value strings for literal membership tests.
        pattern: Compiled ``format``/``pattern`` regex, or ``None``.
        tau_d: Type-distinctiveness weight in ``[0, 1]``.
    """

    label_terms: tuple[str, ...]
    type_classes: tuple[str, ...]
    enum_folded: tuple[str, ...]
    pattern: re.Pattern[str] | None
    tau_d: float


def build_glean_index(segments: list[Segment]) -> GleanIndex:
    """Build the paired BMX + morphology index over *segments*.

    Args:
        segments: Document segments to index.

    Returns:
        A :class:`GleanIndex` ready for :func:`glean_rescore`.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> idx = build_glean_index([
        ...     Segment(text="enrolled 4591 patients", start=0, end=22,
        ...             segment_type="unstructured", segment_id=0),
        ... ])
        >>> idx.morphology.segments[0].type_positions["number"]
        [1]
    """
    return GleanIndex(
        segments=segments,
        lexical=build_bmx_index(segments),
        morphology=build_morphology_index(segments),
    )


def glean_rescore(
    index: GleanIndex,
    fields: list[Field],
    query: str,
    *,
    top_k: int,
) -> list[tuple[Segment, float]]:
    """Rank segments for *fields* by fusing BMX with schema-typed morphology.

    Args:
        index: A :class:`GleanIndex` from :func:`build_glean_index`.
        fields: The group's fields (supply the morphological/type signal).
        query: The group's lexical retrieval query (same string BMX would use).
        top_k: Maximum number of results.

    Returns:
        ``(segment, fused_score)`` pairs sorted by score descending; empty when
        ``top_k <= 0`` or nothing matches. Reduces to the BMX ranking when the
        group carries no morphological evidence.

    Example:
        >>> from formatshield.schema._types import Field, Segment
        >>> segs = [
        ...     Segment(text="study enrollment was 4591 people", start=0, end=32,
        ...             segment_type="unstructured", segment_id=0),
        ...     Segment(text="enrollment criteria are described here", start=32,
        ...             end=70, segment_type="unstructured", segment_id=1),
        ... ]
        >>> idx = build_glean_index(segs)
        >>> f = Field(path="enrollment", type="integer", constraints={},
        ...           parent_path="", schema_node={"type": "integer",
        ...           "description": "enrollment count"})
        >>> glean_rescore(idx, [f], "enrollment count", top_k=1)[0][0].segment_id
        0
    """
    if top_k <= 0 or not index.segments:
        return []

    lex_scores = bmx_scores(index.lexical, query)
    rank_lex = _rank_map(lex_scores)

    probes = [p for p in (_build_probe(f) for f in fields) if _has_morphology(p)]
    morph_scores = _morph_scores(index.morphology, probes)
    max_morph = max(morph_scores.values(), default=0.0)

    # Lexical RRF base + bounded normalized morph boost. A chunk with no typed
    # evidence keeps its lexical RRF score, preserving non-typed order.
    fused: dict[int, float] = {}
    for doc in set(rank_lex) | set(morph_scores):
        score = 0.0
        if doc in rank_lex:
            score += 1.0 / (_RRF_K + rank_lex[doc])
        if max_morph > 0.0 and doc in morph_scores:
            score += _MORPH_BOOST * (morph_scores[doc] / max_morph)
        fused[doc] = score

    ranked = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [(index.segments[doc], score) for doc, score in ranked]


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _rank_map(scores: dict[int, float]) -> dict[int, int]:
    """Convert ``doc -> score`` into ``doc -> 1-based rank`` (best score = 1).

    Args:
        scores: Per-doc scores.

    Returns:
        Per-doc rank; empty input yields an empty map.
    """
    ordered = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    return {doc: rank for rank, (doc, _) in enumerate(ordered, start=1)}


def _morph_scores(
    morphology: MorphologyIndex,
    probes: list[_FieldProbe],
) -> dict[int, float]:
    """Per-doc morphological score: ``sum_f tau_d(f) * M(doc, f)`` over *probes*.

    Args:
        morphology: The morphology index.
        probes: Pre-built field probes with morphological evidence.

    Returns:
        ``doc_index -> score`` for docs with positive evidence; empty when no
        probe has morphological signal (preserving the BMX degradation path).
    """
    if not probes:
        return {}
    scores: dict[int, float] = {}
    for doc, seg in enumerate(morphology.segments):
        total = 0.0
        for probe in probes:
            evidence = _field_evidence(seg, probe)
            if evidence > 0.0:
                total += probe.tau_d * evidence
        if total > 0.0:
            scores[doc] = total
    return scores


def _field_evidence(seg: MorphologySegment, probe: _FieldProbe) -> float:
    """Morphological evidence ``M(doc, f)`` in ``[0, 1]`` for one field in one seg.

    The strongest of: a literal enum hit, a format/pattern match, or
    label↔value proximity (LMC). ``max`` so one decisive signal saturates the
    score instead of being diluted.

    Args:
        seg: The segment's morphological features.
        probe: The field's pre-built morphological artifacts.

    Returns:
        Evidence strength in ``[0, 1]``.
    """
    if probe.enum_folded and any(v in seg.folded_text for v in probe.enum_folded):
        return 1.0
    if probe.pattern is not None and probe.pattern.search(seg.folded_text):
        return 1.0
    return _lmc(seg, probe)


def _lmc(seg: MorphologySegment, probe: _FieldProbe) -> float:
    """Lexical-morphological co-location: ``exp(-gap/sigma)`` for *probe* in *seg*.

    Args:
        seg: The segment's morphological features.
        probe: The field probe (label terms + expected type-classes).

    Returns:
        Proximity score in ``[0, 1]``; ``0.0`` when the label or a typed token is
        absent from the segment.
    """
    label_positions = _merge_positions(seg.token_positions, probe.label_terms)
    if not label_positions:
        return 0.0
    type_positions = _merge_positions(seg.type_positions, probe.type_classes)
    if not type_positions:
        return 0.0
    gap = nearest_gap(label_positions, type_positions)
    if gap is None:
        return 0.0
    return math.exp(-gap / _LMC_SIGMA)


def _merge_positions(table: dict[str, list[int]], keys: tuple[str, ...]) -> list[int]:
    """Sorted union of the position lists in *table* for *keys*.

    Args:
        table: ``key -> ascending token indices``.
        keys: Keys to union (terms or type-classes).

    Returns:
        Ascending, de-duplicated token indices (possibly empty).
    """
    merged: set[int] = set()
    for key in keys:
        positions = table.get(key)
        if positions:
            merged.update(positions)
    return sorted(merged)


def _build_probe(field: Field) -> _FieldProbe:
    """Pre-compute a field's morphological artifacts once for all segments.

    Args:
        field: The schema field.

    Returns:
        The field's :class:`_FieldProbe`.
    """
    return _FieldProbe(
        label_terms=_label_terms(field),
        type_classes=field_type_classes(field),
        enum_folded=_enum_values(field),
        pattern=_compile_pattern(field),
        tau_d=_type_distinctiveness(field),
    )


def _has_morphology(probe: _FieldProbe) -> bool:
    """True when a probe can produce non-zero evidence for some segment.

    Args:
        probe: The field probe.

    Returns:
        Whether the field carries any morphological signal.
    """
    return bool(probe.enum_folded or probe.pattern or probe.type_classes)


def _label_terms(field: Field) -> tuple[str, ...]:
    """Folded label tokens for a field: path fragments + leading description words.

    Args:
        field: The schema field.

    Returns:
        Order-preserving, de-duplicated folded tokens.
    """
    raw = field.path.replace(".", " ")
    desc = field.schema_node.get("description", "")
    if isinstance(desc, str) and desc:
        raw = f"{raw} {' '.join(desc.split()[:_FIELD_QUERY_MAX_DESC_WORDS])}"
    return tuple(dict.fromkeys(tokenize(raw)))


def _enum_values(field: Field) -> tuple[str, ...]:
    """Folded, non-empty enum value strings for a field, if it enumerates values.

    Args:
        field: The schema field.

    Returns:
        Folded enum strings (possibly empty).
    """
    enum = field.schema_node.get("enum")
    if enum is None:
        enum = field.constraints.get("enum")
    if not isinstance(enum, list):
        return ()
    folded = (fold_diacritics(str(v).lower()).strip() for v in enum)
    return tuple(v for v in folded if v)


def _compile_pattern(field: Field) -> re.Pattern[str] | None:
    """Compile a field's ``pattern`` constraint against folded text, if valid.

    Args:
        field: The schema field.

    Returns:
        The compiled regex, or ``None`` when absent or invalid.
    """
    pattern = field.schema_node.get("pattern") or field.constraints.get("pattern")
    if not isinstance(pattern, str) or not pattern:
        return None
    try:
        return re.compile(fold_diacritics(pattern.lower()))
    except re.error:
        return None


def _type_distinctiveness(field: Field) -> float:
    """Type-distinctiveness ``tau_d`` for a field, from its schema type/constraints.

    Args:
        field: The schema field.

    Returns:
        Weight in ``[0, 1]``: enum 1.0, format/pattern 0.8, number 0.6,
        boolean 0.5, plain string 0.1, container 0.0.
    """
    if _enum_values(field):
        return _TAU_ENUM
    has_format = bool(field.schema_node.get("format") or field.constraints.get("format"))
    has_pattern = bool(field.schema_node.get("pattern") or field.constraints.get("pattern"))
    if has_format or has_pattern:
        return _TAU_FORMAT
    if field.type in {"integer", "number"}:
        return _TAU_NUMBER
    if field.type == "boolean":
        return _TAU_BOOLEAN
    if field.type == "string":
        return _TAU_STRING
    return _TAU_NONE
