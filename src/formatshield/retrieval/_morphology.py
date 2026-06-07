"""Schema-typed morphological evidence index (Stage 2.5).

Records, per segment, where typed surface forms occur: number/date/email/uri/uuid/
boolean tokens by type-class, every term's token positions, and the accent-folded
text for literal enum/pattern tests. The GLEAN scorer (:mod:`_glean`) uses these to
score label↔value co-location and typed-value hits.

Type-classes derive only from JSON Schema ``type``/``format``, not domain vocabulary.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from formatshield.retrieval._tokenize import fold_diacritics

if TYPE_CHECKING:
    from formatshield.schema._types import Field, Segment

__all__ = [
    "TYPE_BOOLEAN",
    "TYPE_DATE",
    "TYPE_EMAIL",
    "TYPE_NUMBER",
    "TYPE_URI",
    "TYPE_UUID",
    "MorphologyIndex",
    "MorphologySegment",
    "build_morphology_index",
    "field_type_classes",
    "nearest_gap",
]

# ---------------------------------------------------------------------------
# Type-class identifiers (a field's expected value shape)
# ---------------------------------------------------------------------------
TYPE_NUMBER: str = "number"
TYPE_DATE: str = "date"
TYPE_EMAIL: str = "email"
TYPE_URI: str = "uri"
TYPE_UUID: str = "uuid"
TYPE_BOOLEAN: str = "boolean"

# Word-token pattern (matches the lexical tokenizer: runs of word characters).
_WORD_RE = re.compile(r"\w+")

# Tokens counted as boolean evidence (folded, lowercase).
_BOOLEAN_TOKENS: frozenset[str] = frozenset({"true", "false", "yes", "no", "y", "n"})

# Surface-form detectors over folded text; matched spans map to token positions.
_DATE_RE = re.compile(
    r"\b("
    r"\d{4}-\d{1,2}-\d{1,2}"  # 2020-04-13
    r"|\d{1,2}/\d{1,2}/\d{2,4}"  # 13/04/2020
    r"|\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{2,4}"
    r"|(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2},?\s+\d{2,4}"
    r")\b"
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URI_RE = re.compile(r"https?://\S+|\bwww\.\S+")
_UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")

# Span-detector classes (regex on text). Token-predicate classes (number,
# boolean) are handled inline while iterating tokens.
_SPAN_DETECTORS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (TYPE_DATE, _DATE_RE),
    (TYPE_EMAIL, _EMAIL_RE),
    (TYPE_URI, _URI_RE),
    (TYPE_UUID, _UUID_RE),
)

# JSON Schema ``format`` values mapped to their type-class.
_FORMAT_TO_CLASS: dict[str, str] = {
    "date": TYPE_DATE,
    "date-time": TYPE_DATE,
    "datetime": TYPE_DATE,
    "time": TYPE_DATE,
    "email": TYPE_EMAIL,
    "idn-email": TYPE_EMAIL,
    "uri": TYPE_URI,
    "url": TYPE_URI,
    "iri": TYPE_URI,
    "uri-reference": TYPE_URI,
    "uuid": TYPE_UUID,
}


@dataclass(frozen=True, slots=True)
class MorphologySegment:
    """Per-segment morphological features for GLEAN scoring.

    Attributes:
        token_positions: ``folded_term -> [token_index, ...]`` (ascending), the
            location of every term in the segment, for label proximity.
        type_positions: ``type_class -> [token_index, ...]`` (ascending), the
            location of every typed token, for value proximity.
        folded_text: Accent-folded, lowercased segment text, for literal enum and
            pattern membership tests.
        n_tokens: Number of word tokens in the segment.
    """

    token_positions: dict[str, list[int]]
    type_positions: dict[str, list[int]]
    folded_text: str
    n_tokens: int


@dataclass(frozen=True, slots=True)
class MorphologyIndex:
    """Morphological features for every segment, aligned with the BMX doc index.

    Attributes:
        segments: One :class:`MorphologySegment` per document segment, positionally
            aligned with the segments passed to :func:`build_morphology_index` (and
            therefore with the BMX index built over the same list).
    """

    segments: list[MorphologySegment]


def build_morphology_index(segments: list[Segment]) -> MorphologyIndex:
    """Index the typed surface forms in *segments* (regex + counting, no model).

    Args:
        segments: Document segments to index (same list given to the BMX index).

    Returns:
        A :class:`MorphologyIndex` positionally aligned with *segments*.

    Example:
        >>> from formatshield.schema._types import Segment
        >>> idx = build_morphology_index([
        ...     Segment(text="enrolled 4591 patients", start=0, end=22,
        ...             segment_type="unstructured", segment_id=0),
        ... ])
        >>> idx.segments[0].type_positions["number"]
        [1]
    """
    return MorphologyIndex(segments=[_index_segment(seg.text) for seg in segments])


def _index_segment(text: str) -> MorphologySegment:
    """Extract term positions, typed-token positions, and folded text for *text*.

    Args:
        text: Raw segment text.

    Returns:
        The segment's :class:`MorphologySegment`.
    """
    folded = fold_diacritics(text.lower())

    token_positions: dict[str, list[int]] = {}
    type_positions: dict[str, list[int]] = {}
    token_spans: list[tuple[int, int]] = []

    for idx, match in enumerate(_WORD_RE.finditer(folded)):
        token = match.group(0)
        token_spans.append(match.span())
        token_positions.setdefault(token, []).append(idx)
        if token.isdigit():
            type_positions.setdefault(TYPE_NUMBER, []).append(idx)
        if token in _BOOLEAN_TOKENS:
            type_positions.setdefault(TYPE_BOOLEAN, []).append(idx)

    for type_class, detector in _SPAN_DETECTORS:
        positions = _tokens_in_spans(token_spans, detector, folded)
        if positions:
            type_positions[type_class] = positions

    return MorphologySegment(
        token_positions=token_positions,
        type_positions=type_positions,
        folded_text=folded,
        n_tokens=len(token_spans),
    )


def _tokens_in_spans(
    token_spans: list[tuple[int, int]],
    detector: re.Pattern[str],
    folded: str,
) -> list[int]:
    """Token indices whose character span overlaps a detector match.

    Args:
        token_spans: ``(char_start, char_end)`` per token, in folded coordinates.
        detector: Compiled surface-form regex.
        folded: The folded/lowercased text the spans index into.

    Returns:
        Ascending token indices covered by at least one match (possibly empty).
    """
    match_spans = [m.span() for m in detector.finditer(folded)]
    if not match_spans:
        return []
    hits: list[int] = []
    for idx, (t_start, t_end) in enumerate(token_spans):
        for m_start, m_end in match_spans:
            if t_start < m_end and m_start < t_end:
                hits.append(idx)
                break
    return hits


def field_type_classes(field: Field) -> tuple[str, ...]:
    """Type-classes whose surface form a field's value is expected to take.

    Derived only from JSON Schema ``type`` and ``format`` — a ``format`` (date,
    email, uri, uuid) takes precedence for string fields; numeric and boolean
    types map directly. Containers and plain strings return ``()``.

    Args:
        field: The schema field.

    Returns:
        Zero or more type-class identifiers (e.g. ``("number",)``).

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field(path="n", type="integer", constraints={}, parent_path="",
        ...           schema_node={"type": "integer"})
        >>> field_type_classes(f)
        ('number',)
    """
    fmt = field.schema_node.get("format") or field.constraints.get("format") or ""
    fmt_class = _FORMAT_TO_CLASS.get(str(fmt).lower())
    if fmt_class is not None:
        return (fmt_class,)
    if field.type in {"integer", "number"}:
        return (TYPE_NUMBER,)
    if field.type == "boolean":
        return (TYPE_BOOLEAN,)
    return ()


def nearest_gap(positions_a: list[int], positions_b: list[int]) -> int | None:
    """Smallest absolute index distance between two ascending position lists.

    Used for label↔value proximity: how close a field's name term (``a``) sits to
    a typed token (``b``). Linear two-pointer over sorted inputs.

    Args:
        positions_a: Ascending token indices.
        positions_b: Ascending token indices.

    Returns:
        The minimum ``|i - j|`` over the two lists, or ``None`` if either is empty.

    Example:
        >>> nearest_gap([2, 9], [4, 10])
        1
        >>> nearest_gap([], [4]) is None
        True
    """
    if not positions_a or not positions_b:
        return None
    i = j = 0
    best = abs(positions_a[0] - positions_b[0])
    while i < len(positions_a) and j < len(positions_b):
        diff = positions_a[i] - positions_b[j]
        if abs(diff) < best:
            best = abs(diff)
        if diff == 0:
            return 0
        if diff < 0:
            i += 1
        else:
            j += 1
    return best
