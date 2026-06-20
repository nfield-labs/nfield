"""Structural slicing: feed each leaf only its own record's document section.

When a schema repeats one shape across many sibling keys (44 ``patient_record_k``,
30 ``subsidiary_k``, …) and the document repeats a matching block that many times,
each leaf only needs its own records' blocks — not the whole document. This turns
the per-leaf excerpt cost from O(D) (whole doc) into O(block), so total input
drops from O(K·D) to O(D + K·header).

The map is built by ORDER, not by word similarity: record ``r`` aligns to the
``r``-th repeating block. The ordinal is exact, so identical records (which defeat
lexical retrieval) are still separated correctly.

Scope: this only overwrites ``leaf.document_excerpt`` after packing and Stage 3.
It never changes K (call count), grouping, or chunking. If the structure is not
unambiguous it is a no-op and the leaf keeps its Stage 3 excerpt — so a normal
heterogeneous document behaves exactly as before.

Domain-agnostic throughout: detection is by repetition and schema shape only,
never by field names or domain words.

Routing a query by document structure rather than a lexical guess follows
structure-aware RAG (RDR2, arXiv:2510.04293); the block -> chunk parent-child
descent follows small-to-big / hierarchical retrieval (arXiv:2510.13217).
"""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import TYPE_CHECKING, NamedTuple

from formatshield.retrieval._chunker import chunk_document
from formatshield.schema._types import Segment

if TYPE_CHECKING:
    from formatshield.schema._types import Field

__all__ = [
    "RecordSegments",
    "detect_blocks",
    "detect_record_axis",
    "group_record_ordinal",
    "record_segments",
]


class RecordSegments(NamedTuple):
    """Parent-child structure of a record document.

    Attributes:
        field_ordinal: field path -> record index.
        block_tokens: record index -> block token cost (for record-aware packing).
        segments: all child segments (header + every record's chunks).
        by_record: record index -> its child segments (the within-record candidates).
        header_segments: the shared header's child segments (every leaf may use them).
    """

    field_ordinal: dict[str, int]
    block_tokens: dict[int, int]
    segments: list[Segment]
    by_record: dict[int, list[Segment]]
    header_segments: list[Segment]


# A repeated shape is only a "record axis" with at least this many siblings.
# Below this, repetition is too weak to trust order-alignment; fall back to the
# whole-document excerpt (accurate, just not sliced).
_MIN_RECORD_BLOCKS: int = 4
# A record axis must hold a MAJORITY of the schema's fields — i.e. the document is
# primarily a list of these records. Otherwise the "axis" is a minor nested list
# (e.g. a study's 12 outcomes = 12% of fields), and slicing by it would strand the
# other 88%. Below this fraction, treat it as not a record document.
_MIN_AXIS_DOMINANCE: float = 0.5
# Block boundaries must be roughly evenly spaced: a per-record line recurs once
# per block, so its gaps cluster around D/R. We accept a candidate only if the
# gap standard deviation is under this fraction of the mean gap (coefficient of
# variation), which rejects lines that happen to recur R times by coincidence.
_MAX_SPACING_CV: float = 0.5
_DIGIT_RUN: re.Pattern[str] = re.compile(r"\d+")
_WHITESPACE_RUN: re.Pattern[str] = re.compile(r"\s+")
# A run of words (letters joined by name punctuation), collapsed to one token so a
# record header keeps the same SHAPE while its identifier (a name, code, title)
# varies. Masking only digits is not enough: a header like "SUBSIDIARY 4 OF 30 --
# Brookstone Dynamics Corp." carries a unique name, so the digit-masked line never
# recurs and the true boundary is missed. Shape-masking lets all R headers share one
# signature regardless of the name's word count or trailing punctuation.
_NAME_RUN: re.Pattern[str] = re.compile(r"[a-z][a-z .,'&/-]*[a-z]\.?")
# A record header carries the record's identity, so its text differs across the R
# occurrences (distinct count ~ R); a constant divider or section label repeats
# verbatim (distinct count = 1). The two are far apart, so any threshold in
# (1/R, 1) separates them; 0.5 sits in that gap with a wide margin on both sides.
_MIN_IDENTIFIER_DISTINCT_FRAC: float = 0.5


def detect_record_axis(fields: list[Field]) -> tuple[dict[str, int], int] | None:
    """Find the schema level that repeats one shape across many siblings.

    Scans every parent prefix of every field path; a prefix is a record axis if
    it has >= ``_MIN_RECORD_BLOCKS`` direct children whose sub-shapes are
    identical (so the children are records, not distinct leaves). The axis with the
    most children wins, but only if its records hold a majority of fields
    (``_MIN_AXIS_DOMINANCE``) — a minor nested list is not the document's structure.
    Each record is numbered by first appearance, which is its document order.

    Args:
        fields: All flattened schema fields (each with a dot-notation ``path``).

    Returns:
        ``(field_path -> record ordinal, record count)`` for fields under the
        axis, or ``None`` if no clear record axis exists.

    Example:
        >>> detect_record_axis([])
        >>>
    """
    # parent prefix (tuple) -> child segment -> set of relative sub-paths.
    child_shapes: dict[tuple[str, ...], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    first_seen: dict[tuple[tuple[str, ...], str], int] = {}
    order = 0
    for f in fields:
        segs = f.path.split(".")
        for i in range(len(segs) - 1):
            parent = tuple(segs[:i])
            child = segs[i]
            child_shapes[parent][child].add(".".join(segs[i + 1 :]))
            key = (parent, child)
            if key not in first_seen:
                first_seen[key] = order
                order += 1

    best_parent: tuple[str, ...] | None = None
    best_children: dict[str, set[str]] = {}
    for parent, children in child_shapes.items():
        if len(children) < _MIN_RECORD_BLOCKS:
            continue
        shapes = list(children.values())
        # Records share an identical sub-shape; a non-empty shape rules out plain
        # leaf siblings. Strictness only forfeits the optimization (safe no-op),
        # never accuracy.
        if not shapes[0] or any(s != shapes[0] for s in shapes[1:]):
            continue
        if len(children) > len(best_children):
            best_parent, best_children = parent, children

    if best_parent is None:
        return None

    ordered = sorted(best_children, key=lambda c: first_seen[(best_parent, c)])
    ordinal = {child: i for i, child in enumerate(ordered)}
    depth = len(best_parent)
    field_ordinal: dict[str, int] = {}
    for f in fields:
        segs = f.path.split(".")
        if len(segs) > depth and tuple(segs[:depth]) == best_parent and segs[depth] in ordinal:
            field_ordinal[f.path] = ordinal[segs[depth]]
    # Dominance guard: the records must cover most fields, else this is a minor
    # nested list, not the document's record structure — don't slice by it.
    if fields and len(field_ordinal) / len(fields) < _MIN_AXIS_DOMINANCE:
        return None
    return field_ordinal, len(best_children)


def detect_blocks(document: str, count: int) -> tuple[str, list[str]] | None:
    """Split the document into ``count`` blocks at a per-record boundary line.

    Looks for a line shape (digits masked) that recurs exactly ``count`` times
    and is evenly spaced; its occurrences are the block starts. Text before the
    first occurrence is the shared header. Tying the line count to the schema's
    record count is the cross-check that the document really has that structure.

    Args:
        document: Raw document text.
        count: Record count from :func:`detect_record_axis`.

    Returns:
        ``(header, [block, ...])`` with ``len(blocks) == count``, or ``None`` if
        no evenly-spaced line recurs exactly ``count`` times.

    Example:
        >>> detect_blocks("", 4)
        >>>
    """
    starts = _block_starts(document, count)
    if starts is None:
        return None
    header = document[: starts[0]]
    bounds = [*starts, len(document)]
    blocks = [document[bounds[i] : bounds[i + 1]] for i in range(count)]
    return header, blocks


def _block_starts(document: str, count: int) -> list[int] | None:
    """Character offsets of the ``count`` per-record boundary lines, or ``None``.

    A line *shape* (digits and word-runs masked) that recurs exactly ``count`` times
    and is evenly spaced marks the record boundaries. Among such families, the record
    header is the one whose raw text VARIES across occurrences — it carries each
    record's identity — so it is preferred over a constant divider or section label
    that also recurs. Cutting at the header makes every block lead with its own
    identity, which is what lets the model bind values to the right record. When no
    family varies (records delimited by a constant marker), the earliest periodic
    family is used.

    Args:
        document: Raw document text.
        count: Record count from :func:`detect_record_axis`.

    Returns:
        Sorted boundary offsets, or ``None`` when no such line exists.
    """
    if count < _MIN_RECORD_BLOCKS:
        return None
    positions: dict[str, list[int]] = defaultdict(list)
    distinct: dict[str, set[str]] = defaultdict(set)
    pos = 0
    for line in document.splitlines(keepends=True):
        stripped = line.strip()
        sig = _block_signature(stripped)
        if sig:
            positions[sig].append(pos)
            distinct[sig].add(stripped)
        pos += len(line)
    even = [
        sig
        for sig, starts in positions.items()
        if len(starts) == count and _spacing_cv(starts) < _MAX_SPACING_CV
    ]
    if not even:
        return None
    # Prefer the identifier header (text varies across records) over a constant
    # divider/label; fall back to all periodic families when none varies.
    threshold = max(2, int(count * _MIN_IDENTIFIER_DISTINCT_FRAC))
    identifiers = [sig for sig in even if len(distinct[sig]) >= threshold]
    pool = identifiers or even
    best = min(pool, key=lambda sig: positions[sig][0])
    return sorted(positions[best])


def record_segments(
    fields: list[Field],
    document: str,
    chars_per_token: float,
    c_usable: float,
) -> RecordSegments | None:
    """Chunk each record block into child segments (parent-child retrieval).

    The record block is the *parent* (structure routes a leaf to it); its chunks are
    the *children* (retrieval ranks within it). A block that fits a leaf whole stays
    ONE segment — chunking it would split a value across a child boundary and lose it
    for no benefit (the whole block is kept anyway). Only an oversized block (cannot
    fit ``c_usable``) is chunked, so within-record retrieval can trim it to budget.
    Child offsets are absolute document positions so Stage 3 reorders coherently.
    Parent-child / small-to-big retrieval (HiREC; RDR2 arXiv:2510.04293).

    Args:
        fields: All flattened schema fields.
        document: Raw document text.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).
        c_usable: Usable input budget in tokens; a block under it stays whole.

    Returns:
        A :class:`RecordSegments`, or ``None`` when no aligned record structure exists.

    Example:
        >>> record_segments([], "", 4.0, 4096.0)
        >>>
    """
    axis = detect_record_axis(fields)
    if axis is None:
        return None
    field_ordinal, count = axis
    starts = _block_starts(document, count)
    if starts is None:
        return None

    cpt = chars_per_token if chars_per_token > 0 else 4.0
    bounds = [*starts, len(document)]
    segments: list[Segment] = []
    by_record: dict[int, list[Segment]] = {}
    counter = [0]

    def add_span(text: str, base: int) -> list[Segment]:
        # Whole block if it fits a leaf; chunk only an oversized block (so a value is
        # never split across children unless the block could not be kept whole anyway).
        block_tokens = math.ceil(len(text) / cpt)
        parts = chunk_document(text) if block_tokens > c_usable else None
        out: list[Segment] = []
        if parts is None:
            seg = Segment(
                text=text,
                start=base,
                end=base + len(text),
                segment_type="unstructured",
                segment_id=counter[0],
            )
            counter[0] += 1
            out.append(seg)
            segments.append(seg)
            return out
        for s in parts:
            seg = Segment(
                text=s.text,
                start=base + s.start,
                end=base + s.end,
                segment_type=s.segment_type,
                segment_id=counter[0],
            )
            counter[0] += 1
            out.append(seg)
            segments.append(seg)
        return out

    header_segments = add_span(document[: starts[0]], 0)
    block_tokens: dict[int, int] = {}
    for r in range(count):
        block_text = document[bounds[r] : bounds[r + 1]]
        by_record[r] = add_span(block_text, bounds[r])
        block_tokens[r] = max(1, math.ceil(len(block_text) / cpt))

    return RecordSegments(field_ordinal, block_tokens, segments, by_record, header_segments)


def group_record_ordinal(field_paths: list[str], field_ordinal: dict[str, int]) -> int:
    """Record index a group belongs to, or ``-1`` for shared/global fields.

    A group shares one ``parent_path``, so all its record-bearing fields map to the
    same record; the first such field decides it.

    Args:
        field_paths: The group's field paths.
        field_ordinal: ``field path -> record index`` from :func:`detect_record_axis`.

    Returns:
        The record index, or ``-1`` when the group has no record-bearing field.
    """
    for path in field_paths:
        if path in field_ordinal:
            return field_ordinal[path]
    return -1


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _block_signature(line: str) -> str:
    """Coarse SHAPE of a line: digit runs and word-runs masked, whitespace collapsed.

    Two record headers that differ only in their identifier (``RECORD 1 -- Acme`` vs
    ``RECORD 2 -- Globex``) collapse to the same signature, so the header family is
    detectable even though no two header lines are textually identical. Digit-only
    masking cannot do this — the name survives and the header never recurs.

    Args:
        line: A single (already stripped) document line.

    Returns:
        Normalized signature; empty for blank lines.
    """
    masked = _NAME_RUN.sub("A", _DIGIT_RUN.sub("0", line.lower()))
    return _WHITESPACE_RUN.sub(" ", masked).strip()


def _spacing_cv(starts: list[int]) -> float:
    """Coefficient of variation of the gaps between consecutive positions.

    Args:
        starts: Sorted occurrence offsets.

    Returns:
        ``std(gaps) / mean(gaps)``; 0.0 for fewer than two gaps.
    """
    if len(starts) < 2:
        return 0.0
    gaps = [starts[i + 1] - starts[i] for i in range(len(starts) - 1)]
    mean = sum(gaps) / len(gaps)
    if mean <= 0:
        return float("inf")
    var = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return float(var**0.5) / mean
