"""Adaptive document chunking for retrieval in Stage 2.5.

Implements three chunking strategies: structured (heading-based), tabular
(row-based), and unstructured (fixed-size with overlap). Auto-detection
chooses the best strategy based on document patterns.
"""

from __future__ import annotations

import re

from formatshield.exceptions import SchemaError
from formatshield.schema._types import (
    SEGMENT_TYPE_STRUCTURED,
    SEGMENT_TYPE_TABULAR,
    SEGMENT_TYPE_UNSTRUCTURED,
    Segment,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Chunking targets a token budget, not a raw char count. Real frameworks
# (LangChain ~1k chars, LlamaIndex 1024 tokens) and the chunking literature
# converge on recursive, boundary-aware splitting around ~256 tokens rather than
# tiny fixed windows: semantic chunking is not worth the cost (arXiv:2410.13070),
# while structure/boundary awareness beats blind fixed-size (arXiv:2603.06976).
# The budget is approximated in chars via an average chars-per-token ratio so the
# chunker stays dependency-free (no tokenizer import in the hot path).
_APPROX_CHARS_PER_TOKEN: int = 4
_DEFAULT_TARGET_TOKENS: int = 256
_DEFAULT_OVERLAP_TOKENS: int = 32
_DEFAULT_CHUNK_SIZE: int = _DEFAULT_TARGET_TOKENS * _APPROX_CHARS_PER_TOKEN  # ~1024 chars
_DEFAULT_CHUNK_OVERLAP: int = _DEFAULT_OVERLAP_TOKENS * _APPROX_CHARS_PER_TOKEN  # ~128 chars
_MIN_SEGMENT_CHARS: int = 50
# Hard cap on a single segment (4x the base chunk). A heading/table section
# larger than this is sub-split so no segment spans an unbounded region, which
# would defeat lexical ranking. Engineering guard, not paper-derived.
_MAX_SEGMENT_CHARS: int = 4 * _DEFAULT_CHUNK_SIZE
# Split boundaries in descending priority: paragraph, line, sentence, clause,
# word. A chunk ends at the best available boundary at or before the size budget
# so a word/name is never cut across chunks (which previously hid tokens like
# "Denísov" from retrieval). If none is found, fall back to a hard split.
_BOUNDARY_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", "? ", "! ", "; ", ", ", " ")
_HEADING_PATTERN: re.Pattern[str] = re.compile(
    r"^(#{1,6}|={3,}|-{3,}|\*{3,})\s+(.+)$", re.MULTILINE
)
_TABLE_ROW_PATTERN: re.Pattern[str] = re.compile(r"^\s*\|[^\n]+\|", re.MULTILINE)


# ---------------------------------------------------------------------------
# Chunking strategy functions
# ---------------------------------------------------------------------------


def segment_structured(text: str) -> list[Segment]:
    """Split text on headings and structural markers.

    Detects headings (Markdown #, === underlines, --- separators) and
    sections. Ensures each segment contains complete sections without
    artificial breaks mid-paragraph.

    Falls back to returning entire text as single segment if no headings
    detected or all detected sections are below MIN_SEGMENT_CHARS threshold.

    Args:
        text: Document text to segment.

    Returns:
        List of Segment objects with segment_type="structured". If no
        structured sections found, returns entire text as one segment.

    Example:
        >>> intro = "# Introduction\\n" + "Background information about the topic. " * 2
        >>> chapter = "## Chapter 1\\n" + "Detailed discussion of the first matter. " * 2
        >>> segs = segment_structured(intro + "\\n" + chapter)
        >>> len(segs) >= 2
        True
        >>> segs[0].segment_type
        'structured'
    """
    segments: list[Segment] = []
    matches = list(_HEADING_PATTERN.finditer(text))

    if not matches:
        # No headings found; return entire text as one segment
        if len(text) >= _MIN_SEGMENT_CHARS:
            segments.append(
                Segment(
                    text=text,
                    start=0,
                    end=len(text),
                    segment_type=SEGMENT_TYPE_STRUCTURED,
                    segment_id=0,
                )
            )
        return segments

    # Split on headings
    segment_id = 0
    for i, match in enumerate(matches):
        heading_start = match.start()
        # Next heading or end of text
        next_heading_start = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        # Store original text (without strip) to maintain boundary contract:
        # text[seg.start:seg.end] == seg.text (always true)
        # Check stripped length to filter blank segments
        segment_text = text[heading_start:next_heading_start]
        if len(segment_text.strip()) >= _MIN_SEGMENT_CHARS:
            segments.append(
                Segment(
                    text=segment_text,
                    start=heading_start,
                    end=next_heading_start,
                    segment_type=SEGMENT_TYPE_STRUCTURED,
                    segment_id=segment_id,
                )
            )
            segment_id += 1

    return (
        segments
        if segments
        else [
            Segment(
                text=text,
                start=0,
                end=len(text),
                segment_type=SEGMENT_TYPE_STRUCTURED,
                segment_id=0,
            )
        ]
    )


def segment_tabular(text: str) -> list[Segment]:
    """Split text on table boundaries (row-based).

    Detects pipe-delimited tables (Markdown-style) and segments by
    contiguous table blocks. Falls back to unstructured chunking if no
    tables are detected.

    Args:
        text: Document text to segment.

    Returns:
        List of Segment objects with segment_type="tabular" for detected tables.
        If no tables found, falls back to unstructured chunking with default
        parameters and returns segments with segment_type="unstructured".

    Example:
        >>> text = "| Name | Age |\\n| John | 30 |"
        >>> segs = segment_tabular(text)
        >>> len(segs) >= 1
        True
    """
    segments: list[Segment] = []
    matches = list(_TABLE_ROW_PATTERN.finditer(text))

    if not matches:
        # No tables found; return unstructured fallback
        return segment_unstructured(text)

    segment_id = 0
    in_table = False
    table_start = 0

    for match in matches:
        if not in_table:
            in_table = True
            table_start = match.start()

        # Check if next line is still a table row (lookahead)
        # Handle edge case: if no newline found (EOF), table ends
        newline_idx = text.find("\n", match.end())

        if newline_idx < 0:
            # No newline after this row; we're at EOF
            # End the current table
            table_text = text[table_start:]
            if len(table_text.strip()) >= _MIN_SEGMENT_CHARS:
                segments.append(
                    Segment(
                        text=table_text,
                        start=table_start,
                        end=len(text),
                        segment_type=SEGMENT_TYPE_TABULAR,
                        segment_id=segment_id,
                    )
                )
                segment_id += 1
            in_table = False
        else:
            next_line_start = newline_idx + 1
            next_line_text = text[next_line_start:]
            if next_line_start < len(text) and not _TABLE_ROW_PATTERN.match(next_line_text):
                # Next line exists and is not a table row; end table
                table_text = text[table_start:next_line_start]
                if len(table_text.strip()) >= _MIN_SEGMENT_CHARS:
                    segments.append(
                        Segment(
                            text=table_text,
                            start=table_start,
                            end=next_line_start,
                            segment_type=SEGMENT_TYPE_TABULAR,
                            segment_id=segment_id,
                        )
                    )
                    segment_id += 1
                in_table = False

    # Final table if document ends mid-table (safety fallback)
    if in_table:
        table_text = text[table_start:]
        if len(table_text.strip()) >= _MIN_SEGMENT_CHARS:
            segments.append(
                Segment(
                    text=table_text,
                    start=table_start,
                    end=len(text),
                    segment_type=SEGMENT_TYPE_TABULAR,
                    segment_id=segment_id,
                )
            )

    return segments if segments else segment_unstructured(text)


def segment_unstructured(
    text: str,
    *,
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Segment]:
    """Split text into boundary-aware, overlapping chunks of about ``chunk_size``.

    Each chunk ends at the best available boundary (paragraph -> line -> sentence
    -> clause -> word) at or before ``chunk_size`` characters, so a word or name
    is never cut across two chunks. If no boundary is found within the budget the
    chunk is hard-split at ``chunk_size``. Consecutive chunks overlap by
    ``overlap`` characters. Every character of *text* is covered by at least one
    chunk (no content is dropped), and the offset contract
    ``text[seg.start:seg.end] == seg.text`` always holds.

    Args:
        text: Document text to segment.
        chunk_size: Target characters per chunk (must be > 0). The default is a
            token budget (~256 tokens) approximated in characters.
        overlap: Characters shared between consecutive chunks (>= 0 and < chunk_size).

    Returns:
        List of Segment objects with segment_type="unstructured".

    Raises:
        SchemaError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.

    Example:
        >>> text = "The cat sat on the mat. " * 100
        >>> segs = segment_unstructured(text, chunk_size=100, overlap=20)
        >>> len(segs) > 1
        True
        >>> all(s.text == text[s.start:s.end] for s in segs)  # offset contract holds
        True
    """
    # Validate parameters
    if chunk_size <= 0:
        raise SchemaError(
            f"chunk_size must be > 0, got {chunk_size}",
            hint="Use a positive integer for chunk_size parameter",
        )
    if overlap < 0:
        raise SchemaError(
            f"overlap must be >= 0, got {overlap}",
            hint="overlap must be non-negative",
        )
    if overlap >= chunk_size:
        raise SchemaError(
            f"overlap ({overlap}) must be < chunk_size ({chunk_size})",
            hint="overlap must be less than chunk_size to avoid infinite loops",
        )

    n = len(text)
    if n <= chunk_size:
        return [
            Segment(
                text=text,
                start=0,
                end=n,
                segment_type=SEGMENT_TYPE_UNSTRUCTURED,
                segment_id=0,
            )
        ]

    segments: list[Segment] = []
    segment_id = 0
    pos = 0

    while pos < n:
        target_end = min(pos + chunk_size, n)
        split_at = n if target_end >= n else _find_split_point(text, pos, target_end)

        chunk_text = text[pos:split_at]
        if chunk_text:
            segments.append(
                Segment(
                    text=chunk_text,
                    start=pos,
                    end=split_at,
                    segment_type=SEGMENT_TYPE_UNSTRUCTURED,
                    segment_id=segment_id,
                )
            )
            segment_id += 1

        if split_at >= n:
            break
        # Step back by overlap from the chosen boundary; +1 floor guarantees progress.
        pos = max(split_at - overlap, pos + 1)

    return segments


def _find_split_point(text: str, start: int, target_end: int) -> int:
    """Return the chunk end aligned to the boundary closest to the size budget.

    Considers every separator and splits just after the occurrence whose end is
    nearest ``target_end`` (the largest valid chunk that still ends on a boundary),
    keeping the separator with the left chunk so no character is lost. Taking the
    boundary closest to the budget — rather than the highest-priority separator —
    avoids tiny chunks when a paragraph/line break happens to fall early in the
    window (common in hard-wrapped text). Falls back to ``target_end`` (a hard
    split) when no boundary is found.

    Args:
        text: The full document text.
        start: Start offset of the current chunk (exclusive lower bound).
        target_end: Upper bound for the chunk end (the size budget).

    Returns:
        A split offset in ``(start, target_end]``.

    Example:
        >>> _find_split_point("the quick brown fox", 0, 12)
        10
    """
    best = -1
    for sep in _BOUNDARY_SEPARATORS:
        idx = text.rfind(sep, start, target_end)
        if idx != -1:
            candidate = idx + len(sep)
            if candidate > start:
                best = max(best, candidate)
    return best if best != -1 else target_end


# ---------------------------------------------------------------------------
# Auto-detection and main entry point
# ---------------------------------------------------------------------------


def _enforce_max_segment_size(segments: list[Segment]) -> list[Segment]:
    """Sub-split any segment longer than ``_MAX_SEGMENT_CHARS``.

    Structured and tabular strategies can emit very large segments when a
    single heading-delimited section (or table) spans a huge span of text.
    Such segments are useless for lexical retrieval and can overflow the model
    context. This post-pass splits each oversized segment into fixed-size
    sub-segments while preserving the global character-offset contract
    (``text[seg.start:seg.end] == seg.text``) and reassigning ``segment_id``
    sequentially across the whole result.

    Args:
        segments: Segments produced by a chunking strategy.

    Returns:
        Segments where every segment is at most ``_MAX_SEGMENT_CHARS`` chars,
        with contiguous ``segment_id`` values starting at 0.
    """
    result: list[Segment] = []
    next_id = 0
    for seg in segments:
        if len(seg.text) <= _MAX_SEGMENT_CHARS:
            result.append(
                Segment(
                    text=seg.text,
                    start=seg.start,
                    end=seg.end,
                    segment_type=seg.segment_type,
                    segment_id=next_id,
                )
            )
            next_id += 1
            continue

        # Oversized: sub-split the section text with fixed-size chunking,
        # then re-map local offsets back to global document offsets.
        sub_segments = segment_unstructured(seg.text)
        for sub in sub_segments:
            result.append(
                Segment(
                    text=sub.text,
                    start=seg.start + sub.start,
                    end=seg.start + sub.end,
                    segment_type=seg.segment_type,
                    segment_id=next_id,
                )
            )
            next_id += 1

    return result


def _detect_document_type(text: str) -> str:
    """Auto-detect document type: structured, tabular, or unstructured.

    Heuristics:
    - structured: >1 heading markers (## or underlines)
    - tabular: >2 pipe-delimited table rows
    - default: unstructured

    Args:
        text: Document text.

    Returns:
        Strategy name: "structured", "tabular", or "unstructured".
    """
    heading_count = len(list(_HEADING_PATTERN.finditer(text)))
    table_row_count = len(list(_TABLE_ROW_PATTERN.finditer(text)))

    if heading_count > 1:
        return "structured"
    if table_row_count > 2:
        return "tabular"
    return "unstructured"


def chunk_document(
    text: str,
    *,
    strategy: str = "auto",
    chunk_size: int = _DEFAULT_CHUNK_SIZE,
    overlap: int = _DEFAULT_CHUNK_OVERLAP,
) -> list[Segment]:
    """Chunk a document into segments for retrieval.

    Supports three strategies: structured (heading-based), tabular
    (row-based), unstructured (fixed-size). "auto" detects the best
    strategy from the document.

    Args:
        text: Document text to chunk (positional).

    Keyword-Only Args:
        strategy: Chunking strategy: "structured", "tabular", "unstructured",
            or "auto" to detect. Defaults to "auto".
        chunk_size: Target characters per segment (unstructured strategy only).
            Defaults to ``_DEFAULT_CHUNK_SIZE`` (~256 tokens ≈ 1024 chars).
        overlap: Character overlap between chunks (unstructured strategy only).
            Defaults to ``_DEFAULT_CHUNK_OVERLAP`` (~32 tokens ≈ 128 chars).

    Returns:
        List of Segment objects with boundaries and types set.

    Raises:
        SchemaError: If strategy is not recognized or parameters are invalid.

    Example:
        >>> doc = "# Section 1\\nContent here\\n## Subsection\\nMore content"
        >>> segs = chunk_document(doc, strategy="auto")
        >>> all(s.segment_type in {"structured", "tabular", "unstructured"} for s in segs)
        True
    """
    if not text or not text.strip():
        return []

    if strategy == "auto":
        strategy = _detect_document_type(text)

    if strategy == "structured":
        raw_segments = segment_structured(text)
    elif strategy == "tabular":
        raw_segments = segment_tabular(text)
    elif strategy == "unstructured":
        raw_segments = segment_unstructured(text, chunk_size=chunk_size, overlap=overlap)
    else:
        raise SchemaError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Must be 'structured', 'tabular', 'unstructured', or 'auto'.",
            hint="Check strategy parameter in chunk_document() call",
        )

    # Guarantee no segment exceeds the max size, regardless of strategy.
    return _enforce_max_segment_size(raw_segments)
