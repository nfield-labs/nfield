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

_DEFAULT_CHUNK_SIZE: int = 512
_DEFAULT_CHUNK_OVERLAP: int = 128
_MIN_SEGMENT_CHARS: int = 50
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
        >>> text = "# Introduction\\nSome text\\n## Chapter 1\\nMore text"
        >>> segs = segment_structured(text)
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
    """Split text into fixed-size overlapping chunks (MVP strategy).

    Implements simple fixed-size chunking with character-based overlap.
    Post-MVP: cosine-dissimilarity based splitting.

    Args:
        text: Document text to segment.
        chunk_size: Target characters per chunk (must be > 0).
        overlap: Characters to repeat at chunk boundaries (must be >= 0 and < chunk_size).

    Returns:
        List of Segment objects with segment_type="unstructured".

    Raises:
        ValueError: If chunk_size <= 0, overlap < 0, or overlap >= chunk_size.

    Example:
        >>> text = "a" * 1000
        >>> segs = segment_unstructured(text, chunk_size=250, overlap=50)
        >>> len(segs) > 1
        True
        >>> segs[0].segment_type
        'unstructured'
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

    if len(text) <= chunk_size:
        return [
            Segment(
                text=text,
                start=0,
                end=len(text),
                segment_type=SEGMENT_TYPE_UNSTRUCTURED,
                segment_id=0,
            )
        ]

    segments: list[Segment] = []
    segment_id = 0
    pos = 0

    while pos < len(text):
        chunk_end = min(pos + chunk_size, len(text))
        chunk_text = text[pos:chunk_end]

        if len(chunk_text) >= _MIN_SEGMENT_CHARS:
            segments.append(
                Segment(
                    text=chunk_text,
                    start=pos,
                    end=chunk_end,
                    segment_type=SEGMENT_TYPE_UNSTRUCTURED,
                    segment_id=segment_id,
                )
            )
            segment_id += 1

        # Advance by stride (chunk_size - overlap) from chunk start.
        # Always use original pos + stride, not chunk_end - overlap,
        # to avoid getting stuck when chunk_end == len(text).
        pos += chunk_size - overlap

    return segments


# ---------------------------------------------------------------------------
# Auto-detection and main entry point
# ---------------------------------------------------------------------------


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
            Defaults to 512.
        overlap: Character overlap between chunks (unstructured strategy only).
            Defaults to 128.

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
        return segment_structured(text)
    elif strategy == "tabular":
        return segment_tabular(text)
    elif strategy == "unstructured":
        return segment_unstructured(text, chunk_size=chunk_size, overlap=overlap)
    else:
        raise SchemaError(
            f"Unknown chunking strategy: {strategy!r}. "
            f"Must be 'structured', 'tabular', 'unstructured', or 'auto'.",
            hint="Check strategy parameter in chunk_document() call",
        )


# ---------------------------------------------------------------------------
# Post-MVP stubs
# ---------------------------------------------------------------------------


def resolve_coreferences(_text: str) -> str:
    """Resolve pronoun coreferences in text (post-MVP).

    Args:
        _text: Document text.

    Returns:
        Text with coreferences resolved (post-MVP feature).

    Raises:
        NotImplementedError: This is a post-MVP feature.
    """
    raise NotImplementedError("Coreference resolution is a post-MVP feature.")


def mark_continuity(_segments: list[Segment]) -> list[Segment]:
    """Mark cross-chunk continuity boundaries (post-MVP).

    Args:
        _segments: List of segments from chunking.

    Returns:
        Segments with continuity metadata (post-MVP feature).

    Raises:
        NotImplementedError: This is a post-MVP feature.
    """
    raise NotImplementedError("Continuity marking is a post-MVP feature.")
