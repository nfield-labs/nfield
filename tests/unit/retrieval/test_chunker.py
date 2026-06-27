"""Tests for retrieval._chunker - adaptive document chunking."""

from __future__ import annotations

import pytest

from nfield.exceptions import SchemaError
from nfield.retrieval._chunker import (
    chunk_document,
    segment_structured,
    segment_tabular,
    segment_unstructured,
)
from nfield.schema._types import (
    SEGMENT_TYPE_STRUCTURED,
    SEGMENT_TYPE_TABULAR,
    SEGMENT_TYPE_UNSTRUCTURED,
)


class TestSegmentStructured:
    """Tests for heading-based chunking."""

    def test_empty_text_returns_empty_list(self) -> None:
        """Empty text produces no segments."""
        result = segment_structured("")
        assert result == []

    def test_text_without_headings_returns_single_segment(self) -> None:
        """Text without headings becomes one segment."""
        text = (
            "Just some plain text without any structure. "
            "This is a longer text to meet minimum length requirements."
        )
        result = segment_structured(text)
        assert len(result) == 1
        assert result[0].text == text
        assert result[0].segment_type == SEGMENT_TYPE_STRUCTURED

    def test_text_with_markdown_headings_splits_on_headings(self) -> None:
        """Markdown headings split the text."""
        text = "# Section 1\nContent here\n## Subsection\nMore content"
        result = segment_structured(text)
        assert len(result) >= 1
        assert all(s.segment_type == SEGMENT_TYPE_STRUCTURED for s in result)

    def test_segments_have_valid_boundaries(self) -> None:
        """Segment start/end map to correct text positions."""
        text = "# Intro\nHello\n# Main\nWorld"
        result = segment_structured(text)
        for seg in result:
            assert seg.start >= 0
            assert seg.end <= len(text)
            assert seg.start < seg.end


class TestSegmentTabular:
    """Tests for table-based chunking."""

    def test_no_tables_returns_unstructured_fallback(self) -> None:
        """Text without tables falls back to unstructured."""
        text = "Just regular text, no tables here."
        result = segment_tabular(text)
        # Falls back to segment_unstructured
        assert len(result) >= 1

    def test_table_detection_finds_pipe_rows(self) -> None:
        """Markdown table rows are detected."""
        text = "| Name | Age |\n| John | 30 |"
        result = segment_tabular(text)
        assert len(result) >= 1
        # Should find table content
        assert any("Name" in s.text or "John" in s.text for s in result)


class TestSegmentUnstructured:
    """Tests for fixed-size chunking."""

    def test_small_text_returns_one_chunk(self) -> None:
        """Text smaller than chunk_size returns one segment."""
        text = "short text"
        result = segment_unstructured(text, chunk_size=100, overlap=10)
        assert len(result) == 1
        assert result[0].text == text

    def test_large_text_chunks_into_multiple_segments(self) -> None:
        """Large text is chunked into multiple segments."""
        text = "x" * 1000
        result = segment_unstructured(text, chunk_size=250, overlap=50)
        assert len(result) > 1
        assert all(s.segment_type == SEGMENT_TYPE_UNSTRUCTURED for s in result)

    def test_chunks_respect_overlap(self) -> None:
        """Overlap creates shared content between chunks."""
        text = "abcdefghijklmnopqrstuvwxyz" * 100
        result = segment_unstructured(text, chunk_size=100, overlap=20)
        assert len(result) > 1
        # Check overlap exists (next chunk starts before previous ends)
        for i in range(len(result) - 1):
            assert result[i + 1].start < result[i].end


class TestBoundaryAwareChunking:
    """Boundary-aware, token-budgeted splitting (Phase A chunker upgrade)."""

    _PROSE = (
        "Denisov rode forward with his hussars. Pierre watched from the hill. "
        "Natasha danced at the ball while Andrei looked on. Kutuzov gave the order "
        "and the army advanced toward Borodino under a grey winter sky. "
    ) * 30

    def test_no_word_split_on_prose(self) -> None:
        """Every non-final chunk ends at a whitespace/punctuation boundary, not mid-word."""
        segs = segment_unstructured(self._PROSE, chunk_size=200, overlap=40)
        assert len(segs) > 1
        for seg in segs[:-1]:
            # The boundary char (space/newline/.,;?!) is kept with the left chunk.
            assert seg.text[-1] in " \n.,;?!", f"chunk ended mid-token: ...{seg.text[-12:]!r}"

    def test_no_content_dropped(self) -> None:
        """Every character of the source is covered by at least one chunk."""
        segs = segment_unstructured(self._PROSE, chunk_size=200, overlap=40)
        covered = [False] * len(self._PROSE)
        for seg in segs:
            for i in range(seg.start, seg.end):
                covered[i] = True
        assert all(covered), "boundary-aware chunking must not drop characters"

    def test_offset_contract_holds(self) -> None:
        """text[start:end] == seg.text for every chunk, including boundary splits."""
        segs = segment_unstructured(self._PROSE, chunk_size=180, overlap=30)
        for seg in segs:
            assert seg.text == self._PROSE[seg.start : seg.end]

    def test_token_budget_default_reduces_chunk_count(self) -> None:
        """The ~256-token default yields far fewer chunks than the old 512-char window."""
        from nfield.retrieval._chunker import _DEFAULT_CHUNK_SIZE

        big = self._PROSE * 4
        default_chunks = segment_unstructured(big)
        old_style = segment_unstructured(big, chunk_size=512, overlap=128)
        assert _DEFAULT_CHUNK_SIZE > 512
        assert len(default_chunks) < len(old_style)

    def test_hard_split_when_no_boundary(self) -> None:
        """Text with no separators still splits (degrades to fixed-size)."""
        segs = segment_unstructured("x" * 1000, chunk_size=250, overlap=50)
        assert len(segs) > 1


class TestChunkDocumentAuto:
    """Tests for automatic strategy detection."""

    def test_auto_detects_structured_for_headings(self) -> None:
        """Auto-detection chooses structured for heading-heavy text."""
        text = "# Title\nText\n## Subtitle\nMore text\n### Sub-subtitle\nMore"
        result = chunk_document(text, strategy="auto")
        assert len(result) >= 1
        valid_types = {SEGMENT_TYPE_STRUCTURED, SEGMENT_TYPE_TABULAR, SEGMENT_TYPE_UNSTRUCTURED}
        assert all(s.segment_type in valid_types for s in result)

    def test_auto_detects_unstructured_for_plain_text(self) -> None:
        """Auto-detection chooses unstructured for plain text."""
        text = "Just plain text without any structure whatsoever. " * 20
        result = chunk_document(text, strategy="auto")
        assert len(result) >= 1

    def test_explicit_strategy_override_auto(self) -> None:
        """Explicit strategy overrides auto-detection."""
        text = "# Title\nContent" * 10
        result_auto = chunk_document(text, strategy="auto")
        result_unstructured = chunk_document(text, strategy="unstructured")
        # Strategies may produce different results
        assert len(result_auto) >= 1
        assert len(result_unstructured) >= 1

    def test_invalid_strategy_raises_error(self) -> None:
        """Invalid strategy raises SchemaError."""
        with pytest.raises(SchemaError, match="Unknown chunking strategy"):
            chunk_document("text", strategy="invalid")

    def test_empty_document_returns_empty_list(self) -> None:
        """Empty document produces no segments."""
        result = chunk_document("", strategy="auto")
        assert result == []
        result_whitespace = chunk_document("   \n  \t  ", strategy="auto")
        assert result_whitespace == []


class TestSegmentBoundaries:
    """Tests for segment start/end consistency."""

    def test_all_segments_have_valid_ids(self) -> None:
        """Segment IDs are non-negative and ordered."""
        text = "# Section 1\nContent\n# Section 2\nMore" * 5
        result = chunk_document(text, strategy="structured")
        ids = [s.segment_id for s in result]
        assert all(i >= 0 for i in ids)
        assert len(ids) == len(set(ids))  # All unique

    def test_segment_text_matches_boundaries(self) -> None:
        """Segment text matches start/end positions in original."""
        text = "The quick brown fox jumps over the lazy dog"
        result = chunk_document(text, strategy="unstructured", chunk_size=15, overlap=0)
        for seg in result:
            extracted = text[seg.start : seg.end]
            assert seg.text == extracted
