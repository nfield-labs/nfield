"""save_html: grounded spans rendered as a standalone, correctly-escaped HTML page."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from pathlib import Path

from nfield.types import ExtractionResult, ExtractionStatus, Metadata
from nfield.viz import save_html

_DOC = "Vendor: Acme <Corp>. Total: 1284.50 USD."


def _result(provenance: dict[str, list[int]] | None) -> ExtractionResult:
    meta = Metadata(
        K=1,
        K_min=1,
        optimality_gap=0.0,
        quality_score=1.0,
        confidence_level="HIGH",
        fields_extracted=2,
        fields_total=2,
        fields_missing=0,
        fields_conflicted=0,
        fields_needs_revalidation=0,
        per_field_confidence={},
        retry_rounds=0,
    )
    return ExtractionResult(
        data={}, metadata=meta, status=ExtractionStatus.SUCCESS, provenance=provenance
    )


class TestSaveHtml:
    def test_requires_provenance(self) -> None:
        with pytest.raises(ValueError, match="provenance"):
            save_html(_result(None), _DOC)

    def test_highlights_each_span_with_its_field(self) -> None:
        page = save_html(_result({"vendor": [8, 19], "total": [28, 35]}), _DOC)
        assert '<mark title="vendor">Acme &lt;Corp&gt;</mark>' in page
        assert '<mark title="total">1284.50</mark>' in page

    def test_document_html_is_escaped(self) -> None:
        page = save_html(_result({"total": [28, 35]}), _DOC)
        # The raw angle brackets from the document never reach the page unescaped.
        assert "<Corp>" not in page
        assert "&lt;Corp&gt;" in page

    def test_field_name_is_escaped_in_the_title(self) -> None:
        page = save_html(_result({'x"><script>': [0, 6]}), _DOC)
        assert "<script>" not in page

    def test_overlapping_span_is_skipped_in_text_but_listed(self) -> None:
        page = save_html(_result({"outer": [8, 19], "inner": [10, 14]}), _DOC)
        assert page.count("<mark") == 1  # only the first span is marked
        assert "<td>inner</td>" in page  # but the table still lists it

    def test_out_of_range_span_is_ignored(self) -> None:
        page = save_html(_result({"bad": [0, len(_DOC) + 50]}), _DOC)
        assert "<mark" not in page

    def test_writes_the_file_when_a_path_is_given(self, tmp_path: Path) -> None:
        out = tmp_path / "extraction.html"
        page = save_html(_result({"total": [28, 35]}), _DOC, out)
        assert out.read_text(encoding="utf-8") == page

    def test_summary_reports_extraction_counts(self) -> None:
        page = save_html(_result({"total": [28, 35]}), _DOC)
        assert "2 of 2 fields extracted" in page
        assert "1 located in the source" in page

    def test_importable_from_the_top_level(self) -> None:
        import nfield

        assert nfield.save_html is save_html

    def test_empty_provenance_renders_a_plain_page(self) -> None:
        page = save_html(_result({}), _DOC)
        assert "<mark" not in page
        assert "&lt;Corp&gt;" in page  # document still shown, escaped

    def test_adjacent_spans_are_both_marked(self) -> None:
        # [8, 12) and [12, 19) touch but do not overlap.
        page = save_html(_result({"first": [8, 12], "second": [12, 19]}), _DOC)
        assert page.count("<mark") == 2

    def test_zero_length_span_is_ignored(self) -> None:
        page = save_html(_result({"empty": [5, 5]}), _DOC)
        assert "<mark" not in page

    def test_malformed_offsets_are_dropped_not_crashed(self) -> None:
        # A hand-edited results file may carry floats or short lists; never crash.
        page = save_html(
            _result({"floats": [1.5, 4.5], "short": [3], "good": [28, 35]}),  # type: ignore[dict-item]
            _DOC,
        )
        assert page.count("<mark") == 1  # only the well-formed span renders

    def test_full_document_span_keeps_every_character(self) -> None:
        page = save_html(_result({"all": [0, len(_DOC)]}), _DOC)
        assert page.count("<mark") == 1
        # Nothing of the document is lost around the mark boundaries.
        assert "Vendor: Acme" in page.replace('<mark title="all">', "")
