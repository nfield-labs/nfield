"""Tests for structural slicing (pipeline/_structure.py)."""

from __future__ import annotations

from formatshield.pipeline._structure import (
    align_path_to_section,
    detect_blocks,
    detect_record_axis,
    detect_section_structure,
    group_record_ordinal,
    record_segments,
)
from formatshield.schema._types import Field


def _field(path: str) -> Field:
    return Field(path=path, type="string", constraints={}, parent_path="", schema_node={})


def _record_fields(n: int) -> list[Field]:
    """n records under ``recs``, each with the identical {name, age} sub-shape."""
    fields: list[Field] = []
    for i in range(1, n + 1):
        fields.append(_field(f"recs.rec_{i}.name"))
        fields.append(_field(f"recs.rec_{i}.age"))
    return fields


_DOC = (
    "HEADER LINE\n"
    "RECORD 1\nname: Ann\nage: 30\n"
    "RECORD 2\nname: Ben\nage: 41\n"
    "RECORD 3\nname: Cleo\nage: 52\n"
    "RECORD 4\nname: Dan\nage: 63\n"
)


class TestDetectRecordAxis:
    def test_finds_repeated_sibling_axis(self):
        result = detect_record_axis(_record_fields(4))
        assert result is not None
        field_ordinal, count = result
        assert count == 4
        assert field_ordinal["recs.rec_1.name"] == 0
        assert field_ordinal["recs.rec_4.age"] == 3

    def test_none_when_too_few_siblings(self):
        assert detect_record_axis(_record_fields(3)) is None

    def test_none_on_heterogeneous_schema(self):
        fields = [_field("a.x"), _field("b.y"), _field("c.z"), _field("d.w")]
        assert detect_record_axis(fields) is None

    def test_rejects_minor_nested_list_below_dominance(self):
        # A 5-item nested list (10 fields) inside a doc of 60 other fields: the list
        # is the biggest sibling group but only ~14% of fields → not a record doc.
        fields = [_field(f"outcomes.item_{i}.measure") for i in range(5)]
        fields += [_field(f"outcomes.item_{i}.value") for i in range(5)]
        fields += [_field(f"study.field_{j}") for j in range(60)]
        assert detect_record_axis(fields) is None

    def test_accepts_axis_above_dominance(self):
        # Same 5-item list but now it holds the majority of fields → real record doc.
        fields = [_field(f"recs.rec_{i}.a") for i in range(5)]
        fields += [_field(f"recs.rec_{i}.b") for i in range(5)]
        fields += [_field("study.title")]
        result = detect_record_axis(fields)
        assert result is not None and result[1] == 5

    def test_ordinal_follows_first_appearance(self):
        # Records out of numeric order still number by document/schema order.
        fields = [
            _field("recs.rec_3.name"),
            _field("recs.rec_3.age"),
            _field("recs.rec_1.name"),
            _field("recs.rec_1.age"),
            _field("recs.rec_2.name"),
            _field("recs.rec_2.age"),
            _field("recs.rec_4.name"),
            _field("recs.rec_4.age"),
        ]
        field_ordinal, _ = detect_record_axis(fields)  # type: ignore[misc]
        assert field_ordinal["recs.rec_3.name"] == 0
        assert field_ordinal["recs.rec_1.name"] == 1


class TestDetectBlocks:
    def test_splits_at_repeated_line(self):
        result = detect_blocks(_DOC, 4)
        assert result is not None
        header, blocks = result
        assert "HEADER LINE" in header
        assert len(blocks) == 4
        assert "Ann" in blocks[0]
        assert "Dan" in blocks[3]
        # Each block carries its own record only.
        assert "Ben" not in blocks[0]

    def test_none_when_count_mismatch(self):
        assert detect_blocks(_DOC, 5) is None

    def test_none_below_min_blocks(self):
        assert detect_blocks(_DOC, 2) is None


class TestRecordSegments:
    def test_chunks_each_record_and_header(self):
        rs = record_segments(_record_fields(4), _DOC, 4.0, 4096.0)
        assert rs is not None
        assert set(rs.by_record) == {0, 1, 2, 3}
        assert rs.field_ordinal["recs.rec_1.name"] == 0
        assert set(rs.block_tokens) == {0, 1, 2, 3}
        # header child carries the header text.
        assert any("HEADER LINE" in s.text for s in rs.header_segments)

    def test_small_block_kept_whole(self):
        # Block fits c_usable → exactly one segment per record (no value split).
        rs = record_segments(_record_fields(4), _DOC, 4.0, 4096.0)
        assert all(len(segs) == 1 for segs in rs.by_record.values())  # type: ignore[union-attr]

    def test_children_are_record_local(self):
        rs = record_segments(_record_fields(4), _DOC, 4.0, 4096.0)
        rec1 = " ".join(s.text for s in rs.by_record[1])  # type: ignore[union-attr]
        assert "Ben" in rec1
        assert "Ann" not in rec1  # record 0's value never leaks into record 1

    def test_oversized_block_is_chunked(self):
        # Big blocks (>chunk size) + tiny c_usable → a record yields multiple children.
        big = "HEADER\n" + "".join(f"RECORD {i}\n" + ("word " * 600) + "\n" for i in range(1, 5))
        rs = record_segments(_record_fields(4), big, 4.0, 1.0)
        assert any(len(segs) > 1 for segs in rs.by_record.values())  # type: ignore[union-attr]

    def test_child_offsets_are_absolute(self):
        rs = record_segments(_record_fields(4), _DOC, 4.0, 4096.0)
        for seg in rs.segments:  # type: ignore[union-attr]
            assert _DOC[seg.start : seg.end] == seg.text  # offsets index the document

    def test_none_on_heterogeneous(self):
        fields = [_field("a.x"), _field("b.y"), _field("c.z"), _field("d.w")]
        assert record_segments(fields, _DOC, 4.0, 4096.0) is None


class TestGroupRecordOrdinal:
    def test_returns_record_of_group_fields(self):
        field_ordinal, _ = detect_record_axis(_record_fields(4))  # type: ignore[misc]
        assert group_record_ordinal(["recs.rec_2.name", "recs.rec_2.age"], field_ordinal) == 1

    def test_minus_one_for_non_record_fields(self):
        field_ordinal, _ = detect_record_axis(_record_fields(4))  # type: ignore[misc]
        assert group_record_ordinal(["study.title"], field_ordinal) == -1


# A heterogeneous (non-record) document: a preamble, then four enumerated headings,
# each introducing denser body lines. No repeating record axis exists.
_HETERO_DOC = (
    "This filing summarises the consolidated results for the year under review in full.\n"
    "1. Income Statement\n"
    "Total revenue reached 1,234,567 dollars and net income was 89,000 dollars after taxes.\n"
    "Operating expenses totalled 500,000 dollars across every division for the period.\n"
    "2. Balance Sheet\n"
    "Total assets stood at 9,876,543 dollars while liabilities were 4,000,000 dollars.\n"
    "Cash and equivalents amounted to 250,000 dollars across operating accounts.\n"
    "3. Cash Flow Statement\n"
    "Net cash from operating activities was 750,000 dollars during the period reviewed.\n"
    "Capital expenditures consumed 300,000 dollars for plant and equipment that year.\n"
    "4. Governance\n"
    "The board comprised nine directors who met quarterly to review the strategy.\n"
    "The audit committee oversaw financial reporting and internal controls all year.\n"
)


class TestDetectSectionStructure:
    def test_finds_enumerated_headings(self):
        structure = detect_section_structure(_HETERO_DOC, 4.0, 4096.0)
        assert structure is not None
        headings = [s.heading for s in structure.sections]
        assert headings == [
            "1. Income Statement",
            "2. Balance Sheet",
            "3. Cash Flow Statement",
            "4. Governance",
        ]

    def test_preamble_kept_separate(self):
        structure = detect_section_structure(_HETERO_DOC, 4.0, 4096.0)
        assert structure is not None
        preamble = " ".join(s.text for s in structure.preamble_segments)
        assert "consolidated results" in preamble
        assert "Income Statement" not in preamble  # body before the first heading only

    def test_section_segments_are_local(self):
        structure = detect_section_structure(_HETERO_DOC, 4.0, 4096.0)
        assert structure is not None
        income = " ".join(s.text for s in structure.by_section[0])
        assert "revenue" in income
        assert "directors" not in income  # governance content never leaks into section 0

    def test_offsets_are_absolute(self):
        structure = detect_section_structure(_HETERO_DOC, 4.0, 4096.0)
        assert structure is not None
        for seg in structure.segments:
            assert _HETERO_DOC[seg.start : seg.end] == seg.text

    def test_oversized_section_is_chunked(self):
        # A section large enough to exceed the chunker's own size splits into children.
        big = (
            "Preamble line of body text that is comfortably long enough to be prose.\n"
            "1. Income Statement\n" + ("revenue and income detail line. " * 600) + "\n"
            "2. Balance Sheet\nassets and liabilities summary line follows here now.\n"
            "3. Cash Flow Statement\noperating and investing cash detail line here now.\n"
            "4. Governance\nthe board of directors met to review strategy this year.\n"
        )
        structure = detect_section_structure(big, 4.0, 1.0)  # tiny budget
        assert structure is not None
        assert any(len(segs) > 1 for segs in structure.by_section.values())

    def test_none_without_headings(self):
        flat = "A long paragraph of plain prose. " * 40
        assert detect_section_structure(flat, 4.0, 4096.0) is None

    def test_none_when_too_few_headings(self):
        doc = "Intro line of body text here.\n1. Only Section\nSome longer body content follows.\n"
        assert detect_section_structure(doc, 4.0, 4096.0) is None

    def test_none_when_headings_not_distinct(self):
        # A boilerplate line that repeats verbatim is a divider, not a heading family.
        doc = "".join(
            "Section\nA longer line of body content beneath the repeated label here.\n"
            for _ in range(6)
        )
        assert detect_section_structure(doc, 4.0, 4096.0) is None


class TestAlignPathToSection:
    def _sections(self):
        structure = detect_section_structure(_HETERO_DOC, 4.0, 4096.0)
        assert structure is not None
        return structure.sections

    def test_path_aligns_to_matching_heading(self):
        index, score = align_path_to_section(
            ["income_statement.total_revenue", "income_statement.net_income"], self._sections()
        )
        assert index == 0
        assert score >= 0.5

    def test_distinct_path_aligns_to_its_section(self):
        index, _ = align_path_to_section(["governance.board_size"], self._sections())
        assert index == 3

    def test_no_match_returns_minus_one(self):
        index, score = align_path_to_section(["unrelated.identifier_code"], self._sections())
        assert index == -1
        assert score == 0.0

    def test_empty_paths_return_minus_one(self):
        assert align_path_to_section([], self._sections()) == (-1, 0.0)
