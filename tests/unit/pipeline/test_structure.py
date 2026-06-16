"""Tests for structural slicing (pipeline/_structure.py)."""

from __future__ import annotations

from formatshield.pipeline._structure import (
    detect_blocks,
    detect_record_axis,
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
