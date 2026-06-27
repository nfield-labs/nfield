"""Unit tests for the schema-typed morphological index."""

from __future__ import annotations

from nfield.retrieval._morphology import (
    TYPE_BOOLEAN,
    TYPE_DATE,
    TYPE_EMAIL,
    TYPE_NUMBER,
    TYPE_URI,
    TYPE_UUID,
    build_morphology_index,
    field_type_classes,
    nearest_gap,
)
from nfield.schema._types import Field, Segment


def _seg(text: str, sid: int = 0) -> Segment:
    return Segment(text=text, start=0, end=len(text), segment_type="unstructured", segment_id=sid)


def _field(path: str, ftype: str, **node: object) -> Field:
    schema_node = {"type": ftype, **node}
    return Field(path=path, type=ftype, constraints={}, parent_path="", schema_node=schema_node)


class TestFieldTypeClasses:
    def test_integer_and_number_map_to_number(self) -> None:
        assert field_type_classes(_field("n", "integer")) == (TYPE_NUMBER,)
        assert field_type_classes(_field("n", "number")) == (TYPE_NUMBER,)

    def test_boolean(self) -> None:
        assert field_type_classes(_field("b", "boolean")) == (TYPE_BOOLEAN,)

    def test_format_takes_precedence(self) -> None:
        assert field_type_classes(_field("d", "string", format="date")) == (TYPE_DATE,)
        assert field_type_classes(_field("e", "string", format="email")) == (TYPE_EMAIL,)
        assert field_type_classes(_field("u", "string", format="uri")) == (TYPE_URI,)
        assert field_type_classes(_field("i", "string", format="uuid")) == (TYPE_UUID,)

    def test_plain_string_and_containers_have_no_class(self) -> None:
        assert field_type_classes(_field("s", "string")) == ()
        assert field_type_classes(_field("o", "object")) == ()
        assert field_type_classes(_field("a", "array")) == ()


class TestBuildMorphologyIndex:
    def test_number_token_positions(self) -> None:
        idx = build_morphology_index([_seg("enrolled 4591 patients")])
        assert idx.segments[0].type_positions[TYPE_NUMBER] == [1]

    def test_boolean_token_positions(self) -> None:
        idx = build_morphology_index([_seg("eligible yes confirmed")])
        assert idx.segments[0].type_positions[TYPE_BOOLEAN] == [1]

    def test_date_span_detected(self) -> None:
        idx = build_morphology_index([_seg("start date 2020-04-13 noted")])
        assert TYPE_DATE in idx.segments[0].type_positions

    def test_email_detected(self) -> None:
        idx = build_morphology_index([_seg("contact john.doe@example.com today")])
        assert TYPE_EMAIL in idx.segments[0].type_positions

    def test_uuid_detected(self) -> None:
        idx = build_morphology_index([_seg("id 123e4567-e89b-12d3-a456-426614174000 here")])
        assert TYPE_UUID in idx.segments[0].type_positions

    def test_folded_text_lowercased_and_accent_stripped(self) -> None:
        idx = build_morphology_index([_seg("Café RÉSUMÉ")])
        assert idx.segments[0].folded_text == "cafe resume"

    def test_term_positions_track_each_token(self) -> None:
        idx = build_morphology_index([_seg("alpha beta alpha")])
        assert idx.segments[0].token_positions["alpha"] == [0, 2]
        assert idx.segments[0].n_tokens == 3

    def test_plain_text_has_no_typed_tokens(self) -> None:
        idx = build_morphology_index([_seg("the quick brown fox")])
        assert idx.segments[0].type_positions == {}


class TestNearestGap:
    def test_basic_distance(self) -> None:
        assert nearest_gap([2, 9], [4, 10]) == 1

    def test_touching_returns_zero(self) -> None:
        assert nearest_gap([3], [3]) == 0

    def test_empty_returns_none(self) -> None:
        assert nearest_gap([], [4]) is None
        assert nearest_gap([1], []) is None
