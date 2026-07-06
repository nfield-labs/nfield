"""Prompt-contract tests: invariants every extraction prompt must satisfy.

Every live model call flows through ``build_extraction_prompt``; these tests pin
what the model is shown for every schema shape, so a change that would confuse
the model fails here instead of surfacing as a benchmark regression.
"""

from __future__ import annotations

import pytest

from nfield.extraction._papt import TemplateType
from nfield.extraction._prompt import build_extraction_prompt
from nfield.schema._flatten import flatten_schema

DOC = "The reporting period ended on March 31. Region North: 120. Region South: 80."

METRIC_ITEMS = {
    "type": "object",
    "properties": {
        "region": {
            "type": "string",
            "enum": ["north", "south", "total"],
            "description": "Reported region.",
        },
        "amount": {"type": "number", "description": "Reported figure."},
    },
}

SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string", "description": "Document title."},
        "grade": {"type": "string", "enum": ["low", "mid", "high"]},
        "tags": {"type": "array", "items": {"type": "string"}},
        "matrix": {
            "type": "array",
            "items": {"type": "array", "items": {"type": "number"}},
        },
        "sales": {"type": "array", "items": METRIC_ITEMS, "description": "Sales rows."},
        "costs": {"type": "array", "items": METRIC_ITEMS, "description": "Cost rows."},
        "people": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "roles": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
        "extra": {"type": "object", "additionalProperties": {"type": "string"}},
    },
}


@pytest.fixture(scope="module")
def fields():
    return flatten_schema(SCHEMA)


@pytest.fixture(scope="module")
def user_msg(fields):
    return build_extraction_prompt(fields, DOC, TemplateType.STANDARD)[1]["content"]


@pytest.fixture(scope="module")
def system_msg(fields):
    return build_extraction_prompt(fields, DOC, TemplateType.STANDARD)[0]["content"]


class TestFieldListCompleteness:
    def test_every_field_listed_exactly_once(self, fields, user_msg):
        field_section = user_msg[user_msg.index("Fields to extract") :]
        for f in fields:
            starts = [
                line for line in field_section.splitlines() if line.startswith(f"{f.path} (")
            ]
            assert len(starts) == 1, f"{f.path} listed {len(starts)} times"

    def test_fields_keep_given_order(self, fields, user_msg):
        positions = [user_msg.index(f"\n{f.path} (") for f in fields]
        assert positions == sorted(positions)

    def test_every_description_reaches_the_model(self, user_msg):
        for text in ("Document title.", "Sales rows.", "Cost rows.", "Reported region."):
            assert text in user_msg

    def test_every_enum_value_reaches_the_model(self, user_msg):
        for value in ("low", "mid", "high", "north", "south", "total"):
            assert value in user_msg


class TestStructureAndOrder:
    def test_document_before_field_list(self, user_msg):
        assert user_msg.index("Document:") < user_msg.index("Fields to extract")

    def test_instructions_come_first_when_given(self, fields):
        msg = build_extraction_prompt(
            fields, DOC, TemplateType.STANDARD, instructions="Focus on totals."
        )[1]["content"]
        assert msg.index("Focus on totals.") < msg.index("Document:")

    def test_dependency_block_between_document_and_fields(self, fields):
        msg = build_extraction_prompt(
            fields, DOC, TemplateType.STANDARD, dependency_values={"title": "Annual Report"}
        )[1]["content"]
        assert (
            msg.index("Document:")
            < msg.index("title = Annual Report")
            < msg.index("Fields to extract")
        )

    def test_field_reason_appended_to_its_line_only(self, fields, user_msg):
        msg = build_extraction_prompt(
            fields, DOC, TemplateType.STANDARD, field_reasons={"sales": "previous pass cut off"}
        )[1]["content"]
        sales_line = next(line for line in msg.splitlines() if line.startswith("sales ("))
        assert "previous pass cut off" in sales_line
        assert msg.count("previous pass cut off") == 1


class TestNoDuplication:
    def test_shared_item_shape_spelled_out_once(self, user_msg):
        assert user_msg.count("region: string") == 1

    def test_sharing_fields_reference_the_named_shape(self, user_msg):
        for path in ("sales", "costs"):
            line = next(line for line in user_msg.splitlines() if line.startswith(f"{path} ("))
            assert "entry shape S1" in line

    def test_unique_object_shape_stays_inline(self, user_msg):
        people_line = next(line for line in user_msg.splitlines() if line.startswith("people ("))
        assert "name: string" in people_line

    def test_dimension_directive_on_each_sharing_field_line(self, user_msg):
        for path in ("sales", "costs"):
            line = next(line for line in user_msg.splitlines() if line.startswith(f"{path} ("))
            assert "enumerate EXHAUSTIVELY" in line, f"{path} lost its directive"

    def test_directive_never_in_shared_definition(self, user_msg):
        defs = user_msg[user_msg.index("Shared entry shapes") : user_msg.index("\nsales (")]
        assert "enumerate EXHAUSTIVELY" not in defs


class TestShapeRendering:
    def test_nested_array_shape_visible(self, user_msg):
        matrix_line = next(line for line in user_msg.splitlines() if line.startswith("matrix ("))
        assert "array of number" in matrix_line

    def test_nested_object_array_shape_visible(self, user_msg):
        people_line = next(line for line in user_msg.splitlines() if line.startswith("people ("))
        assert "roles: array of string" in people_line


class TestSystemContract:
    def test_names_the_shared_shape_convention(self, system_msg):
        assert "entry shape" in system_msg

    def test_covers_all_output_kinds(self, system_msg):
        for phrase in ("boolean", "integer", "number", "array", "enum", "NULL"):
            assert phrase in system_msg

    def test_strict_mode_grounds_null_in_document(self, system_msg):
        assert "genuinely not stated" in system_msg

    def test_closed_book_never_mentions_a_document(self, fields):
        msgs = build_extraction_prompt(fields, "", TemplateType.STANDARD, closed_book=True)
        assert "document above" not in msgs[1]["content"]
        assert "Document:" not in msgs[1]["content"]
