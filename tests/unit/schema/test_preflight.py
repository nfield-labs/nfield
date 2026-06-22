"""Tests for the schema preflight (schema/_preflight)."""

from __future__ import annotations

import pytest

from formatshield.exceptions import SchemaError
from formatshield.schema._preflight import preflight_schema


def _obj(name: str, node: dict) -> dict:
    """Wrap a leaf node as a one-property object schema."""
    return {"type": "object", "properties": {name: node}}


# ---------------------------------------------------------------------------
# Valid schemas are NEVER rejected (zero false rejections)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node",
    [
        {"type": "integer", "minimum": 0, "maximum": 100},
        {"type": "integer", "minimum": 5, "maximum": 5},  # equal bounds OK
        {"type": "string", "minLength": 2, "maxLength": 10},
        {"type": "string", "pattern": r"^\d{4}$"},
        {"type": "string", "enum": ["a", "b", "c"]},
        {"type": "integer", "enum": [1, 2, 3]},
        {"type": "integer", "multipleOf": 5, "minimum": 0, "maximum": 100},
        {"type": "number", "multipleOf": 0.1, "minimum": 0, "maximum": 1},  # float skipped
        {"type": "array", "minItems": 1, "maxItems": 3},
        {"type": "string", "const": "fixed"},
        {"type": "integer", "const": 42, "minimum": 0, "maximum": 100},
    ],
)
def test_valid_schemas_pass(node: dict) -> None:
    preflight_schema(_obj("f", node))  # must not raise


def test_empty_schema_passes() -> None:
    preflight_schema({"type": "object", "properties": {}})


def test_union_type_with_string_pattern_not_rejected() -> None:
    # A pattern on a (string|integer) union is valid — strings can satisfy it.
    preflight_schema(_obj("f", {"type": ["string", "integer"], "pattern": "^a"}))


# ---------------------------------------------------------------------------
# Contradictions ARE rejected, with the field path
# ---------------------------------------------------------------------------


def test_min_greater_than_max() -> None:
    with pytest.raises(SchemaError) as exc:
        preflight_schema(_obj("age", {"type": "integer", "minimum": 100, "maximum": 10}))
    assert "minimum" in str(exc.value)
    assert exc.value.field == "age"
    assert exc.value.hint is not None


def test_exclusive_bounds_conflict() -> None:
    with pytest.raises(SchemaError):
        preflight_schema(
            _obj("x", {"type": "number", "exclusiveMinimum": 5, "exclusiveMaximum": 5})
        )


def test_minlength_greater_than_maxlength() -> None:
    with pytest.raises(SchemaError, match="minLength"):
        preflight_schema(_obj("s", {"type": "string", "minLength": 10, "maxLength": 2}))


def test_min_items_greater_than_max_items() -> None:
    with pytest.raises(SchemaError, match="minItems"):
        preflight_schema(_obj("a", {"type": "array", "minItems": 5, "maxItems": 1}))


def test_empty_enum() -> None:
    with pytest.raises(SchemaError, match="enum is empty"):
        preflight_schema(_obj("e", {"type": "string", "enum": []}))


def test_enum_members_fail_type() -> None:
    with pytest.raises(SchemaError, match="declared type"):
        preflight_schema(_obj("e", {"type": "integer", "enum": ["a", "b"]}))


def test_enum_members_fail_pattern() -> None:
    with pytest.raises(SchemaError, match="pattern"):
        preflight_schema(_obj("e", {"type": "string", "enum": ["abc"], "pattern": r"^\d+$"}))


def test_uncompilable_pattern() -> None:
    with pytest.raises(SchemaError, match="valid regular expression"):
        preflight_schema(_obj("s", {"type": "string", "pattern": "([unclosed"}))


def test_const_violates_type() -> None:
    with pytest.raises(SchemaError, match="const"):
        preflight_schema(_obj("c", {"type": "integer", "const": "not_an_int"}))


def test_const_below_minimum() -> None:
    with pytest.raises(SchemaError, match="below minimum"):
        preflight_schema(_obj("c", {"type": "integer", "const": 1, "minimum": 10}))


def test_required_absent_with_no_additional() -> None:
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "ghost"],
        "additionalProperties": False,
    }
    with pytest.raises(SchemaError, match="ghost"):
        preflight_schema(schema)


def test_required_absent_but_additional_allowed_is_ok() -> None:
    # additionalProperties not false -> the required key can still appear; no contradiction.
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "extra"],
    }
    preflight_schema(schema)  # must not raise


def test_multiple_of_empty_range() -> None:
    with pytest.raises(SchemaError, match="multiple of"):
        preflight_schema(
            _obj("n", {"type": "integer", "multipleOf": 10, "minimum": 1, "maximum": 9})
        )


def test_nested_contradiction_reports_dotted_path() -> None:
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"inner": {"type": "integer", "minimum": 5, "maximum": 1}},
            }
        },
    }
    with pytest.raises(SchemaError) as exc:
        preflight_schema(schema)
    assert exc.value.field == "outer.inner"
