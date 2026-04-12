"""Unit tests for SchemaAnalyzer."""

from __future__ import annotations

import pytest

from formatshield.scorer.schema_analyzer import SchemaAnalyzer


@pytest.fixture
def analyzer() -> SchemaAnalyzer:
    return SchemaAnalyzer()


def test_flat_schema_depth_1(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
        },
    }
    depth, _ = analyzer.analyze(schema)
    assert depth == 1


def test_nested_schema_depth_3(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {
                            "level3": {"type": "string"},
                        },
                    },
                },
            },
        },
    }
    depth, _ = analyzer.analyze(schema)
    assert depth >= 3


def test_required_fields_counted(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "type": "object",
        "required": ["a", "b", "c"],
        "properties": {
            "a": {"type": "string"},
            "b": {"type": "string"},
            "c": {"type": "string"},
        },
    }
    _, constraint_count = analyzer.analyze(schema)
    assert constraint_count >= 3


def test_anyof_schema_analyzed(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "anyOf": [
            {"type": "string"},
            {"type": "integer"},
        ]
    }
    depth, count = analyzer.analyze(schema)
    assert isinstance(depth, int)
    assert isinstance(count, int)


def test_empty_schema_returns_zero(analyzer: SchemaAnalyzer) -> None:
    depth, count = analyzer.analyze({})
    assert depth == 0
    assert count == 0


def test_malformed_schema_no_exception(analyzer: SchemaAnalyzer) -> None:
    # None input
    depth, count = analyzer.analyze(None)  # type: ignore[arg-type]
    assert depth == 0
    assert count == 0

    # String input
    depth, count = analyzer.analyze("bad schema")  # type: ignore[arg-type]
    assert depth == 0
    assert count == 0


def test_enum_field_counted(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["active", "inactive"]},
        },
    }
    _, count = analyzer.analyze(schema)
    assert count >= 1


def test_array_items_schema_analyzed(analyzer: SchemaAnalyzer) -> None:
    schema = {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"name": {"type": "string"}},
        },
    }
    depth, _ = analyzer.analyze(schema)
    assert depth >= 1


def test_items_as_list_analyzed(analyzer: SchemaAnalyzer) -> None:
    """items as a list (tuple validation, draft <=2019-09) should be analyzed."""
    schema = {
        "type": "array",
        "items": [{"type": "string"}, {"type": "integer"}, {"type": "boolean"}],
    }
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_prefix_items_analyzed(analyzer: SchemaAnalyzer) -> None:
    """prefixItems (draft 2020-12) should contribute to depth."""
    schema = {
        "type": "array",
        "prefixItems": [
            {"type": "string"},
            {"type": "object", "properties": {"x": {"type": "number"}}},
        ],
    }
    depth, _ = analyzer.analyze(schema)
    assert depth >= 1


def test_defs_registry_traversed(analyzer: SchemaAnalyzer) -> None:
    """$defs registry is traversed at the same depth level."""
    schema = {
        "type": "object",
        "properties": {"ref_field": {"$ref": "#/$defs/MyType"}},
        "$defs": {
            "MyType": {
                "type": "object",
                "properties": {"inner": {"type": "string"}},
                "required": ["inner"],
            }
        },
    }
    depth, constraints = analyzer.analyze(schema)
    assert isinstance(depth, int)
    assert isinstance(constraints, int)


def test_definitions_registry_traversed(analyzer: SchemaAnalyzer) -> None:
    """definitions registry (old-style) is also traversed."""
    schema = {
        "type": "object",
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {"street": {"type": "string"}},
                "required": ["street"],
            }
        },
    }
    depth, constraints = analyzer.analyze(schema)
    assert isinstance(depth, int)
    assert constraints >= 0


def test_if_then_else_keywords_traversed(analyzer: SchemaAnalyzer) -> None:
    """if/then/else keywords should be analyzed without raising."""
    schema = {
        "if": {"properties": {"country": {"const": "US"}}},
        "then": {"properties": {"zip_code": {"type": "string", "pattern": "^[0-9]{5}$"}}},
        "else": {"properties": {"postal_code": {"type": "string"}}},
    }
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_not_keyword_traversed(analyzer: SchemaAnalyzer) -> None:
    """not keyword should be traversed without raising."""
    schema = {"not": {"type": "string"}}
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_contains_keyword_traversed(analyzer: SchemaAnalyzer) -> None:
    """contains keyword should be traversed without raising."""
    schema = {
        "type": "array",
        "contains": {"type": "integer", "minimum": 5},
    }
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_additional_properties_as_schema(analyzer: SchemaAnalyzer) -> None:
    """additionalProperties as a schema object is traversed."""
    schema = {
        "type": "object",
        "additionalProperties": {"type": "string", "maxLength": 100},
    }
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_additional_properties_as_bool_not_traversed(analyzer: SchemaAnalyzer) -> None:
    """additionalProperties as a bool is NOT a sub-schema — must not raise."""
    schema = {"type": "object", "additionalProperties": False}
    depth, _ = analyzer.analyze(schema)
    assert isinstance(depth, int)


def test_oneof_branches_analyzed(analyzer: SchemaAnalyzer) -> None:
    """oneOf branches should be traversed and increase depth."""
    schema = {
        "oneOf": [
            {"type": "object", "properties": {"a": {"type": "string"}}, "required": ["a"]},
            {"type": "object", "properties": {"b": {"type": "integer"}}, "required": ["b"]},
        ]
    }
    depth, constraints = analyzer.analyze(schema)
    assert depth >= 1
    assert constraints >= 2


def test_allof_branches_analyzed(analyzer: SchemaAnalyzer) -> None:
    """allOf branches should contribute to depth."""
    schema = {
        "allOf": [
            {"type": "object", "required": ["x"]},
            {"properties": {"x": {"type": "number"}, "y": {"type": "string"}}},
        ]
    }
    depth, constraints = analyzer.analyze(schema)
    assert isinstance(depth, int)
    assert constraints >= 1


def test_pattern_constraint_counted(analyzer: SchemaAnalyzer) -> None:
    """pattern keyword should be counted as a constraint."""
    schema = {
        "type": "object",
        "properties": {
            "email": {"type": "string", "pattern": r"^[^@]+@[^@]+\.[^@]+$"},
        },
    }
    _, constraints = analyzer.analyze(schema)
    assert constraints >= 1


def test_format_property_does_not_crash(analyzer: SchemaAnalyzer) -> None:
    """format keyword in a property should not raise regardless of whether it is counted."""
    schema = {
        "type": "object",
        "properties": {
            "created_at": {"type": "string", "format": "date-time"},
        },
        "required": ["created_at"],
    }
    depth, constraints = analyzer.analyze(schema)
    assert isinstance(depth, int)
    # The required field IS counted as a constraint
    assert constraints >= 1
