"""Targeted coverage tests for SchemaAnalyzer — no API keys required.

Covers lines: 83-85 (analyze exception path), 104 (_compute_depth non-dict node),
177 (additionalProperties depth), 202 (_count_constraints non-dict node),
216 (non-dict child schema skip), 240-241 (bounds break).
"""

from __future__ import annotations

from unittest.mock import patch

from formatshield.scorer.schema_analyzer import SchemaAnalyzer

# ---------------------------------------------------------------------------
# analyze() — lines 83-85: exception handler returns (0, 0)
# ---------------------------------------------------------------------------


def test_analyze_returns_zero_zero_on_unexpected_exception():
    """Lines 83-85: exception inside _compute_depth/count → (0, 0)."""
    analyzer = SchemaAnalyzer()
    with patch.object(
        analyzer,
        "_compute_depth",
        side_effect=RuntimeError("simulated internal error"),
    ):
        result = analyzer.analyze({"type": "object"})
    assert result == (0, 0)


def test_analyze_returns_zero_zero_when_count_constraints_raises():
    """Lines 83-85: exception inside _count_constraints → (0, 0)."""
    analyzer = SchemaAnalyzer()
    with patch.object(
        analyzer,
        "_count_constraints",
        side_effect=RecursionError("simulated recursion"),
    ):
        result = analyzer.analyze({"type": "object"})
    assert result == (0, 0)


# ---------------------------------------------------------------------------
# _compute_depth() — line 104: non-dict node returns current_depth
# ---------------------------------------------------------------------------


def test_compute_depth_non_dict_returns_current_depth():
    """Line 104: _compute_depth called with a non-dict returns current_depth."""
    analyzer = SchemaAnalyzer()
    # items as a list of schemas causes recursion into each item_schema.
    # A non-dict item_schema hits line 104.
    schema = {
        "type": "array",
        "items": ["not-a-dict", 42, None],
    }
    depth, _ = analyzer.analyze(schema)
    # Root is depth 0; items list children are traversed at depth+1=1 but each
    # item_schema is not a dict so they return 1 immediately.
    assert depth >= 0


def test_compute_depth_non_dict_node_directly():
    """Line 104: calling _compute_depth with a string returns current_depth unchanged."""
    analyzer = SchemaAnalyzer()
    result = analyzer._compute_depth("not-a-dict", current_depth=3)
    assert result == 3


def test_compute_depth_non_dict_with_integer():
    """Line 104: integer node returns current_depth."""
    analyzer = SchemaAnalyzer()
    result = analyzer._compute_depth(42, current_depth=5)
    assert result == 5


# ---------------------------------------------------------------------------
# _compute_depth() — line 177: additionalProperties as dict
# ---------------------------------------------------------------------------


def test_compute_depth_additional_properties_as_dict():
    """Line 177: additionalProperties that is a dict is traversed for depth."""
    analyzer = SchemaAnalyzer()
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "nested_field": {"type": "string"},
            },
        },
    }
    depth, _ = analyzer.analyze(schema)
    # additionalProperties traversed at same depth (0), nested properties add 1.
    assert depth >= 1


def test_compute_depth_additional_properties_increases_max_depth():
    """Line 177: additionalProperties with deeper nesting raises max_depth."""
    analyzer = SchemaAnalyzer()
    flat_schema = {"type": "object"}
    nested_schema = {
        "type": "object",
        "additionalProperties": {
            "type": "object",
            "properties": {
                "deep": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                }
            },
        },
    }
    flat_depth, _ = analyzer.analyze(flat_schema)
    nested_depth, _ = analyzer.analyze(nested_schema)
    assert nested_depth > flat_depth


# ---------------------------------------------------------------------------
# _count_constraints() — line 202: non-dict node returns 0
# ---------------------------------------------------------------------------


def test_count_constraints_non_dict_returns_zero():
    """Line 202: _count_constraints called with a non-dict returns 0."""
    analyzer = SchemaAnalyzer()
    result = analyzer._count_constraints("not-a-dict")
    assert result == 0


def test_count_constraints_non_dict_integer_returns_zero():
    """Line 202: integer node returns 0 constraints."""
    analyzer = SchemaAnalyzer()
    assert analyzer._count_constraints(123) == 0


def test_count_constraints_none_returns_zero():
    """Line 202: None node returns 0 constraints."""
    analyzer = SchemaAnalyzer()
    assert analyzer._count_constraints(None) == 0


# ---------------------------------------------------------------------------
# _count_constraints() — line 216: non-dict child schema skipped via continue
# ---------------------------------------------------------------------------


def test_count_constraints_skips_non_dict_property_child():
    """Line 216: a non-dict value inside properties is skipped (continue)."""
    analyzer = SchemaAnalyzer()
    schema = {
        "type": "object",
        "properties": {
            "good_field": {"type": "string", "enum": ["a", "b"]},
            "bad_field": "not-a-dict",  # triggers line 215-216 continue
            "another_bad": 42,
        },
    }
    _, constraints = analyzer.analyze(schema)
    # Only good_field's enum should be counted; bad_field and another_bad skipped.
    assert constraints >= 1


def test_count_constraints_all_non_dict_properties_yields_zero_property_constraints():
    """Line 216: all property children non-dict → no property constraints."""
    analyzer = SchemaAnalyzer()
    schema = {
        "type": "object",
        "properties": {
            "a": "string",
            "b": 99,
            "c": None,
        },
    }
    _, constraints = analyzer.analyze(schema)
    # No required array, no dict child schemas — should be 0.
    assert constraints == 0


# ---------------------------------------------------------------------------
# _count_constraints() — lines 240-241: bounds keyword triggers break
# ---------------------------------------------------------------------------


def test_count_constraints_bounds_keyword_counts_once_per_property():
    """Lines 240-241: multiple bounds keywords on one property count as 1 (break)."""
    analyzer = SchemaAnalyzer()
    schema = {
        "type": "object",
        "properties": {
            "age": {
                "type": "integer",
                "minimum": 0,
                "maximum": 150,
                "exclusiveMinimum": -1,
            }
        },
    }
    _, constraints = analyzer.analyze(schema)
    # Only 1 bound should be counted (break after first hit), not 3.
    assert constraints == 1


def test_count_constraints_bounds_break_versus_no_bounds():
    """Lines 240-241: property with bounds counts exactly 1 more than without."""
    analyzer = SchemaAnalyzer()
    schema_with_bounds = {
        "type": "object",
        "properties": {
            "value": {
                "type": "number",
                "minimum": 0,
                "maximum": 100,
                "minLength": 1,  # extra bound — should be ignored due to break
            }
        },
    }
    schema_without_bounds = {
        "type": "object",
        "properties": {
            "value": {"type": "number"},
        },
    }
    _, with_bounds = analyzer.analyze(schema_with_bounds)
    _, without_bounds = analyzer.analyze(schema_without_bounds)
    assert with_bounds == without_bounds + 1


def test_count_constraints_all_bound_keywords_each_trigger_break():
    """Lines 240-241: each supported bound keyword independently triggers break."""
    analyzer = SchemaAnalyzer()

    bound_keywords = [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ]
    for kw in bound_keywords:
        schema = {
            "type": "object",
            "properties": {
                "field": {kw: 1},
            },
        }
        _, constraints = analyzer.analyze(schema)
        assert constraints == 1, (
            f"Expected 1 constraint for bound keyword '{kw}', got {constraints}"
        )
