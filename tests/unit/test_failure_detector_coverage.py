"""Targeted coverage tests for FailureModeDetector — no API keys required.

Covers lines: 147-152 (detect() exception handler), 362 (_count_enum_values
non-dict returns 0), 381 (_count_enum_values items/additionalProperties path).
"""

from __future__ import annotations

from unittest.mock import patch

from formatshield.scorer.features import ComplexityFeatures
from formatshield.ttf.failure_detector import (
    FailureModeDetector,
    _count_enum_values,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    *,
    token_entropy: float = 0.5,
    schema_depth: int = 2,
    required_reasoning_ops: int = 2,
    instruction_tune_score: float = 0.5,
    prompt_length_bucket: int = 2,
    schema_constraint_count: int = 3,
) -> ComplexityFeatures:
    return ComplexityFeatures(
        token_entropy=token_entropy,
        schema_depth=schema_depth,
        required_reasoning_ops=required_reasoning_ops,
        instruction_tune_score=instruction_tune_score,
        prompt_length_bucket=prompt_length_bucket,
        schema_constraint_count=schema_constraint_count,
    )


# ---------------------------------------------------------------------------
# detect() — lines 147-152: exception handler returns []
# ---------------------------------------------------------------------------


def test_detect_returns_empty_list_on_unexpected_exception():
    """Lines 147-152: when _detect_impl raises, detect() catches and returns []."""
    detector = FailureModeDetector()
    features = _make_features()

    with patch.object(
        detector,
        "_detect_impl",
        side_effect=RuntimeError("simulated internal error"),
    ):
        result = detector.detect(features, "groq/llama-3.3-70b-versatile", schema={})

    assert result == []


def test_detect_returns_empty_list_on_attribute_error():
    """Lines 147-152: AttributeError in _detect_impl is caught and [] returned."""
    detector = FailureModeDetector()
    features = _make_features()

    with patch.object(
        detector,
        "_detect_impl",
        side_effect=AttributeError("simulated attribute error"),
    ):
        result = detector.detect(features, "openai/gpt-4o", schema=None)

    assert result == []


def test_detect_returns_empty_list_on_recursion_error():
    """Lines 147-152: RecursionError in _detect_impl is caught and [] returned."""
    detector = FailureModeDetector()
    features = _make_features()

    with patch.object(
        detector,
        "_detect_impl",
        side_effect=RecursionError("max recursion depth exceeded"),
    ):
        result = detector.detect(features, "ollama/llama3.1")

    assert result == []


def test_detect_exception_path_does_not_reraise():
    """Lines 147-152: verify the exception path swallows the error cleanly."""
    detector = FailureModeDetector()
    features = _make_features()

    # Should NOT raise even for a totally unexpected error type.
    with patch.object(detector, "_detect_impl", side_effect=ValueError("unexpected")):
        result = detector.detect(features, "vllm/mistral-7b")

    assert isinstance(result, list)
    assert len(result) == 0


# ---------------------------------------------------------------------------
# _count_enum_values() — line 362: non-dict input returns 0
# ---------------------------------------------------------------------------


def test_count_enum_values_with_non_dict_string_returns_zero():
    """Line 362: passing a string returns 0."""
    assert _count_enum_values("not-a-dict") == 0  # type: ignore[arg-type]


def test_count_enum_values_with_none_returns_zero():
    """Line 362: passing None returns 0."""
    assert _count_enum_values(None) == 0  # type: ignore[arg-type]


def test_count_enum_values_with_integer_returns_zero():
    """Line 362: passing an integer returns 0."""
    assert _count_enum_values(42) == 0  # type: ignore[arg-type]


def test_count_enum_values_with_list_returns_zero():
    """Line 362: passing a list (not a dict) returns 0."""
    assert _count_enum_values(["a", "b", "c"]) == 0  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _count_enum_values() — line 381: items / additionalProperties / if/then/else
# ---------------------------------------------------------------------------


def test_count_enum_values_in_items_sub_schema():
    """Line 381: enum values inside an 'items' sub-schema are counted."""
    schema = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": ["red", "green", "blue"],
        },
    }
    assert _count_enum_values(schema) == 3


def test_count_enum_values_in_additional_properties_sub_schema():
    """Line 381: enum values inside 'additionalProperties' are counted."""
    schema = {
        "type": "object",
        "additionalProperties": {
            "type": "string",
            "enum": ["yes", "no"],
        },
    }
    assert _count_enum_values(schema) == 2


def test_count_enum_values_in_if_then_else_sub_schemas():
    """Line 381: enum values in 'if', 'then', and 'else' sub-schemas counted."""
    schema = {
        "if": {"enum": ["a"]},
        "then": {"enum": ["b", "c"]},
        "else": {"enum": ["d", "e", "f"]},
    }
    result = _count_enum_values(schema)
    # 1 + 2 + 3 = 6
    assert result == 6


def test_count_enum_values_in_not_sub_schema():
    """Line 381: enum values in 'not' sub-schema are counted."""
    schema = {
        "not": {
            "enum": ["forbidden_value_1", "forbidden_value_2"],
        }
    }
    assert _count_enum_values(schema) == 2


def test_count_enum_values_items_non_dict_not_counted():
    """Line 381: 'items' that is not a dict is not traversed (no crash)."""
    schema = {
        "type": "array",
        "items": "not-a-dict",
    }
    # items is a string, not a dict, so the isinstance check at line 380 fails.
    assert _count_enum_values(schema) == 0


def test_count_enum_values_combined_paths():
    """Lines 362 and 381: combining multiple paths — nested items + top-level enum."""
    schema = {
        "enum": ["top1", "top2"],
        "items": {
            "enum": ["item1"],
        },
        "additionalProperties": {
            "enum": ["add1", "add2", "add3"],
        },
    }
    result = _count_enum_values(schema)
    # top-level: 2, items: 1, additionalProperties: 3 → total 6
    assert result == 6


# ---------------------------------------------------------------------------
# Integration: _count_enum_values triggers schema_too_constrained via detect()
# ---------------------------------------------------------------------------


def test_schema_too_constrained_via_enum_count_uses_count_enum_values():
    """Lines 362/381: large enum in 'items' makes schema_too_constrained fire."""
    detector = FailureModeDetector()
    # Build a schema where items has > 50 enum values.
    schema = {
        "type": "array",
        "items": {
            "type": "string",
            "enum": [f"value_{i}" for i in range(51)],
        },
    }
    features = _make_features(schema_constraint_count=3)
    modes = detector.detect(features, "groq/llama-3.3-70b-versatile", schema=schema)
    assert "schema_too_constrained" in modes


def test_count_enum_values_none_items_not_traversed():
    """Line 381 guard: items=None (not a dict) → 0 from that path."""
    schema: dict = {"items": None}
    assert _count_enum_values(schema) == 0
