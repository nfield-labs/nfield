"""Unit tests for output_type casting in core.py — no API keys required."""

from __future__ import annotations

import enum
from typing import Literal

import pytest

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield, _build_schema_from_output_type, _cast_parsed

# ---------------------------------------------------------------------------
# _build_schema_from_output_type tests
# ---------------------------------------------------------------------------


def test_build_schema_int() -> None:
    """int maps to JSON Schema integer type."""
    assert _build_schema_from_output_type(int) == {"type": "integer"}


def test_build_schema_float() -> None:
    """float maps to JSON Schema number type."""
    assert _build_schema_from_output_type(float) == {"type": "number"}


def test_build_schema_bool() -> None:
    """bool maps to JSON Schema boolean type."""
    assert _build_schema_from_output_type(bool) == {"type": "boolean"}


def test_build_schema_str() -> None:
    """str maps to JSON Schema string type."""
    assert _build_schema_from_output_type(str) == {"type": "string"}


def test_build_schema_enum() -> None:
    """Enum subclass maps to JSON Schema with enum values."""

    class Color(enum.Enum):
        RED = "red"
        BLUE = "blue"

    schema = _build_schema_from_output_type(Color)
    assert schema["enum"] == ["red", "blue"]


def test_build_schema_literal() -> None:
    """Literal type maps to JSON Schema enum."""
    schema = _build_schema_from_output_type(Literal["yes", "no"])  # type: ignore[arg-type]
    assert set(schema["enum"]) == {"yes", "no"}


def test_build_schema_list_of_str() -> None:
    """list[str] maps to JSON Schema array of string."""
    schema = _build_schema_from_output_type(list[str])  # type: ignore[arg-type]
    assert schema["type"] == "array"
    assert schema["items"] == {"type": "string"}


def test_build_schema_list_of_int() -> None:
    """list[int] maps to JSON Schema array of integer."""
    schema = _build_schema_from_output_type(list[int])  # type: ignore[arg-type]
    assert schema == {"type": "array", "items": {"type": "integer"}}


def test_build_schema_unknown_returns_empty() -> None:
    """Unknown type returns empty dict (graceful fallback)."""
    assert _build_schema_from_output_type(dict) == {}  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _cast_parsed tests
# ---------------------------------------------------------------------------


def test_cast_int() -> None:
    """JSON integer is cast to Python int."""
    assert _cast_parsed("42", int) == 42
    assert isinstance(_cast_parsed("42", int), int)


def test_cast_float() -> None:
    """JSON number is cast to Python float."""
    result = _cast_parsed("3.14", float)
    assert abs(result - 3.14) < 1e-9


def test_cast_bool_true() -> None:
    """JSON true is cast to Python True."""
    assert _cast_parsed("true", bool) is True


def test_cast_bool_false() -> None:
    """JSON false is cast to Python False."""
    assert _cast_parsed("false", bool) is False


def test_cast_str() -> None:
    """JSON string is cast to Python str."""
    assert _cast_parsed('"hello"', str) == "hello"


def test_cast_enum() -> None:
    """JSON value is cast to the matching Enum member."""

    class Direction(enum.Enum):
        NORTH = "north"
        SOUTH = "south"

    result = _cast_parsed('"north"', Direction)
    assert result == Direction.NORTH


def test_cast_literal_valid() -> None:
    """Literal value within allowed set is returned unchanged."""
    result = _cast_parsed('"yes"', Literal["yes", "no"])  # type: ignore[arg-type]
    assert result == "yes"


def test_cast_literal_invalid_raises() -> None:
    """Literal value outside allowed set raises ValueError."""
    with pytest.raises(ValueError, match="not in the allowed Literal"):
        _cast_parsed('"maybe"', Literal["yes", "no"])  # type: ignore[arg-type]


def test_cast_list_of_int() -> None:
    """JSON array is cast to list of int."""
    result = _cast_parsed("[1, 2, 3]", list[int])  # type: ignore[arg-type]
    assert result == [1, 2, 3]
    assert all(isinstance(x, int) for x in result)


def test_cast_invalid_json_returns_raw() -> None:
    """Invalid JSON string is returned as-is (graceful degradation)."""
    result = _cast_parsed("not json", int)
    assert result == "not json"


def test_cast_bool_before_int() -> None:
    """bool cast returns a bool, not a plain int — issubclass(bool, int) is True in Python."""
    result = _cast_parsed("true", bool)
    assert result is True
    # type() is bool (not int), confirming the bool path was taken
    assert type(result) is bool


# ---------------------------------------------------------------------------
# Integration tests with FormatShield + DryRunBackend
# ---------------------------------------------------------------------------


def test_generate_with_output_type_int() -> None:
    """generate_sync with output_type=int returns int in parsed."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    # DryRunBackend returns {"result": ..., "confidence": ...} for no schema,
    # but with output_type=int it derives {"type": "integer"} schema,
    # so DryRunBackend returns integer 0.
    result = shield.generate_sync("What is 2+2?", output_type=int)
    assert result is not None
    assert result.schema_valid is True
    # DryRunBackend returns 0 for integer schema (minimum default)
    assert isinstance(result.parsed, int)


def test_generate_with_output_type_bool() -> None:
    """generate_sync with output_type=bool returns bool in parsed."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync("Is the sky blue?", output_type=bool)
    assert result is not None
    assert isinstance(result.parsed, bool)


def test_generate_with_output_type_str() -> None:
    """generate_sync with output_type=str returns str in parsed."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync("Say hello", output_type=str)
    assert result is not None
    assert isinstance(result.parsed, str)


def test_generate_output_type_overridden_by_explicit_schema() -> None:
    """When both schema and output_type are given, schema takes precedence."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
    result = shield.generate_sync("Test", schema=schema, output_type=int)
    # schema is provided → result is parsed as dict (schema wins)
    assert isinstance(result.parsed, dict)


def test_generate_with_literal_output_type() -> None:
    """output_type=Literal generates enum-constrained schema."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    # DryRunBackend picks the first enum value for string type
    result = shield.generate_sync("Yes or no?", output_type=Literal["yes", "no"])  # type: ignore[arg-type]
    assert result is not None


def test_generate_with_list_output_type() -> None:
    """output_type=list[str] generates array schema."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync("List three items", output_type=list[str])  # type: ignore[arg-type]
    assert result is not None


def test_generate_with_sampling_params() -> None:
    """generate_sync accepts and passes through sampling params."""
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    # DryRunBackend ignores sampling params, but they should not raise
    result = shield.generate_sync(
        "What is 2+2?",
        temperature=0.7,
        max_tokens=100,
        seed=42,
    )
    assert result is not None
    assert result.output != ""
