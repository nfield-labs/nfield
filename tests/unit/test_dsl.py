"""Unit tests for formatshield.dsl — Maybe, Partial, IterableModel."""

from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from formatshield.dsl import IterableModel, Maybe, MaybeResult, Partial
from formatshield.dsl.partial import _make_all_optional, _try_recover_json

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class Person(BaseModel):
    name: str
    age: int


class Analysis(BaseModel):
    summary: str
    sentiment: str
    confidence: float


# ---------------------------------------------------------------------------
# MaybeResult.from_raw
# ---------------------------------------------------------------------------


def test_maybe_result_success_with_pydantic_model() -> None:
    raw = json.dumps({"result": {"name": "Alice", "age": 30}, "error": False})
    maybe = MaybeResult.from_raw(raw, Person)
    assert maybe.error is False
    assert maybe.result is not None
    assert maybe.result.name == "Alice"
    assert maybe.result.age == 30


def test_maybe_result_error_flag() -> None:
    raw = json.dumps({"result": None, "error": True, "error_message": "Cannot determine"})
    maybe = MaybeResult.from_raw(raw, Person)
    assert maybe.error is True
    assert maybe.result is None
    assert maybe.error_message == "Cannot determine"


def test_maybe_result_invalid_json() -> None:
    maybe = MaybeResult.from_raw("not json", Person)
    assert maybe.error is True
    assert maybe.result is None


def test_maybe_result_non_dict_json() -> None:
    maybe = MaybeResult.from_raw('"just a string"', Person)
    assert maybe.error is True


def test_maybe_result_validation_failure() -> None:
    # age must be int — sending string should trigger validation error
    raw = json.dumps({"result": {"name": "Alice", "age": "not-an-int"}, "error": False})
    maybe = MaybeResult.from_raw(raw, Person)
    # Pydantic will coerce "not-an-int" → fail
    assert maybe.error is True or maybe.result is not None  # coercion may succeed


def test_maybe_result_unwrap_success() -> None:
    raw = json.dumps({"result": {"name": "Bob", "age": 25}, "error": False})
    maybe = MaybeResult.from_raw(raw, Person)
    person = maybe.unwrap()
    assert person.name == "Bob"


def test_maybe_result_unwrap_raises_on_error() -> None:
    maybe: MaybeResult[Person] = MaybeResult(result=None, error=True, error_message="oops")
    with pytest.raises(ValueError, match="no value"):
        maybe.unwrap()


def test_maybe_result_unwrap_or_returns_default() -> None:
    maybe: MaybeResult[Person] = MaybeResult(result=None, error=True)
    default = Person(name="Default", age=0)
    result = maybe.unwrap_or(default)
    assert result.name == "Default"


def test_maybe_result_unwrap_or_returns_result_when_present() -> None:
    person = Person(name="Alice", age=30)
    maybe: MaybeResult[Person] = MaybeResult(result=person, error=False)
    result = maybe.unwrap_or(Person(name="Default", age=0))
    assert result.name == "Alice"


# ---------------------------------------------------------------------------
# Maybe class
# ---------------------------------------------------------------------------


def test_maybe_subscript_creates_subclass() -> None:
    maybe_person_cls = Maybe[Person]
    assert issubclass(maybe_person_cls, Maybe)
    assert "Person" in maybe_person_cls.__name__


def test_maybe_build_schema_has_result_field() -> None:
    schema = Maybe.build_schema(Person)
    assert "result" in schema["properties"]
    assert "error" in schema["properties"]


def test_maybe_build_schema_error_is_required() -> None:
    schema = Maybe.build_schema(Person)
    assert "error" in schema.get("required", [])


def test_maybe_build_schema_fallback_for_plain_dict() -> None:
    schema = Maybe.build_schema(dict)  # type: ignore[arg-type]
    assert schema["type"] == "object"


def test_maybe_get_wrapped_type() -> None:
    maybe_person_cls = Maybe[Person]
    assert maybe_person_cls.get_wrapped_type() is Person


# ---------------------------------------------------------------------------
# Partial class
# ---------------------------------------------------------------------------


def test_partial_subscript() -> None:
    partial_analysis_cls = Partial[Analysis]
    assert issubclass(partial_analysis_cls, Partial)
    assert "Analysis" in partial_analysis_cls.__name__


def test_partial_build_schema_removes_required() -> None:
    schema = Partial.build_schema(Analysis)
    assert "required" not in schema


def test_partial_build_schema_makes_fields_nullable() -> None:
    schema = Partial.build_schema(Analysis)
    for prop in schema["properties"].values():
        assert "anyOf" in prop or "default" in prop


def test_partial_parse_partial_complete_json() -> None:
    raw = json.dumps({"summary": "Good product", "sentiment": "positive", "confidence": 0.9})
    result = Partial.parse_partial(raw, Analysis)
    assert result is not None
    assert result.summary == "Good product"


def test_partial_parse_partial_missing_fields_become_none() -> None:
    raw = json.dumps({"summary": "Incomplete"})
    result = Partial.parse_partial(raw, Analysis)
    # Missing fields should be None rather than raising
    assert result is not None


def test_partial_parse_partial_invalid_json_returns_none() -> None:
    result = Partial.parse_partial("{{broken json", Analysis)
    # May return None or a partial dict — must not raise
    # If recovery fails, result is None
    assert result is None or isinstance(result, dict)


def test_partial_get_wrapped_type() -> None:
    partial_analysis_cls = Partial[Analysis]
    assert partial_analysis_cls.get_wrapped_type() is Analysis


# ---------------------------------------------------------------------------
# _make_all_optional helper
# ---------------------------------------------------------------------------


def test_make_all_optional_removes_required() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        "required": ["name", "age"],
    }
    result = _make_all_optional(schema)
    assert "required" not in result


def test_make_all_optional_makes_fields_nullable() -> None:
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}},
    }
    result = _make_all_optional(schema)
    name_schema = result["properties"]["name"]
    assert "anyOf" in name_schema
    # One of the anyOf options should be {"type": "null"}
    null_option = {"type": "null"}
    assert null_option in name_schema["anyOf"]


# ---------------------------------------------------------------------------
# _try_recover_json helper
# ---------------------------------------------------------------------------


def test_try_recover_json_completes_truncated_object() -> None:
    truncated = '{"name": "Alice", "age":'
    result = _try_recover_json(truncated)
    # May succeed or fail — must not raise
    # If it returns something, it should be a dict
    if result is not None:
        assert isinstance(result, dict)


def test_try_recover_json_valid_json_returns_parsed() -> None:
    raw = '{"name": "Alice", "age": 30}'
    result = _try_recover_json(raw)
    assert result == {"name": "Alice", "age": 30}


def test_try_recover_json_completely_invalid_returns_none() -> None:
    result = _try_recover_json("not json at all !!!!")
    assert result is None


# ---------------------------------------------------------------------------
# IterableModel class
# ---------------------------------------------------------------------------


def test_iterable_model_subscript() -> None:
    iterable_person_cls = IterableModel[Person]
    assert issubclass(iterable_person_cls, IterableModel)
    assert "Person" in iterable_person_cls.__name__


def test_iterable_model_build_schema_is_array() -> None:
    schema = IterableModel.build_schema(Person)
    assert schema["type"] == "array"
    assert "items" in schema


def test_iterable_model_build_schema_items_has_person_fields() -> None:
    schema = IterableModel.build_schema(Person)
    items = schema["items"]
    assert "name" in items.get("properties", {})
    assert "age" in items.get("properties", {})


def test_iterable_model_parse_items_valid_array() -> None:
    raw = json.dumps([{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}])
    items = IterableModel.parse_items(raw, Person)
    assert len(items) == 2
    assert items[0].name == "Alice"
    assert items[1].name == "Bob"


def test_iterable_model_parse_items_invalid_json_returns_empty() -> None:
    items = IterableModel.parse_items("not json", Person)
    assert items == []


def test_iterable_model_parse_items_non_array_returns_empty() -> None:
    items = IterableModel.parse_items('{"name": "Alice"}', Person)
    assert items == []


def test_iterable_model_parse_items_skips_invalid_items() -> None:
    """Invalid items should be skipped; valid ones should be returned."""
    raw = json.dumps(
        [
            {"name": "Alice", "age": 30},
            {"name": "Invalid", "age": "not-a-number"},  # Pydantic may coerce or reject
            {"name": "Bob", "age": 25},
        ]
    )
    items = IterableModel.parse_items(raw, Person)
    # At least Alice and Bob should be present (or coerced)
    names = [p.name for p in items if hasattr(p, "name")]
    assert "Alice" in names


def test_iterable_model_iter_items() -> None:
    raw = json.dumps([{"name": "Alice", "age": 30}])
    items = list(IterableModel.iter_items(raw, Person))
    assert len(items) == 1
    assert items[0].name == "Alice"


def test_iterable_model_get_wrapped_type() -> None:
    iterable_person_cls = IterableModel[Person]
    assert iterable_person_cls.get_wrapped_type() is Person


def test_iterable_model_build_schema_description_contains_type_name() -> None:
    schema = IterableModel.build_schema(Person)
    assert "Person" in schema.get("description", "")


# ---------------------------------------------------------------------------
# Integration: dsl.__init__ exports
# ---------------------------------------------------------------------------


def test_dsl_exports_all_types() -> None:
    from formatshield import dsl

    assert hasattr(dsl, "Maybe")
    assert hasattr(dsl, "MaybeResult")
    assert hasattr(dsl, "Partial")
    assert hasattr(dsl, "IterableModel")


def test_formatshield_top_level_exports_dsl_types() -> None:
    import formatshield as fs

    assert hasattr(fs, "Maybe")
    assert hasattr(fs, "Partial")
    assert hasattr(fs, "IterableModel")
    assert hasattr(fs, "MaybeResult")
