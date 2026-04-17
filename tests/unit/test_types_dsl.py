"""Tests for the FormatShield type DSL."""

import dataclasses

import pytest

from formatshield.types.dsl import (
    CFG,
    Alternatives,
    Choice,
    JsonSchema,
    KleenePlus,
    KleeneStar,
    Optional,
    QuantifyBetween,
    QuantifyExact,
    QuantifyMaximum,
    QuantifyMinimum,
    Regex,
    Sequence,
    String,
    at_least,
    at_most,
    between,
    cfg,
    either,
    exactly,
    json_schema,
    one_or_more,
    optional,
    python_types_to_terms,
    regex,
    to_regex,
    zero_or_more,
)

# --- Basic Term classes ---


def test_regex_term():
    r = Regex(r"\d+")
    assert r.pattern == r"\d+"


def test_string_term():
    s = String("hello")
    assert s.value == "hello"


def test_choice_term():
    c = Choice(["yes", "no"])
    assert c.items == ["yes", "no"]


def test_json_schema_from_dict():
    import json

    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    js = JsonSchema(schema)
    assert json.loads(js.schema)["type"] == "object"


def test_json_schema_from_string():
    import json

    s = '{"type": "string"}'
    js = JsonSchema(s)
    assert json.loads(js.schema)["type"] == "string"


# --- DSL operators ---


def test_or_operator():
    a = Regex(r"\d+")
    b = Regex(r"[a-z]+")
    result = a | b
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 2


def test_add_operator():
    a = Regex(r"\d+")
    b = String(",")
    result = a + b
    assert isinstance(result, Sequence)
    assert len(result.terms) == 2


def test_or_with_string():
    a = Regex(r"\d+")
    result = a | "hello"
    assert isinstance(result, Alternatives)


def test_add_with_string():
    a = Regex(r"\d+")
    result = a + ","
    assert isinstance(result, Sequence)


# --- DSL factory functions ---


def test_either():
    result = either("yes", "no", "maybe")
    assert isinstance(result, Alternatives)
    assert len(result.terms) == 3
    assert all(isinstance(t, String) for t in result.terms)


def test_optional():
    r = Regex(r"\d+")
    result = optional(r)
    assert isinstance(result, Optional)


def test_optional_with_string():
    result = optional("hello")
    assert isinstance(result, Optional)
    assert isinstance(result.term, String)


def test_exactly():
    r = Regex(r"\d")
    result = exactly(3, r)
    assert isinstance(result, QuantifyExact)
    assert result.count == 3


def test_at_least():
    result = at_least(2, Regex(r"\d"))
    assert isinstance(result, QuantifyMinimum)
    assert result.count == 2


def test_at_most():
    result = at_most(5, Regex(r"\d"))
    assert isinstance(result, QuantifyMaximum)
    assert result.count == 5


def test_between():
    result = between(2, 5, Regex(r"\d"))
    assert isinstance(result, QuantifyBetween)
    assert result.min_count == 2
    assert result.max_count == 5


def test_zero_or_more():
    result = zero_or_more(Regex(r"\d"))
    assert isinstance(result, KleeneStar)


def test_one_or_more():
    result = one_or_more(Regex(r"\d"))
    assert isinstance(result, KleenePlus)


# --- to_regex ---


def test_to_regex_string():
    assert to_regex(String("hello")) == r"hello"


def test_to_regex_regex():
    assert to_regex(Regex(r"\d+")) == r"\d+"


def test_to_regex_choice():
    result = to_regex(Choice(["yes", "no"]))
    assert "yes" in result
    assert "no" in result


def test_to_regex_alternatives():
    a = Regex(r"\d+") | Regex(r"[a-z]+")
    pattern = to_regex(a)
    assert "|" in pattern


def test_to_regex_sequence():
    s = Regex(r"\d+") + String(",") + Regex(r"\d+")
    pattern = to_regex(s)
    assert r"\d+" in pattern


def test_to_regex_optional():
    result = to_regex(optional(Regex(r"\d+")))
    assert "?" in result


def test_to_regex_kleene_star():
    result = to_regex(zero_or_more(Regex(r"\d")))
    assert "*" in result


def test_to_regex_kleene_plus():
    result = to_regex(one_or_more(Regex(r"\d")))
    assert "+" in result


def test_to_regex_quantify_exact():
    result = to_regex(exactly(3, Regex(r"\d")))
    assert "{3}" in result


def test_to_regex_between():
    result = to_regex(between(2, 5, Regex(r"\d")))
    assert "{2,5}" in result


def test_to_regex_json_schema_raises():
    with pytest.raises(TypeError):
        to_regex(JsonSchema({"type": "object"}))


def test_to_regex_cfg_raises():
    with pytest.raises(TypeError):
        to_regex(cfg("start: 'hello'"))


# --- python_types_to_terms ---


def test_python_int_type():
    term = python_types_to_terms(int)
    assert isinstance(term, Regex)
    assert term.matches("42")
    assert term.matches("-5")


def test_python_float_type():
    term = python_types_to_terms(float)
    assert isinstance(term, Regex)
    assert term.matches("3.14")


def test_python_bool_type():
    term = python_types_to_terms(bool)
    assert isinstance(term, Choice)
    assert "true" in term.items
    assert "false" in term.items


def test_python_str_type():
    term = python_types_to_terms(str)
    assert isinstance(term, Regex)


def test_enum_type():
    from enum import StrEnum

    class Color(StrEnum):
        red = "red"
        blue = "blue"

    term = python_types_to_terms(Color)
    assert isinstance(term, Choice)
    assert "red" in term.items
    assert "blue" in term.items


def test_literal_type():
    from typing import Literal

    term = python_types_to_terms(Literal["yes", "no"])
    assert isinstance(term, Choice)
    assert "yes" in term.items


def test_optional_type():
    term = python_types_to_terms(int | None)
    assert isinstance(term, Optional)


def test_union_type():
    term = python_types_to_terms(int | str)
    assert isinstance(term, Alternatives)
    assert len(term.terms) == 2


def test_pydantic_model():
    from pydantic import BaseModel

    class Person(BaseModel):
        name: str
        age: int

    term = python_types_to_terms(Person)
    assert isinstance(term, JsonSchema)


def test_dict_schema():
    schema = {"type": "object", "properties": {"name": {"type": "string"}}}
    term = python_types_to_terms(schema)
    assert isinstance(term, JsonSchema)


def test_term_passthrough():
    r = Regex(r"\d+")
    assert python_types_to_terms(r) is r


def test_recursion_depth_protection():
    from formatshield.types.dsl import _MAX_RECURSION_DEPTH

    with pytest.raises(RecursionError):
        python_types_to_terms(int, recursion_depth=_MAX_RECURSION_DEPTH + 1)


# --- Pre-built domain types ---


def test_domain_types_exist():
    from formatshield import types

    for attr in [
        "string",
        "integer",
        "number",
        "boolean",
        "date",
        "time",
        "datetime",
        "digit",
        "char",
        "hex_str",
        "uuid4",
        "ipv4",
        "email",
        "isbn",
        "sentence",
        "paragraph",
        "whitespace",
        "newline",
    ]:
        assert hasattr(types, attr), f"types.{attr} missing"


def test_integer_type_is_regex():
    from formatshield.types import integer

    assert isinstance(integer, Regex)


def test_email_type_is_regex():
    from formatshield.types import email

    assert isinstance(email, Regex)


def test_uuid4_type_is_regex():
    from formatshield.types import uuid4

    assert isinstance(uuid4, Regex)
    # valid UUID4 should match
    assert uuid4.matches("550e8400-e29b-41d4-a716-446655440000")


def test_ipv4_type_is_regex():
    from formatshield.types import ipv4

    assert isinstance(ipv4, Regex)


def test_date_type_matches():
    from formatshield.types import date

    assert date.matches("2024-01-15")


def test_term_matches_method():
    r = Regex(r"\d+")
    assert r.matches("123")
    assert not r.matches("abc")


def test_term_validate_raises():
    r = Regex(r"\d+")
    with pytest.raises(ValueError, match="does not match constraint"):
        r.validate("abc")


# --- Factory function aliases ---


def test_regex_factory():
    r = regex(r"\d+")
    assert isinstance(r, Regex)
    assert r.pattern == r"\d+"


def test_cfg_factory():
    c = cfg("start: 'hello'")
    assert isinstance(c, CFG)
    assert c.definition == "start: 'hello'"


def test_json_schema_factory_dict():
    import json

    schema = {"type": "string"}
    js = json_schema(schema)
    assert isinstance(js, JsonSchema)
    assert json.loads(js.schema)["type"] == "string"


# --- JsonSchema helpers ---


def test_json_schema_is_json_schema_true():
    assert JsonSchema.is_json_schema({"type": "object"})
    assert JsonSchema.is_json_schema({"properties": {"x": {"type": "string"}}})
    assert JsonSchema.is_json_schema({"$schema": "http://json-schema.org/draft-07/schema"})
    assert JsonSchema.is_json_schema({"anyOf": [{"type": "string"}]})


def test_json_schema_is_json_schema_false():
    assert not JsonSchema.is_json_schema("not a dict")
    assert not JsonSchema.is_json_schema(42)
    assert not JsonSchema.is_json_schema({"key": "value"})


# --- Term method chaining ---


def test_term_optional_method():
    r = Regex(r"\d+")
    result = r.optional()
    assert isinstance(result, Optional)


def test_term_exactly_method():
    r = Regex(r"\d")
    result = r.exactly(4)
    assert isinstance(result, QuantifyExact)
    assert result.count == 4


def test_term_at_least_method():
    r = Regex(r"\d")
    result = r.at_least(2)
    assert isinstance(result, QuantifyMinimum)


def test_term_at_most_method():
    r = Regex(r"\d")
    result = r.at_most(5)
    assert isinstance(result, QuantifyMaximum)


def test_term_between_method():
    r = Regex(r"\d")
    result = r.between(1, 3)
    assert isinstance(result, QuantifyBetween)


def test_term_one_or_more_method():
    r = Regex(r"\d")
    result = r.one_or_more()
    assert isinstance(result, KleenePlus)


def test_term_zero_or_more_method():
    r = Regex(r"\d")
    result = r.zero_or_more()
    assert isinstance(result, KleeneStar)


def test_ror_operator():
    a = Regex(r"\d+")
    result = "hello" | a  # type: ignore[operator]
    assert isinstance(result, Alternatives)


def test_radd_operator():
    a = Regex(r"\d+")
    result = "prefix" + a  # type: ignore[operator]
    assert isinstance(result, Sequence)


# --- Uncovered lines: matches/validate/from_file/to_regex edge cases ---


def test_term_validate_success_returns_value() -> None:
    """validate() returns the value when it matches — covers line 100."""
    r = Regex(r"\d+")
    result = r.validate("123")
    assert result == "123"


def test_term_matches_exception_returns_false() -> None:
    """matches() catches exceptions from to_regex — covers lines 83-84.

    JsonSchema raises TypeError in to_regex, so matches() must return False.
    """
    js = JsonSchema({"type": "object"})
    # to_regex(js) raises TypeError; matches() must catch it and return False
    assert js.matches("anything") is False


def test_cfg_from_file(tmp_path) -> None:
    """CFG.from_file() reads grammar from a file — covers lines 142-143."""
    grammar_file = tmp_path / "test.lark"
    grammar_file.write_text("start: 'hello'")
    result = CFG.from_file(str(grammar_file))
    assert isinstance(result, CFG)
    assert "hello" in result.definition


def test_json_schema_from_file(tmp_path) -> None:
    """JsonSchema.from_file() reads schema from a file — covers lines 185-186."""
    import json

    schema_file = tmp_path / "schema.json"
    schema_file.write_text(json.dumps({"type": "string"}))
    result = JsonSchema.from_file(str(schema_file))
    assert isinstance(result, JsonSchema)


def test_json_schema_init_with_type() -> None:
    """JsonSchema.__init__ with a Pydantic model class — covers line 168."""
    from pydantic import BaseModel

    class MyModel(BaseModel):
        name: str
        age: int

    js = JsonSchema(MyModel)
    assert isinstance(js, JsonSchema)
    assert "name" in js.schema


def test_to_regex_quantify_minimum() -> None:
    """to_regex handles QuantifyMinimum — covers line 459."""
    term = at_least(2, Regex(r"\d"))
    pattern = to_regex(term)
    assert "{2,}" in pattern


def test_to_regex_quantify_maximum() -> None:
    """to_regex handles QuantifyMaximum — covers line 461."""
    term = at_most(5, Regex(r"\d"))
    pattern = to_regex(term)
    assert "{0,5}" in pattern


def test_to_regex_unknown_type_raises() -> None:
    """to_regex raises TypeError for unknown Term — covers line 471."""

    class FakeTerm:
        pass

    with pytest.raises(TypeError, match="Unknown Term type"):
        to_regex(FakeTerm())  # type: ignore[arg-type]


def test_python_types_to_terms_dict_type() -> None:
    """dict type → JsonSchema({type: object}) — covers line 595."""
    result = python_types_to_terms(dict)
    assert isinstance(result, JsonSchema)
    assert "object" in result.schema


def test_python_types_to_terms_list_with_inner() -> None:
    """list[int] → JsonSchema array — covers lines 599-607."""
    result = python_types_to_terms(list[int])
    assert isinstance(result, JsonSchema)
    assert "array" in result.schema


def test_python_types_to_terms_list_no_args() -> None:
    """list without type args hits the array branch with no inner — covers line 607.

    list[str] where inner is not JsonSchema falls through; bare list raises TypeError.
    """
    # list[str] → str is a Regex, not JsonSchema → returns plain array JsonSchema
    result = python_types_to_terms(list[str])
    assert isinstance(result, JsonSchema)


def test_python_types_to_terms_unknown_type_raises() -> None:
    """Unknown type raises TypeError — covers line 613."""

    class MyWeirdThing:
        pass

    with pytest.raises(TypeError, match="Cannot convert"):
        python_types_to_terms(MyWeirdThing)  # type: ignore[arg-type]


def test_python_types_to_terms_dataclass() -> None:
    """Dataclass type → JsonSchema — covers lines 569-579."""

    @dataclasses.dataclass
    class Point:
        x: float
        y: float

    result = python_types_to_terms(Point)
    assert isinstance(result, JsonSchema)
    assert "object" in result.schema


def test_json_schema_init_with_dataclass_covers_schema_from_type() -> None:
    """JsonSchema(DataclassType) triggers _schema_from_type's dataclass branch — covers 639-650."""

    @dataclasses.dataclass
    class Coord:
        x: float
        y: float

    js = JsonSchema(Coord)
    assert isinstance(js, JsonSchema)
    assert "object" in js.schema


def test_json_schema_init_non_type_triggers_type_error_branch() -> None:
    """JsonSchema with a non-type non-str non-dict falls into _schema_from_type.

    issubclass(42, BaseModel) raises TypeError → except branch covers lines 639-640.
    """
    js = JsonSchema(42)  # type: ignore[arg-type]
    assert isinstance(js, JsonSchema)
    assert "object" in js.schema


def test_python_types_to_terms_plain_dict_instance() -> None:
    """A plain dict instance → JsonSchema — covers line 610-611."""
    result = python_types_to_terms({"type": "string"})  # type: ignore[arg-type]
    assert isinstance(result, JsonSchema)


def test_python_types_to_terms_typed_dict() -> None:
    """TypedDict → JsonSchema — covers lines 583-591."""
    from typing import TypedDict

    class Point(TypedDict):
        x: int
        y: int

    result = python_types_to_terms(Point)
    assert isinstance(result, JsonSchema)
    assert "object" in result.schema


def test_python_types_to_terms_list_of_dict() -> None:
    """list[dict] → JsonSchema array with items — covers lines 602-606."""
    result = python_types_to_terms(list[dict])
    assert isinstance(result, JsonSchema)
    import json

    schema = json.loads(result.schema)
    assert schema.get("type") == "array"
    assert "items" in schema


def test_python_types_to_terms_list_of_str() -> None:
    """list[str] where inner is Regex → plain array schema — covers line 607."""
    result = python_types_to_terms(list[str])
    assert isinstance(result, JsonSchema)
    import json

    schema = json.loads(result.schema)
    assert schema.get("type") == "array"
