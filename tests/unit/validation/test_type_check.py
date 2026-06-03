"""Unit tests for validation._type_check -- field type and constraint validation."""

from __future__ import annotations

from formatshield.schema._types import Field
from formatshield.validation._type_check import constraint_check, validate_field


def make_field(path: str, ftype: str, constraints: dict | None = None) -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints=constraints or {},
        parent_path="",
        schema_node={},
    )


class TestValidateFieldTypes:
    def test_valid_string(self):
        f = make_field("name", "string")
        assert validate_field("Alice", f) == (True, None)

    def test_valid_integer(self):
        f = make_field("age", "integer")
        assert validate_field(25, f) == (True, None)

    def test_valid_number(self):
        f = make_field("score", "number")
        assert validate_field(3.14, f) == (True, None)

    def test_valid_boolean_true(self):
        f = make_field("active", "boolean")
        assert validate_field(True, f) == (True, None)

    def test_valid_boolean_false(self):
        f = make_field("active", "boolean")
        assert validate_field(False, f) == (True, None)

    def test_valid_null_value(self):
        f = make_field("notes", "string")
        assert validate_field(None, f) == (True, None)

    def test_valid_array(self):
        f = make_field("tags", "array")
        assert validate_field(["a", "b"], f) == (True, None)

    def test_valid_empty_array(self):
        f = make_field("tags", "array")
        assert validate_field([], f) == (True, None)

    def test_invalid_integer_string(self):
        f = make_field("age", "integer")
        valid, err = validate_field("not_a_number", f)
        assert not valid
        assert err is not None
        assert "integer" in err

    def test_invalid_boolean_string(self):
        f = make_field("active", "boolean")
        valid, err = validate_field("maybe", f)
        assert not valid
        assert err is not None

    def test_invalid_array_is_string(self):
        f = make_field("tags", "array")
        valid, _err = validate_field("not_an_array", f)
        assert not valid

    def test_bool_rejected_as_integer(self):
        f = make_field("count", "integer")
        valid, err = validate_field(True, f)
        assert not valid
        assert "bool" in (err or "")

    def test_bool_rejected_as_number(self):
        f = make_field("score", "number")
        valid, _err = validate_field(False, f)
        assert not valid


class TestValidateFieldCoercion:
    def test_string_integer_coerced(self):
        f = make_field("age", "integer")
        valid, err = validate_field("42", f)
        assert valid
        assert err is None

    def test_float_string_integer_coerced(self):
        f = make_field("age", "integer")
        valid, _err = validate_field("30.0", f)
        assert valid

    def test_string_number_coerced(self):
        f = make_field("price", "number")
        valid, _err = validate_field("9.99", f)
        assert valid

    def test_string_true_coerced_as_boolean(self):
        f = make_field("flag", "boolean")
        valid, _err = validate_field("true", f)
        assert valid

    def test_string_false_coerced_as_boolean(self):
        f = make_field("flag", "boolean")
        valid, _err = validate_field("false", f)
        assert valid


class TestValidateFieldConstraints:
    def test_min_value_pass(self):
        f = make_field("age", "integer", {"minimum": 0})
        assert validate_field(5, f) == (True, None)

    def test_min_value_fail(self):
        f = make_field("age", "integer", {"minimum": 0})
        valid, err = validate_field(-1, f)
        assert not valid
        assert "minimum" in (err or "")

    def test_max_value_pass(self):
        f = make_field("score", "number", {"maximum": 100})
        assert validate_field(99.9, f) == (True, None)

    def test_max_value_fail(self):
        f = make_field("score", "number", {"maximum": 100})
        valid, err = validate_field(101, f)
        assert not valid
        assert "maximum" in (err or "")

    def test_min_length_pass(self):
        f = make_field("code", "string", {"minLength": 3})
        assert validate_field("abc", f) == (True, None)

    def test_min_length_fail(self):
        f = make_field("code", "string", {"minLength": 3})
        valid, err = validate_field("ab", f)
        assert not valid
        assert "minLength" in (err or "")

    def test_max_length_pass(self):
        f = make_field("code", "string", {"maxLength": 5})
        assert validate_field("hello", f) == (True, None)

    def test_max_length_fail(self):
        f = make_field("code", "string", {"maxLength": 5})
        valid, err = validate_field("toolong", f)
        assert not valid
        assert "maxLength" in (err or "")

    def test_pattern_pass(self):
        f = make_field("zip", "string", {"pattern": r"^\d{5}$"})
        assert validate_field("12345", f) == (True, None)

    def test_pattern_fail(self):
        f = make_field("zip", "string", {"pattern": r"^\d{5}$"})
        valid, err = validate_field("1234X", f)
        assert not valid
        assert "pattern" in (err or "")

    def test_enum_membership_pass(self):
        f = make_field("status", "string", {"enum": ["active", "inactive"]})
        assert validate_field("active", f) == (True, None)

    def test_enum_membership_fail(self):
        f = make_field("status", "string", {"enum": ["active", "inactive"]})
        valid, _err = validate_field("unknown", f)
        assert not valid

    def test_format_email_pass(self):
        f = make_field("email", "string", {"format": "email"})
        assert validate_field("user@example.com", f) == (True, None)

    def test_format_email_fail(self):
        f = make_field("email", "string", {"format": "email"})
        valid, err = validate_field("not_an_email", f)
        assert not valid
        assert "format" in (err or "")

    def test_format_date_pass(self):
        f = make_field("dob", "string", {"format": "date"})
        assert validate_field("2024-01-15", f) == (True, None)

    def test_format_date_fail(self):
        f = make_field("dob", "string", {"format": "date"})
        valid, _err = validate_field("15-01-2024", f)
        assert not valid

    def test_exclusive_minimum_pass(self):
        f = make_field("x", "number", {"exclusiveMinimum": 0})
        assert validate_field(0.001, f) == (True, None)

    def test_exclusive_minimum_fail(self):
        f = make_field("x", "number", {"exclusiveMinimum": 0})
        valid, _err = validate_field(0.0, f)
        assert not valid

    def test_min_items_array(self):
        f = make_field("tags", "array", {"minItems": 2})
        valid, _err = validate_field(["only_one"], f)
        assert not valid

    def test_max_items_array(self):
        f = make_field("tags", "array", {"maxItems": 2})
        valid, _err = validate_field([1, 2, 3], f)
        assert not valid


class TestConstraintCheck:
    def test_no_violations_returns_empty(self):
        f = make_field("score", "number", {"minimum": 0, "maximum": 100})
        assert constraint_check(50, f) == []

    def test_multiple_violations_returned(self):
        f = make_field("s", "string", {"minLength": 5, "maxLength": 3})
        violations = constraint_check("abcd", f)
        assert len(violations) >= 1

    def test_returns_strings(self):
        f = make_field("x", "integer", {"minimum": 10})
        violations = constraint_check(5, f)
        assert all(isinstance(v, str) for v in violations)

    def test_no_constraints_returns_empty(self):
        f = make_field("x", "string")
        assert constraint_check("anything", f) == []
