"""Edge-case tests for validation._type_check.

Covers cases the main suite does not:
- NEEDS_REVALIDATION checked via repr() is fragile
- multipleOf float-precision handling
- large integer/float edge cases
- constraint checks on coerced values (not the original type)
- an unknown format string passes silently
- a pattern with regex special characters in the value
"""

from __future__ import annotations

import pytest

from nfield.schema._types import Field
from nfield.validation._type_check import validate_field


def make_field(path: str, ftype: str, constraints: dict | None = None) -> Field:
    return Field(
        path=path, type=ftype, constraints=constraints or {}, parent_path="", schema_node={}
    )


class TestNeedsRevalidationCheck:
    def test_real_sentinel_passes_validation(self):
        """Real NEEDS_REVALIDATION sentinel must pass (not fail) validation."""
        from nfield.extraction._sfep import NEEDS_REVALIDATION

        f = make_field("score", "number")
        valid, err = validate_field(NEEDS_REVALIDATION, f)
        assert valid is True
        assert err is None

    def test_fake_object_with_same_repr_should_not_pass(self):
        """An object with repr 'NEEDS_REVALIDATION' is not the real sentinel."""

        class FakeRepr:
            def __repr__(self) -> str:
                return "NEEDS_REVALIDATION"

        f = make_field("score", "number")
        fake = FakeRepr()
        valid, _err = validate_field(fake, f)
        if valid:
            pytest.xfail(
                "BUG H1: validate_field passes any object with repr 'NEEDS_REVALIDATION'. "
                "Fix: use 'value is NEEDS_REVALIDATION' instead of repr() comparison."
            )


class TestMultipleOfFloatPrecision:
    def test_multiple_of_float_0_1(self):
        """0.3 is a multiple of 0.1 mathematically -- should pass."""
        f = make_field("x", "number", {"multipleOf": 0.1})
        valid, _err = validate_field(0.3, f)
        if not valid:
            pytest.xfail(
                "multipleOf float precision: 0.3 % 0.1 != 0 due to floating point. "
                "Fix: use round(value % multiple, 10) == 0 or math.isclose()."
            )

    def test_multiple_of_float_0_25(self):
        """0.75 is a multiple of 0.25 -- exact in binary, must pass."""
        f = make_field("x", "number", {"multipleOf": 0.25})
        valid, _err = validate_field(0.75, f)
        assert valid is True

    def test_multiple_of_integer_100(self):
        """300 is a multiple of 100 -- must pass."""
        f = make_field("salary", "integer", {"multipleOf": 100})
        valid, _err = validate_field(300, f)
        assert valid is True

    def test_not_multiple_of_raises_violation(self):
        """301 is NOT a multiple of 100 -- must fail."""
        f = make_field("salary", "integer", {"multipleOf": 100})
        valid, err = validate_field(301, f)
        assert not valid
        assert "multipleOf" in (err or "")


class TestUnknownFormatConstraint:
    def test_unknown_format_passes(self):
        """An unknown format like 'phone' has no validator -- must not fail."""
        f = make_field("phone", "string", {"format": "phone"})
        valid, _err = validate_field("+1-555-0100", f)
        assert valid is True

    def test_known_format_email_validates(self):
        """Known email format should still validate correctly."""
        f = make_field("email", "string", {"format": "email"})
        valid, _err = validate_field("notanemail", f)
        assert not valid


class TestPatternConstraint:
    def test_pattern_value_with_parentheses(self):
        r"""Pattern '^\(\d{3}\)' matches '(555)' correctly."""
        f = make_field("phone", "string", {"pattern": r"^\(\d{3}\)"})
        valid, _err = validate_field("(555)0100", f)
        assert valid is True

    def test_pattern_must_use_re_search_not_match(self):
        r"""re.search used (not re.match) -- pattern \d+ matches within string."""
        f = make_field("ref", "string", {"pattern": r"\d+"})
        valid, _err = validate_field("REF-123", f)
        assert valid is True

    def test_invalid_regex_pattern_does_not_crash(self):
        """Invalid regex pattern in constraints should not crash the validator."""
        f = make_field("x", "string", {"pattern": "[invalid"})
        try:
            _valid, _err = validate_field("test", f)
        except Exception as exc:
            pytest.xfail(f"Invalid regex crashes validator: {exc}")


class TestConstraintOnCoercedValue:
    def test_string_number_coerced_and_constraint_checked(self):
        """'110' as number field with maximum=100 should fail constraint."""
        f = make_field("score", "number", {"maximum": 100})
        valid, err = validate_field("110", f)
        assert not valid
        assert "maximum" in (err or "")

    def test_string_integer_within_range_passes(self):
        """'50' for integer field with minimum=0, maximum=100 -- must pass."""
        f = make_field("age", "integer", {"minimum": 0, "maximum": 100})
        valid, _err = validate_field("50", f)
        assert valid is True

    def test_boolean_coerced_from_string_no_conflict(self):
        """'true' coerced to True for boolean field -- must pass."""
        f = make_field("active", "boolean")
        valid, _err = validate_field("true", f)
        assert valid is True


class TestNoneValueConstraints:
    def test_none_skips_all_constraints(self):
        """None value should pass even with tight constraints."""
        f = make_field("code", "string", {"minLength": 100, "pattern": r"^\d+$"})
        valid, err = validate_field(None, f)
        assert valid is True
        assert err is None

    def test_none_skips_minimum_constraint(self):
        """None value skips minimum constraint."""
        f = make_field("age", "integer", {"minimum": 0})
        valid, _err = validate_field(None, f)
        assert valid is True


class TestArrayItemTypeNotValidated:
    def test_array_with_wrong_item_types_still_passes(self):
        """[1, 2, 'oops'] for integer array -- item types NOT checked by validate_field."""
        f = make_field("ids", "array", {"items": {"type": "integer"}})
        valid, _err = validate_field([1, 2, "oops"], f)
        assert valid is True
