"""Tests for value normalization (validation/_normalize.py)."""

from __future__ import annotations

import pytest

from formatshield.schema._types import Field
from formatshield.validation._normalize import normalize_value


def _field(ftype: str, constraints: dict | None = None) -> Field:
    return Field(
        path="f", type=ftype, constraints=constraints or {}, parent_path="", schema_node={}
    )


class TestNumbers:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("$1,234,568", 1234568.0),
            ("1,234,568", 1234568.0),
            ("(2,300)", -2300.0),
            ("12.5%", 12.5),
            ("1234.56", 1234.56),
            ("-42", -42.0),
            ("€1,000", 1000.0),
            ("-$50", -50.0),
            ("$-50", -50.0),
            ("($1,234.56)", -1234.56),
        ],
    )
    def test_number_formats(self, raw, expected):
        assert normalize_value(raw, _field("number")) == expected

    def test_integer_returns_int(self):
        assert normalize_value("$1,234,568", _field("integer")) == 1234568
        assert isinstance(normalize_value("$1,234,568", _field("integer")), int)

    def test_integer_declines_non_integral(self):
        # 42.7 is not an integer → decline (return original for the validator to reject).
        assert normalize_value("42.7", _field("integer")) == "42.7"

    @pytest.mark.parametrize("raw", ["1,23", "1,2345", "nope", "12-34", ""])
    def test_declines_ambiguous_or_nonnumeric(self, raw):
        assert normalize_value(raw, _field("number")) == raw


class TestBooleans:
    @pytest.mark.parametrize(
        ("raw", "expected"), [("Yes", True), ("n", False), ("TRUE", True), ("off", False)]
    )
    def test_bool_forms(self, raw, expected):
        assert normalize_value(raw, _field("boolean")) is expected

    def test_bool_declines_unknown(self):
        assert normalize_value("maybe", _field("boolean")) == "maybe"


class TestStringsEnums:
    def test_strips_surrounding_quotes_and_space(self):
        assert normalize_value('  "hi"  ', _field("string")) == "hi"
        assert normalize_value("'x'", _field("string")) == "x"

    def test_enum_canonical_casing(self):
        f = _field("enum", {"enum": ["Male", "Female"]})
        assert normalize_value("female", f) == "Female"

    def test_enum_no_match_unchanged(self):
        f = _field("enum", {"enum": ["Male", "Female"]})
        assert normalize_value("Other", f) == "Other"


class TestLossless:
    @pytest.mark.parametrize(
        ("value", "ftype"),
        [
            (42, "integer"),
            (3.14, "number"),
            (True, "boolean"),
            (False, "boolean"),
            ("hello", "string"),
        ],
    )
    def test_canonical_value_unchanged(self, value, ftype):
        # normalize(canonical) == canonical: an already-correct value is never altered.
        assert normalize_value(value, _field(ftype)) == value

    def test_idempotent(self):
        f = _field("number")
        once = normalize_value("$1,234,568", f)
        assert normalize_value(once, f) == once

    def test_non_string_passthrough(self):
        assert normalize_value(1234568, _field("number")) == 1234568
        assert normalize_value(None, _field("number")) is None
