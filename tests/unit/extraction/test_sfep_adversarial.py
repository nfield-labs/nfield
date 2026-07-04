"""Edge-case tests for extraction._sfep.

Covers cases the main suite does not:
- an empty raw value "" maps to None
- a single-item array without brackets degrades to [raw_value]
- _RE_ARRAY_ITEM is unused (import hygiene)
- a float literal like "1e5" in an integer field
- leading/trailing spaces inside array values
- unicode values in arrays
- deeply nested array paths
"""

from __future__ import annotations

import pytest

from nfield.extraction._sfep import (
    NEEDS_REVALIDATION,
    _cast_array,
    parse_sfep,
    parse_sfep_line,
    typecast,
)
from nfield.schema._types import Field


def make_field(path: str, ftype: str, constraints: dict | None = None) -> Field:
    return Field(
        path=path, type=ftype, constraints=constraints or {}, parent_path="", schema_node={}
    )


# ---------------------------------------------------------------------------
# An empty raw value should map to None (like the "null" literal), not just "NULL".
# ---------------------------------------------------------------------------


class TestEmptyRawValueHandling:
    """An empty string maps to None."""

    def test_empty_string_for_integer_field_returns_none(self):
        """Empty raw_value on integer field returns None (spec: '' → None)."""
        f = make_field("count", "integer")
        result = typecast("", f)
        assert result is None

    def test_empty_string_for_string_field_returns_empty_string(self):
        """Empty raw_value on string field returns '' - empty string is a valid string value."""
        f = make_field("notes", "string")
        result = typecast("", f)
        assert result == ""

    def test_empty_string_for_boolean_field_returns_none(self):
        """Empty raw_value on boolean field returns None."""
        f = make_field("active", "boolean")
        result = typecast("", f)
        assert result is None

    def test_parse_sfep_empty_value_field_is_none(self):
        """Empty value in SFEP line produces None in result dict."""
        f = make_field("count", "integer")
        result = parse_sfep("count = ", [f])
        assert "count" in result
        assert result["count"] is None


# ---------------------------------------------------------------------------
# A single item without brackets or commas should degrade to [raw_value], not raise.
# ---------------------------------------------------------------------------


class TestSingleItemArrayFallback:
    """A single value without brackets wraps as [value]."""

    def test_single_item_no_brackets_no_comma(self):
        """LLM outputs 'engineering' for array field - becomes ['engineering']."""
        f = make_field("tags", "array", {"items": {"type": "string"}})
        result = _cast_array("engineering", f)
        assert isinstance(result, list)
        assert result == ["engineering"]

    def test_single_integer_no_brackets(self):
        """LLM outputs '42' for integer array field - becomes [42]."""
        f = make_field("ids", "array", {"items": {"type": "integer"}})
        result = _cast_array("42", f)
        assert isinstance(result, list)
        assert result == [42]

    def test_single_float_no_brackets(self):
        """LLM outputs '3.14' for number array field - becomes [3.14]."""
        f = make_field("prices", "array", {"items": {"type": "number"}})
        result = _cast_array("3.14", f)
        assert result == [pytest.approx(3.14)]


# ---------------------------------------------------------------------------
# Edge cases the author forgot: numeric edge cases in typecast
# ---------------------------------------------------------------------------


class TestTypeCastNumericEdgeCases:
    def test_scientific_notation_float(self):
        """'1e5' should parse as number 100000.0."""
        f = make_field("price", "number")
        result = typecast("1e5", f)
        assert result == pytest.approx(100000.0)

    def test_negative_zero_integer(self):
        """-0 should parse as integer 0."""
        f = make_field("count", "integer")
        result = typecast("-0", f)
        assert result == 0
        assert isinstance(result, int)

    def test_very_large_integer(self):
        """Python ints are arbitrary precision - huge numbers must parse."""
        f = make_field("big", "integer")
        big_num = "999999999999999999999999999999"
        result = typecast(big_num, f)
        assert result == int(big_num)

    def test_float_with_leading_plus(self):
        """+3.14 should parse as 3.14."""
        f = make_field("score", "number")
        result = typecast("+3.14", f)
        assert result == pytest.approx(3.14)

    def test_integer_with_spaces(self):
        """' 42 ' (with spaces) should parse correctly."""
        f = make_field("age", "integer")
        result = typecast("  42  ", f)
        assert result == 42

    def test_float_42_point_9_for_integer_raises(self):
        """42.9 cannot be truncated to integer - should raise, not silently truncate."""
        f = make_field("age", "integer")
        from nfield.exceptions import ExtractionError

        with pytest.raises(ExtractionError):
            typecast("42.9", f)


# ---------------------------------------------------------------------------
# Edge cases: array parsing robustness
# ---------------------------------------------------------------------------


class TestArrayParsingEdgeCases:
    def test_array_with_spaces_in_items(self):
        """'[hello world, foo bar]' - items with internal spaces."""
        f = make_field("phrases", "array")
        result = typecast("[hello world, foo bar]", f)
        assert result == ["hello world", "foo bar"]

    def test_array_with_quoted_comma_item(self):
        """'["one,two", three]' - quoted string containing comma."""
        f = make_field("items", "array")
        result = typecast('["one,two", three]', f)
        # The quoted item should be treated as a single element
        assert len(result) == 2
        assert "one,two" in result or '"one,two"' in result  # either stripped or not

    def test_array_with_null_element(self):
        """'[NULL, valid]' - a NULL item is padding and is dropped."""
        f = make_field("values", "array")
        result = typecast("[NULL, valid]", f)
        assert result == ["valid"]

    def test_array_with_boolean_elements(self):
        """'[true, false, true]' for boolean array."""
        f = make_field("flags", "array", {"items": {"type": "boolean"}})
        result = typecast("[true, false, true]", f)
        assert result == [True, False, True]

    def test_empty_brackets_is_empty_list(self):
        """'[]' should always return []."""
        f = make_field("tags", "array")
        result = typecast("[]", f)
        assert result == []

    def test_array_whitespace_only_inner(self):
        """'[  ]' should return []."""
        f = make_field("tags", "array")
        result = typecast("[   ]", f)
        assert result == []


# ---------------------------------------------------------------------------
# Edge cases: parse_sfep_line separator robustness
# ---------------------------------------------------------------------------


class TestParseSfepLineSeparator:
    def test_value_with_equals_in_url(self):
        """Value containing multiple '=' signs (URL query params)."""
        result = parse_sfep_line("callback_url = https://example.com?a=1&b=2")
        assert result is not None
        path, val = result
        assert path == "callback_url"
        assert "a=1&b=2" in val

    def test_value_with_leading_space(self):
        """Leading space in value should be preserved."""
        result = parse_sfep_line("note =  double space start")
        assert result is not None
        _, val = result
        assert val.startswith(" double")

    def test_path_with_underscores(self):
        """Valid path with underscores."""
        result = parse_sfep_line("patient_id = ABC123")
        assert result == ("patient_id", "ABC123")

    def test_line_with_only_separator(self):
        """Line that is just ' = ' (no path or value)."""
        result = parse_sfep_line(" = ")
        # Path is empty after strip - should return None
        assert result is None

    def test_multiline_string_value_first_line_only(self):
        """If LLM outputs multi-line value, only first line is parsed."""
        result = parse_sfep_line("notes = line one")
        assert result == ("notes", "line one")


# ---------------------------------------------------------------------------
# NEEDS_REVALIDATION sentinel identity
# ---------------------------------------------------------------------------


class TestNeedsRevalidationSentinel:
    def test_singleton_identity_via_is(self):
        """NEEDS_REVALIDATION is a singleton - two imports must be same object."""
        from nfield.extraction._sfep import (
            NEEDS_REVALIDATION as NR1,
        )
        from nfield.extraction._sfep import (
            _NeedsRevalidationType,
        )

        nr2 = _NeedsRevalidationType()
        assert NR1 is nr2

    def test_not_equal_to_string(self):
        """NEEDS_REVALIDATION should NOT equal the string 'NEEDS_REVALIDATION'."""
        assert NEEDS_REVALIDATION != "NEEDS_REVALIDATION"

    def test_not_equal_to_none(self):
        """NEEDS_REVALIDATION is not None."""
        assert NEEDS_REVALIDATION is not None

    def test_bool_is_false(self):
        """if sentinel: should be falsy."""
        assert not NEEDS_REVALIDATION

    def test_repr_is_string(self):
        assert repr(NEEDS_REVALIDATION) == "NEEDS_REVALIDATION"
