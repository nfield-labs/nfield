"""Unit tests for extraction._sfep - SFEP parser and typecast."""

from __future__ import annotations

import pytest

from nfield.extraction._sfep import (
    NEEDS_REVALIDATION,
    count_unknown_paths,
    parse_sfep,
    parse_sfep_failures,
    parse_sfep_line,
    typecast,
)
from nfield.schema._types import Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_field(path: str, ftype: str, constraints: dict | None = None) -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints=constraints or {},
        parent_path="",
        schema_node={},
    )


class TestMultilineObjectArray:
    """parse_sfep joins a JSON array/object value the model pretty-printed over lines."""

    def _rows_field(self) -> Field:
        items = {
            "type": "object",
            "properties": {"rank": {"type": "integer"}, "name": {"type": "string"}},
        }
        return make_field("rows", "array", {"items": items})

    def test_multiline_array_is_joined_and_parsed(self) -> None:
        field = self._rows_field()
        text = 'rows = [\n  {"rank": 1, "name": "Alice"},\n  {"rank": 2, "name": "Bob"}\n]'
        result = parse_sfep(text, [field])
        assert result["rows"] == [{"rank": 1, "name": "Alice"}, {"rank": 2, "name": "Bob"}]

    def test_multiline_array_stops_at_next_field(self) -> None:
        rows = self._rows_field()
        title = make_field("title", "string")
        # A truncated (never-closing) array must not swallow the following field line.
        text = 'rows = [\n  {"rank": 1, "name": "Alice"},\ntitle = Champs'
        result = parse_sfep(text, [rows, title])
        assert result.get("title") == "Champs"

    def test_singleline_array_still_parses(self) -> None:
        field = self._rows_field()
        result = parse_sfep('rows = [{"rank": 1, "name": "Alice"}]', [field])
        assert result["rows"] == [{"rank": 1, "name": "Alice"}]


class TestCountUnknownPaths:
    """count_unknown_paths - the format-drift signal for out-of-schema paths."""

    def test_no_unknowns_when_all_paths_known(self) -> None:
        fields = [make_field("name", "string"), make_field("age", "integer")]
        assert count_unknown_paths("name = Alice\nage = 30", fields) == 0

    def test_counts_path_outside_schema(self) -> None:
        fields = [make_field("name", "string")]
        assert count_unknown_paths("name = Alice\nfavorite_color = blue", fields) == 1

    def test_unparseable_lines_are_not_counted(self) -> None:
        # Prose / lines without a separator are noise, not invented fields.
        fields = [make_field("name", "string")]
        assert count_unknown_paths("name = Alice\nhere is some prose\n\n", fields) == 0

    def test_multiple_unknowns(self) -> None:
        fields = [make_field("a", "string")]
        assert count_unknown_paths("a = 1\nb = 2\nc = 3", fields) == 2


class TestParseSfepFailures:
    """parse_sfep_failures - capture the raw text of values that could not be cast."""

    def test_uncastable_value_is_captured(self) -> None:
        fields = [make_field("age", "integer")]
        assert parse_sfep_failures("age = abc", fields) == {"age": "abc"}

    def test_castable_value_is_not_a_failure(self) -> None:
        fields = [make_field("age", "integer")]
        assert parse_sfep_failures("age = 30", fields) == {}

    def test_null_is_not_a_failure(self) -> None:
        # NULL coerces to None for every type - it never fails the cast.
        fields = [make_field("age", "integer")]
        assert parse_sfep_failures("age = NULL", fields) == {}

    def test_unknown_path_is_not_a_failure(self) -> None:
        # A path outside the schema is format drift (count_unknown_paths), not a cast error.
        fields = [make_field("age", "integer")]
        assert parse_sfep_failures("weight = abc", fields) == {}

    def test_invalid_enum_member_is_captured(self) -> None:
        fields = [make_field("status", "enum", {"enum": ["open", "closed"]})]
        assert parse_sfep_failures("status = pending", fields) == {"status": "pending"}

    def test_mixed_good_and_bad(self) -> None:
        fields = [make_field("name", "string"), make_field("age", "integer")]
        assert parse_sfep_failures("name = Alice\nage = abc", fields) == {"age": "abc"}


# ---------------------------------------------------------------------------
# parse_sfep_line
# ---------------------------------------------------------------------------


class TestParseSfepLine:
    def test_basic_pair(self):
        assert parse_sfep_line("name = Alice") == ("name", "Alice")

    def test_nested_path(self):
        assert parse_sfep_line("address.city = New York") == ("address.city", "New York")

    def test_value_contains_equals(self):
        assert parse_sfep_line("url = https://example.com?a=1") == (
            "url",
            "https://example.com?a=1",
        )

    def test_leading_whitespace_stripped(self):
        assert parse_sfep_line("  name = Bob  ") == ("name", "Bob  ")

    def test_blank_line_returns_none(self):
        assert parse_sfep_line("") is None
        assert parse_sfep_line("   ") is None

    def test_no_separator_returns_none(self):
        assert parse_sfep_line("name: Alice") is None
        assert parse_sfep_line("just some text") is None

    def test_comment_line_returns_none(self):
        assert parse_sfep_line("# this is a comment") is None

    def test_empty_value_is_valid(self):
        result = parse_sfep_line("field = ")
        assert result == ("field", "")

    def test_array_value(self):
        assert parse_sfep_line("tags = [a, b, c]") == ("tags", "[a, b, c]")


# ---------------------------------------------------------------------------
# typecast - all 8 types
# ---------------------------------------------------------------------------


class TestTypecastBoolean:
    def test_true_lowercase(self):
        f = make_field("active", "boolean")
        assert typecast("true", f) is True

    def test_false_lowercase(self):
        f = make_field("active", "boolean")
        assert typecast("false", f) is False

    def test_true_uppercase(self):
        f = make_field("active", "boolean")
        assert typecast("TRUE", f) is True

    def test_yes_no(self):
        f = make_field("flag", "boolean")
        assert typecast("yes", f) is True
        assert typecast("no", f) is False

    def test_invalid_boolean_raises(self):
        from nfield.exceptions import ExtractionError

        f = make_field("flag", "boolean")
        with pytest.raises(ExtractionError, match="Cannot cast"):
            typecast("maybe", f)


class TestTypecastInteger:
    def test_basic_integer(self):
        f = make_field("age", "integer")
        assert typecast("42", f) == 42

    def test_float_string_integer(self):
        f = make_field("age", "integer")
        assert typecast("30.0", f) == 30

    def test_negative_integer(self):
        f = make_field("balance", "integer")
        assert typecast("-5", f) == -5

    def test_invalid_integer_raises(self):
        from nfield.exceptions import ExtractionError

        f = make_field("age", "integer")
        with pytest.raises(ExtractionError):
            typecast("thirty", f)


class TestTypecastNumber:
    def test_float_value(self):
        f = make_field("price", "number")
        assert typecast("3.14", f) == pytest.approx(3.14)

    def test_integer_as_float(self):
        f = make_field("score", "number")
        assert typecast("10", f) == pytest.approx(10.0)

    def test_invalid_number_raises(self):
        from nfield.exceptions import ExtractionError

        f = make_field("score", "number")
        with pytest.raises(ExtractionError):
            typecast("not_a_number", f)


class TestTypecastString:
    def test_basic_string(self):
        f = make_field("name", "string")
        assert typecast("Alice Smith", f) == "Alice Smith"

    def test_strips_outer_whitespace(self):
        f = make_field("name", "string")
        assert typecast("  Bob  ", f) == "Bob"

    def test_empty_string(self):
        f = make_field("notes", "string")
        assert typecast("", f) == ""


class TestTypecastNull:
    def test_null_sentinel(self):
        f = make_field("value", "null")
        assert typecast("NULL", f) is None

    def test_null_case_insensitive(self):
        f = make_field("value", "string")
        assert typecast("null", f) is None
        assert typecast("NULL", f) is None
        assert typecast("Null", f) is None

    def test_null_on_any_type(self):
        f = make_field("age", "integer")
        assert typecast("NULL", f) is None


class TestTypecastEnum:
    def test_valid_enum(self):
        f = make_field("status", "enum", {"enum": ["active", "inactive", "pending"]})
        assert typecast("active", f) == "active"

    def test_enum_case_insensitive(self):
        f = make_field("status", "enum", {"enum": ["Active", "Inactive"]})
        assert typecast("active", f) == "Active"

    def test_invalid_enum_raises(self):
        from nfield.exceptions import ExtractionError

        f = make_field("status", "enum", {"enum": ["active", "inactive"]})
        with pytest.raises(ExtractionError, match="not a valid enum"):
            typecast("unknown", f)

    def test_empty_enum_constraint_accepts_any(self):
        f = make_field("status", "enum", {})
        assert typecast("anything", f) == "anything"


class TestTypecastArray:
    def test_string_array(self):
        f = make_field("tags", "array", {"items": {"type": "string"}})
        assert typecast("[alpha, beta, gamma]", f) == ["alpha", "beta", "gamma"]

    def test_integer_array(self):
        f = make_field("ids", "array", {"items": {"type": "integer"}})
        result = typecast("[1, 2, 3]", f)
        assert result == [1, 2, 3]

    def test_empty_array(self):
        f = make_field("tags", "array")
        assert typecast("[]", f) == []

    def test_single_element_array(self):
        f = make_field("tags", "array")
        assert typecast("[only]", f) == ["only"]

    def test_bare_comma_list(self):
        f = make_field("tags", "array")
        result = typecast("a, b, c", f)
        assert result == ["a", "b", "c"]

    def test_truncated_quoted_array_salvages_items(self):
        # The model ran out of output mid-list, leaving no closing bracket. Without
        # repair the whole blob collapses into one item; the complete quoted items
        # must survive and the trailing partial item is dropped.
        f = make_field("tags", "array", {"items": {"type": "string"}})
        truncated = '["first item", "second item", "third ite'
        result = typecast(truncated, f)
        assert result == ["first item", "second item"]

    def test_truncated_single_quoted_array_salvages_items(self):
        # Same output cut, but the model quoted items with single quotes; without
        # repair the whole blob collapses into ONE stringified-list item.
        f = make_field("tags", "array", {"items": {"type": "string"}})
        truncated = "['first item, with comma', 'second item', 'third ite"
        result = typecast(truncated, f)
        assert result == ["first item, with comma", "second item"]


class TestTypecastObjectArray:
    def _field(self) -> Field:
        return make_field(
            "rows",
            "array",
            {"items": {"type": "object", "properties": {"seg": {"type": "string"}}}},
        )

    def test_parses_json_array_of_objects(self):
        result = typecast('[{"seg": "ISG", "v": 1}, {"seg": "CSG", "v": 2}]', self._field())
        assert result == [{"seg": "ISG", "v": 1}, {"seg": "CSG", "v": 2}]

    def test_object_array_via_ref_items(self):
        f = make_field("rows", "array", {"items": {"$ref": "#/$defs/Entry"}})
        assert typecast('[{"a": 1}]', f) == [{"a": 1}]

    def test_empty_object_array(self):
        assert typecast("[]", self._field()) == []

    def test_tolerates_text_around_array(self):
        assert typecast('here: [{"seg": "X"}] end', self._field()) == [{"seg": "X"}]

    def test_truncated_array_salvages_complete_rows(self):
        # The model ran out of output mid-list; the complete objects must survive.
        truncated = '[{"seg": "A"}, {"seg": "B"}, {"seg": "C'
        result = typecast(truncated, self._field())
        assert result[:2] == [{"seg": "A"}, {"seg": "B"}]

    def test_truncated_array_with_brace_in_string_value(self):
        # A "}" inside a string value must not fool the repair's brace counter.
        truncated = '[{"seg": "a}b"}, {"seg": "next'
        assert {"seg": "a}b"} in typecast(truncated, self._field())

    def test_misbraced_nested_array_is_repaired(self):
        # LLMs miscount braces in deeply nested single-line JSON ("}}}]" for "}}]}");
        # both entries must still be recovered, not lost to a parse failure.
        misbraced = '[{"seg": "A", "sub": [{"x": 1}]}}}], {"seg": "B", "sub": []}]'
        result = typecast(misbraced, self._field())
        assert [row["seg"] for row in result] == ["A", "B"]

    def test_inner_values_typecast_per_item_schema(self):
        f = make_field(
            "rows",
            "array",
            {
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "integer"},
                        "scale": {"type": "number"},
                        "seg": {"type": "string"},
                    },
                }
            },
        )
        result = typecast('[{"value": "25,026", "scale": "1000000", "seg": "ISG"}]', f)
        assert result == [{"value": 25026, "scale": 1000000, "seg": "ISG"}]
        assert isinstance(result[0]["value"], int)

    def test_untyped_numeric_item_values_coerced_but_strings_kept(self):
        # Loosely-typed item schema: numeric-looking values coerce (comma stripped),
        # while dates / names / codes stay strings.
        f = make_field(
            "rows",
            "array",
            {
                "items": {
                    "type": "object",
                    "properties": {
                        "value": {"description": "amount"},
                        "period": {"type": "string"},
                        "name": {"type": "string"},
                    },
                }
            },
        )
        result = typecast('[{"value": "19,715", "period": "FY2025 Q2", "name": "ISG Group"}]', f)
        assert result == [{"value": 19715, "period": "FY2025 Q2", "name": "ISG Group"}]

    def test_scalar_elements_dropped_not_fatal(self):
        assert typecast('[{"seg": "X"}, 7]', self._field()) == [{"seg": "X"}]


class TestTypecastSentinels:
    def test_needs_revalidation_sentinel(self):
        f = make_field("score", "number")
        result = typecast("NEEDS_REVALIDATION", f)
        assert result is NEEDS_REVALIDATION

    def test_needs_revalidation_is_singleton(self):
        from nfield.extraction._sfep import _NeedsRevalidationType

        assert NEEDS_REVALIDATION is _NeedsRevalidationType()

    def test_needs_revalidation_bool_is_false(self):
        assert not NEEDS_REVALIDATION

    def test_needs_revalidation_repr(self):
        assert repr(NEEDS_REVALIDATION) == "NEEDS_REVALIDATION"


# ---------------------------------------------------------------------------
# parse_sfep - full document parsing
# ---------------------------------------------------------------------------


class TestParseSfep:
    def test_basic_three_fields(self):
        f_name = make_field("name", "string")
        f_age = make_field("age", "integer")
        f_active = make_field("active", "boolean")
        result = parse_sfep(
            "name = Alice\nage = 30\nactive = true",
            [f_name, f_age, f_active],
        )
        assert result == {"name": "Alice", "age": 30, "active": True}

    def test_null_handling(self):
        f = make_field("notes", "string")
        result = parse_sfep("notes = NULL", [f])
        assert result == {"notes": None}

    def test_unknown_paths_skipped(self):
        f = make_field("name", "string")
        result = parse_sfep("name = Alice\nhallucinated = oops", [f])
        assert result == {"name": "Alice"}
        assert "hallucinated" not in result

    def test_malformed_lines_skipped(self):
        f = make_field("name", "string")
        result = parse_sfep("not a valid line\nname = Bob", [f])
        assert result == {"name": "Bob"}

    def test_blank_lines_skipped(self):
        f = make_field("x", "integer")
        result = parse_sfep("\n\nx = 5\n\n", [f])
        assert result == {"x": 5}

    def test_deep_nested_path(self):
        f = make_field("a.b.c.d", "string")
        result = parse_sfep("a.b.c.d = deep", [f])
        assert result == {"a.b.c.d": "deep"}

    def test_needs_revalidation_in_result(self):
        f = make_field("score", "number")
        result = parse_sfep("score = NEEDS_REVALIDATION", [f])
        assert result["score"] is NEEDS_REVALIDATION

    def test_array_in_full_parse(self):
        f = make_field("tags", "array", {"items": {"type": "string"}})
        result = parse_sfep("tags = [python, json, api]", [f])
        assert result == {"tags": ["python", "json", "api"]}

    def test_empty_text_returns_empty_dict(self):
        f = make_field("x", "string")
        assert parse_sfep("", [f]) == {}

    def test_empty_fields_list(self):
        assert parse_sfep("name = Alice", []) == {}

    def test_multiple_types_round_trip(self):
        fields = [
            make_field("s", "string"),
            make_field("i", "integer"),
            make_field("n", "number"),
            make_field("b", "boolean"),
            make_field("nil", "null"),
        ]
        text = "s = hello\ni = 7\nn = 2.5\nb = false\nnil = NULL"
        result = parse_sfep(text, fields)
        assert result == {"s": "hello", "i": 7, "n": pytest.approx(2.5), "b": False, "nil": None}


class TestCommonSuffixStrip:
    def test_repeated_role_clause_stripped(self):
        from nfield.extraction._sfep import _strip_common_item_suffix

        items = [
            "BANK OF AMERICA, N.A., as a Lender",
            "HSBC BANK USA, NATIONAL ASSOCIATION, as a Lender",
            "DEUTSCHE BANK AG, NEW YORK BRANCH, as a Lender",
            "MORGAN STANLEY SENIOR FUNDING, INC., as a Lender",
        ]
        out = _strip_common_item_suffix(items)
        assert out[0] == "BANK OF AMERICA, N.A."
        assert out[2] == "DEUTSCHE BANK AG, NEW YORK BRANCH"

    def test_repeated_entity_descriptor_stripped(self):
        from nfield.extraction._sfep import _strip_common_item_suffix

        items = [
            "ORION FUND II, L.P., a Delaware limited partnership",
            "ORION FUND II PV, L.P., a Delaware limited partnership",
            "ORION GCE CO-INVEST, L.P., a Delaware limited partnership",
            "LABOR IMPACT FUND, L.P.",
        ]
        out = _strip_common_item_suffix(items)
        assert out[0] == "ORION FUND II, L.P."
        assert out[3] == "LABOR IMPACT FUND, L.P."  # untouched, no shared tail

    def test_short_name_suffix_never_stripped(self):
        from nfield.extraction._sfep import _strip_common_item_suffix

        items = ["Acme, Inc.", "Globex, Inc.", "Initech, Inc.", "Umbrella, Inc."]
        assert _strip_common_item_suffix(items) == items  # 1 word, part of the name

    def test_citations_with_distinct_tails_untouched(self):
        from nfield.extraction._sfep import _strip_common_item_suffix

        items = [
            "Alpha and Beta. Routing at scale. Journal A, 2021.",
            "Gamma and Delta. Windowed extraction. Journal B, 2022.",
            "Epsilon and Zeta. Bounded outputs. Journal C, 2023.",
        ]
        assert _strip_common_item_suffix(items) == items
