"""Tests for value-based numeric-token folding in the array merge dedupe."""

from __future__ import annotations

import pytest

from nfield.pipeline.s4_extract import _canonical_number, _merge_window_items


class TestCanonicalNumber:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("1,817", "1817"),  # digit-grouping comma
            ("42 820", "42820"),  # digit-grouping space
            ("1,234,567", "1234567"),
            ("( 42,820 )", "-42820"),  # accounting-parenthesis negative
            ("(36)", "-36"),
            ("-42820", "-42820"),
            ("36.10", "36.1"),  # trailing decimal zero
            ("12.50%", "12.5%"),
            ("15 %", "15%"),  # spacing before the percent mark
            ("$1,817", "$1817"),  # currency mark kept, grouping folded
            ("+36", "36"),  # explicit plus is formatting
            ("007", "7"),  # leading zeros are formatting
            ("3.14159", "3.14159"),
            ("0.50", "0.5"),
        ],
    )
    def test_formatting_folds_to_one_value_form(self, raw: str, expected: str) -> None:
        assert _canonical_number(raw) == expected

    @pytest.mark.parametrize(
        "raw",
        [
            "1,5",  # locale decimal comma - ambiguous, never folded
            "1,23,456",  # non-3-digit grouping - ambiguous, never folded
            "Smith, John",  # real text comma
            "2021-05-01",  # date, not a number
            "abc",
            "",
            "36.",  # dangling decimal point
            "1817 dollars",  # number embedded in text is not a numeric token
        ],
    )
    def test_non_numeric_tokens_stay_none(self, raw: str) -> None:
        assert _canonical_number(raw) is None

    def test_meaning_marks_never_fold_away(self) -> None:
        # Sign, percent, and currency change the value's meaning; formatting folds
        # must never merge across them.
        assert _canonical_number("15%") != _canonical_number("15")
        assert _canonical_number("-36") != _canonical_number("36")
        assert _canonical_number("$1817") != _canonical_number("1817")


def _row(fact: str) -> dict[str, str]:
    return {"Fact": fact, "Type": "monetaryItemType"}


class TestObjectRowVariantDedupe:
    def test_grouping_variant_is_one_row(self) -> None:
        merged: list = [_row("1,817")]
        added = _merge_window_items(merged, [_row("1817")])
        assert added == 0
        assert len(merged) == 1

    def test_accounting_negative_matches_minus(self) -> None:
        merged: list = [_row("( 42,820 )")]
        added = _merge_window_items(merged, [_row("-42820")])
        assert added == 0

    def test_sign_difference_is_two_rows(self) -> None:
        merged: list = [_row("42820")]
        added = _merge_window_items(merged, [_row("-42820")])
        assert added == 1
        assert len(merged) == 2

    def test_distinct_values_both_kept(self) -> None:
        merged: list = [_row("1,817")]
        added = _merge_window_items(merged, [_row("1,818")])
        assert added == 1


class TestStringItemNumericDedupe:
    def test_reformatted_numeric_string_is_duplicate(self) -> None:
        merged: list = ["1,817"]
        added = _merge_window_items(merged, ["1817"])
        assert added == 0

    def test_numeric_token_never_contained_in_text(self) -> None:
        # "1817" appears inside the sentence, but a value is not a copy of prose.
        sentence = "The company reported revenue of 1817 in the fourth quarter."
        merged: list = [sentence]
        added = _merge_window_items(merged, ["1817"])
        assert added == 1
        assert merged == [sentence, "1817"]

    def test_text_containment_dedupe_still_works(self) -> None:
        full = "Alpha and Beta. A study of segment routing in large systems. Journal A, 2021."
        merged: list = [full]
        added = _merge_window_items(merged, ["Alpha and Beta. A study of segment routing"])
        assert added == 0
        assert merged == [full]  # the fuller copy stays
