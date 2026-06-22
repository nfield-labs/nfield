"""Tests for the source-grounding scorer (validation/_grounding)."""

from __future__ import annotations

import pytest

from formatshield.schema._types import Field
from formatshield.validation._grounding import (
    GROUNDABLE_TYPES,
    grounding_score,
    is_groundable,
    is_grounded,
)


def _field(field_type: str, constraints: dict | None = None) -> Field:
    return Field("f", field_type, constraints or {}, "", {})


# ---------------------------------------------------------------------------
# is_groundable — type-awareness (do-no-harm on inferred types)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("field_type", sorted(GROUNDABLE_TYPES))
def test_groundable_types_are_grounded(field_type: str) -> None:
    assert is_groundable(_field(field_type), "value") is True


@pytest.mark.parametrize("field_type", ["boolean", "null", "object", "array"])
def test_inferred_and_structural_types_are_skipped(field_type: str) -> None:
    # Booleans/null/object/array are inferred or structural, not quoted verbatim.
    assert is_groundable(_field(field_type), True) is False


def test_none_value_is_never_grounded() -> None:
    assert is_groundable(_field("string"), None) is False


# ---------------------------------------------------------------------------
# grounding_score — the ladder
# ---------------------------------------------------------------------------


def test_exact_substring_scores_one() -> None:
    assert grounding_score("Acme Corp", "Issued by Acme Corp on Friday.", "string") == 1.0


def test_exact_is_case_insensitive() -> None:
    assert grounding_score("acme corp", "ACME CORP LLC", "string") == 1.0


def test_all_words_present_scores_high_band() -> None:
    # Every token present but not as a contiguous phrase.
    score = grounding_score("Acme Limited", "Acme Holdings Limited group", "string")
    assert score == pytest.approx(0.85)


def test_fuzzy_coverage_band() -> None:
    # Most tokens align in order but one is missing -> fuzzy tier, not all-words.
    text = "the quick brown fox jumped"
    score = grounding_score("quick brown red fox", text, "string")
    assert score == pytest.approx(0.7)


def test_partial_support_scores_low() -> None:
    # One significant word matches; not enough for the fuzzy threshold.
    score = grounding_score("Wellington Aerospace Dynamics", "Wellington is a city", "string")
    assert score == pytest.approx(0.4)


def test_density_gate_rejects_scattered_match() -> None:
    # 3 of 4 value tokens appear (coverage 0.75 passes) but spread across noise, so the
    # density gate rejects the fuzzy tier — it drops to PARTIAL, not FUZZY.
    text = "alpha lorem ipsum dolor sit amet consectetur adipiscing elit sed do beta gamma"
    score = grounding_score("alpha beta gamma delta", text, "string")
    assert score == pytest.approx(0.4)  # PARTIAL, not 0.7 FUZZY


def test_density_gate_accepts_tight_match() -> None:
    # Same coverage but the matched tokens are tight (no noise between) -> FUZZY passes.
    score = grounding_score("alpha beta gamma delta", "alpha beta gamma here", "string")
    assert score == pytest.approx(0.7)


def test_no_support_scores_zero() -> None:
    assert grounding_score("Zeta", "nothing relevant appears here", "string") == 0.0


def test_empty_text_scores_zero() -> None:
    assert grounding_score("anything", "", "string") == 0.0


# ---------------------------------------------------------------------------
# Numeric grounding — formatted variants
# ---------------------------------------------------------------------------


def test_integer_matches_comma_grouped_form() -> None:
    # 1234568 extracted, document writes "1,234,568".
    assert grounding_score(1234568, "revenue was 1,234,568 dollars", "integer") == 1.0


def test_integer_matches_plain_form() -> None:
    assert grounding_score(1947, "founded in 1947", "integer") == 1.0


def test_number_not_in_text_scores_zero() -> None:
    assert grounding_score(9999, "no such figure here", "number") == 0.0


def test_non_finite_numbers_do_not_crash() -> None:
    # An extracted "1e500" casts to inf; int(inf)/int(nan) raise — grounding must not.
    assert grounding_score(float("inf"), "revenue was 100", "number") == 0.0
    assert grounding_score(float("nan"), "revenue was 100", "number") == 0.0
    # inf still grounds via its raw string form if the text happens to contain it.
    assert grounding_score(float("inf"), "value is inf here", "number") == 1.0


# ---------------------------------------------------------------------------
# is_grounded — threshold
# ---------------------------------------------------------------------------


def test_is_grounded_admits_exact_and_words_and_fuzzy() -> None:
    assert is_grounded("Acme Corp", "Acme Corp", "string", min_score=0.5) is True
    assert is_grounded("Acme Limited", "Acme big Limited", "string", min_score=0.5) is True


def test_is_grounded_rejects_partial_and_absent() -> None:
    assert (
        is_grounded("Wellington Aerospace Dynamics", "Wellington city", "string", min_score=0.5)
        is False
    )
    assert is_grounded("Zeta", "nothing here", "string", min_score=0.5) is False
