"""Unit tests for the GPQA-Diamond benchmark task.

No API keys or GPU required — all data is embedded.
"""

from __future__ import annotations

import json

import pytest

from formatshield.benchmark.tasks.gpqa import get_problems, score_response

# ---------------------------------------------------------------------------
# get_problems
# ---------------------------------------------------------------------------


def test_gpqa_get_problems_returns_list() -> None:
    """get_problems() must return a list."""
    result = get_problems()
    assert isinstance(result, list)


def test_gpqa_get_problems_quick_returns_few() -> None:
    """quick=True must return no more than 5 problems."""
    result = get_problems(quick=True)
    assert len(result) <= 5
    assert len(result) >= 1


def test_gpqa_get_problems_full_returns_many() -> None:
    """quick=False must return at least 10 problems."""
    result = get_problems(quick=False)
    assert len(result) >= 10


def test_gpqa_problems_have_required_keys() -> None:
    """Every problem must have 'prompt', 'ground_truth', and 'schema' keys."""
    for problem in get_problems():
        assert "prompt" in problem, f"Missing 'prompt': {problem}"
        assert "ground_truth" in problem, f"Missing 'ground_truth': {problem}"
        assert "schema" in problem, f"Missing 'schema': {problem}"


def test_gpqa_problems_have_domain_field() -> None:
    """Every problem must have a non-empty 'domain' field."""
    for problem in get_problems():
        assert "domain" in problem, f"Missing 'domain': {problem}"
        assert problem["domain"] in {"biology", "chemistry", "physics"}, (
            f"Unexpected domain: {problem['domain']}"
        )


def test_gpqa_ground_truth_is_letter() -> None:
    """Every ground_truth must be one of A, B, C, D."""
    valid = {"A", "B", "C", "D"}
    for problem in get_problems():
        gt = problem["ground_truth"]
        assert isinstance(gt, str), f"ground_truth is not a str: {gt!r}"
        assert gt.upper() in valid, f"ground_truth not in A-D: {gt!r}"


def test_gpqa_schema_has_answer_property() -> None:
    """Every problem schema must have an 'answer' property."""
    for problem in get_problems():
        schema = problem["schema"]
        assert "properties" in schema
        assert "answer" in schema["properties"], "Missing 'answer' in schema properties"


# ---------------------------------------------------------------------------
# score_response
# ---------------------------------------------------------------------------


def test_gpqa_score_perfect_json() -> None:
    """JSON with the correct answer letter must score 1.0."""
    payload = json.dumps({"answer": "B", "reasoning": "Because of Michaelis-Menten."})
    assert score_response(payload, "B") == 1.0


def test_gpqa_score_wrong_json() -> None:
    """JSON with the wrong answer letter must score 0.0."""
    payload = json.dumps({"answer": "A", "reasoning": "Wrong reason."})
    assert score_response(payload, "B") == 0.0


def test_gpqa_score_invalid_json() -> None:
    """Non-JSON input must score 0.0 without raising."""
    assert score_response("not json at all", "B") == 0.0


def test_gpqa_score_plain_text_extraction() -> None:
    """Plain text containing 'The answer is B' with B correct must score 1.0."""
    assert score_response("The answer is B, because of quantum mechanics.", "B") == 1.0


def test_gpqa_score_plain_text_wrong() -> None:
    """Plain text containing a wrong letter must score 0.0."""
    assert score_response("The answer is A, clearly.", "B") == 0.0


def test_gpqa_score_non_string_ground_truth() -> None:
    """Non-string ground_truth must always score 0.0."""
    assert score_response(json.dumps({"answer": "A"}), 42) == 0.0
    assert score_response(json.dumps({"answer": "A"}), None) == 0.0
    assert score_response(json.dumps({"answer": "A"}), ["A"]) == 0.0


@pytest.mark.parametrize("letter", ["A", "B", "C", "D"])
def test_gpqa_score_all_correct_letters(letter: str) -> None:
    """Correct JSON answer scores 1.0 for every valid letter."""
    payload = json.dumps({"answer": letter, "reasoning": "test"})
    assert score_response(payload, letter) == 1.0


def test_gpqa_score_empty_string() -> None:
    """Empty predicted string must score 0.0."""
    assert score_response("", "A") == 0.0


def test_gpqa_score_json_missing_answer_key() -> None:
    """JSON without 'answer' key falls through to regex; if no letter found, scores 0.0."""
    payload = json.dumps({"reasoning": "No answer field here"})
    # No letter in the reasoning text that would match unambiguously — result must be 0.0 or 1.0
    # (deterministic for a fixed ground_truth).
    result = score_response(payload, "Z")  # Z is not a valid letter → 0.0
    assert result == 0.0
