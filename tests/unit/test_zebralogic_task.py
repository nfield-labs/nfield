"""Unit tests for the ZebraLogic benchmark task.

No API keys or GPU required — all data is embedded.
"""

from __future__ import annotations

import json

import pytest

from formatshield.benchmark.tasks.zebralogic import get_problems, score_response

# ---------------------------------------------------------------------------
# get_problems
# ---------------------------------------------------------------------------


def test_zebralogic_get_problems_returns_list() -> None:
    """get_problems() must return a list."""
    result = get_problems()
    assert isinstance(result, list)


def test_zebralogic_get_problems_quick_returns_few() -> None:
    """quick=True must return no more than 5 problems."""
    result = get_problems(quick=True)
    assert len(result) >= 1
    assert len(result) <= 5


def test_zebralogic_get_problems_full_returns_many() -> None:
    """quick=False must return at least 8 problems."""
    result = get_problems(quick=False)
    assert len(result) >= 8


def test_zebralogic_problems_have_required_keys() -> None:
    """Every problem must have 'prompt', 'ground_truth', and 'schema' keys."""
    for problem in get_problems():
        assert "prompt" in problem, f"Missing 'prompt': {problem}"
        assert "ground_truth" in problem, f"Missing 'ground_truth': {problem}"
        assert "schema" in problem, f"Missing 'schema': {problem}"


def test_zebralogic_problems_have_puzzle_type() -> None:
    """Every problem must have a non-empty 'puzzle_type' field."""
    for problem in get_problems():
        assert "puzzle_type" in problem, f"Missing 'puzzle_type': {problem}"
        assert isinstance(problem["puzzle_type"], str)
        assert len(problem["puzzle_type"]) > 0


def test_zebralogic_schema_has_reasoning_required() -> None:
    """Every problem schema must list 'reasoning' in its required fields."""
    for problem in get_problems():
        schema = problem["schema"]
        required = schema.get("required", [])
        assert "reasoning" in required, (
            f"'reasoning' not in required for puzzle_type={problem.get('puzzle_type')}"
        )


# ---------------------------------------------------------------------------
# score_response — numeric fields
# ---------------------------------------------------------------------------


def test_zebralogic_score_perfect_numeric() -> None:
    """Correct integer answer must score 1.0."""
    ground_truth = {"min_crossings": 7}
    payload = json.dumps({"min_crossings": 7, "reasoning": "standard solution"})
    assert score_response(payload, ground_truth) == 1.0


def test_zebralogic_score_wrong_numeric() -> None:
    """Wrong integer answer must score 0.0."""
    ground_truth = {"min_crossings": 7}
    payload = json.dumps({"min_crossings": 5, "reasoning": "incorrect"})
    assert score_response(payload, ground_truth) == 0.0


# ---------------------------------------------------------------------------
# score_response — failure modes
# ---------------------------------------------------------------------------


def test_zebralogic_score_invalid_json() -> None:
    """Non-JSON input must score 0.0 without raising."""
    ground_truth = {"min_crossings": 7}
    assert score_response("not json", ground_truth) == 0.0


def test_zebralogic_score_non_dict_ground_truth() -> None:
    """Non-dict ground_truth must score 0.0."""
    assert score_response(json.dumps({"answer": 7}), "seven") == 0.0
    assert score_response(json.dumps({"answer": 7}), 7) == 0.0
    assert score_response(json.dumps({"answer": 7}), None) == 0.0


# ---------------------------------------------------------------------------
# score_response — partial credit
# ---------------------------------------------------------------------------


def test_zebralogic_score_partial_credit() -> None:
    """Matching one of two scoreable fields gives 0.5."""
    ground_truth = {"fish_owner_house": 4, "nationality": "German"}
    # Only fish_owner_house is correct; nationality is wrong.
    payload = json.dumps(
        {"fish_owner_house": 4, "nationality": "Norwegian", "reasoning": "partial"}
    )
    result = score_response(payload, ground_truth)
    assert 0.0 < result < 1.0


def test_zebralogic_score_all_fields_correct() -> None:
    """All scoreable fields correct must give 1.0."""
    ground_truth = {"fish_owner_house": 4, "nationality": "German"}
    payload = json.dumps(
        {"fish_owner_house": 4, "nationality": "German", "reasoning": "full solution"}
    )
    assert score_response(payload, ground_truth) == 1.0


# ---------------------------------------------------------------------------
# score_response — string fields (case-insensitive)
# ---------------------------------------------------------------------------


def test_zebralogic_score_string_field_case_insensitive() -> None:
    """String comparison must be case-insensitive."""
    ground_truth = {"box": "Mixed"}
    payload_lower = json.dumps({"box": "mixed", "reasoning": "logic"})
    payload_upper = json.dumps({"box": "MIXED", "reasoning": "logic"})
    assert score_response(payload_lower, ground_truth) == 1.0
    assert score_response(payload_upper, ground_truth) == 1.0


def test_zebralogic_score_string_field_wrong() -> None:
    """Wrong string value must score 0.0."""
    ground_truth = {"box": "Mixed"}
    payload = json.dumps({"box": "Apples", "reasoning": "wrong"})
    assert score_response(payload, ground_truth) == 0.0


# ---------------------------------------------------------------------------
# score_response — list fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("predicted_ranking", "expected_score"),
    [
        (["A", "B", "C", "D", "E"], 1.0),
        (["B", "A", "C", "D", "E"], 0.0),
    ],
)
def test_zebralogic_score_list_field(predicted_ranking: list[str], expected_score: float) -> None:
    """List comparison is order-sensitive and case-insensitive."""
    ground_truth = {"ranking": ["A", "B", "C", "D", "E"]}
    payload = json.dumps({"ranking": predicted_ranking, "reasoning": "tournament"})
    assert score_response(payload, ground_truth) == expected_score


# ---------------------------------------------------------------------------
# score_response — additional edge cases (uncovered lines)
# ---------------------------------------------------------------------------


def test_zebralogic_score_json_list_not_dict() -> None:
    """JSON that parses to a list (not dict) must score 0.0 — covers line 300."""
    ground_truth = {"min_crossings": 7}
    payload = json.dumps([1, 2, 3])
    assert score_response(payload, ground_truth) == 0.0


def test_zebralogic_score_all_reasoning_fields() -> None:
    """ground_truth with only 'reasoning' key yields 0.0 — covers line 305."""
    ground_truth = {"reasoning": "some explanation"}
    payload = json.dumps({"reasoning": "some explanation"})
    assert score_response(payload, ground_truth) == 0.0


def test_zebralogic_score_numeric_conversion_error() -> None:
    """predicted_val that can't be int-converted scores 0 — covers lines 320-321."""
    ground_truth = {"min_crossings": 7}
    payload = json.dumps({"min_crossings": "not-a-number", "reasoning": "bad"})
    assert score_response(payload, ground_truth) == 0.0


def test_zebralogic_score_nested_dict_partial_credit() -> None:
    """Nested dict ground truth gives 0.5 partial credit — covers lines 325-328."""
    ground_truth = {"wins": {"A": 4, "B": 3}}
    payload = json.dumps({"wins": {"A": 4, "B": 3}, "reasoning": "correct"})
    result = score_response(payload, ground_truth)
    assert result == 0.5


def test_zebralogic_score_nested_dict_not_predicted_as_dict() -> None:
    """Nested dict ground truth with non-dict predicted gives 0.0."""
    ground_truth = {"wins": {"A": 4}}
    payload = json.dumps({"wins": "A wins everything", "reasoning": "wrong"})
    result = score_response(payload, ground_truth)
    assert result == 0.0
