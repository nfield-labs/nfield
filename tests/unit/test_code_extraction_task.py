"""Unit tests for CodeExtractionTask — covers CodeExtractionTask class methods."""

from __future__ import annotations

import json
from typing import Any

from formatshield.benchmark.tasks.code_extraction import CodeExtractionTask, _score_code_extraction


class TestCodeExtractionTask:
    """Tests for the CodeExtractionTask class (covers lines 407-433)."""

    def setup_method(self) -> None:
        self.task = CodeExtractionTask()

    def test_task_name(self) -> None:
        assert self.task.name == "code_extraction"

    def test_task_complexity(self) -> None:
        assert self.task.complexity == "MEDIUM"

    def test_task_expected_ttf_benefit(self) -> None:
        assert self.task.expected_ttf_benefit is True

    def test_get_problems_returns_list(self) -> None:
        problems = self.task.get_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0

    def test_get_problems_quick(self) -> None:
        problems = self.task.get_problems(quick=True)
        assert 1 <= len(problems) <= 3

    def test_get_problems_have_required_keys(self) -> None:
        for p in self.task.get_problems(quick=True):
            assert "prompt" in p
            assert "ground_truth" in p
            assert "schema" in p

    def test_get_problems_prompt_contains_code(self) -> None:
        for p in self.task.get_problems(quick=True):
            assert "python" in p["prompt"].lower() or "def " in p["prompt"]

    def test_score_response_delegates(self) -> None:
        gt: dict[str, Any] = {"function_name": "add", "arguments": []}
        payload = json.dumps({"function_name": "add", "arguments": []})
        result = self.task.score_response(payload, gt)
        assert 0.0 <= result <= 1.0


class TestScoreCodeExtraction:
    """Tests for _score_code_extraction (covers lines 357-384)."""

    def test_perfect_function_name_match(self) -> None:
        gt: dict[str, Any] = {
            "function_name": "calculate",
            "arguments": [{"name": "x"}, {"name": "y"}],
        }
        payload = json.dumps(
            {
                "function_name": "calculate",
                "arguments": [{"name": "x"}, {"name": "y"}],
            }
        )
        result = _score_code_extraction(payload, gt)
        assert result == 1.0

    def test_invalid_json(self) -> None:
        assert _score_code_extraction("not json", {}) == 0.0

    def test_non_dict_json(self) -> None:
        assert _score_code_extraction(json.dumps([1, 2]), {}) == 0.0

    def test_wrong_function_name(self) -> None:
        gt: dict[str, Any] = {"function_name": "add", "arguments": []}
        payload = json.dumps({"function_name": "subtract", "arguments": []})
        result = _score_code_extraction(payload, gt)
        assert result < 1.0

    def test_partial_arg_match(self) -> None:
        gt: dict[str, Any] = {
            "function_name": "fn",
            "arguments": [{"name": "a"}, {"name": "b"}],
        }
        payload = json.dumps(
            {
                "function_name": "fn",
                "arguments": [{"name": "a"}, {"name": "c"}],
            }
        )
        result = _score_code_extraction(payload, gt)
        assert 0.5 < result < 1.0

    def test_args_non_list_in_prediction(self) -> None:
        gt: dict[str, Any] = {
            "function_name": "fn",
            "arguments": [{"name": "a"}],
        }
        payload = json.dumps({"function_name": "fn", "arguments": "not a list"})
        result = _score_code_extraction(payload, gt)
        assert 0.0 <= result <= 1.0

    def test_no_args_in_ground_truth(self) -> None:
        gt: dict[str, Any] = {"function_name": "fn", "arguments": []}
        payload = json.dumps({"function_name": "fn", "arguments": []})
        result = _score_code_extraction(payload, gt)
        assert result == 0.5  # function name matches, no args to check
