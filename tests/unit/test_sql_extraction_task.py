"""Unit tests for SQLExtractionTask — covers SQLExtractionTask class methods."""

from __future__ import annotations

import json
from typing import Any

from formatshield.benchmark.tasks.sql_extraction import SQLExtractionTask, _score_sql_extraction


class TestSQLExtractionTask:
    """Tests for the SQLExtractionTask class (lines 325-370)."""

    def setup_method(self) -> None:
        self.task = SQLExtractionTask()

    def test_task_name_attribute(self) -> None:
        assert self.task.name == "sql_extraction"

    def test_task_complexity_attribute(self) -> None:
        assert self.task.complexity == "HIGH"

    def test_task_expected_ttf_benefit(self) -> None:
        assert self.task.expected_ttf_benefit is True

    def test_get_problems_returns_list(self) -> None:
        problems = self.task.get_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0

    def test_get_problems_quick_returns_few(self) -> None:
        problems = self.task.get_problems(quick=True)
        assert 1 <= len(problems) <= 3

    def test_get_problems_full_returns_more(self) -> None:
        problems_quick = self.task.get_problems(quick=True)
        problems_full = self.task.get_problems(quick=False)
        assert len(problems_full) >= len(problems_quick)

    def test_get_problems_have_required_keys(self) -> None:
        for problem in self.task.get_problems(quick=True):
            assert "prompt" in problem
            assert "ground_truth" in problem
            assert "schema" in problem

    def test_get_problems_prompt_is_string(self) -> None:
        for problem in self.task.get_problems(quick=True):
            assert isinstance(problem["prompt"], str)
            assert len(problem["prompt"]) > 0

    def test_get_problems_ground_truth_is_dict(self) -> None:
        for problem in self.task.get_problems(quick=True):
            assert isinstance(problem["ground_truth"], dict)

    def test_get_problems_schema_is_dict(self) -> None:
        for problem in self.task.get_problems(quick=True):
            assert isinstance(problem["schema"], dict)

    def test_score_response_delegates_to_scorer(self) -> None:
        gt = {"query": "SELECT * FROM users", "tables": ["users"]}
        payload = json.dumps({"query": "SELECT * FROM users", "tables": ["users"]})
        result = self.task.score_response(payload, gt)
        assert 0.0 <= result <= 1.0


class TestScoreSQLExtraction:
    """Tests for the module-level _score_sql_extraction function."""

    def test_perfect_match_returns_one(self) -> None:
        gt: dict[str, Any] = {
            "query": "SELECT name FROM employees WHERE dept = 'HR'",
            "tables": ["employees"],
        }
        payload = json.dumps({
            "query": "SELECT name FROM employees WHERE dept = 'HR'",
            "tables": ["employees"],
        })
        assert _score_sql_extraction(payload, gt) == 1.0

    def test_invalid_json_returns_zero(self) -> None:
        assert _score_sql_extraction("not json", {"tables": ["x"]}) == 0.0

    def test_non_dict_json_returns_zero(self) -> None:
        assert _score_sql_extraction(json.dumps([1, 2, 3]), {"tables": ["x"]}) == 0.0

    def test_empty_tables_in_ground_truth_returns_zero(self) -> None:
        gt: dict[str, Any] = {"query": "SELECT 1", "tables": []}
        payload = json.dumps({"query": "SELECT 1", "tables": []})
        assert _score_sql_extraction(payload, gt) == 0.0

    def test_wrong_table_partial_credit(self) -> None:
        gt: dict[str, Any] = {"query": "SELECT * FROM users", "tables": ["users"]}
        payload = json.dumps({"query": "SELECT * FROM orders", "tables": ["orders"]})
        result = _score_sql_extraction(payload, gt)
        assert result == 0.0

    def test_correct_tables_wrong_query_partial_credit(self) -> None:
        gt: dict[str, Any] = {
            "query": "SELECT name FROM employees",
            "tables": ["employees"],
        }
        payload = json.dumps({
            "query": "SELECT * FROM products",
            "tables": ["employees"],
        })
        result = _score_sql_extraction(payload, gt)
        # tables match but query doesn't mention table → partial
        assert 0.0 < result < 1.0

    def test_missing_tables_key_in_prediction(self) -> None:
        gt: dict[str, Any] = {"query": "SELECT 1", "tables": ["users"]}
        payload = json.dumps({"query": "SELECT 1"})
        result = _score_sql_extraction(payload, gt)
        assert result == 0.0
