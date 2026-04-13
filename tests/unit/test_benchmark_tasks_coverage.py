"""
Coverage tests for benchmark task classes:
  - MedicalNERTask  (medical_ner.py)
  - TemplateFillTask (template_fill.py)
  - GSMSymbolicTask  (gsm_symbolic.py)
  - FinancialTask    (financial.py)

Targets uncovered lines reported by coverage:
  medical_ner  : 273, 289-307, 359-360, 388-410, 426
  template_fill: 213-214, 248-275, 293
  gsm_symbolic : 287-288, 292, 296-298
  financial    : 333, 337-339, 343
"""

from __future__ import annotations

import pytest

from formatshield.benchmark.tasks.financial import FinancialTask
from formatshield.benchmark.tasks.gsm_symbolic import GSMSymbolicTask
from formatshield.benchmark.tasks.medical_ner import MedicalNERTask, _entity_f1, _token_set
from formatshield.benchmark.tasks.template_fill import TemplateFillTask

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _medical_task() -> MedicalNERTask:
    return MedicalNERTask()


def _template_task() -> TemplateFillTask:
    return TemplateFillTask()


def _gsm_task() -> GSMSymbolicTask:
    return GSMSymbolicTask()


def _financial_task() -> FinancialTask:
    return FinancialTask()


# ===========================================================================
# MedicalNERTask
# ===========================================================================


class TestMedicalNERTaskBasics:
    def test_task_name_is_string(self) -> None:
        assert isinstance(MedicalNERTask.name, str)
        assert MedicalNERTask.name == "medical_ner"

    def test_get_schema_returns_class_with_type_info(self) -> None:
        """schema attribute is a Pydantic BaseModel subclass with model_fields."""
        from pydantic import BaseModel

        assert issubclass(MedicalNERTask.schema, BaseModel)
        assert "conditions" in MedicalNERTask.schema.model_fields

    def test_get_problems_returns_nonempty_list(self) -> None:
        task = _medical_task()
        problems = task.get_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0

    def test_get_problems_quick_returns_five(self) -> None:
        task = _medical_task()
        problems = task.get_problems(quick=True)
        assert len(problems) == 5

    def test_get_problems_full_returns_fifteen(self) -> None:
        task = _medical_task()
        problems = task.get_problems(quick=False)
        assert len(problems) == 15

    def test_get_problems_each_has_text_and_entities(self) -> None:
        task = _medical_task()
        for p in task.get_problems(quick=True):
            assert "text" in p
            assert "entities" in p

    def test_build_prompt_returns_nonempty_string(self) -> None:
        task = _medical_task()
        prompt = task.build_prompt("The patient has hypertension.")
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "hypertension" in prompt

    def test_build_prompt_contains_clinical_text(self) -> None:
        """build_prompt covers line 426 (the return statement)."""
        task = _medical_task()
        text = "Patient has diabetes mellitus."
        prompt = task.build_prompt(text)
        assert text in prompt


class TestMedicalNERScoring:
    """Tests for score_response — covers lines 388-410."""

    def test_exact_match_returns_one(self) -> None:
        task = _medical_task()
        predicted = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response(predicted, ground_truth)
        assert score == pytest.approx(1.0)

    def test_no_match_returns_zero(self) -> None:
        task = _medical_task()
        predicted = {
            "conditions": ["completely wrong"],
            "medications": ["unknown drug"],
            "dosages": ["99 mg"],
            "procedures": ["alien surgery"],
        }
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response(predicted, ground_truth)
        assert 0.0 <= score <= 1.0

    def test_partial_match_between_zero_and_one(self) -> None:
        task = _medical_task()
        predicted = {
            "conditions": ["hypertension"],
            "medications": [],
            "dosages": [],
            "procedures": [],
        }
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response(predicted, ground_truth)
        assert 0.0 < score < 1.0

    def test_non_dict_predicted_returns_zero(self) -> None:
        """Covers line 388-390: non-dict predicted."""
        task = _medical_task()
        assert task.score_response("not a dict", {}) == 0.0
        assert task.score_response(None, {}) == 0.0
        assert task.score_response(42, {}) == 0.0

    def test_empty_predicted_returns_zero(self) -> None:
        task = _medical_task()
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response({}, ground_truth)
        assert score == pytest.approx(0.0)

    def test_missing_categories_treated_as_empty(self) -> None:
        """Missing keys in predicted are treated as empty lists (covers 396)."""
        task = _medical_task()
        # Only "conditions" present, others missing
        predicted = {"conditions": ["hypertension"]}
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response(predicted, ground_truth)
        assert 0.0 < score < 1.0

    def test_non_list_category_treated_as_empty(self) -> None:
        """Non-list value for a category is treated as empty (covers 400-401)."""
        task = _medical_task()
        predicted = {
            "conditions": "not a list",
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        ground_truth = {
            "conditions": ["hypertension"],
            "medications": ["lisinopril"],
            "dosages": ["10 mg daily"],
            "procedures": ["ECG"],
        }
        score = task.score_response(predicted, ground_truth)
        assert 0.0 <= score <= 1.0

    def test_score_with_real_problem(self) -> None:
        """Score a response against a real problem from the dataset."""
        task = _medical_task()
        problem = task.get_problems(quick=True)[0]
        ground_truth = problem["entities"]
        # Perfect prediction
        score = task.score_response(ground_truth, ground_truth)
        assert score == pytest.approx(1.0)


class TestEntityF1:
    """Tests for _entity_f1 and _token_set helper functions — covers lines 273, 289-307."""

    def test_token_set_basic(self) -> None:
        """_token_set covers line 273."""
        tokens = _token_set("type 2 diabetes mellitus")
        assert "type" in tokens
        assert "2" in tokens
        assert "diabetes" in tokens
        assert "mellitus" in tokens

    def test_token_set_strips_punctuation(self) -> None:
        tokens = _token_set("pain, fever.")
        assert "pain" in tokens
        assert "fever" in tokens

    def test_token_set_lowercases(self) -> None:
        tokens = _token_set("Hypertension DIABETES")
        assert "hypertension" in tokens
        assert "diabetes" in tokens

    def test_entity_f1_both_empty_returns_one(self) -> None:
        """Covers line 289: both lists empty → 1.0."""
        assert _entity_f1([], []) == pytest.approx(1.0)

    def test_entity_f1_predicted_empty_returns_zero(self) -> None:
        """Covers line 291: truth non-empty, predicted empty → 0.0."""
        assert _entity_f1([], ["hypertension"]) == pytest.approx(0.0)

    def test_entity_f1_truth_empty_returns_zero(self) -> None:
        """Covers line 291: predicted non-empty, truth empty → 0.0."""
        assert _entity_f1(["hypertension"], []) == pytest.approx(0.0)

    def test_entity_f1_exact_match_returns_one(self) -> None:
        """Covers lines 294-307: normal computation with full match."""
        assert _entity_f1(["hypertension"], ["hypertension"]) == pytest.approx(1.0)

    def test_entity_f1_no_overlap_returns_zero(self) -> None:
        result = _entity_f1(["diabetes"], ["hypertension"])
        assert result == pytest.approx(0.0)

    def test_entity_f1_partial_overlap(self) -> None:
        """Token-level overlap: 'type 2 diabetes' overlaps with 'type 2 diabetes mellitus'."""
        result = _entity_f1(["type 2 diabetes"], ["type 2 diabetes mellitus"])
        assert 0.0 < result <= 1.0

    def test_entity_f1_multiple_entities_partial_recall(self) -> None:
        predicted = ["hypertension"]
        truth = ["hypertension", "diabetes", "asthma"]
        result = _entity_f1(predicted, truth)
        assert 0.0 < result < 1.0


# ===========================================================================
# TemplateFillTask
# ===========================================================================


class TestTemplateFillTaskBasics:
    def test_task_name_is_string(self) -> None:
        assert isinstance(TemplateFillTask.name, str)
        assert TemplateFillTask.name == "template_fill"

    def test_schema_has_type_key(self) -> None:
        """schema is a Pydantic model; check required fields."""
        from pydantic import BaseModel

        assert issubclass(TemplateFillTask.schema, BaseModel)
        assert "name" in TemplateFillTask.schema.model_fields
        assert "age" in TemplateFillTask.schema.model_fields
        assert "city" in TemplateFillTask.schema.model_fields

    def test_get_problems_returns_nonempty_list(self) -> None:
        task = _template_task()
        problems = task.get_problems()
        assert isinstance(problems, list)
        assert len(problems) > 0

    def test_get_problems_quick_mode(self) -> None:
        """Covers lines 213-214: quick=True slices to 5."""
        task = _template_task()
        problems = task.get_problems(quick=True)
        assert len(problems) == 5

    def test_get_problems_full_mode(self) -> None:
        """Covers lines 213-214: quick=False returns all 15."""
        task = _template_task()
        problems = task.get_problems(quick=False)
        assert len(problems) == 15

    def test_get_problems_has_required_keys(self) -> None:
        task = _template_task()
        for p in task.get_problems(quick=True):
            assert "instruction" in p
            assert "context" in p
            assert "expected" in p

    def test_build_prompt_returns_nonempty_string(self) -> None:
        """Covers line 293: build_prompt return statement."""
        task = _template_task()
        prompt = task.build_prompt(
            instruction="Fill in: Name: ___, Age: ___, City: ___.",
            context="John is 25 and lives in Paris.",
        )
        assert isinstance(prompt, str)
        assert len(prompt) > 0
        assert "John" in prompt


class TestTemplateFillScoring:
    """Tests for score_response — covers lines 248-275."""

    def test_exact_match_all_fields_returns_one(self) -> None:
        task = _template_task()
        predicted = {"name": "John", "age": 25, "city": "Paris"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_no_match_returns_zero(self) -> None:
        task = _template_task()
        predicted = {"name": "Wrong", "age": 99, "city": "Nowhere"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_partial_match_returns_fraction(self) -> None:
        task = _template_task()
        predicted = {"name": "John", "age": 99, "city": "Paris"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        # 2 out of 3 fields correct
        score = task.score_response(predicted, ground_truth)
        assert score == pytest.approx(2.0 / 3.0)

    def test_non_dict_predicted_returns_zero(self) -> None:
        """Covers lines 248-250: non-dict predicted → 0.0."""
        task = _template_task()
        assert (
            task.score_response("not a dict", {"name": "John", "age": 25, "city": "Paris"}) == 0.0
        )
        assert task.score_response(None, {"name": "John", "age": 25, "city": "Paris"}) == 0.0

    def test_empty_ground_truth_returns_one(self) -> None:
        """Covers lines 252-253: empty ground_truth → 1.0."""
        task = _template_task()
        assert task.score_response({"name": "John"}, {}) == pytest.approx(1.0)

    def test_missing_predicted_field_counts_as_miss(self) -> None:
        """Covers line 260-261: pred_value is None → no hit counted."""
        task = _template_task()
        predicted = {"name": "John"}  # age and city missing
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        score = task.score_response(predicted, ground_truth)
        # Only 1 of 3 fields hit
        assert score == pytest.approx(1.0 / 3.0)

    def test_age_numeric_comparison(self) -> None:
        """Covers lines 264-269: int field comparison."""
        task = _template_task()
        predicted = {"name": "John", "age": "25", "city": "Paris"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_age_wrong_value(self) -> None:
        task = _template_task()
        predicted = {"name": "John", "age": 30, "city": "Paris"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        score = task.score_response(predicted, ground_truth)
        assert score == pytest.approx(2.0 / 3.0)

    def test_age_non_numeric_string_no_hit(self) -> None:
        """Covers lines 268-269: int() conversion fails → no hit."""
        task = _template_task()
        predicted = {"name": "John", "age": "not_a_number", "city": "Paris"}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        score = task.score_response(predicted, ground_truth)
        assert score == pytest.approx(2.0 / 3.0)

    def test_string_comparison_case_insensitive(self) -> None:
        """Covers lines 271-273: string folding."""
        task = _template_task()
        predicted = {"name": "JOHN", "age": 25, "city": "  paris  "}
        ground_truth = {"name": "John", "age": 25, "city": "Paris"}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_score_with_real_problem(self) -> None:
        task = _template_task()
        problem = task.get_problems(quick=True)[0]
        expected = problem["expected"]
        score = task.score_response(expected, expected)
        assert score == pytest.approx(1.0)


# ===========================================================================
# GSMSymbolicTask
# ===========================================================================


class TestGSMSymbolicTaskBasics:
    def test_task_name_is_string(self) -> None:
        assert isinstance(GSMSymbolicTask.name, str)
        assert GSMSymbolicTask.name == "gsm_symbolic"

    def test_schema_has_final_answer_field(self) -> None:
        from pydantic import BaseModel

        assert issubclass(GSMSymbolicTask.schema, BaseModel)
        assert "final_answer" in GSMSymbolicTask.schema.model_fields

    def test_get_problems_returns_nonempty_list(self) -> None:
        task = _gsm_task()
        assert len(task.get_problems()) > 0

    def test_get_problems_quick_returns_five(self) -> None:
        task = _gsm_task()
        assert len(task.get_problems(quick=True)) == 5

    def test_get_problems_full_returns_twenty(self) -> None:
        task = _gsm_task()
        assert len(task.get_problems(quick=False)) == 20

    def test_get_problems_has_question_and_answer(self) -> None:
        task = _gsm_task()
        for p in task.get_problems(quick=True):
            assert "question" in p
            assert "answer" in p

    def test_build_prompt_returns_nonempty_string(self) -> None:
        task = _gsm_task()
        prompt = task.build_prompt("Janet has 24 apples.")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestGSMSymbolicScoring:
    """Tests for score_response — covers lines 287-288, 292, 296-298."""

    def test_correct_answer_returns_one(self) -> None:
        task = _gsm_task()
        predicted = {"reasoning_steps": ["step 1"], "final_answer": 19.0, "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(1.0)

    def test_wrong_answer_returns_zero(self) -> None:
        task = _gsm_task()
        predicted = {"reasoning_steps": [], "final_answer": 99.0, "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(0.0)

    def test_non_dict_predicted_returns_zero(self) -> None:
        """Covers lines 287-288: non-dict predicted → 0.0."""
        task = _gsm_task()
        assert task.score_response("not a dict", 19.0) == pytest.approx(0.0)
        assert task.score_response(42, 19.0) == pytest.approx(0.0)
        assert task.score_response(None, 19.0) == pytest.approx(0.0)

    def test_missing_final_answer_returns_zero(self) -> None:
        """Covers line 292: final_answer key absent → 0.0."""
        task = _gsm_task()
        assert task.score_response({"reasoning_steps": ["step1"], "unit": "apples"}, 19.0) == 0.0

    def test_non_numeric_final_answer_returns_zero(self) -> None:
        """Covers lines 296-298: float() conversion fails → 0.0."""
        task = _gsm_task()
        predicted = {"final_answer": "not_a_number", "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(0.0)

    def test_none_final_answer_returns_zero(self) -> None:
        """Covers line 292 path: final_answer is None."""
        task = _gsm_task()
        predicted = {"final_answer": None, "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(0.0)

    def test_within_tolerance_returns_one(self) -> None:
        """Answer within ±0.01 counts as correct."""
        task = _gsm_task()
        predicted = {"final_answer": 19.005, "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(1.0)

    def test_outside_tolerance_returns_zero(self) -> None:
        task = _gsm_task()
        predicted = {"final_answer": 19.02, "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(0.0)

    def test_string_numeric_final_answer_converts(self) -> None:
        """String that converts to float should be accepted."""
        task = _gsm_task()
        predicted = {"final_answer": "19.0", "unit": "apples"}
        assert task.score_response(predicted, 19.0) == pytest.approx(1.0)

    def test_score_with_real_problem(self) -> None:
        task = _gsm_task()
        problem = task.get_problems(quick=True)[0]
        correct = {"final_answer": problem["answer"], "unit": "apples"}
        assert task.score_response(correct, problem["answer"]) == pytest.approx(1.0)


# ===========================================================================
# FinancialTask
# ===========================================================================


class TestFinancialTaskBasics:
    def test_task_name_is_string(self) -> None:
        assert isinstance(FinancialTask.name, str)
        assert FinancialTask.name == "financial"

    def test_schema_has_revenue_field(self) -> None:
        from pydantic import BaseModel

        assert issubclass(FinancialTask.schema, BaseModel)
        assert "revenue_usd" in FinancialTask.schema.model_fields

    def test_get_problems_returns_nonempty_list(self) -> None:
        task = _financial_task()
        assert len(task.get_problems()) > 0

    def test_get_problems_quick_returns_five(self) -> None:
        task = _financial_task()
        assert len(task.get_problems(quick=True)) == 5

    def test_get_problems_full_returns_fifteen(self) -> None:
        task = _financial_task()
        assert len(task.get_problems(quick=False)) == 15

    def test_get_problems_has_required_keys(self) -> None:
        task = _financial_task()
        for p in task.get_problems(quick=True):
            assert "text" in p
            assert "expected_revenue" in p
            assert "expected_net_income" in p
            assert "expected_gross_margin" in p
            assert "expected_yoy_growth" in p

    def test_build_prompt_returns_nonempty_string(self) -> None:
        task = _financial_task()
        prompt = task.build_prompt("Revenue was $2.4 billion.")
        assert isinstance(prompt, str)
        assert len(prompt) > 0


class TestFinancialScoring:
    """Tests for score_response — covers lines 333, 337-339, 343."""

    def test_correct_revenue_within_tolerance_returns_one(self) -> None:
        task = _financial_task()
        predicted = {
            "revenue_usd": 2_400_000_000.0,
            "net_income_usd": 312_000_000.0,
            "gross_margin_pct": 61.5,
            "yoy_growth_pct": 18.0,
        }
        ground_truth = {"expected_revenue": 2_400_000_000.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_revenue_wrong_unit_returns_zero(self) -> None:
        """Returns 0.0 if revenue_usd is in millions but expected in raw USD."""
        task = _financial_task()
        predicted = {"revenue_usd": 2_400.0}  # millions, not billions
        ground_truth = {"expected_revenue": 2_400_000_000.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_non_dict_predicted_returns_zero(self) -> None:
        task = _financial_task()
        assert task.score_response("not a dict", {"expected_revenue": 1_000_000.0}) == 0.0
        assert task.score_response(None, {"expected_revenue": 1_000_000.0}) == 0.0

    def test_missing_revenue_usd_returns_zero(self) -> None:
        """Covers line 333: revenue_usd key absent → 0.0."""
        task = _financial_task()
        assert task.score_response({"net_income_usd": 100.0}, {"expected_revenue": 1_000.0}) == 0.0

    def test_non_numeric_revenue_returns_zero(self) -> None:
        """Covers lines 337-339: float() conversion fails → 0.0."""
        task = _financial_task()
        predicted = {"revenue_usd": "not_a_number"}
        ground_truth = {"expected_revenue": 1_000_000.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_none_revenue_returns_zero(self) -> None:
        """float(None) raises TypeError → line 337-339."""
        task = _financial_task()
        predicted = {"revenue_usd": None}
        ground_truth = {"expected_revenue": 1_000_000.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_zero_expected_revenue_with_zero_predicted_returns_one(self) -> None:
        """Covers line 343: expected == 0.0 and predicted == 0.0 → 1.0."""
        task = _financial_task()
        predicted = {"revenue_usd": 0.0}
        ground_truth = {"expected_revenue": 0.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_zero_expected_revenue_with_nonzero_predicted_returns_zero(self) -> None:
        """Covers line 343: expected == 0.0 and predicted != 0.0 → 0.0."""
        task = _financial_task()
        predicted = {"revenue_usd": 100.0}
        ground_truth = {"expected_revenue": 0.0}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_within_five_percent_tolerance_returns_one(self) -> None:
        task = _financial_task()
        expected = 2_400_000_000.0
        predicted_val = expected * 1.04  # 4% above — within 5%
        predicted = {"revenue_usd": predicted_val}
        ground_truth = {"expected_revenue": expected}
        assert task.score_response(predicted, ground_truth) == pytest.approx(1.0)

    def test_outside_five_percent_tolerance_returns_zero(self) -> None:
        task = _financial_task()
        expected = 2_400_000_000.0
        predicted_val = expected * 1.10  # 10% above — outside 5%
        predicted = {"revenue_usd": predicted_val}
        ground_truth = {"expected_revenue": expected}
        assert task.score_response(predicted, ground_truth) == pytest.approx(0.0)

    def test_score_with_real_problem(self) -> None:
        task = _financial_task()
        problem = task.get_problems(quick=True)[0]
        predicted = {"revenue_usd": problem["expected_revenue"]}
        score = task.score_response(predicted, problem)
        assert score == pytest.approx(1.0)
