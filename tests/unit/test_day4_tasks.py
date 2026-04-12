"""
Unit tests for FormatShield Day 4 benchmark tasks.

Covers all six task classes:
  - LegalExtractTask
  - FinancialTask
  - Math500Task
  - ClassificationTask
  - AgentStateTask
  - ToolCallTask

Each task is tested across 8 dimensions:
  1. name attribute
  2. expected_ttf_benefit attribute
  3. complexity attribute
  4. get_problems full count
  5. get_problems quick count
  6. score_response correct (returns 1.0)
  7. score_response wrong (returns 0.0)
  8. build_prompt non-empty and contains input

Plus a shared not-a-dict test for each task.
"""

from __future__ import annotations

from formatshield.benchmark.tasks.agent_state import AgentStateTask
from formatshield.benchmark.tasks.classification import ClassificationTask
from formatshield.benchmark.tasks.financial import FinancialTask
from formatshield.benchmark.tasks.legal_extract import LegalExtractTask
from formatshield.benchmark.tasks.math500 import Math500Task
from formatshield.benchmark.tasks.tool_call import ToolCallTask

# ===========================================================================
# LegalExtractTask
# ===========================================================================


def test_legal_extract_name() -> None:
    """LegalExtractTask must have name == 'legal_extract'."""
    task = LegalExtractTask()
    assert task.name == "legal_extract"


def test_legal_extract_expected_ttf_benefit() -> None:
    """LegalExtractTask.expected_ttf_benefit must be True."""
    task = LegalExtractTask()
    assert task.expected_ttf_benefit is True


def test_legal_extract_complexity() -> None:
    """LegalExtractTask.complexity must be 'HIGH'."""
    task = LegalExtractTask()
    assert task.complexity == "HIGH"


def test_legal_extract_get_problems_full() -> None:
    """LegalExtractTask.get_problems() must return 15 problems."""
    task = LegalExtractTask()
    problems = task.get_problems()
    assert len(problems) == 15


def test_legal_extract_get_problems_quick() -> None:
    """LegalExtractTask.get_problems(quick=True) must return 5 problems."""
    task = LegalExtractTask()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_legal_extract_score_response_correct() -> None:
    """score_response returns 1.0 when parties and obligations are non-empty."""
    task = LegalExtractTask()
    predicted = {
        "parties": ["Alice Corp", "Bob Inc"],
        "effective_date": "2024-01-01",
        "obligations": ["Party A shall pay"],
        "termination_conditions": [],
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 1.0


def test_legal_extract_score_response_wrong() -> None:
    """score_response returns 0.0 when parties and obligations are empty."""
    task = LegalExtractTask()
    predicted = {
        "parties": [],
        "effective_date": "",
        "obligations": [],
        "termination_conditions": [],
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 0.0


def test_legal_extract_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the input text."""
    task = LegalExtractTask()
    sample_text = "This Agreement is entered into between Alpha Inc. and Beta LLC."
    prompt = task.build_prompt(sample_text)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert sample_text in prompt


def test_legal_extract_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = LegalExtractTask()
    ground_truth = task.get_problems()[0]
    assert task.score_response(None, ground_truth) == 0.0
    assert task.score_response("some string", ground_truth) == 0.0


# ===========================================================================
# FinancialTask
# ===========================================================================


def test_financial_name() -> None:
    """FinancialTask must have name == 'financial'."""
    task = FinancialTask()
    assert task.name == "financial"


def test_financial_expected_ttf_benefit() -> None:
    """FinancialTask.expected_ttf_benefit must be True."""
    task = FinancialTask()
    assert task.expected_ttf_benefit is True


def test_financial_complexity() -> None:
    """FinancialTask.complexity must be 'HIGH'."""
    task = FinancialTask()
    assert task.complexity == "HIGH"


def test_financial_get_problems_full() -> None:
    """FinancialTask.get_problems() must return 15 problems."""
    task = FinancialTask()
    problems = task.get_problems()
    assert len(problems) == 15


def test_financial_get_problems_quick() -> None:
    """FinancialTask.get_problems(quick=True) must return 5 problems."""
    task = FinancialTask()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_financial_score_response_correct() -> None:
    """score_response returns 1.0 when revenue_usd is within 5% of expected."""
    task = FinancialTask()
    # First problem: expected_revenue = 2_400_000_000.0
    predicted = {
        "revenue_usd": 1_000_000.0,
        "net_income_usd": 100_000.0,
        "gross_margin_pct": 50.0,
        "yoy_growth_pct": 10.0,
    }
    ground_truth = {"expected_revenue": 1_000_000.0}
    assert task.score_response(predicted, ground_truth) == 1.0


def test_financial_score_response_wrong() -> None:
    """score_response returns 0.0 when revenue_usd is far from expected."""
    task = FinancialTask()
    predicted = {
        "revenue_usd": 999.0,
        "net_income_usd": 100.0,
        "gross_margin_pct": 50.0,
        "yoy_growth_pct": 5.0,
    }
    ground_truth = {"expected_revenue": 1_000_000.0}
    assert task.score_response(predicted, ground_truth) == 0.0


def test_financial_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the input text."""
    task = FinancialTask()
    sample_text = "Revenue for the quarter reached $2.4 billion, up 18%."
    prompt = task.build_prompt(sample_text)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert sample_text in prompt


def test_financial_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = FinancialTask()
    ground_truth = {"expected_revenue": 1_000_000.0}
    assert task.score_response(None, ground_truth) == 0.0
    assert task.score_response("not a dict", ground_truth) == 0.0


# ===========================================================================
# Math500Task
# ===========================================================================


def test_math500_name() -> None:
    """Math500Task must have name == 'math500'."""
    task = Math500Task()
    assert task.name == "math500"


def test_math500_expected_ttf_benefit() -> None:
    """Math500Task.expected_ttf_benefit must be True."""
    task = Math500Task()
    assert task.expected_ttf_benefit is True


def test_math500_complexity() -> None:
    """Math500Task.complexity must be 'HIGH'."""
    task = Math500Task()
    assert task.complexity == "HIGH"


def test_math500_get_problems_full() -> None:
    """Math500Task.get_problems() must return 15 problems."""
    task = Math500Task()
    problems = task.get_problems()
    assert len(problems) == 15


def test_math500_get_problems_quick() -> None:
    """Math500Task.get_problems(quick=True) must return 5 problems."""
    task = Math500Task()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_math500_score_response_correct() -> None:
    """score_response returns 1.0 when final_answer matches first problem's answer.

    The first problem is the linear system 3x+2y=16, 5x-y=9, answer: 'x=2, y=5'.
    Scoring is case-insensitive and whitespace-insensitive.
    """
    task = Math500Task()
    first_problem = task.get_problems()[0]
    # first problem answer is "x=2, y=5" — normalised: "x=2,y=5"
    predicted = {
        "solution_steps": ["Multiply second equation by 2", "Add equations"],
        "final_answer": "x=2, y=5",
        "answer_type": "integer",
    }
    assert task.score_response(predicted, first_problem["answer"]) == 1.0


def test_math500_score_response_wrong() -> None:
    """score_response returns 0.0 when final_answer is incorrect."""
    task = Math500Task()
    first_problem = task.get_problems()[0]
    predicted = {
        "solution_steps": [],
        "final_answer": "wrong",
        "answer_type": "integer",
    }
    assert task.score_response(predicted, first_problem["answer"]) == 0.0


def test_math500_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the question text."""
    task = Math500Task()
    question = "Solve the system: x + y = 5, x - y = 1."
    prompt = task.build_prompt(question)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert question in prompt


def test_math500_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = Math500Task()
    first_problem = task.get_problems()[0]
    assert task.score_response(None, first_problem["answer"]) == 0.0
    assert task.score_response("x=2, y=5", first_problem["answer"]) == 0.0


# ===========================================================================
# ClassificationTask
# ===========================================================================


def test_classification_name() -> None:
    """ClassificationTask must have name == 'classification'."""
    task = ClassificationTask()
    assert task.name == "classification"


def test_classification_expected_ttf_benefit() -> None:
    """ClassificationTask.expected_ttf_benefit must be False."""
    task = ClassificationTask()
    assert task.expected_ttf_benefit is False


def test_classification_complexity() -> None:
    """ClassificationTask.complexity must be 'LOW'."""
    task = ClassificationTask()
    assert task.complexity == "LOW"


def test_classification_get_problems_full() -> None:
    """ClassificationTask.get_problems() must return 15 problems."""
    task = ClassificationTask()
    problems = task.get_problems()
    assert len(problems) == 15


def test_classification_get_problems_quick() -> None:
    """ClassificationTask.get_problems(quick=True) must return 5 problems."""
    task = ClassificationTask()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_classification_score_response_correct() -> None:
    """score_response returns 1.0 when primary_category matches first problem.

    The first problem's ground truth is 'politics' (Senate climate bill headline).
    """
    task = ClassificationTask()
    first_problem = task.get_problems()[0]
    # first problem primary_category == "politics"
    predicted = {
        "primary_category": "politics",
        "secondary_categories": [],
        "confidence": 0.95,
        "reasoning": "The headline is about a Senate legislative vote.",
    }
    assert task.score_response(predicted, first_problem["primary_category"]) == 1.0


def test_classification_score_response_wrong() -> None:
    """score_response returns 0.0 when primary_category is wrong."""
    task = ClassificationTask()
    first_problem = task.get_problems()[0]
    # first problem is "politics"; predict "sports" instead
    predicted = {
        "primary_category": "sports",
        "secondary_categories": [],
        "confidence": 0.5,
        "reasoning": "Misclassified as sports.",
    }
    assert task.score_response(predicted, first_problem["primary_category"]) == 0.0


def test_classification_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the input text."""
    task = ClassificationTask()
    sample_text = "Scientists discover a new species of deep-sea fish."
    prompt = task.build_prompt(sample_text)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert sample_text in prompt


def test_classification_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = ClassificationTask()
    first_problem = task.get_problems()[0]
    assert task.score_response(None, first_problem["primary_category"]) == 0.0
    assert task.score_response("politics", first_problem["primary_category"]) == 0.0


# ===========================================================================
# AgentStateTask
# ===========================================================================


def test_agent_state_name() -> None:
    """AgentStateTask must have name == 'agent_state'."""
    task = AgentStateTask()
    assert task.name == "agent_state"


def test_agent_state_expected_ttf_benefit() -> None:
    """AgentStateTask.expected_ttf_benefit must be True."""
    task = AgentStateTask()
    assert task.expected_ttf_benefit is True


def test_agent_state_complexity() -> None:
    """AgentStateTask.complexity must be 'HIGH'."""
    task = AgentStateTask()
    assert task.complexity == "HIGH"


def test_agent_state_get_problems_full() -> None:
    """AgentStateTask.get_problems() must return 12 problems."""
    task = AgentStateTask()
    problems = task.get_problems()
    assert len(problems) == 12


def test_agent_state_get_problems_quick() -> None:
    """AgentStateTask.get_problems(quick=True) must return 5 problems."""
    task = AgentStateTask()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_agent_state_score_response_correct() -> None:
    """score_response returns 1.0 when current_goal non-empty and completed_steps >= 1."""
    task = AgentStateTask()
    predicted = {
        "current_goal": "deploy service",
        "completed_steps": ["step 1"],
        "pending_steps": [],
        "blockers": [],
        "confidence": 0.9,
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 1.0


def test_agent_state_score_response_wrong() -> None:
    """score_response returns 0.0 when current_goal is empty or completed_steps is empty."""
    task = AgentStateTask()
    predicted = {
        "current_goal": "",
        "completed_steps": [],
        "pending_steps": [],
        "blockers": [],
        "confidence": 0.0,
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 0.0


def test_agent_state_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the narrative text."""
    task = AgentStateTask()
    narrative = "The agent has completed data ingestion and is now awaiting approval."
    prompt = task.build_prompt(narrative)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert narrative in prompt


def test_agent_state_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = AgentStateTask()
    ground_truth = task.get_problems()[0]
    assert task.score_response(None, ground_truth) == 0.0
    assert task.score_response("agent state string", ground_truth) == 0.0


# ===========================================================================
# ToolCallTask
# ===========================================================================


def test_tool_call_name() -> None:
    """ToolCallTask must have name == 'tool_call'."""
    task = ToolCallTask()
    assert task.name == "tool_call"


def test_tool_call_expected_ttf_benefit() -> None:
    """ToolCallTask.expected_ttf_benefit must be True."""
    task = ToolCallTask()
    assert task.expected_ttf_benefit is True


def test_tool_call_complexity() -> None:
    """ToolCallTask.complexity must be 'MEDIUM'."""
    task = ToolCallTask()
    assert task.complexity == "MEDIUM"


def test_tool_call_get_problems_full() -> None:
    """ToolCallTask.get_problems() must return 15 problems."""
    task = ToolCallTask()
    problems = task.get_problems()
    assert len(problems) == 15


def test_tool_call_get_problems_quick() -> None:
    """ToolCallTask.get_problems(quick=True) must return 5 problems."""
    task = ToolCallTask()
    problems = task.get_problems(quick=True)
    assert len(problems) == 5


def test_tool_call_score_response_correct() -> None:
    """score_response returns 1.0 when tool_name matches first problem's expected_tool.

    The first problem is 'What is the weather like in Tokyo right now? Use Celsius.'
    with expected_tool == 'get_weather'.
    """
    task = ToolCallTask()
    first_problem = task.get_problems()[0]
    # first problem expected_tool == "get_weather"
    predicted = {
        "tool_name": "get_weather",
        "arguments": {"city": "Tokyo", "units": "metric"},
        "reasoning": "The user asked for weather information.",
    }
    assert task.score_response(predicted, first_problem) == 1.0


def test_tool_call_score_response_wrong() -> None:
    """score_response returns 0.0 when tool_name is incorrect."""
    task = ToolCallTask()
    first_problem = task.get_problems()[0]
    # first problem expects "get_weather"; use wrong tool name
    predicted = {
        "tool_name": "search_web",
        "arguments": {"query": "Tokyo weather"},
        "reasoning": "Searched the web instead.",
    }
    assert task.score_response(predicted, first_problem) == 0.0


def test_tool_call_build_prompt() -> None:
    """build_prompt returns a non-empty string containing the request text."""
    task = ToolCallTask()
    request = "What is 42 times 7?"
    prompt = task.build_prompt(request)
    assert isinstance(prompt, str)
    assert len(prompt) > 0
    assert request in prompt


def test_tool_call_score_response_not_dict() -> None:
    """score_response returns 0.0 when predicted is not a dict."""
    task = ToolCallTask()
    first_problem = task.get_problems()[0]
    assert task.score_response(None, first_problem) == 0.0
    assert task.score_response("get_weather", first_problem) == 0.0


# ===========================================================================
# Cross-task sanity checks
# ===========================================================================


def test_all_tasks_have_schema_attribute() -> None:
    """Every task class must expose a 'schema' attribute pointing to a Pydantic model."""
    tasks = [
        LegalExtractTask(),
        FinancialTask(),
        Math500Task(),
        ClassificationTask(),
        AgentStateTask(),
        ToolCallTask(),
    ]
    for task in tasks:
        assert hasattr(task, "schema"), f"{task.name} is missing a 'schema' attribute"
        assert task.schema is not None


def test_all_tasks_quick_problems_are_subset_of_full() -> None:
    """Quick problems must be the first N problems of the full set."""
    tasks = [
        LegalExtractTask(),
        FinancialTask(),
        Math500Task(),
        ClassificationTask(),
        AgentStateTask(),
        ToolCallTask(),
    ]
    for task in tasks:
        full = task.get_problems(quick=False)
        quick = task.get_problems(quick=True)
        assert quick == full[: len(quick)], (
            f"{task.name}: quick problems are not the leading slice of full problems"
        )


def test_all_tasks_problems_are_dicts() -> None:
    """Every problem returned by get_problems must be a dict."""
    tasks = [
        LegalExtractTask(),
        FinancialTask(),
        Math500Task(),
        ClassificationTask(),
        AgentStateTask(),
        ToolCallTask(),
    ]
    for task in tasks:
        for problem in task.get_problems():
            assert isinstance(problem, dict), (
                f"{task.name}: problem is not a dict: {type(problem)}"
            )


def test_math500_first_problem_answer_is_correct() -> None:
    """Verify the first Math500 problem answer string is 'x=2, y=5'."""
    task = Math500Task()
    first = task.get_problems()[0]
    assert first["answer"] == "x=2, y=5"


def test_classification_first_problem_category_is_politics() -> None:
    """Verify the first ClassificationTask problem primary_category is 'politics'."""
    task = ClassificationTask()
    first = task.get_problems()[0]
    assert first["primary_category"] == "politics"


def test_tool_call_first_problem_tool_is_get_weather() -> None:
    """Verify the first ToolCallTask problem expected_tool is 'get_weather'."""
    task = ToolCallTask()
    first = task.get_problems()[0]
    assert first["expected_tool"] == "get_weather"


def test_agent_state_problem_count_is_twelve_not_fifteen() -> None:
    """AgentStateTask is special — it has 12 problems, not 15."""
    task = AgentStateTask()
    problems = task.get_problems()
    assert len(problems) == 12


def test_financial_score_exact_boundary_within_tolerance() -> None:
    """score_response returns 1.0 for revenue exactly at the 5% upper boundary."""
    task = FinancialTask()
    expected = 1_000_000.0
    # Exactly 5% above expected — should still pass (relative_error == 0.05)
    predicted = {"revenue_usd": expected * 1.05}
    ground_truth = {"expected_revenue": expected}
    assert task.score_response(predicted, ground_truth) == 1.0


def test_financial_score_just_outside_tolerance() -> None:
    """score_response returns 0.0 for revenue just beyond the 5% boundary."""
    task = FinancialTask()
    expected = 1_000_000.0
    # 5.1% above expected — just outside the 5% tolerance
    predicted = {"revenue_usd": expected * 1.051}
    ground_truth = {"expected_revenue": expected}
    assert task.score_response(predicted, ground_truth) == 0.0


def test_math500_score_response_case_insensitive() -> None:
    """score_response normalises case — 'X=2, Y=5' matches 'x=2, y=5'."""
    task = Math500Task()
    first_problem = task.get_problems()[0]
    predicted = {
        "solution_steps": ["step"],
        "final_answer": "X=2, Y=5",
        "answer_type": "integer",
    }
    assert task.score_response(predicted, first_problem["answer"]) == 1.0


def test_legal_extract_score_response_whitespace_only_fields() -> None:
    """score_response returns 0.0 when parties/obligations contain only whitespace."""
    task = LegalExtractTask()
    predicted = {
        "parties": ["   "],
        "effective_date": "2024-01-01",
        "obligations": ["\t"],
        "termination_conditions": [],
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 0.0


def test_agent_state_score_response_whitespace_goal_is_wrong() -> None:
    """score_response returns 0.0 when current_goal is only whitespace."""
    task = AgentStateTask()
    predicted = {
        "current_goal": "   ",
        "completed_steps": ["done something"],
        "pending_steps": [],
        "blockers": [],
        "confidence": 0.5,
    }
    ground_truth = task.get_problems()[0]
    assert task.score_response(predicted, ground_truth) == 0.0
