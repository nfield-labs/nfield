"""
ToolCallTask — Tool/function call generation benchmark task for FormatShield.

This task contains 15 hardcoded natural language requests.  Models must
identify which tool to invoke, supply the correct arguments, and explain
their reasoning — all returned as a structured :class:`ToolCall` response.

Selecting the right tool and populating arguments correctly requires
understanding the request intent and mapping it to a schema, making this
a MEDIUM complexity task with an expected benefit from TTF routing.

Complexity: MEDIUM
Expected TTF benefit: True

Available tools described in each prompt:
  - search_web(query: str)
  - get_weather(city: str, units: str)
  - calculate(expression: str)
  - send_email(to: str, subject: str, body: str)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_TOOL_DESCRIPTIONS: str = (
    "search_web(query: str) — Search the internet for information on a topic.\n"
    "get_weather(city: str, units: str) — Get current weather for a city. "
    "units must be 'metric' or 'imperial'.\n"
    "calculate(expression: str) — Evaluate a mathematical expression and return the result.\n"
    "send_email(to: str, subject: str, body: str) — Send an email to a recipient."
)


class ToolCall(BaseModel):
    """Structured schema for a generated tool or function call."""

    tool_name: str
    """Name of the tool to invoke."""

    arguments: dict[str, Any]
    """Key-value arguments to pass to the tool."""

    reasoning: str
    """Brief explanation of why this tool and these arguments were chosen."""


# ---------------------------------------------------------------------------
# 15 hardcoded natural-language requests with ground-truth tool names
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "request": "What is the weather like in Tokyo right now? Use Celsius.",
        "expected_tool": "get_weather",
    },
    {
        "request": "Can you look up the latest news about the 2024 Paris Olympics?",
        "expected_tool": "search_web",
    },
    {
        "request": "What is 347 multiplied by 58?",
        "expected_tool": "calculate",
    },
    {
        "request": (
            "Please email john.smith@example.com to let him know that the project deadline "
            "has been moved to next Friday."
        ),
        "expected_tool": "send_email",
    },
    {
        "request": (
            "I need to know if it will rain in London today. Show me temperatures in Fahrenheit."
        ),
        "expected_tool": "get_weather",
    },
    {
        "request": "Find me information about the Python programming language.",
        "expected_tool": "search_web",
    },
    {
        "request": "What is the square root of 1764?",
        "expected_tool": "calculate",
    },
    {
        "request": (
            "Send an email to hr@company.org with the subject 'Annual Leave Request' "
            "asking for two weeks off starting March 10th."
        ),
        "expected_tool": "send_email",
    },
    {
        "request": "How cold is it in Reykjavik right now? Give me the temperature in Celsius.",
        "expected_tool": "get_weather",
    },
    {
        "request": "Search for recent papers on large language model alignment.",
        "expected_tool": "search_web",
    },
    {
        "request": (
            "If I invest $5,000 at 7% annual interest compounded annually"
            " for 10 years, what is (5000 * 1.07^10)?"
        ),
        "expected_tool": "calculate",
    },
    {
        "request": (
            "Draft and send an email to alice@marketing.io telling her the Q3 campaign "
            "report is ready for her review."
        ),
        "expected_tool": "send_email",
    },
    {
        "request": "What's the weather forecast for Sydney, Australia? Use metric units.",
        "expected_tool": "get_weather",
    },
    {
        "request": "Look up who won the Nobel Prize in Physics in 2023.",
        "expected_tool": "search_web",
    },
    {
        "request": "Calculate (128 / 4) + (17 * 3) - 9 for me.",
        "expected_tool": "calculate",
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class ToolCallTask:
    """
    Tool/function call generation benchmark task.

    Contains 15 hardcoded natural language requests, each mapping to one of
    four available tools: ``search_web``, ``get_weather``, ``calculate``, and
    ``send_email``.

    Models must produce a structured :class:`ToolCall` with the correct
    ``tool_name``, populated ``arguments``, and a brief ``reasoning`` string.
    Scoring checks only whether ``tool_name`` matches the expected tool.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because correctly interpreting the intent of a request and
        mapping it to the right tool benefits from step-by-step reasoning.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "tool_call"
    expected_ttf_benefit: bool = True
    schema = ToolCall
    complexity: str = "MEDIUM"

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        """
        Return the list of benchmark problems.

        Parameters
        ----------
        quick:
            When ``True`` returns only the first 5 problems, enabling fast
            smoke-test runs without hitting rate limits or wasting API budget.

        Returns
        -------
        list[dict]
            Each element has keys:

            ``"request"`` : str
                The natural language request passed to the model.
            ``"expected_tool"`` : str
                The ground-truth tool name used by :meth:`score_response`.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [{"request": p["request"], "expected_tool": p["expected_tool"]} for p in problems]

    def score_response(self, predicted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """
        Score a model response against the ground-truth tool name.

        Only the ``tool_name`` field is evaluated.  The comparison is exact
        (case-sensitive).  Argument correctness is not scored.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`ToolCall` instance produced
            by the model.  Must contain the key ``"tool_name"``.
        ground_truth:
            The ground-truth dict for the request.  Must contain the key
            ``"expected_tool"`` with the correct tool name as a string.

        Returns
        -------
        float
            ``1.0`` if ``tool_name`` exactly matches ``expected_tool``,
            otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        predicted_tool = predicted.get("tool_name")
        if predicted_tool is None:
            return 0.0

        expected_tool = ground_truth.get("expected_tool", "")
        result = 1.0 if str(predicted_tool) == str(expected_tool) else 0.0

        if result == 0.0:
            logger.debug(
                "score_response: tool mismatch — predicted=%r expected=%r",
                predicted_tool,
                expected_tool,
            )

        return result

    def build_prompt(self, request: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        request:
            The natural language request that the model must map to a tool call.

        Returns
        -------
        str
            A formatted prompt listing the available tools and instructing the
            model to generate a structured JSON tool call.
        """
        return (
            "You are an intelligent assistant with access to the following tools:\n\n"
            f"{_TOOL_DESCRIPTIONS}\n\n"
            "Given the user request below, determine which tool to call and what arguments "
            "to pass.  Return your answer as a structured JSON object with exactly these "
            "three fields:\n"
            "  - tool_name: the name of the tool to invoke (must be one of: search_web, "
            "get_weather, calculate, send_email)\n"
            "  - arguments: a JSON object containing the key-value arguments for the tool\n"
            "  - reasoning: a brief one-sentence explanation of why you chose this tool\n\n"
            "Return only the JSON object.  Do not include text outside the JSON.\n\n"
            f"User request: {request}"
        )
