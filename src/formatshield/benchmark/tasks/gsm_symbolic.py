"""
GSMSymbolicTask — Grade-School Math reasoning benchmark task for FormatShield.

This task contains 20 hardcoded multi-step arithmetic word problems drawn from
the GSM8K-style distribution.  It measures whether Think-Then-Format (TTF)
routing improves structured output accuracy for high-complexity reasoning tasks.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class GSMAnswer(BaseModel):
    """Structured answer schema for a GSM-style math problem."""

    reasoning_steps: list[str]
    """Ordered list of intermediate reasoning steps leading to the final answer."""

    final_answer: float
    """The numeric result of the calculation."""

    unit: str
    """The unit of measurement for the final answer (e.g. 'dollars', 'apples')."""


# ---------------------------------------------------------------------------
# 20 hardcoded grade-school math problems
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": (
            "Janet has 24 apples. She gives half of them to her brother and then "
            "buys 7 more. How many apples does Janet have now?"
        ),
        "answer": 19.0,
        "unit": "apples",
    },
    {
        "question": (
            "A baker makes 3 dozen cookies every morning and sells them for $2 each. "
            "How much money does the baker earn in 5 days?"
        ),
        "answer": 360.0,
        "unit": "dollars",
    },
    {
        "question": (
            "Tom drives at 60 miles per hour for 2.5 hours, then at 80 miles per hour "
            "for 1.5 hours. What is the total distance Tom drove?"
        ),
        "answer": 270.0,
        "unit": "miles",
    },
    {
        "question": (
            "A classroom has 32 students. 3/8 of them are girls. How many boys "
            "are in the classroom?"
        ),
        "answer": 20.0,
        "unit": "boys",
    },
    {
        "question": (
            "Maria earns $15 per hour. She works 8 hours on Monday, 6 hours on "
            "Tuesday, and 9 hours on Wednesday. How much does she earn in total?"
        ),
        "answer": 345.0,
        "unit": "dollars",
    },
    {
        "question": (
            "A swimming pool holds 5000 gallons of water. It drains at 200 gallons "
            "per hour but is simultaneously refilled at 50 gallons per hour. "
            "How many hours will it take to empty the pool completely?"
        ),
        "answer": 33.33,
        "unit": "hours",
    },
    {
        "question": (
            "A store sells notebooks for $3.50 each and pens for $1.25 each. "
            "Sarah buys 4 notebooks and 6 pens. How much does she spend in total?"
        ),
        "answer": 21.5,
        "unit": "dollars",
    },
    {
        "question": (
            "A train travels 450 km in 3 hours. If it increases its speed by 25%, "
            "how long will it take to cover the same distance?"
        ),
        "answer": 2.4,
        "unit": "hours",
    },
    {
        "question": (
            "There are 5 boxes. Each box contains 8 small bags. Each bag contains "
            "12 marbles. How many marbles are there in total?"
        ),
        "answer": 480.0,
        "unit": "marbles",
    },
    {
        "question": (
            "James has $200. He spends 40% on groceries, 25% on utilities, and "
            "saves the rest. How much money does James save?"
        ),
        "answer": 70.0,
        "unit": "dollars",
    },
    {
        "question": (
            "A factory produces 1200 units per day. If the factory runs for 5 days "
            "per week for 4 weeks, and 3% of units are defective, how many "
            "non-defective units are produced in total?"
        ),
        "answer": 23280.0,
        "unit": "units",
    },
    {
        "question": (
            "Emma jogs every morning. On Monday she jogs 3.5 km, on Tuesday 4.2 km, "
            "on Wednesday she rests, on Thursday 5.1 km, and on Friday 2.8 km. "
            "What is her total jogging distance for the week?"
        ),
        "answer": 15.6,
        "unit": "km",
    },
    {
        "question": (
            "A bookshelf has 6 shelves. Each shelf can hold 25 books. The bookshelf "
            "is currently 60% full. How many more books can fit on the bookshelf?"
        ),
        "answer": 60.0,
        "unit": "books",
    },
    {
        "question": (
            "Carlos earns a base salary of $3000 per month plus a 5% commission on "
            "all sales. If his sales this month totalled $28000, what is his total "
            "monthly income?"
        ),
        "answer": 4400.0,
        "unit": "dollars",
    },
    {
        "question": (
            "A recipe requires 2.5 cups of flour to make 24 cookies. How much flour "
            "is needed to make 60 cookies?"
        ),
        "answer": 6.25,
        "unit": "cups",
    },
    {
        "question": (
            "A tank is 1/4 full. After adding 150 litres it becomes 3/4 full. "
            "What is the total capacity of the tank?"
        ),
        "answer": 300.0,
        "unit": "litres",
    },
    {
        "question": (
            "Three friends share a pizza bill equally. The total bill is $45.60 and "
            "they also leave a 20% tip. How much does each person pay in total?"
        ),
        "answer": 18.24,
        "unit": "dollars",
    },
    {
        "question": (
            "A rectangular garden measures 15 metres by 8 metres. A path 1 metre "
            "wide runs around the inside edge of the garden. What is the area of "
            "the path?"
        ),
        "answer": 44.0,
        "unit": "square metres",
    },
    {
        "question": (
            "Alice can paint a wall in 4 hours. Bob can paint the same wall in "
            "6 hours. If they work together, how long will it take them to paint "
            "the wall?"
        ),
        "answer": 2.4,
        "unit": "hours",
    },
    {
        "question": (
            "A school orders 15 boxes of pencils. Each box contains 144 pencils. "
            "The school distributes the pencils equally among 180 students. "
            "How many pencils does each student receive?"
        ),
        "answer": 12.0,
        "unit": "pencils",
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class GSMSymbolicTask:
    """
    Grade-School Math (GSM-Symbolic) benchmark task.

    Contains 20 hardcoded multi-step arithmetic word problems.  Each problem
    requires the model to emit a structured :class:`GSMAnswer` with explicit
    reasoning steps, a numeric final answer, and a unit string.

    This task is classified as HIGH complexity — it benefits from the
    Think-Then-Format (TTF) routing path in FormatShield.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because multi-step reasoning significantly improves when
        the model is allowed to reason freely before committing to JSON.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "gsm_symbolic"
    expected_ttf_benefit: bool = True
    schema = GSMAnswer
    complexity: str = "HIGH"

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
            Each element has the keys:

            ``"question"`` : str
                The word problem text passed to the model as the prompt.
            ``"answer"`` : float
                The expected numeric result, used by :meth:`score_response`.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [{"question": p["question"], "answer": p["answer"]} for p in problems]

    def score_response(self, predicted: dict[str, Any], ground_truth: float) -> float:
        """
        Score a model response against the ground-truth numeric answer.

        The predicted value is accepted as correct when its ``final_answer``
        field is within ``0.01`` (absolute tolerance) of *ground_truth*.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`GSMAnswer` produced by the model,
            typically the result of ``GSMAnswer(**response).model_dump()``.
            Must contain the key ``"final_answer"`` with a numeric value.
        ground_truth:
            The expected numeric answer as a float.

        Returns
        -------
        float
            ``1.0`` if correct (within ±0.01), ``0.0`` otherwise.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        raw = predicted.get("final_answer")
        if raw is None:
            return 0.0

        try:
            predicted_value = float(raw)
        except (TypeError, ValueError):
            logger.debug("score_response: could not convert %r to float", raw)
            return 0.0

        return 1.0 if abs(predicted_value - ground_truth) <= 0.01 else 0.0

    def build_prompt(self, question: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        question:
            The raw word problem text.

        Returns
        -------
        str
            A formatted prompt instructing the model to show its work and
            return a structured JSON answer.
        """
        return (
            "Solve the following math problem step by step.  Show every "
            "intermediate calculation as a separate reasoning step.  After your "
            "reasoning, provide a structured JSON answer with three fields:\n"
            "  - reasoning_steps: a list of strings, one per step\n"
            "  - final_answer: the numeric result (float)\n"
            "  - unit: the unit of the answer (e.g. 'dollars', 'apples')\n\n"
            f"Problem: {question}"
        )
