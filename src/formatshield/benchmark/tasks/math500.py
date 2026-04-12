"""
Math500Task — Advanced mathematics benchmark task for FormatShield.

This task contains 15 hardcoded higher-difficulty mathematics problems spanning
algebra, quadratics, sequences, combinatorics, and basic calculus.  Problems
are deliberately harder than GSM-style arithmetic word problems and require
multi-step reasoning to solve.

Models must produce a structured :class:`MathAnswer` with explicit solution
steps, a final answer string, and an answer-type classification.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class MathAnswer(BaseModel):
    """Structured schema for an advanced mathematics problem response."""

    solution_steps: list[str]
    """Ordered list of solution steps, each explaining one transformation."""

    final_answer: str
    """The final answer as a string (integer, fraction, decimal, or expression)."""

    answer_type: Literal["integer", "fraction", "decimal", "expression"]
    """Classification of the answer format."""


# ---------------------------------------------------------------------------
# 15 hardcoded advanced math problems with ground-truth answers
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "question": (
            "Solve the system of equations:\n"
            "  3x + 2y = 16\n"
            "  5x - y = 9\n"
            "Find x and y."
        ),
        "answer": "x=2, y=5",
        "answer_type": "integer",
    },
    {
        "question": (
            "Find all real solutions to the quadratic equation:\n"
            "  2x² - 7x + 3 = 0"
        ),
        "answer": "x=3, x=1/2",
        "answer_type": "fraction",
    },
    {
        "question": (
            "The sum of the first n terms of an arithmetic sequence is given by "
            "S_n = 3n² + 2n.  Find the 10th term of the sequence."
        ),
        "answer": "61",
        "answer_type": "integer",
    },
    {
        "question": (
            "How many ways can a committee of 3 people be chosen from a group of "
            "10 people, if two specific people (Alice and Bob) must not both be "
            "on the same committee?"
        ),
        "answer": "96",
        "answer_type": "integer",
    },
    {
        "question": (
            "A geometric sequence has first term a₁ = 4 and common ratio r = 3. "
            "Find the sum of the first 6 terms."
        ),
        "answer": "1456",
        "answer_type": "integer",
    },
    {
        "question": (
            "Find the derivative of f(x) = 3x⁴ - 5x³ + 2x - 7 and evaluate "
            "it at x = 2."
        ),
        "answer": "54",
        "answer_type": "integer",
    },
    {
        "question": (
            "Simplify the expression:\n"
            "  (x² - 9) / (x² - x - 6)\n"
            "State the result and any values of x for which the expression is undefined."
        ),
        "answer": "(x+3)/(x+2), undefined at x=3 and x=-2",
        "answer_type": "expression",
    },
    {
        "question": (
            "In how many ways can the letters of the word MATHEMATICS be arranged "
            "in a row?  (Account for repeated letters.)"
        ),
        "answer": "4989600",
        "answer_type": "integer",
    },
    {
        "question": (
            "Find the area enclosed between the parabola y = x² and the line y = 4."
        ),
        "answer": "32/3",
        "answer_type": "fraction",
    },
    {
        "question": (
            "Solve for x:\n"
            "  log₂(x + 3) + log₂(x - 1) = 5"
        ),
        "answer": "5",
        "answer_type": "integer",
    },
    {
        "question": (
            "A ball is thrown upward from ground level with an initial velocity of "
            "20 m/s.  Its height in metres after t seconds is h(t) = 20t - 5t². "
            "Find the maximum height reached and the time at which it occurs."
        ),
        "answer": "20 metres at t=2 seconds",
        "answer_type": "integer",
    },
    {
        "question": (
            "Find the number of integer solutions to the inequality:\n"
            "  |2x - 5| < 9"
        ),
        "answer": "9",
        "answer_type": "integer",
    },
    {
        "question": (
            "The roots of the quadratic x² + px + q = 0 are α and β.  "
            "If α + β = 5 and α² + β² = 17, find the values of p and q."
        ),
        "answer": "p=-5, q=4",
        "answer_type": "integer",
    },
    {
        "question": (
            "Evaluate the definite integral:\n"
            "  ∫₁³ (2x² + 3x - 1) dx"
        ),
        "answer": "32",
        "answer_type": "integer",
    },
    {
        "question": (
            "A box contains 6 red balls, 4 blue balls, and 2 green balls.  Three "
            "balls are drawn at random without replacement.  What is the probability "
            "that exactly 2 of the drawn balls are red?  Express as a simplified fraction."
        ),
        "answer": "15/44",
        "answer_type": "fraction",
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5

def _normalise(text: str) -> str:
    """Return a lower-cased, whitespace-stripped comparison string."""
    return text.lower().translate(str.maketrans("", "", " \t"))


class Math500Task:
    """
    Advanced mathematics benchmark task.

    Contains 15 hardcoded higher-difficulty mathematics problems across
    algebra, quadratics, sequences, combinatorics, and basic calculus.
    Problems require multi-step reasoning that goes beyond grade-school
    arithmetic.

    Models must produce a structured :class:`MathAnswer` with explicit
    solution steps, a ``final_answer`` string, and an ``answer_type``
    classification.

    Scoring compares the ``final_answer`` string against the ground-truth
    after whitespace normalisation and lower-casing.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because multi-step algebraic and calculus reasoning
        benefits strongly from the think-then-format routing path.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "math500"
    expected_ttf_benefit: bool = True
    schema = MathAnswer
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
            Each element has keys:

            ``"question"`` : str
                The mathematics problem passed to the model.
            ``"answer"`` : str
                Ground-truth answer string used by :meth:`score_response`.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [{"question": p["question"], "answer": p["answer"]} for p in problems]

    def score_response(self, predicted: dict[str, Any], ground_truth: str) -> float:
        """
        Score a model response against the ground-truth answer string.

        Both the predicted ``final_answer`` and *ground_truth* are normalised
        (lower-cased, whitespace removed) before comparison, so minor
        formatting differences do not penalise otherwise correct answers.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`MathAnswer` instance produced
            by the model.  Must contain the key ``"final_answer"`` with a
            string value.
        ground_truth:
            The expected answer string for the problem.

        Returns
        -------
        float
            ``1.0`` if the normalised predicted answer matches the normalised
            ground truth, otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        raw = predicted.get("final_answer")
        if raw is None:
            return 0.0

        predicted_norm = _normalise(str(raw))
        truth_norm = _normalise(str(ground_truth))

        if predicted_norm == truth_norm:
            return 1.0

        logger.debug(
            "score_response: mismatch — predicted=%r truth=%r",
            predicted_norm,
            truth_norm,
        )
        return 0.0

    def build_prompt(self, question: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        question:
            The mathematics problem text.

        Returns
        -------
        str
            A formatted prompt instructing the model to show its working and
            return a structured JSON answer.
        """
        return (
            "You are an expert mathematics tutor.  Solve the following problem step by step, "
            "showing every transformation or calculation as a separate step.  After your "
            "working, provide a structured JSON answer with three fields:\n"
            "  - solution_steps: list of strings, one per solution step\n"
            "  - final_answer: the final answer as a string (may be an integer, fraction, "
            "decimal, or algebraic expression)\n"
            "  - answer_type: one of 'integer', 'fraction', 'decimal', or 'expression'\n\n"
            "Return only the JSON object.  Do not include text outside the JSON.\n\n"
            f"Problem:\n{question}"
        )
