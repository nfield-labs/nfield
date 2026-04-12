"""
ClassificationTask — Multi-label news classification benchmark task for FormatShield.

This task contains 15 hardcoded news headline and snippet problems.  Models
must assign a primary category, zero or more secondary categories, a confidence
score, and a brief reasoning string.

Because classification into a fixed taxonomy is a relatively shallow reasoning
task, this task is classified LOW complexity.  The TTF routing path is
*not* expected to help here — forcing structured output directly tends to
produce better calibrated responses for simple classification.

Complexity: LOW
Expected TTF benefit: False
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

_VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "politics",
        "technology",
        "business",
        "science",
        "sports",
        "entertainment",
        "health",
    }
)


class ClassificationResult(BaseModel):
    """Structured schema for multi-label news classification."""

    primary_category: str
    """The single most relevant category for the news item."""

    secondary_categories: list[str]
    """Zero or more additional relevant categories."""

    confidence: float
    """Model confidence in the primary category assignment, in [0.0, 1.0]."""

    reasoning: str
    """Brief explanation of why the primary category was chosen."""


# ---------------------------------------------------------------------------
# 15 hardcoded news headline / snippet problems with ground-truth categories
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "text": (
            "Senate passes landmark climate bill requiring 50% reduction in carbon "
            "emissions by 2035, sending it to the President for signature."
        ),
        "primary_category": "politics",
    },
    {
        "text": (
            "Apple unveils its next-generation M4 chip with a 40% performance leap, "
            "promising all-day battery life for the new MacBook Pro lineup."
        ),
        "primary_category": "technology",
    },
    {
        "text": (
            "Goldman Sachs reports record quarterly profit of $4.2 billion, beating "
            "analyst expectations as investment banking fees surge amid a busy M&A market."
        ),
        "primary_category": "business",
    },
    {
        "text": (
            "Astronomers detect a potentially habitable exoplanet just 12 light-years "
            "away, describing it as the most Earth-like world found to date."
        ),
        "primary_category": "science",
    },
    {
        "text": (
            "Manchester City clinches its fourth consecutive Premier League title after "
            "a 3-1 win over Arsenal on the final day of the season."
        ),
        "primary_category": "sports",
    },
    {
        "text": (
            "Taylor Swift's Eras Tour becomes the highest-grossing concert tour in "
            "history, surpassing $2 billion in ticket sales."
        ),
        "primary_category": "entertainment",
    },
    {
        "text": (
            "New study published in The Lancet links ultra-processed food consumption "
            "to a 35% increased risk of cardiovascular disease in adults over 50."
        ),
        "primary_category": "health",
    },
    {
        "text": (
            "European Parliament votes to ban the sale of new petrol and diesel cars "
            "by 2035, clearing the final legislative hurdle for the green transition policy."
        ),
        "primary_category": "politics",
    },
    {
        "text": (
            "NVIDIA's market capitalisation crosses $3 trillion as AI chip demand "
            "continues to outpace supply, making it the world's most valuable company."
        ),
        "primary_category": "technology",
    },
    {
        "text": (
            "Amazon announces acquisition of smart-home device maker iHome for "
            "$1.8 billion, expanding its ecosystem of connected consumer devices."
        ),
        "primary_category": "business",
    },
    {
        "text": (
            "Researchers at MIT develop a new class of antibiotics effective against "
            "drug-resistant bacteria, using AI to screen over a million compounds."
        ),
        "primary_category": "science",
    },
    {
        "text": (
            "Novak Djokovic withdraws from the Australian Open with a knee injury, "
            "casting doubt on his participation in the upcoming Grand Slam season."
        ),
        "primary_category": "sports",
    },
    {
        "text": (
            "HBO's adaptation of 'The Last of Us' wins five Emmy Awards including "
            "Outstanding Drama Series, cementing its place as the year's defining show."
        ),
        "primary_category": "entertainment",
    },
    {
        "text": (
            "WHO declares the end of the mpox public health emergency of international "
            "concern after case counts decline sharply across all affected regions."
        ),
        "primary_category": "health",
    },
    {
        "text": (
            "SpaceX successfully launches and lands its Starship megarocket for the "
            "third time, marking a pivotal milestone in commercial deep-space travel."
        ),
        "primary_category": "science",
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class ClassificationTask:
    """
    Multi-label news classification benchmark task.

    Contains 15 hardcoded news headlines and snippets, each pre-labelled with
    a ground-truth primary category drawn from seven options: politics,
    technology, business, science, sports, entertainment, and health.

    Models must produce a structured :class:`ClassificationResult` with a
    primary category, optional secondary categories, a confidence score, and
    brief reasoning.  Scoring checks only whether the ``primary_category``
    exactly matches the ground truth.

    This task is LOW complexity — it does not benefit from the TTF routing
    path because the reasoning overhead of think-then-format can hurt the
    calibration of straightforward classification.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``False`` because simple category classification does not require
        extended multi-step reasoning and structured output is sufficient.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "classification"
    expected_ttf_benefit: bool = False
    schema = ClassificationResult
    complexity: str = "LOW"

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

            ``"text"`` : str
                The news headline or snippet passed to the model.
            ``"primary_category"`` : str
                Ground-truth primary category used by :meth:`score_response`.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [{"text": p["text"], "primary_category": p["primary_category"]} for p in problems]

    def score_response(self, predicted: dict[str, Any], ground_truth: str) -> float:
        """
        Score a model response against the ground-truth primary category.

        The comparison is exact (case-sensitive).  The predicted
        ``primary_category`` must match the ground truth string exactly to
        receive a score of 1.0.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`ClassificationResult` produced
            by the model.  Must contain the key ``"primary_category"``.
        ground_truth:
            The expected primary category string for the news item.

        Returns
        -------
        float
            ``1.0`` if ``primary_category`` exactly matches *ground_truth*,
            otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        predicted_cat = predicted.get("primary_category")
        if predicted_cat is None:
            return 0.0

        result = 1.0 if str(predicted_cat) == str(ground_truth) else 0.0

        if result == 0.0:
            logger.debug(
                "score_response: category mismatch — predicted=%r truth=%r",
                predicted_cat,
                ground_truth,
            )

        return result

    def build_prompt(self, text: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        text:
            The news headline or snippet to classify.

        Returns
        -------
        str
            A formatted prompt instructing the model to classify the text and
            return a structured JSON result.
        """
        categories_str = ", ".join(sorted(_VALID_CATEGORIES))
        return (
            "You are a news classification assistant.  Classify the following news headline "
            "or snippet into the most appropriate category and return the result as structured "
            "JSON.\n\n"
            f"Available categories: {categories_str}\n\n"
            "Return a JSON object with exactly these four fields:\n"
            "  - primary_category: the single most relevant category (must be one of the "
            "available categories listed above)\n"
            "  - secondary_categories: list of additional relevant categories (may be empty)\n"
            "  - confidence: your confidence in the primary category as a float in [0.0, 1.0]\n"
            "  - reasoning: a brief one-sentence explanation of your classification choice\n\n"
            "Return only the JSON object.  Do not include text outside the JSON.\n\n"
            f"News text:\n{text}"
        )
