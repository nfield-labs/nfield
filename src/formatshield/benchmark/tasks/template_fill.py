"""
TemplateFillTask — Simple template-filling benchmark task for FormatShield.

This task is the NEGATIVE CONTROL in the FormatShield benchmark suite.
It contains 15 trivially simple template-filling problems where the answer
can be extracted by direct pattern matching — no multi-step reasoning is needed.

TTF routing should NOT help here.  If TTF routing is applied to these problems,
it will add overhead without accuracy benefit.  The FailureModeDetector should
correctly identify these cases and suppress TTF.

Complexity: LOW
Expected TTF benefit: False (negative control — validates FailureModeDetector)
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class TemplateData(BaseModel):
    """Generic template schema for simple fill-in-the-blank extraction."""

    name: str
    """The person's full name."""

    age: int
    """The person's age in years."""

    city: str
    """The city where the person lives or works."""


# ---------------------------------------------------------------------------
# Extended schemas for varied template types
# ---------------------------------------------------------------------------


class ProductInfo(BaseModel):
    """Schema for product information templates."""

    product_name: str
    price: float
    category: str


class EventInfo(BaseModel):
    """Schema for event information templates."""

    event_name: str
    date: str
    location: str


# ---------------------------------------------------------------------------
# 15 hardcoded template-filling problems
# Each problem is deliberately trivial — answers appear literally in the text
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "John is 25 years old and lives in Paris.",
        "schema_type": "person",
        "expected": {"name": "John", "age": 25, "city": "Paris"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Maria is 32 years old. She is from Barcelona.",
        "schema_type": "person",
        "expected": {"name": "Maria", "age": 32, "city": "Barcelona"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Ahmed, aged 19, currently resides in Cairo.",
        "schema_type": "person",
        "expected": {"name": "Ahmed", "age": 19, "city": "Cairo"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "The customer is Li Wei, 45 years old, calling from Shanghai.",
        "schema_type": "person",
        "expected": {"name": "Li Wei", "age": 45, "city": "Shanghai"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Emma Johnson is a 28-year-old resident of New York.",
        "schema_type": "person",
        "expected": {"name": "Emma Johnson", "age": 28, "city": "New York"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "The applicant is Carlos Ruiz, who is 37 and based in Mexico City.",
        "schema_type": "person",
        "expected": {"name": "Carlos Ruiz", "age": 37, "city": "Mexico City"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Yuki Tanaka is 23 years old and lives in Tokyo.",
        "schema_type": "person",
        "expected": {"name": "Yuki Tanaka", "age": 23, "city": "Tokyo"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "The patient, Fatima Al-Hassan, is 61 years old and resides in Dubai.",
        "schema_type": "person",
        "expected": {"name": "Fatima Al-Hassan", "age": 61, "city": "Dubai"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "David Kim, 54, is registered at our Toronto branch.",
        "schema_type": "person",
        "expected": {"name": "David Kim", "age": 54, "city": "Toronto"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Sophie Müller is 30 years old and currently living in Berlin.",
        "schema_type": "person",
        "expected": {"name": "Sophie Müller", "age": 30, "city": "Berlin"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Raj Patel, 42 years old, is a software engineer from Bengaluru.",
        "schema_type": "person",
        "expected": {"name": "Raj Patel", "age": 42, "city": "Bengaluru"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Our new intern Olivia Chen is 21 and comes from Hong Kong.",
        "schema_type": "person",
        "expected": {"name": "Olivia Chen", "age": 21, "city": "Hong Kong"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "The account belongs to Marco Rossi, aged 48, domiciled in Rome.",
        "schema_type": "person",
        "expected": {"name": "Marco Rossi", "age": 48, "city": "Rome"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "Amara Diallo is 26 years old and lives in Dakar.",
        "schema_type": "person",
        "expected": {"name": "Amara Diallo", "age": 26, "city": "Dakar"},
    },
    {
        "instruction": "Fill in the template fields: Name: ___, Age: ___, City: ___.",
        "context": "The speaker, Hans Zimmermann, is 67 years old and is from Vienna.",
        "schema_type": "person",
        "expected": {"name": "Hans Zimmermann", "age": 67, "city": "Vienna"},
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class TemplateFillTask:
    """
    Simple template-filling benchmark task — negative control.

    Contains 15 trivially straightforward extraction problems where the answer
    fields appear almost verbatim in the source text.  No arithmetic or
    multi-step reasoning is required.

    This task exists to validate that FormatShield's routing logic does **not**
    activate TTF unnecessarily.  If TTF is applied here, it will introduce
    latency overhead without improving accuracy, which the
    :class:`~formatshield.oracle.FailureModeDetector` should flag.

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``False`` — this is a negative control.  TTF is not expected to help.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "template_fill"
    expected_ttf_benefit: bool = False
    schema = TemplateData
    complexity: str = "LOW"

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        """
        Return the list of benchmark problems.

        Parameters
        ----------
        quick:
            When ``True`` returns only the first 5 problems.

        Returns
        -------
        list[dict]
            Each element has keys:

            ``"instruction"`` : str
                The template instruction shown to the model.
            ``"context"`` : str
                The source text containing the field values.
            ``"expected"`` : dict
                Ground-truth dict with keys ``name``, ``age``, ``city``.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [
            {
                "instruction": p["instruction"],
                "context": p["context"],
                "expected": p["expected"],
            }
            for p in problems
        ]

    def score_response(
        self,
        predicted: dict[str, Any],
        ground_truth: dict[str, Any],
    ) -> float:
        """
        Score a model response against the ground-truth template fields.

        Scoring is exact-match per field, averaged across all fields present
        in *ground_truth*.  String comparisons are case-folded and stripped.
        Numeric fields (``age``) allow a tolerance of ±0.

        Parameters
        ----------
        predicted:
            A dict produced by the model.  Expected keys: ``name``, ``age``,
            ``city``.  Extra keys are ignored.
        ground_truth:
            The annotated field dict for the template problem.

        Returns
        -------
        float
            Fraction of correctly matched fields in [0.0, 1.0].
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        if not ground_truth:
            return 1.0

        hits = 0
        total = len(ground_truth)

        for field, truth_value in ground_truth.items():
            pred_value = predicted.get(field)
            if pred_value is None:
                continue

            # Numeric comparison (age)
            if isinstance(truth_value, (int, float)):
                try:
                    if int(pred_value) == int(truth_value):
                        hits += 1
                except (TypeError, ValueError):
                    pass
            else:
                # String comparison: case-fold and strip whitespace
                if str(pred_value).strip().lower() == str(truth_value).strip().lower():
                    hits += 1

        return hits / total if total > 0 else 0.0

    def build_prompt(self, instruction: str, context: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        instruction:
            The template fill instruction (describes the output schema).
        context:
            The source sentence or paragraph containing the field values.

        Returns
        -------
        str
            A formatted prompt requesting direct JSON field extraction.
        """
        return (
            f"{instruction}\n\n"
            f"Source text: {context}\n\n"
            "Extract the values and return them as a JSON object with exactly "
            "these keys: name (string), age (integer), city (string).\n"
            "Do not include any explanation — return only the JSON object."
        )
