"""
FinancialTask — Financial metric extraction benchmark task for FormatShield.

This task contains 15 hardcoded earnings-report text snippets.  Models must
extract four key financial metrics — revenue, net income, gross margin, and
year-over-year growth — into a structured :class:`FinancialMetrics` response.

Parsing financial prose requires identifying the correct figures among many
numbers and handling varied units (millions, billions, percentages), making
this task HIGH complexity with an expected benefit from TTF routing.

Complexity: HIGH
Expected TTF benefit: True
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class FinancialMetrics(BaseModel):
    """Structured schema for financial metric extraction from earnings reports."""

    revenue_usd: float
    """Total revenue in US dollars (raw value, not abbreviated)."""

    net_income_usd: float
    """Net income in US dollars (raw value, not abbreviated)."""

    gross_margin_pct: float
    """Gross margin expressed as a percentage (e.g. 42.5 for 42.5%)."""

    yoy_growth_pct: float
    """Year-over-year revenue growth expressed as a percentage."""


# ---------------------------------------------------------------------------
# 15 hardcoded earnings-report snippets with known ground-truth values
# All monetary values stored as raw USD floats.
# ---------------------------------------------------------------------------

_PROBLEMS: list[dict[str, Any]] = [
    {
        "text": (
            "TechNova Corp. today reported third-quarter fiscal 2024 results.  Revenue for "
            "the quarter reached $2.4 billion, up 18% compared to the same period last year. "
            "The company posted net income of $312 million.  Gross margin improved to 61.5% "
            "from 58.3% in the prior year quarter."
        ),
        "expected_revenue": 2_400_000_000.0,
        "expected_net_income": 312_000_000.0,
        "expected_gross_margin": 61.5,
        "expected_yoy_growth": 18.0,
    },
    {
        "text": (
            "Meridian Retail Group reported full-year 2023 revenues of $890 million, "
            "representing a 7.3% increase over fiscal 2022.  Net income came in at "
            "$54 million.  The company achieved a gross margin of 34.8% for the year, "
            "compared to 33.1% in the prior year."
        ),
        "expected_revenue": 890_000_000.0,
        "expected_net_income": 54_000_000.0,
        "expected_gross_margin": 34.8,
        "expected_yoy_growth": 7.3,
    },
    {
        "text": (
            "CloudStream Inc. announced first-quarter 2024 results with total revenue of "
            "$175 million, a 42% jump from Q1 2023.  Despite strong top-line growth, the "
            "company reported a net loss of $22 million due to heavy investment in "
            "infrastructure.  Gross margin stood at 68.2%."
        ),
        "expected_revenue": 175_000_000.0,
        "expected_net_income": -22_000_000.0,
        "expected_gross_margin": 68.2,
        "expected_yoy_growth": 42.0,
    },
    {
        "text": (
            "Pinnacle Industrial Holdings released its second-quarter 2023 earnings.  "
            "Quarterly revenue totalled $1.1 billion, up 3.5% year over year.  Net income "
            "was $88 million.  Gross margin contracted slightly to 29.4% from 30.1% in the "
            "comparable quarter of the prior year."
        ),
        "expected_revenue": 1_100_000_000.0,
        "expected_net_income": 88_000_000.0,
        "expected_gross_margin": 29.4,
        "expected_yoy_growth": 3.5,
    },
    {
        "text": (
            "BioVantage Pharmaceuticals reported fiscal year 2023 revenue of $3.75 billion, "
            "growing 11.2% compared to fiscal 2022.  Net income for the year was $820 million, "
            "reflecting strong demand for its flagship oncology portfolio.  Gross margin for "
            "the period was 74.6%."
        ),
        "expected_revenue": 3_750_000_000.0,
        "expected_net_income": 820_000_000.0,
        "expected_gross_margin": 74.6,
        "expected_yoy_growth": 11.2,
    },
    {
        "text": (
            "For the fourth quarter ended December 31, 2023, Atlas Logistics Inc. reported "
            "revenue of $620 million, a 5.8% decline versus Q4 2022 due to lower freight "
            "volumes.  Net income was $31 million.  Gross margin for the quarter was 22.7%, "
            "broadly in line with the same quarter last year."
        ),
        "expected_revenue": 620_000_000.0,
        "expected_net_income": 31_000_000.0,
        "expected_gross_margin": 22.7,
        "expected_yoy_growth": -5.8,
    },
    {
        "text": (
            "Solaris Energy Partners announced annual results for 2023: total revenues of "
            "$5.2 billion, representing a 21% year-over-year increase driven by higher "
            "commodity prices.  Net income reached $1.05 billion.  Gross margin expanded "
            "to 38.9% from 33.5% in 2022."
        ),
        "expected_revenue": 5_200_000_000.0,
        "expected_net_income": 1_050_000_000.0,
        "expected_gross_margin": 38.9,
        "expected_yoy_growth": 21.0,
    },
    {
        "text": (
            "DataStream Analytics posted Q2 2024 revenues of $95 million, up 31% from the "
            "same quarter a year ago.  The company recorded net income of $8 million.  "
            "Gross margin was 72.3%, compared to 69.8% in Q2 2023, reflecting improved "
            "software mix and pricing."
        ),
        "expected_revenue": 95_000_000.0,
        "expected_net_income": 8_000_000.0,
        "expected_gross_margin": 72.3,
        "expected_yoy_growth": 31.0,
    },
    {
        "text": (
            "Heritage Consumer Brands reported third-quarter 2023 net revenues of "
            "$2.85 billion, a 2.1% increase compared to Q3 2022.  Net earnings were "
            "$195 million.  The gross margin for the quarter was 44.1%, up 90 basis points "
            "versus the prior-year period."
        ),
        "expected_revenue": 2_850_000_000.0,
        "expected_net_income": 195_000_000.0,
        "expected_gross_margin": 44.1,
        "expected_yoy_growth": 2.1,
    },
    {
        "text": (
            "Frontier Semiconductor reported fiscal fourth-quarter 2024 revenue of "
            "$780 million, down 9.4% year over year as enterprise customers reduced "
            "inventory levels.  Net income was $72 million.  Gross margin came in at 51.8%, "
            "compared to 55.2% in the fiscal fourth quarter of 2023."
        ),
        "expected_revenue": 780_000_000.0,
        "expected_net_income": 72_000_000.0,
        "expected_gross_margin": 51.8,
        "expected_yoy_growth": -9.4,
    },
    {
        "text": (
            "NovaCare Health Systems reported full-year 2023 revenues of $12.3 billion, "
            "growing 6.7% from 2022.  Net income for the fiscal year was $540 million. "
            "Gross margin was 28.6%, reflecting ongoing cost pressures in the healthcare "
            "services segment."
        ),
        "expected_revenue": 12_300_000_000.0,
        "expected_net_income": 540_000_000.0,
        "expected_gross_margin": 28.6,
        "expected_yoy_growth": 6.7,
    },
    {
        "text": (
            "Luminary Games reported record full-year 2023 revenue of $425 million, "
            "surging 58% year over year following the release of two major game titles. "
            "Net income was $62 million, compared to a net loss of $14 million in 2022. "
            "Gross margin improved dramatically to 65.9% from 52.4% in the prior year."
        ),
        "expected_revenue": 425_000_000.0,
        "expected_net_income": 62_000_000.0,
        "expected_gross_margin": 65.9,
        "expected_yoy_growth": 58.0,
    },
    {
        "text": (
            "Quantum Mobility Inc. disclosed Q1 2024 results: revenue of $310 million, "
            "13.5% higher than Q1 2023, supported by strong electric vehicle component "
            "demand.  Net income was $19 million.  Gross margin for the quarter was 33.2%, "
            "consistent with the company's annual guidance range of 32–35%."
        ),
        "expected_revenue": 310_000_000.0,
        "expected_net_income": 19_000_000.0,
        "expected_gross_margin": 33.2,
        "expected_yoy_growth": 13.5,
    },
    {
        "text": (
            "Pacific Agritech Corporation reported fiscal 2023 annual revenues of $540 million, "
            "4.9% above fiscal 2022 levels.  Net income was $48 million.  Gross margin was "
            "41.3% for the year, slightly below the 42.0% achieved in the prior year, due to "
            "higher input costs partially offset by pricing actions."
        ),
        "expected_revenue": 540_000_000.0,
        "expected_net_income": 48_000_000.0,
        "expected_gross_margin": 41.3,
        "expected_yoy_growth": 4.9,
    },
    {
        "text": (
            "Apex Fintech Solutions released its Q3 2023 results showing revenue of "
            "$88 million, a 27% increase versus Q3 2022 driven by new enterprise "
            "customer additions.  The company reported a net loss of $5 million as it "
            "continues to invest in go-to-market expansion.  Gross margin was 76.4%."
        ),
        "expected_revenue": 88_000_000.0,
        "expected_net_income": -5_000_000.0,
        "expected_gross_margin": 76.4,
        "expected_yoy_growth": 27.0,
    },
]

# Quick-mode uses the first 5 problems only
_QUICK_SLICE = 5


class FinancialTask:
    """
    Financial metric extraction benchmark task.

    Contains 15 hardcoded earnings-report text snippets annotated with four
    key financial metrics: revenue, net income, gross margin, and year-over-year
    growth.  Revenue values span millions to tens of billions of dollars,
    requiring unit disambiguation.

    Scoring checks whether the extracted ``revenue_usd`` is within 5% of the
    ground-truth value, which tolerates minor rounding differences while
    penalising unit confusion (e.g. reading millions as raw dollars).

    Attributes
    ----------
    name:
        Stable task identifier used in benchmark result records.
    expected_ttf_benefit:
        ``True`` because correctly identifying the right figure among many
        numbers and resolving unit abbreviations benefits from step-by-step
        reasoning.
    schema:
        The Pydantic model class that defines the expected output shape.
    complexity:
        Qualitative complexity label consumed by the harness for reporting.
    """

    name: str = "financial"
    expected_ttf_benefit: bool = True
    schema = FinancialMetrics
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

            ``"text"`` : str
                Earnings-report snippet passed to the model as input.
            ``"expected_revenue"`` : float
                Ground-truth revenue in raw USD used by :meth:`score_response`.
            ``"expected_net_income"`` : float
                Ground-truth net income in raw USD.
            ``"expected_gross_margin"`` : float
                Ground-truth gross margin percentage.
            ``"expected_yoy_growth"`` : float
                Ground-truth year-over-year growth percentage.
        """
        problems = _PROBLEMS[:_QUICK_SLICE] if quick else _PROBLEMS
        return [
            {
                "text": p["text"],
                "expected_revenue": p["expected_revenue"],
                "expected_net_income": p["expected_net_income"],
                "expected_gross_margin": p["expected_gross_margin"],
                "expected_yoy_growth": p["expected_yoy_growth"],
            }
            for p in problems
        ]

    def score_response(self, predicted: dict[str, Any], ground_truth: dict[str, Any]) -> float:
        """
        Score a model response against the ground-truth revenue figure.

        The predicted ``revenue_usd`` is accepted as correct when it is within
        5% (relative tolerance) of the ground-truth ``expected_revenue``.  This
        tolerates minor rounding while penalising unit confusion.

        Parameters
        ----------
        predicted:
            A dict representation of a :class:`FinancialMetrics` instance
            produced by the model.  Must contain the key ``"revenue_usd"``.
        ground_truth:
            The annotated metric dict for the earnings snippet.  Must contain
            the key ``"expected_revenue"`` with the true revenue as a float.

        Returns
        -------
        float
            ``1.0`` if ``revenue_usd`` is within 5% of ``expected_revenue``,
            otherwise ``0.0``.
        """
        if not isinstance(predicted, dict):
            logger.debug("score_response: predicted is not a dict, got %r", type(predicted))
            return 0.0

        raw = predicted.get("revenue_usd")
        if raw is None:
            return 0.0

        try:
            predicted_revenue = float(raw)
        except (TypeError, ValueError):
            logger.debug("score_response: could not convert %r to float", raw)
            return 0.0

        expected_revenue = float(ground_truth.get("expected_revenue", 0.0))
        if expected_revenue == 0.0:
            return 1.0 if predicted_revenue == 0.0 else 0.0

        relative_error = abs(predicted_revenue - expected_revenue) / abs(expected_revenue)
        return 1.0 if relative_error <= 0.05 else 0.0

    def build_prompt(self, text: str) -> str:
        """
        Construct the full prompt string sent to the model.

        Parameters
        ----------
        text:
            The earnings-report text snippet.

        Returns
        -------
        str
            A formatted prompt instructing the model to extract financial
            metrics as structured JSON.
        """
        return (
            "You are a financial analyst assistant.  Extract the key financial metrics "
            "from the following earnings report text and return them as structured JSON.\n\n"
            "Extract exactly these four fields:\n"
            "  - revenue_usd: total revenue as a raw float in US dollars (e.g. 2400000000.0 "
            "for $2.4 billion; do not abbreviate)\n"
            "  - net_income_usd: net income as a raw float in US dollars (negative for a loss)\n"
            "  - gross_margin_pct: gross margin as a percentage float (e.g. 61.5 for 61.5%)\n"
            "  - yoy_growth_pct: year-over-year revenue growth as a percentage float "
            "(negative if revenue declined)\n\n"
            "Return only the JSON object.  Do not include explanation text outside the JSON.\n\n"
            f"Earnings report:\n{text}"
        )
