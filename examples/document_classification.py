"""
FormatShield Example: Multi-Label Document Classification for Legal Document Management

Demonstrates FormatShield debug mode and routing decision inspection for a
legal document management system. Classifying jurisdiction, applicable
regulations, and confidentiality level requires careful reasoning — watch
the routing trace to see how FormatShield scores this request.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/document_classification.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

import formatshield as fs


class DocumentClassification(BaseModel):
    document_type: str = Field(description="Type of legal document, e.g. 'Employment Contract'")
    jurisdiction: str | None = Field(
        None,
        description="Primary legal jurisdiction, e.g. 'California, USA'. Null if not determinable.",
    )
    applicable_regulations: list[str] = Field(
        default_factory=list,
        description="Regulations and statutes that apply to this document",
    )
    confidentiality_level: str = Field(
        description="Confidentiality classification: PUBLIC, INTERNAL, CONFIDENTIAL, or RESTRICTED"
    )
    requires_legal_review: bool = Field(
        description="Whether this document must be reviewed by a qualified attorney before execution"
    )


DOCUMENT_EXCERPT = """
EMPLOYMENT AGREEMENT

This Employment Agreement ("Agreement") is entered into as of January 15, 2025,
between Nexora Technologies, Inc., a Delaware corporation with its principal place
of business in San Francisco, California ("Company"), and the individual identified
in Exhibit A ("Employee").

1. POSITION AND DUTIES. Employee is hired for the position of Senior Machine Learning
Engineer. Employee agrees to devote substantially all of Employee's business time and
attention to the performance of Employee's duties and to comply with all Company
policies, including the Company's Confidential Information and Invention Assignment
Agreement ("CIIA"), which Employee must execute concurrently with this Agreement.

2. COMPENSATION. Employee shall receive a base salary of $185,000 per annum,
subject to applicable withholding taxes. Employee is eligible for an annual
discretionary bonus of up to 20 % of base salary and shall be granted an option
to purchase 15,000 shares of the Company's common stock under the 2022 Equity
Incentive Plan, vesting over four years with a one-year cliff.

3. AT-WILL EMPLOYMENT. Employee's employment with the Company is at-will and may
be terminated by either party at any time, with or without cause, subject to the
WARN Act obligations where applicable under California Labor Code § 1400 et seq.
"""


async def main() -> None:
    print("FormatShield Document Classification Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
    )

    print("\nClassifying legal document...")
    print(f"\nDocument preview:\n{DOCUMENT_EXCERPT[:200]}...\n")

    result = await shield.generate(
        prompt=(
            "Classify the following legal document for a document management system. "
            "Identify the document type, jurisdiction, all applicable regulations or statutes, "
            "the appropriate confidentiality level, and whether legal review is required "
            "before execution:\n\n" + DOCUMENT_EXCERPT
        ),
        schema=DocumentClassification,
    )

    print(f"\n{'=' * 60}")
    print(f"Routing:  {result.routing.strategy.upper()}")
    print(f"Complexity: {result.complexity_score:.3f}")
    print(f"Latency:  {result.latency_ms:.0f}ms")
    print(f"Failure modes detected: {result.failure_modes or 'none'}")

    print("\n--- Routing Decision Detail ---")
    print(f"  Strategy:          {result.routing.strategy}")
    print(f"  Confidence:        {result.routing.confidence:.2f}")
    print(f"  Expected Δ acc:    {result.routing.expected_accuracy_delta:+.3f}")
    print(f"  Estimated overhead:{result.routing.expected_overhead_pct:.0f}%")
    print(f"  Explanation:       {result.routing.explanation}")

    if result.thinking:
        print(f"\nThinking (first 400 chars):\n{result.thinking[:400]}...")

    if result.parsed and isinstance(result.parsed, DocumentClassification):
        doc = result.parsed
        print("\n--- Document Classification ---")
        print(f"Document type:          {doc.document_type}")
        print(f"Jurisdiction:           {doc.jurisdiction or 'Undetermined'}")
        print(f"Confidentiality level:  {doc.confidentiality_level}")
        print(f"Requires legal review:  {doc.requires_legal_review}")
        print(f"\nApplicable regulations ({len(doc.applicable_regulations)}):")
        for reg in doc.applicable_regulations:
            print(f"  - {reg}")


if __name__ == "__main__":
    asyncio.run(main())
