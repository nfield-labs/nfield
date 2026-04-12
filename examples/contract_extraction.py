"""
FormatShield Example: Contract Obligation Extraction

Demonstrates FormatShield routing for complex legal text extraction.
Legal contracts are high-complexity prompts — FormatShield will route to TTF.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/contract_extraction.py
"""
from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field

import formatshield as fs


class Obligation(BaseModel):
    party: str = Field(description="The party responsible for this obligation")
    action: str = Field(description="What they must do")
    deadline: str | None = Field(None, description="When it must be done")
    conditions: list[str] = Field(default_factory=list, description="Conditions that trigger this obligation")


class ContractAnalysis(BaseModel):
    obligations: list[Obligation]
    key_dates: list[str]
    penalty_clauses: list[str]
    risk_level: str = Field(description="LOW, MEDIUM, or HIGH")


CONTRACT_CLAUSE = """
SECTION 4.2 — DELIVERY AND ACCEPTANCE

Vendor shall deliver the Software in three phases: Phase 1 (core modules) within
sixty (60) calendar days of contract execution; Phase 2 (analytics dashboard)
within one hundred twenty (120) calendar days; Phase 3 (API integrations) within
one hundred eighty (180) calendar days.

Client shall provide written acceptance or rejection within ten (10) business days
of each delivery. Failure to respond within this period shall constitute deemed
acceptance. In the event of rejection, Vendor shall have thirty (30) days to cure
defects, after which Client may terminate this Agreement with full refund of fees
paid for the rejected phase.

Late delivery penalties: 2% of Phase contract value per week up to a maximum of
20%, after which Client may exercise termination rights under Section 12.3.
Vendor may request one (1) extension per Phase not exceeding thirty (30) days
upon fourteen (14) days written notice, provided that delays are attributable to
Client's failure to provide required access, data, or approvals.
"""


async def main() -> None:
    print("FormatShield Contract Extraction Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
        expose_thinking=True,
    )

    print("\nAnalyzing contract clause...")
    print(f"\nClause preview:\n{CONTRACT_CLAUSE[:200]}...\n")

    result = await shield.generate(
        prompt=f"Analyze this contract clause and extract all obligations, dates, and risks:\n\n{CONTRACT_CLAUSE}",
        schema=ContractAnalysis,
    )

    print(f"\n{'='*60}")
    print(f"Routing: {result.routing.strategy.upper()}")
    print(f"Complexity: {result.complexity_score:.3f}")
    print(f"Latency: {result.latency_ms:.0f}ms")
    print(f"Failure modes detected: {result.failure_modes or 'none'}")

    if result.thinking:
        print(f"\nThinking (first 400 chars):\n{result.thinking[:400]}...")

    if result.parsed and isinstance(result.parsed, ContractAnalysis):
        analysis = result.parsed
        print(f"\n--- Contract Analysis ---")
        print(f"Risk Level: {analysis.risk_level}")
        print(f"\nObligations ({len(analysis.obligations)}):")
        for ob in analysis.obligations:
            deadline = f" (by {ob.deadline})" if ob.deadline else ""
            print(f"  [{ob.party}] {ob.action}{deadline}")
        print(f"\nKey Dates: {analysis.key_dates}")
        print(f"\nPenalty Clauses: {analysis.penalty_clauses}")


if __name__ == "__main__":
    asyncio.run(main())
