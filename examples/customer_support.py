"""
FormatShield Example: Customer Support Ticket Classification and Routing

Demonstrates FormatShield routing for a customer support ticket.
Simple classification tasks are low-complexity template fills — FormatShield
will route directly (no TTF overhead).

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/customer_support.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

import formatshield as fs


class TicketAnalysis(BaseModel):
    category: str = Field(description="Support category, e.g. 'billing', 'technical', 'account'")
    priority: str = Field(description="Ticket priority: LOW, MEDIUM, HIGH, or URGENT")
    sentiment: float = Field(
        description="Customer sentiment score from 0.0 (negative) to 1.0 (positive)"
    )
    required_actions: list[str] = Field(
        default_factory=list,
        description="List of concrete actions the support team must take",
    )
    escalate_to_human: bool = Field(
        description="Whether this ticket requires human agent escalation"
    )


SUPPORT_TICKET = """
Subject: Incorrect charge on my account — THIRD time this month!

Hi,

I'm absolutely furious right now. For the third consecutive month I've been
charged $149.99 instead of the $29.99 plan I signed up for. I raised this
twice before (tickets #48821 and #49103) and was promised a refund both times,
but the refund never appeared and now I've been charged the wrong amount AGAIN.

I've already spent over 2 hours on hold across two separate calls and nobody
has actually fixed the root cause. I am a 6-year customer and this is completely
unacceptable. If this is not resolved by end of business today with a full
refund of all three overcharges ($360 total) and a written explanation of what
went wrong, I will be disputing all charges with my credit card company and
cancelling my subscription immediately.

Account email: sarah.chen@example.com
Order refs: ORD-2024-0311, ORD-2024-0411, ORD-2024-0511
"""


async def main() -> None:
    print("FormatShield Customer Support Ticket Classification Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
    )

    print("\nAnalyzing support ticket...")
    print(f"\nTicket preview:\n{SUPPORT_TICKET[:200]}...\n")

    result = await shield.generate(
        prompt=(
            "Analyze the following customer support ticket and classify it. "
            "Identify the category, urgency level, customer sentiment, required actions, "
            "and whether a human agent must handle this:\n\n" + SUPPORT_TICKET
        ),
        schema=TicketAnalysis,
    )

    print(f"\n{'=' * 60}")
    print(f"Routing:  {result.routing.strategy.upper()}")
    print(f"Complexity: {result.complexity_score:.3f}")
    print(f"Latency:  {result.latency_ms:.0f}ms")
    print(f"Failure modes detected: {result.failure_modes or 'none'}")

    if result.thinking:
        print(f"\nThinking (first 400 chars):\n{result.thinking[:400]}...")

    if result.parsed and isinstance(result.parsed, TicketAnalysis):
        ticket = result.parsed
        print("\n--- Ticket Analysis ---")
        print(f"Category:           {ticket.category}")
        print(f"Priority:           {ticket.priority}")
        print(f"Sentiment score:    {ticket.sentiment:.2f}  (0=negative, 1=positive)")
        print(f"Escalate to human:  {ticket.escalate_to_human}")
        print(f"\nRequired actions ({len(ticket.required_actions)}):")
        for action in ticket.required_actions:
            print(f"  - {action}")


if __name__ == "__main__":
    asyncio.run(main())
