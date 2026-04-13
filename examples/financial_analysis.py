"""
FormatShield Example: Financial Report Analysis — Earnings Call Parsing

Demonstrates FormatShield streaming mode on an earnings call transcript.
Structured financial extraction with nested metrics and multi-field reasoning
is routed through the TTF pipeline; token-by-token streaming lets downstream
consumers display partial results immediately.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/financial_analysis.py
"""
from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field

import formatshield as fs


class Metric(BaseModel):
    name: str = Field(description="Name of the financial metric, e.g. 'Revenue', 'EBITDA'")
    value: str = Field(description="Reported value with units, e.g. '$4.2B', '18.3%'")
    yoy_change: str = Field(description="Year-over-year change, e.g. '+12%', '-3.1pp'")


class EarningsAnalysis(BaseModel):
    revenue_mentioned: bool = Field(description="Whether revenue figures were explicitly stated")
    guidance_raised: bool = Field(description="Whether full-year guidance was raised")
    key_metrics: list[Metric] = Field(
        default_factory=list,
        description="All quantitative financial metrics mentioned",
    )
    risks: list[str] = Field(
        default_factory=list,
        description="Risk factors or headwinds highlighted by management",
    )
    sentiment_score: float = Field(
        description="Overall tone of the excerpt, 0.0 (bearish) to 1.0 (bullish)"
    )
    analyst_recommendation: str = Field(
        description="Inferred analyst stance based on the excerpt: 'Buy', 'Hold', or 'Sell'"
    )


EARNINGS_TRANSCRIPT = """
Good morning, everyone. We are pleased to report Q3 revenue of $5.84 billion,
up 14 % year-over-year and ahead of our $5.71 billion consensus estimate, driven
by record cloud-services attach rates of 73 % across our enterprise segment.
Adjusted EBITDA margin expanded by 220 basis points to 31.4 %, and free cash
flow came in at $1.1 billion, a 28 % year-over-year improvement. Based on this
momentum, we are raising full-year revenue guidance to a range of $22.6–$22.9
billion from the prior $21.8–$22.2 billion, implying approximately 12 % annual
growth. That said, we continue to monitor macro headwinds in EMEA, where
foreign-exchange volatility reduced reported revenue by roughly $140 million this
quarter, and we remain cautious about enterprise discretionary-spend cycles
heading into Q4.
"""


async def main() -> None:
    print("FormatShield Financial Analysis — Earnings Call Streaming Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
        expose_thinking=True,
    )

    print("\nStreaming earnings call analysis...")
    print(f"\nTranscript excerpt:\n{EARNINGS_TRANSCRIPT.strip()}\n")
    print("=" * 60)

    tokens_received = 0
    thinking_captured = ""
    final_json: dict | None = None  # type: ignore[type-arg]

    print("\n[Streaming tokens] ", end="", flush=True)
    async for event in shield.stream(
        prompt=(
            "Analyze this earnings call excerpt. Extract all financial metrics, "
            "assess whether guidance was raised, identify risks, and provide a "
            "sentiment score and analyst recommendation:\n\n"
            + EARNINGS_TRANSCRIPT
        ),
        schema=EarningsAnalysis,
    ):
        if event.type == "thinking" and event.content:
            thinking_captured = event.content
        elif event.type == "output" and event.token:
            print(event.token, end="", flush=True)
            tokens_received += 1
        elif event.type == "complete":
            final_json = event.json

    print(f"\n\n{'=' * 60}")
    print(f"Tokens streamed: {tokens_received}")

    if thinking_captured:
        print(f"\nThinking (first 400 chars):\n{thinking_captured[:400]}...")

    if final_json:
        try:
            analysis = EarningsAnalysis.model_validate(final_json)
            print(f"\n--- Earnings Analysis ---")
            print(f"Revenue mentioned:      {analysis.revenue_mentioned}")
            print(f"Guidance raised:        {analysis.guidance_raised}")
            print(f"Sentiment score:        {analysis.sentiment_score:.2f}  (0=bearish, 1=bullish)")
            print(f"Analyst recommendation: {analysis.analyst_recommendation}")
            print(f"\nKey metrics ({len(analysis.key_metrics)}):")
            for m in analysis.key_metrics:
                print(f"  {m.name:20s}  {m.value:12s}  YoY: {m.yoy_change}")
            print(f"\nRisks ({len(analysis.risks)}):")
            for risk in analysis.risks:
                print(f"  - {risk}")
        except Exception:
            print(f"\nRaw JSON: {final_json}")


if __name__ == "__main__":
    asyncio.run(main())
