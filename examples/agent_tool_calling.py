"""
FormatShield Example: AI Agent Tool Call Extraction

Demonstrates FormatShield working transparently inside an agent loop.
A natural-language instruction is parsed into structured ToolCallPlan objects
three times, simulating the planner step of a tool-calling agent. FormatShield
handles routing decisions per call so the agent never thinks about JSON modes.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/agent_tool_calling.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

import formatshield as fs


class ToolCallPlan(BaseModel):
    tool_name: str = Field(description="Name of the tool to invoke, e.g. 'search_papers'")
    parameters: dict[str, str] = Field(
        default_factory=dict,
        description="Key-value string parameters to pass to the tool",
    )
    confidence: float = Field(description="Confidence that this is the right tool, 0.0 to 1.0")
    fallback_strategy: str | None = Field(
        None,
        description="What to do if the tool call fails, or null if no fallback is needed",
    )
    explanation: str = Field(
        description="One-sentence rationale for choosing this tool and parameters"
    )


# Three sequential agent steps derived from a single user instruction.
AGENT_STEPS = [
    {
        "step": 1,
        "context": "User instruction received. Plan the first tool call.",
        "user_instruction": (
            "Search for recent papers about transformer attention mechanisms "
            "and summarize the top 3 results"
        ),
        "history": "No prior tool calls.",
    },
    {
        "step": 2,
        "context": "search_papers returned 3 results. Plan the next tool call to fetch full abstracts.",
        "user_instruction": (
            "Search for recent papers about transformer attention mechanisms "
            "and summarize the top 3 results"
        ),
        "history": (
            "Step 1: search_papers(query='transformer attention mechanisms', limit='3') "
            "→ returned IDs [arxiv:2401.00001, arxiv:2402.00042, arxiv:2403.00118]"
        ),
    },
    {
        "step": 3,
        "context": "Abstracts fetched. Plan the summarization tool call.",
        "user_instruction": (
            "Search for recent papers about transformer attention mechanisms "
            "and summarize the top 3 results"
        ),
        "history": (
            "Step 1: search_papers → 3 results. "
            "Step 2: fetch_abstracts(ids='arxiv:2401.00001,arxiv:2402.00042,arxiv:2403.00118') "
            "→ 3 abstracts retrieved."
        ),
    },
]


def _build_prompt(step_info: dict) -> str:  # type: ignore[type-arg]
    return (
        f"You are an AI agent planner. Given the user instruction and the prior tool-call "
        f"history, determine the single best tool call to make next.\n\n"
        f"User instruction: {step_info['user_instruction']}\n\n"
        f"Context: {step_info['context']}\n\n"
        f"Tool call history:\n{step_info['history']}\n\n"
        f"Available tools: search_papers, fetch_abstracts, summarize_texts, render_report\n\n"
        f"Return a ToolCallPlan for step {step_info['step']}."
    )


async def main() -> None:
    print("FormatShield Agent Tool Calling Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
    )

    print("\nUser instruction:")
    print(f'  "{AGENT_STEPS[0]["user_instruction"]}"')
    print(f"\nRunning agent loop ({len(AGENT_STEPS)} steps)...")

    for step_info in AGENT_STEPS:
        step = step_info["step"]
        print(f"\n{'=' * 60}")
        print(f"Agent Step {step}/3")
        print(f"Context: {step_info['context']}")

        result = await shield.generate(
            prompt=_build_prompt(step_info),
            schema=ToolCallPlan,
        )

        print(f"\nRouting:  {result.routing.strategy.upper()}")
        print(f"Complexity: {result.complexity_score:.3f}")
        print(f"Latency:  {result.latency_ms:.0f}ms")
        print(f"Failure modes detected: {result.failure_modes or 'none'}")

        if result.thinking:
            print(f"\nThinking (first 300 chars):\n{result.thinking[:300]}...")

        if result.parsed and isinstance(result.parsed, ToolCallPlan):
            plan = result.parsed
            print(f"\n--- Tool Call Plan (step {step}) ---")
            print(f"Tool:              {plan.tool_name}")
            print(f"Confidence:        {plan.confidence:.2f}")
            print(f"Fallback strategy: {plan.fallback_strategy or 'none'}")
            print(f"Explanation:       {plan.explanation}")
            print("Parameters:")
            for k, v in plan.parameters.items():
                print(f"  {k}: {v}")

    print(f"\n{'=' * 60}")
    print("Agent loop complete.")


if __name__ == "__main__":
    asyncio.run(main())
