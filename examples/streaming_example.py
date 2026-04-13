"""Streaming example for FormatShield.

Demonstrates how to stream structured generation results token-by-token
using FormatShield's async streaming API with DryRunBackend (no API keys needed).

For production use, replace DryRunBackend with a real backend:
    shield = FormatShield(model="groq/llama-3.3-70b-versatile")
"""

from __future__ import annotations

import asyncio
from typing import Any

from pydantic import BaseModel

from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield

# ---------------------------------------------------------------------------
# Example 1: Basic streaming with no schema
# ---------------------------------------------------------------------------


async def stream_unstructured() -> None:
    """Stream plain-text generation without a schema constraint."""
    print("=== Example 1: Unstructured streaming ===")

    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

    token_count = 0
    async for event in shield.stream("Explain the TTF algorithm in simple terms"):
        if event.type == "output" and event.token:
            print(event.token, end="", flush=True)
            token_count += 1
        elif event.type == "complete":
            print(f"\n[Complete] {token_count} token events, latency={event.latency_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Example 2: Streaming with a Pydantic schema
# ---------------------------------------------------------------------------


class AnalysisResult(BaseModel):
    """Structured analysis output."""

    summary: str
    sentiment: str
    confidence: float
    key_points: list[str]


async def stream_structured() -> None:
    """Stream structured generation with a Pydantic schema."""
    print("\n=== Example 2: Structured streaming with Pydantic schema ===")

    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

    prompt = (
        "Analyse the following customer review and extract sentiment, "
        "key points, and a summary:\n\n"
        "The product exceeded my expectations! The build quality is excellent "
        "and the performance is outstanding. A few minor UI quirks but overall "
        "a 5-star experience."
    )

    accumulated = ""
    async for event in shield.stream(prompt, schema=AnalysisResult):
        if event.type == "output" and event.token:
            accumulated += event.token
            print(".", end="", flush=True)  # progress dot per token
        elif event.type == "complete":
            print(f"\n[Complete] Full output:\n{event.content or accumulated}")
            print(f"Latency: {event.latency_ms:.1f}ms")


# ---------------------------------------------------------------------------
# Example 3: Streaming with a dict schema
# ---------------------------------------------------------------------------


async def stream_with_dict_schema() -> None:
    """Stream using a plain JSON Schema dict instead of a Pydantic model."""
    print("\n=== Example 3: Streaming with dict schema ===")

    schema: dict[str, Any] = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "skills": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": ["name", "age", "skills"],
    }

    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())

    events = [
        e async for e in shield.stream("Extract person info: Alice, 30, Python/Rust", schema=schema)
    ]

    output_events = [e for e in events if e.type == "output"]
    complete_events = [e for e in events if e.type == "complete"]

    print(f"Output events received: {len(output_events)}")
    print(f"Complete event content: {complete_events[0].content if complete_events else 'N/A'}")


# ---------------------------------------------------------------------------
# Example 4: Comparing streaming vs. non-streaming latency
# ---------------------------------------------------------------------------


async def compare_streaming_vs_batch() -> None:
    """Compare latency: streaming vs. batch generation."""
    print("\n=== Example 4: Streaming vs. batch comparison ===")

    import time

    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    prompt = "List 5 benefits of the Think-Then-Format approach for structured generation."

    # Batch (non-streaming)
    t0 = time.monotonic()
    result = await shield.generate(prompt)
    batch_ms = (time.monotonic() - t0) * 1000

    # Streaming
    t1 = time.monotonic()
    first_token_ms: float | None = None
    async for event in shield.stream(prompt):
        if event.type == "output" and first_token_ms is None:
            first_token_ms = (time.monotonic() - t1) * 1000
    stream_total_ms = (time.monotonic() - t1) * 1000

    print(f"Batch total latency:         {batch_ms:.1f}ms")
    print(f"Streaming first-token:       {first_token_ms or 0:.1f}ms")
    print(f"Streaming total latency:     {stream_total_ms:.1f}ms")
    print(f"Route used:                  {result.routing.strategy}")
    print(f"Complexity score:            {result.complexity_score:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all streaming examples."""
    await stream_unstructured()
    await stream_structured()
    await stream_with_dict_schema()
    await compare_streaming_vs_batch()
    print("\nAll streaming examples completed.")


if __name__ == "__main__":
    asyncio.run(main())
