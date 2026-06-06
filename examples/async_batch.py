"""Async extraction over many documents with AsyncFormatShield.

Reuses one engine (schema cached, model calibrated once) and runs documents
concurrently with ``asyncio.gather``.

Run:
    export GROQ_API_KEY=...
    python examples/async_batch.py
"""

from __future__ import annotations

import asyncio

from formatshield import AsyncFormatShield

SCHEMA = {
    "type": "object",
    "properties": {"company": {"type": "string"}, "revenue": {"type": "number"}},
    "required": ["company"],
}

DOCUMENTS = [
    "Acme Corp reported revenue of 12.4 million.",
    "Globex Inc revenue: 8.1 million this quarter.",
    "Initech posted 3.7 million in revenue.",
]


async def main() -> None:
    async with AsyncFormatShield(
        "groq/llama-3.1-8b-instant",
        SCHEMA,
        context_window=131_072,
        max_output_tokens=32_768,
    ) as engine:
        results = await asyncio.gather(*(engine.extract(doc) for doc in DOCUMENTS))
    for result in results:
        print(result.data)


if __name__ == "__main__":
    asyncio.run(main())
