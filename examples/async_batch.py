"""Extract from many documents concurrently with one reused engine.

``extract_batch`` runs the documents through the same calibrated engine, bounded by a
semaphore so a large batch stays under provider rate limits. Set GROQ_API_KEY, then run
this file.
"""

import asyncio

from nfield import AsyncNField

schema = {
    "type": "object",
    "properties": {"company": {"type": "string"}, "revenue": {"type": "number"}},
    "required": ["company"],
}

documents = [
    "Acme Corp reported revenue of 12.4 million.",
    "Globex Inc revenue: 8.1 million this quarter.",
    "Initech posted 3.7 million in revenue.",
]


async def main() -> None:
    async with AsyncNField("groq/llama-3.1-8b-instant", schema) as engine:
        results = await engine.extract_batch(documents)
    for result in results:
        print(result.data)


asyncio.run(main())
