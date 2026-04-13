"""
FormatShield Example: RAG Pipeline — Structured Fact Extraction

Demonstrates FormatShield in a Retrieval-Augmented Generation pipeline.
Named-entity recognition over scientific text is high-complexity — FormatShield
detects nested schema + reasoning ops and routes to TTF.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/rag_extraction.py
"""
from __future__ import annotations

import asyncio
from pydantic import BaseModel, Field

import formatshield as fs


class Entity(BaseModel):
    name: str = Field(description="The entity name as it appears in the text")
    type: str = Field(description="Entity type, e.g. 'organization', 'person', 'location', 'concept'")
    relevance_score: float = Field(description="Relevance to the document topic, 0.0 to 1.0")


class DocumentFacts(BaseModel):
    key_facts: list[str] = Field(
        default_factory=list,
        description="The most important factual claims made in the document",
    )
    entities: list[Entity] = Field(
        default_factory=list,
        description="Named entities extracted from the text with type and relevance",
    )
    summary: str = Field(description="A single-sentence summary of the document chunk")
    confidence: float = Field(description="Confidence in the extraction quality, 0.0 to 1.0")


CLIMATE_CHUNK = """
A landmark study published in Nature Climate Change by researchers at the
Potsdam Institute for Climate Impact Research (PIK) and the University of
Exeter has quantified the cascading risk of tipping-point interactions in the
Earth's climate system. Lead author Dr. Ricarda Winkelmann and her team applied
a network model to fourteen identified tipping elements — including the Greenland
Ice Sheet, West Antarctic Ice Sheet, Amazon rainforest dieback, and Atlantic
Meridional Overturning Circulation (AMOC) — finding that exceeding any single
tipping threshold significantly raises the probability of triggering adjacent
elements, a phenomenon the paper terms "tipping cascades."

The analysis, which drew on paleoclimate records spanning the last 2.6 million
years alongside CMIP6 model ensembles, found that at 1.5 °C of global warming
the probability of at least one major tipping event occurring within the next
century stands at approximately 38 %. That figure rises sharply to 69 % at
2 °C and exceeds 90 % beyond 3 °C. Particularly concerning is the AMOC, whose
weakening — already observable in satellite sea-surface temperature anomalies —
could amplify European temperature extremes by up to 4 °C independent of
background warming.

The Intergovernmental Panel on Climate Change (IPCC) AR6 report acknowledged
tipping risks but did not assign quantitative probabilities to cascade scenarios.
The PIK study directly addresses this gap and has already been cited by the
United Nations Environment Programme (UNEP) in pre-COP briefing materials.
Policy analysts at the Grantham Research Institute on Climate Change and the
Environment argue that these findings make the economic case for limiting warming
to 1.5 °C substantially stronger than previously modelled, as the non-linear
damage functions implied by cascade dynamics are not captured in standard
Integrated Assessment Models (IAMs) such as DICE or FUND.
"""


async def main() -> None:
    print("FormatShield RAG Extraction Example")
    print("=" * 60)

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
        expose_thinking=True,
    )

    print("\nProcessing retrieved document chunk...")
    print(f"\nChunk preview:\n{CLIMATE_CHUNK[:200]}...\n")

    result = await shield.generate(
        prompt=(
            "Extract structured facts from the following retrieved document chunk. "
            "Identify key factual claims, named entities with their types and relevance, "
            "a one-sentence summary, and your confidence in the extraction:\n\n"
            + CLIMATE_CHUNK
        ),
        schema=DocumentFacts,
    )

    print(f"\n{'=' * 60}")
    print(f"Routing:  {result.routing.strategy.upper()}")
    print(f"Complexity: {result.complexity_score:.3f}")
    print(f"Latency:  {result.latency_ms:.0f}ms")
    print(f"Failure modes detected: {result.failure_modes or 'none'}")

    if result.thinking:
        print(f"\nThinking (first 400 chars):\n{result.thinking[:400]}...")

    if result.parsed and isinstance(result.parsed, DocumentFacts):
        facts = result.parsed
        print(f"\n--- Extracted Document Facts ---")
        print(f"Summary:    {facts.summary}")
        print(f"Confidence: {facts.confidence:.2f}")
        print(f"\nKey facts ({len(facts.key_facts)}):")
        for i, fact in enumerate(facts.key_facts, 1):
            print(f"  {i}. {fact}")
        print(f"\nEntities ({len(facts.entities)}):")
        for entity in facts.entities:
            print(f"  [{entity.type:12s}] {entity.name}  (relevance={entity.relevance_score:.2f})")


if __name__ == "__main__":
    asyncio.run(main())
