"""
FormatShield Example: Medical NER with TTF routing

Demonstrates how FormatShield automatically detects that medical entity extraction
requires complex reasoning and routes to TTF mode.

Usage:
    export GROQ_API_KEY=your_key_here
    python examples/medical_ner.py
"""

from __future__ import annotations

import asyncio

from pydantic import BaseModel, Field

import formatshield as fs


class MedicalEntities(BaseModel):
    conditions: list[str] = Field(description="Medical conditions and diagnoses")
    medications: list[str] = Field(description="Drug names and treatments")
    dosages: list[str] = Field(description="Dosage information")
    procedures: list[str] = Field(description="Medical procedures or tests")


CLINICAL_TEXTS = [
    """
    Patient presents with severe migraine headaches occurring 3-4x/week for the past month.
    PMH significant for hypertension managed with lisinopril 10mg daily. Allergic to penicillin.
    Plan: Start propranolol 40mg BID for migraine prophylaxis. Continue lisinopril.
    Follow up in 6 weeks or sooner if symptoms worsen. Consider MRI brain if no improvement.
    """,
    """
    62F with Type 2 diabetes mellitus (HbA1c 8.2%), hyperlipidemia, and CAD s/p CABG 2019.
    Current medications: metformin 1000mg BID, atorvastatin 40mg daily, aspirin 81mg,
    metoprolol succinate 50mg. Today's visit for annual diabetic review.
    Labs show microalbuminuria. Adding lisinopril 5mg for nephroprotection.
    Referral to ophthalmology for diabetic retinopathy screening.
    """,
    """
    28M presents with acute onset fever, productive cough, dyspnea. CXR shows RLL infiltrate.
    Diagnosis: Community-acquired pneumonia. WBC 14.2, CRP elevated.
    Plan: Azithromycin 500mg day 1, then 250mg days 2-5. Supportive care.
    Return to ED if worsening dyspnea, hemoptysis, or fever >39C persists.
    """,
]


async def main() -> None:
    print("FormatShield Medical NER Example")
    print("=" * 60)
    print()

    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,
    )

    for i, text in enumerate(CLINICAL_TEXTS, 1):
        print(f"\n--- Clinical Note {i} ---")
        print(text.strip()[:120] + "...")
        print()

        result = await shield.generate(prompt=text.strip(), schema=MedicalEntities)

        print(f"\nRouting: {result.routing.strategy.upper()}")
        print(f"Complexity score: {result.complexity_score:.3f}")
        print(f"Latency: {result.latency_ms:.0f}ms")

        if result.thinking:
            print("\nThinking (first 200 chars):")
            print(result.thinking[:200] + "...")

        if result.parsed and isinstance(result.parsed, MedicalEntities):
            print("\nExtracted entities:")
            print(f"  Conditions:  {result.parsed.conditions}")
            print(f"  Medications: {result.parsed.medications}")
            print(f"  Dosages:     {result.parsed.dosages}")
            print(f"  Procedures:  {result.parsed.procedures}")

        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())
