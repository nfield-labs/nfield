"""Score values against the source and read character-level provenance.

``ground_values`` reports how well the document supports each value; ``provenance`` records
the ``[start, end)`` span each value came from. Both are opt-in and neither changes the values
you get back. Set GROQ_API_KEY, then run this file.
"""

from nfield import ExtractionConfig, nfield

document = """
INVOICE #4471
Vendor: Acme Corporation
Total Due: 1284.50 USD
Date: 2026-01-15
"""

schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "currency": {"type": "string"},
    },
    "required": ["vendor", "total"],
}

result = nfield(
    document,
    schema,
    "groq/llama-3.3-70b-versatile",
    config=ExtractionConfig(ground_values=True, provenance=True),
)

print(result.data)
print("hallucination rate:", result.metadata.hallucination_rate)
for field, (start, end) in (result.provenance or {}).items():
    print(f"{field}: {document[start:end]!r}")
