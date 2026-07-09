"""Use a plain JSON Schema dict and tune the run through ExtractionConfig.

No Pydantic required. ExtractionConfig adjusts behaviour such as the retry budget. Set
GROQ_API_KEY, then run this file.
"""

from nfield import ExtractionConfig, nfield

document = """
Patient: Jane Doe, age 54.
Diagnosis: Type 2 diabetes.
Blood pressure: 130/85. Active medication: metformin.
"""

schema = {
    "type": "object",
    "properties": {
        "patient_name": {"type": "string"},
        "age": {"type": "integer"},
        "diagnosis": {"type": "string"},
        "active": {"type": "boolean"},
        "medications": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["patient_name", "diagnosis"],
}

result = nfield(
    document,
    schema,
    "groq/llama-3.1-8b-instant",
    config=ExtractionConfig(max_retry_rounds=1),
)
print(result.data)
