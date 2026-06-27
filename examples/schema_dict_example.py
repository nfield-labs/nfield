"""Plain JSON Schema dict as input - no Pydantic required.

Run:
    export GROQ_API_KEY=...
    python examples/schema_dict_example.py
"""

from __future__ import annotations

from nfield import ExtractionConfig, nfield

DOCUMENT = """
Patient: Jane Doe, age 54.
Diagnosis: Type 2 diabetes.
Blood pressure: 130/85. Active medication: metformin.
"""

SCHEMA = {
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


def main() -> None:
    config = ExtractionConfig(max_retry_rounds=1)
    result = nfield(
        DOCUMENT,
        SCHEMA,
        "groq/llama-3.1-8b-instant",
        context_window=131_072,
        max_output_tokens=32_768,
        config=config,
    )
    print(result.data)


if __name__ == "__main__":
    main()
