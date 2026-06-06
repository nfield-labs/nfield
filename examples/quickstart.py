"""Quickstart: extract a few fields from a document in five lines.

Run:
    export GROQ_API_KEY=...
    python examples/quickstart.py
"""

from __future__ import annotations

from formatshield import nfield

DOCUMENT = """
INVOICE #4471
Vendor: Acme Corporation
Total Due: 1284.50 USD
Date: 2026-01-15
"""

SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "date": {"type": "string", "format": "date"},
    },
    "required": ["vendor", "total", "date"],
}


def main() -> None:
    # Tell FormatShield the model's real limits so capacity planning uses the
    # full window instead of a conservative default.
    result = nfield(
        DOCUMENT,
        SCHEMA,
        "groq/llama-3.1-8b-instant",
        context_window=131_072,
        max_output_tokens=32_768,
    )
    print(result.data)
    print(f"status={result.status.value} fields={result.metadata.fields_extracted}")


if __name__ == "__main__":
    main()
