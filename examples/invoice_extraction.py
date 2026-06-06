"""Invoice extraction with a Pydantic schema and a reusable engine.

Shows the class-based API (``FormatShield``) which caches the schema across
documents and calibrates the model only once.

Run:
    export GROQ_API_KEY=...
    python examples/invoice_extraction.py
"""

from __future__ import annotations

import pydantic

from formatshield import FormatShield


class LineItem(pydantic.BaseModel):
    description: str
    quantity: int
    unit_price: float


class Invoice(pydantic.BaseModel):
    invoice_number: str
    vendor: str
    total: float
    currency: str
    line_items: list[LineItem]


DOCUMENTS = [
    "INVOICE A-1\nVendor: Acme\nTotal: 100.00 EUR\n- 2x widget @ 50.00",
    "INVOICE A-2\nVendor: Globex\nTotal: 75.50 USD\n- 1x gadget @ 75.50",
]


def main() -> None:
    engine = FormatShield(
        "groq/llama-3.1-8b-instant",
        Invoice,
        context_window=131_072,
        max_output_tokens=32_768,
    )
    for doc in DOCUMENTS:
        result = engine(doc)
        print(result.data)


if __name__ == "__main__":
    main()
