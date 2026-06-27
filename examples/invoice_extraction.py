"""Reuse one engine across documents with a Pydantic schema.

Building NField once parses the schema and calibrates the model a single time, then
every call reuses that work. Set GROQ_API_KEY, then run this file.
"""

import pydantic

from nfield import NField


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


documents = [
    "INVOICE A-1\nVendor: Acme\nTotal: 100.00 EUR\n- 2x widget @ 50.00",
    "INVOICE A-2\nVendor: Globex\nTotal: 75.50 USD\n- 1x gadget @ 75.50",
]

engine = NField("groq/llama-3.1-8b-instant", Invoice)
for doc in documents:
    print(engine(doc).data)
