"""Cache model responses so a repeated extraction is free.

The cache is keyed on the exact request, so a hit is always the text the model would have
returned. Here the same engine extracts the same document twice: the first run calls the
model, the second reads the on-disk cache. Set GROQ_API_KEY, then run this file twice - the
second process reuses the first process's entries. Delete ``.nfield_cache`` to reset.
"""

import time

from nfield import DiskCache, ExtractionConfig, NField

document = """
INVOICE #4471
Vendor: Acme Corporation
Total Due: 1284.50 USD
"""

schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
    },
    "required": ["vendor", "total"],
}

engine = NField(
    "groq/llama-3.3-70b-versatile",
    schema,
    config=ExtractionConfig(cache=DiskCache(".nfield_cache")),
)

start = time.perf_counter()
first = engine.extract(document)
print(f"cold run: {time.perf_counter() - start:.2f}s", first.data)

start = time.perf_counter()
second = engine.extract(document)
print(f"warm run: {time.perf_counter() - start:.2f}s", second.data)

assert first.data == second.data
