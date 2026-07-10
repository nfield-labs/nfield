"""Every headline feature in one run: grounding, provenance, caching, cost, viewer.

One engine extracts one invoice with everything turned on. The run reports what it
spent in tokens and dollars, labels how well the source supports each value, records
where each value came from, caches every response, and renders a reviewable HTML page.
Run it twice: the second run answers from the cache, so its cost is $0.00 - and the
metadata proves it. Set GROQ_API_KEY first. Delete ``.nfield_cache`` to reset.
"""

from nfield import DiskCache, ExtractionConfig, NField, save_html

document = """
INVOICE #4471
Vendor: Acme Corporation
Total Due: 1284.50 USD
Date: 2026-01-15
Status: PAID
"""

schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "invoice_number": {"type": "string"},
        "total": {"type": "number"},
        "currency": {"type": "string"},
        "paid": {"type": "boolean"},
    },
    "required": ["vendor", "total"],
}

cache = DiskCache(".nfield_cache")
config = ExtractionConfig(
    ground_values=True,  # score each value against the source
    provenance=True,  # record each value's [start, end) offsets
    cache=cache,  # repeated requests answer from disk
    pricing=(0.05, 0.08),  # (input, output) USD per million tokens
    fallback_model="groq/llama-3.3-70b-versatile",  # stronger model for stragglers
)
engine = NField("groq/llama-3.1-8b-instant", schema, config=config)

result = engine.extract(document)

print("data       :", result.data)
print("status     :", result.status.value)
print(
    "tokens     :", result.metadata.tokens_prompt, "in /", result.metadata.tokens_completion, "out"
)
print(f"cost       : ${result.metadata.cost:.6f}")
print("halluc.rate:", result.metadata.hallucination_rate)
print("provenance :", result.provenance)
print("cache      :", cache.stats())

save_html(result, document, "invoice_review.html")
print("viewer     : invoice_review.html (open it in a browser)")
