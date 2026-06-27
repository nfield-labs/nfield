"""Extract a few fields from a document. Set GROQ_API_KEY, then run this file."""

from nfield import nfield

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
        "total": {"type": "number"},
        "date": {"type": "string", "format": "date"},
    },
    "required": ["vendor", "total", "date"],
}

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
# {'vendor': 'Acme Corporation', 'total': 1284.5, 'date': '2026-01-15'}
