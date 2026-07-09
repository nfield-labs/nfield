"""Wrap nfield as a tool an agent can call.

The extraction call becomes a plain function described by a JSON Schema, the same shape
tool-calling APIs expect. A small dispatcher stands in for the agent framework; swap it for
your own and the tool stays the same. This is a usage pattern built from the public API, not a
separate tool-calling feature. Set GROQ_API_KEY, then run this file.
"""

from typing import Any

from nfield import NField

INVOICE_SCHEMA = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "currency": {"type": "string"},
        "due_date": {"type": "string"},
    },
    "required": ["vendor", "total"],
}

_engine = NField("groq/llama-3.3-70b-versatile", INVOICE_SCHEMA)


def extract_invoice(document: str) -> dict[str, Any]:
    """Pull structured invoice fields out of raw text."""
    return _engine.extract(document).data


TOOLS = {"extract_invoice": extract_invoice}


def dispatch(tool_call: dict[str, Any]) -> dict[str, Any]:
    """Run the named tool with its arguments, standing in for an agent framework."""
    return TOOLS[tool_call["name"]](**tool_call["arguments"])


document = """
INVOICE #A-2231
Vendor: Globex Industrial
Amount Due: 4820.00 EUR
Payment due by: 2026-03-01
"""

# stands in for the tool call a model would emit
call = {"name": "extract_invoice", "arguments": {"document": document}}
print(dispatch(call))
