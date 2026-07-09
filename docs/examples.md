# Examples

These all ran against `groq/llama-3.1-8b-instant`, and the output shown is what came back. Set
your key first:

```bash
export GROQ_API_KEY="gsk_..."
```

Prefer to run them? The [`examples/` folder](https://github.com/nfield-labs/nfield/tree/main/examples)
has the same ideas as single-file scripts and as Jupyter notebooks you can step through, each
saved with its output. This page is the copy-paste version.

The shared document and schema used below:

```python
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
```

## 1. Basic extraction

```python
from nfield import nfield

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
```

```python
{'vendor': 'Acme Corporation', 'invoice_number': '#4471',
 'total': 1284.5, 'currency': 'USD', 'paid': True}
```

## 2. A Pydantic model as the schema

```python
from pydantic import BaseModel
from nfield import nfield

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    paid: bool

result = nfield(document, Invoice, "groq/llama-3.1-8b-instant")
print(result.data)
```

```python
{'vendor': 'Acme Corporation', 'total': 1284.5, 'currency': 'USD', 'paid': True}
```

A `dataclass` works the same way. The result is always plain nested JSON in `result.data`.

## 3. Explicit API key

Prefer the environment variable in production; pass the key directly for a vault or a
multi-tenant service (it takes precedence over the env var and is never logged):

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", schema, api_key="gsk_...")
result = engine.extract(document)
print(result.data)
```

```python
{'vendor': 'Acme Corporation', 'invoice_number': '4471',
 'total': 1284.5, 'currency': 'USD', 'paid': True}
```

## 4. Grounding and provenance

Score how well the source supports each value, and get the exact character span each value came
from:

```python
from nfield import nfield, ExtractionConfig

config = ExtractionConfig(ground_values=True, provenance=True)
result = nfield(document, schema, "groq/llama-3.1-8b-instant", config=config)

print("hallucination_rate:", result.metadata.hallucination_rate)
print("grounded / ungrounded:", result.metadata.fields_grounded, result.metadata.fields_ungrounded)
print("provenance:", result.provenance)
```

```python
hallucination_rate: 0.0
grounded / ungrounded: 4 0
provenance: {'vendor': [22, 38], 'invoice_number': [8, 13], 'total': [50, 56], 'currency': [58, 61]}
```

Booleans and enum choices are exempt from grounding (a boolean is not a span; an enum is already
validated), which is why four of the five values are grounded here.

## 5. Working with files

```python
from nfield import nfield, load_document, load_schema, save_results

schema = load_schema("schemas/invoice.json")
document = load_document("records/invoice_4471.txt")

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
save_results([result], "out/invoices.jsonl")     # JSON Lines, one record per line
```

Read them back with the same shape:

```python
from nfield import load_results

results = load_results("out/invoices.jsonl")
print(results[0].status.value, results[0].data["vendor"])
```

## 6. Many documents, one engine

Build the engine once so the schema is parsed and the model calibrated a single time:

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", schema)
for doc in documents:
    print(engine(doc).data)
```

Run them concurrently with the async engine:

```python
from nfield import AsyncNField

async with AsyncNField("groq/llama-3.1-8b-instant", schema) as engine:
    results = await engine.extract_batch(documents, max_concurrent=8)
```

## 7. Closed-book (no document)

Fill a schema from the model's own knowledge, with no source document:

```python
from nfield import nfield, ExtractionConfig

schema = {"type": "object",
          "properties": {"capital": {"type": "string"}, "continent": {"type": "string"}},
          "required": ["capital"]}

result = nfield(
    "", schema, "groq/llama-3.1-8b-instant",
    config=ExtractionConfig(closed_book=True),
    instructions="Subject: France",
)
print(result.data)
```

```python
{'capital': 'Paris', 'continent': 'Europe'}
```

Closed-book has no source to ground against, so grounding and provenance are forced off in this
mode. Add `self_consistency=True` to sample each field twice and keep only agreeing answers.

## 8. From the command line

```bash
# Offline schema analysis (no API call)
nfield inspect schema.json

# Extract, with a run summary on stderr
nfield extract invoice.txt -s schema.json -m groq/llama-3.1-8b-instant --show-metadata

# A whole folder, streamed to JSON Lines
nfield batch ./docs -s schema.json -m groq/llama-3.1-8b-instant -o out.jsonl
```

See the [CLI reference](cli.md) for every flag.

## 9. Large schemas on large models

Pass the model's real limits so nfield plans across the full window:

```python
result = nfield(
    document, wide_schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072,
    max_output_tokens=32_768,
)
print(result.metadata.K, "calls for", result.metadata.fields_total, "fields")
```

nfield splits the schema into as many bounded calls as the budget requires and reassembles the
result, so a schema with hundreds or thousands of fields comes back intact. See the
[benchmarks](https://github.com/nfield-labs/nfield/tree/main/benchmark) for how coverage holds
as the field count climbs past a thousand.
