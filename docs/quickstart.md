# Quickstart

## Install

```bash
pip install "nfield[groq]"
export GROQ_API_KEY=...
```

## One-shot extraction

```python
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
print(result.data)        # nested dict matching the schema
print(result.status)      # SUCCESS / PARTIAL / FAILED
print(result.metadata)    # field counts, quality_score, per-field confidence, ...
```

## Pydantic and dataclass schemas

```python
import pydantic
from nfield import nfield

class Invoice(pydantic.BaseModel):
    vendor: str
    total: float

result = nfield(document, Invoice, "groq/llama-3.1-8b-instant")
```

Plain dataclasses work the same way.

## Reusing an engine

For many documents on the same schema, build the engine once. The schema is parsed a single
time and the model is calibrated once, then every call reuses that work.

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", Invoice)
for doc in documents:
    print(engine(doc).data)
```

Async is symmetric, and `extract_batch` runs a batch concurrently under a rate-limit bound:

```python
from nfield import AsyncNField

async with AsyncNField("groq/llama-3.1-8b-instant", Invoice) as engine:
    results = await engine.extract_batch(documents)
```

## Loading from files

A real schema is a file with hundreds of fields, and a real document is a file too. The
`nfield.io` helpers read them and persist results, so you do not hand-roll file I/O.

```python
from nfield import nfield, load_document, load_schema, save_results, load_results

schema = load_schema("schemas/clinical_trial.json")   # parsed JSON Schema dict
document = load_document("records/trial_4471.txt")     # decoded UTF-8 text

result = nfield(document, schema, "groq/llama-3.3-70b-versatile")

save_results([result], "out/trials.jsonl")   # JSON Lines, one record per line
again = load_results("out/trials.jsonl")      # round-trips back to ExtractionResult
```

Loading is text and JSON only; PDF or DOCX parsing stays your job. `load_schema` raises
`SchemaError` on malformed JSON, and `save_results` creates parent directories as needed.

## Command line

```bash
nfield inspect schema.json                 # offline: field count, types, K_min estimate
nfield extract doc.txt --schema schema.json --model groq/llama-3.1-8b-instant
nfield extract doc.txt -s schema.json -m groq/llama-3.1-8b-instant \
    --ground-values --show-metadata --format jsonl     # opt-ins + run summary on stderr
nfield batch ./docs -s schema.json -m groq/llama-3.1-8b-instant -o out.jsonl
```

With no `--model`, nfield reads `NFIELD_MODEL` from the environment, then
`ExtractionConfig(default_model=...)`.

Every `ExtractionConfig` setting is a flag on `extract` (see `nfield extract --help`, grouped
into panels). `--format` is `json`, `jsonl`, or `csv`. `batch` runs a directory or a list of
files through one reused engine and streams JSON Lines that round-trip with `load_results`.
Both commands exit non-zero when an API/call failure leaves the result incomplete.

## Model limits for large schemas

You name the model as `"provider/model-name"`. For a wide schema on a large model, pass the
model's real limits so nfield plans across the full window instead of a safe default:

```python
result = nfield(
    document, schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072,
    max_output_tokens=32_768,
)
```

The same `context_window` / `max_output_tokens` arguments work on `NField`, `AsyncNField`,
and `nfield_async`, and as `--context-window` / `--max-output-tokens` on the CLI. Omit them
to use the provider's conservative default.

## Domain and task context

Add your own framing without changing the extraction format:

```python
result = nfield(
    document, schema, "groq/llama-3.1-8b-instant",
    instructions="Extracting from clinical trial records. Prefer ISO-8601 dates; "
                 "leave unknown fields NULL.",
)
```

`instructions` is prepended to the built-in SFEP prompt (which is always kept, so parsing
stays valid) and is counted in capacity planning, so a long value correctly reduces the
per-call document budget. On the CLI this is `--instructions`.
