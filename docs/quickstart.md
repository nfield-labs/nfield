# Quickstart

## Install

```bash
pip install "nfield[groq]"
export GROQ_API_KEY=...
```

## One-shot extraction

```python
from nfield import nfield

schema = {
    "type": "object",
    "properties": {
        "vendor": {"type": "string"},
        "total": {"type": "number"},
        "date": {"type": "string", "format": "date"},
    },
    "required": ["vendor", "total", "date"],
}

result = nfield(document_text, schema, "groq/llama-3.1-8b-instant")
print(result.data)        # nested dict matching the schema
print(result.status)      # SUCCESS / PARTIAL / FAILED
print(result.metadata)    # K, K_min, quality_score, per-field confidence, ...
```

## Pydantic schemas

```python
import pydantic
from nfield import nfield

class Invoice(pydantic.BaseModel):
    vendor: str
    total: float

result = nfield(document_text, Invoice, "groq/llama-3.1-8b-instant")
```

Plain dataclasses work too.

## Reusable engine

For many documents on the same schema, build the engine once - the schema is
normalised a single time and the model is calibrated once.

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", Invoice)
for doc in documents:
    print(engine(doc).data)
```

Async is symmetric:

```python
from nfield import AsyncNField

async with AsyncNField("groq/llama-3.1-8b-instant", Invoice) as engine:
    result = await engine.extract(document_text)
```

## Command line

```bash
nfield inspect schema.json
nfield extract doc.txt --schema schema.json --model groq/llama-3.1-8b-instant
```

If no model is passed, NField reads `NFIELD_MODEL` from the
environment, then `ExtractionConfig(default_model=...)`.

## Model specs (context window & output ceiling)

You name the model as `"provider/model-name"` and tell NField its real
limits, so capacity planning uses the full window:

```python
result = nfield(
    document_text, schema, "groq/llama-3.1-8b-instant",
    context_window=131_072,
    max_output_tokens=32_768,
)
```

The same `context_window` / `max_output_tokens` arguments work on `NField`,
`AsyncNField`, and `nfield_async`, and as `--context-window` /
`--max-output-tokens` on the CLI. Omit them to use the provider's conservative
default.

## Domain / task context (system & user prompts)

Add your own framing without touching the extraction format:

```python
result = nfield(
    document_text, schema, "groq/llama-3.1-8b-instant",
    context_window=131_072, max_output_tokens=32_768,
    instructions="Extracting from clinical trial records. Prefer ISO-8601 dates; "
                 "leave unknown fields NULL.",
)
```

`instructions` is **prepended** to NField's built-in SFEP prompt (which is
always kept, so output parsing stays valid) and is counted in capacity planning - a
long value correctly reduces the per-call document budget. CLI: `--instructions`.
