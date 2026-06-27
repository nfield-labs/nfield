<div align="center">

# nfield

**Pull N structured fields out of a document, reliably.**

*N is whatever your schema has, from a handful to thousands.*

[![PyPI](https://img.shields.io/pypi/v/nfield?style=flat-square&color=2563eb)](https://pypi.org/project/nfield/)
[![Python](https://img.shields.io/pypi/pyversions/nfield?style=flat-square&color=2563eb)](https://pypi.org/project/nfield/)
[![CI](https://img.shields.io/github/actions/workflow/status/nfield-labs/nfield/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/nfield-labs/nfield/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)

[**Quickstart**](docs/quickstart.md) · [**How it works**](docs/concepts/pipeline.md) · [**API**](docs/api/nfield.md)

</div>

---

Ask an LLM to fill one big JSON
schema in a single call and the answers get worse as the schema grows. The model spends its
output budget on brackets, commas, and quotes instead of the values, and a wide schema can
overflow the context window before it finishes. Most structured-output tools are built for a
handful of fields and quietly fall apart past that.

nfield is built for the wide case. It splits the schema into groups that fit the model,
finds the part of the document each group needs, extracts plain `key = value` lines instead
of nested JSON, validates every field against the text, retries the ones that fail, and
reassembles the clean nested JSON you asked for. Schemas with thousands of fields come back
intact, on models that would choke on a single-call request.

## Install

```bash
pip install nfield
pip install "nfield[groq]"      # Groq provider
pip install "nfield[openai]"    # OpenAI / any OpenAI-compatible endpoint
pip install "nfield[cli]"       # command-line interface
```

## Quickstart

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
print(result.data)
# {'vendor': 'Acme Corporation', 'total': 1284.5, 'date': '2026-01-15'}
```

Set your provider key first (`export GROQ_API_KEY=...`). The model is a
`"provider/model-name"` string. A Pydantic model or a dataclass works as the schema too.

For large schemas on large models, pass the model's real limits so nfield can plan across
the full window instead of a safe default:

```python
result = nfield(
    document, schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072,     # the model's real context window
    max_output_tokens=32_768,   # the model's real output limit
)
```

## Working with files

Real schemas and documents live on disk, not in string literals. The `nfield.io` helpers
load them and persist the results, which is where nfield is meant to be used: point it at a
wide schema file and a long document and it plans the calls for you.

```python
from nfield import nfield, load_document, load_schema, save_results

schema = load_schema("schemas/clinical_trial.json")   # a JSON Schema with hundreds of fields
document = load_document("records/trial_4471.txt")

result = nfield(
    document, schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072, max_output_tokens=32_768,
)
save_results([result], "out/trials.jsonl")            # JSON Lines, one record per line
```

## Benchmarks

The repository ships a field-count scaling benchmark (`benchmark/`) that measures how
extraction holds up as the schema grows. It runs on real documents with schemas from a
couple hundred fields up to over a thousand (clinical-trial records, country factbooks) and
on synthetic fixtures into the thousands of fields. Every number is for a single named model
on a given date: accuracy depends on the model you choose, so the benchmark reports the
method's behaviour rather than a leaderboard. See [benchmark/README.md](benchmark/README.md).

## Why nfield

- **Built for wide schemas.** Thousands of fields, not a handful. nfield plans how to split
  the schema across calls so the model never sees more than it can handle at once.
- **Grounded, not guessed.** Every value is validated against the document; fields that fail
  are retried surgically rather than left wrong.
- **Any model.** Groq, OpenAI, or any OpenAI-compatible endpoint (Together, Fireworks,
  DeepSeek, vLLM, Ollama, LM Studio) through one `base_url`.
- **Typed in, typed out.** Give it a JSON Schema, a Pydantic model, or a dataclass; get back
  nested JSON in the same shape.
- **Fully typed, no required dependencies.** `import nfield` pulls in nothing until you pick
  a provider; ships with `py.typed` and passes `mypy --strict`.

## How it works

A seven-stage pipeline turns one wide request into many small, grounded ones:

```
calibrate -> analyze schema -> group fields -> retrieve text (BM25)
          -> pack to capacity -> extract (key = value) -> validate + retry -> assemble JSON
```

Only the extraction and retry stages call the model; everything else is local. See
[docs/concepts/pipeline.md](docs/concepts/pipeline.md) for the full walkthrough.

## Reusing an engine

For many documents, build the engine once so the schema is parsed and the model calibrated a
single time:

```python
from nfield import NField, AsyncNField

engine = NField("groq/llama-3.1-8b-instant", schema)
for doc in documents:
    print(engine(doc).data)

# or run them concurrently
async with AsyncNField("groq/llama-3.1-8b-instant", schema) as engine:
    results = await engine.extract_batch(documents)
```

## Command line

```bash
nfield inspect schema.json
nfield extract doc.txt --schema schema.json --model groq/llama-3.1-8b-instant
```

## Reasoning models

For a reasoning model (Qwen3, DeepSeek-R1, QwQ), pass
`config=ExtractionConfig(reasoning_model=True)` so its thinking is turned off per call and
does not eat the answer's output budget.

## Documentation

- [Quickstart](docs/quickstart.md)
- [How the pipeline works](docs/concepts/pipeline.md)
- [The SFEP format](docs/concepts/sfep.md)
- [API reference](docs/api/nfield.md)

## Contributing

Issues and pull requests are welcome. Adding a provider is a single registry entry, and the
development workflow is `uv` + `ruff` + `mypy --strict` + `pytest`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
