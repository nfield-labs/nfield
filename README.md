<div align="center">

# nfield

### Pull N structured fields out of a document, reliably.

**From a handful of fields to thousands, on models that choke on a single-call request.**

[![PyPI](https://img.shields.io/pypi/v/nfield?style=flat-square&color=2563eb)](https://pypi.org/project/nfield/)
[![Python](https://img.shields.io/pypi/pyversions/nfield?style=flat-square&color=2563eb)](https://pypi.org/project/nfield/)
[![CI](https://img.shields.io/github/actions/workflow/status/nfield-labs/nfield/ci.yml?branch=main&style=flat-square&label=CI)](https://github.com/nfield-labs/nfield/actions/workflows/ci.yml)
[![Typed](https://img.shields.io/badge/typed-mypy%20strict-2563eb?style=flat-square)](https://mypy.readthedocs.io/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue?style=flat-square)](LICENSE)

[**Quickstart**](#quickstart) · [**Benchmarks**](#benchmarks) · [**CLI**](#command-line) · [**Docs**](docs/index.md) · [**How it works**](#how-it-works)

</div>

---

Ask an LLM to fill one big JSON schema in a single call and the answers get worse as the
schema grows. The model spends its output budget on brackets, commas, and quotes instead of
values, and a wide schema overflows the context window before it finishes. Most
structured-output tools are built for a handful of fields and quietly fall apart past that.

**nfield is built for the wide case.** It splits the schema into groups that fit the model,
finds the part of the document each group needs, extracts plain `key = value` lines instead of
nested JSON, validates every field against the text, retries the ones that fail, and
reassembles the clean nested JSON you asked for. Schemas with thousands of fields come back
intact.

> On ExtractBench's 369-key SEC filings, **every** frontier model in the paper (GPT-5, Opus 4.5,
> Gemini 3 Pro) scores **0%**: the schema is too wide to emit in one response. nfield reaches
> **81-85%** on a 27B open model, because it never asks the model to emit the whole schema at
> once. [See the numbers.](#benchmarks)

## Contents

- [Quickstart](#quickstart) · [Install](#install) · [Set your API key](#set-your-api-key)
- [Benchmarks](#benchmarks) · [Why nfield](#why-nfield) · [How nfield compares](#how-nfield-compares)
- [Working with files](#working-with-files) · [Reusing an engine](#reusing-an-engine) · [Command line](#command-line)
- [Grounding](#grounding-and-provenance) · [Configuration](#configuration) · [Reasoning models](#reasoning-models)
- [How it works](#how-it-works) · [Documentation](#documentation) · [Contributing](#contributing)

## Quickstart

```python
from nfield import nfield

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

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
# {'vendor': 'Acme Corporation', 'invoice_number': '#4471',
#  'total': 1284.5, 'currency': 'USD', 'paid': True}
```

The model is a `"provider/model-name"` string. A **Pydantic model** or a **dataclass** works as
the schema too, and comes back as the same nested JSON:

```python
from pydantic import BaseModel
from nfield import nfield

class Invoice(BaseModel):
    vendor: str
    total: float
    currency: str
    paid: bool

result = nfield(document, Invoice, "groq/llama-3.1-8b-instant")
# {'vendor': 'Acme Corporation', 'total': 1284.5, 'currency': 'USD', 'paid': True}
```

> **Wide schema on a big model?** Pass the model's real `context_window` and
> `max_output_tokens`. nfield plans the calls from these numbers, so giving it the true limits
> (instead of the safe default) lets it use the whole window and split the schema into as few
> calls as the budget allows.
>
> ```python
> result = nfield(
>     document, schema, "groq/llama-3.3-70b-versatile",
>     context_window=131_072,     # the model's real context window
>     max_output_tokens=32_768,   # the model's real output limit
> )
> ```

## Install

```bash
pip install nfield
```

nfield's core has one small dependency; providers and features are opt-in extras:

| Extra | Install | Adds |
|-------|---------|------|
| `groq` | `pip install "nfield[groq]"` | Groq provider (`groq/...`) |
| `openai` | `pip install "nfield[openai]"` | OpenAI + any OpenAI-compatible endpoint via `base_url` |
| `google` | `pip install "nfield[google]"` | Google Gemini provider (`google/...`) |
| `anthropic` | `pip install "nfield[anthropic]"` | Anthropic Claude provider (`anthropic/...`) |
| `openrouter` | `pip install "nfield[openrouter]"` | OpenRouter (`openrouter/...`): one key, many model families |
| `cli` | `pip install "nfield[cli]"` | The `nfield` command-line tool |
| `export` | `pip install "nfield[export]"` | pandas / CSV export |

OpenAI-compatible presets (`deepseek`, `together`, `fireworks`, `mistral`, `xai`, `perplexity`,
`cerebras`, `ollama`) install the same way, e.g. `pip install "nfield[deepseek]"`, and route with
their own prefix and key. Full list in [docs/configuration.md](docs/configuration.md).

Combine them: `pip install "nfield[groq,cli,export]"`.

## Set your API key

nfield reads the provider's standard environment variable, so the common path is zero config:

```bash
export GROQ_API_KEY="gsk_..."       # or OPENAI_API_KEY, etc.
```

A `.env` file works the same way once it is loaded into the environment (nfield reads
`os.environ`; use `python-dotenv` or your shell to load it). Prefer the env var in production.

For a secret vault or a multi-tenant service, pass the key explicitly instead (it is never
logged):

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", schema, api_key="gsk_...")
result = engine.extract(document)
```

Point at any OpenAI-compatible endpoint with `base_url`:

```python
engine = NField("openai/llama-3.1-8b", schema, base_url="http://localhost:11434/v1")
```

Native prefixes are `groq/`, `openai/`, `google/` (Gemini), and `anthropic/` (Claude). Or use one
`OPENROUTER_API_KEY` to reach many vendors through `openrouter/`, using OpenRouter's own model
slugs:

```python
nfield(document, schema, "openrouter/anthropic/claude-3-haiku")     # Claude
nfield(document, schema, "openrouter/deepseek/deepseek-chat")       # DeepSeek
```

See [docs/configuration.md](docs/configuration.md) for every provider and preset.

## Benchmarks

Every number is one named model on one date: nfield measures the method, not a leaderboard.
Accuracy is the model's job; completeness at scale is nfield's. Full detail, reproduction
steps, and manifests in **[benchmark/README.md](benchmark/README.md)**.

**[ExtractBench](https://github.com/ContextualAI/extract-bench)** (Contextual AI, real
documents, human-checked gold) on **`qwen/3.6-27b` on Groq**:

| Domain | Coverage | Value accuracy | Judged | Best frontier model (paper) |
|--------|---------:|---------------:|-------:|----------------------------:|
| Research | 96.8% | 95.5% | 95.9% | 49.0% |
| Credit | 94.3% | 82.3% | 86.4% | 86.9% |
| Resumes | 87.9% | 81.6% | 87.3% | 24.0% |
| Sports | 99.0% | 98.6% | 98.8% | 18.3% |
| **SEC 10-K/Q** (369 keys) | **91.2%** | **81.1%** | **85.4%** | **0.0%** |
| **Overall** | **92%** | **84.2%** | **87.7%** | 6.9% |

On the 369-key SEC filings every frontier model in the paper scored zero: the schema is too
wide to emit in one response, so the JSON truncates and the whole document fails. nfield never
asks the model to emit the whole schema at once.

**Staying complete as the schema widens** (synthetic schemas, one run each): coverage holds at
~100% while the number of model calls tracks the computed minimum, with no call storm.

| Fields | Coverage | Calls (minimum) |
|-------:|---------:|----------------:|
| 2,523 | 100% | 62 (61) |
| 4,000 | 100% | 95 (94) |
| 5,641 | ~100% | 127 (124) |

## Why nfield

<table>
<tr>
<td width="50%" valign="top">

**Built for wide schemas**
Thousands of fields, not a handful. nfield plans how to split the schema across calls so the
model never sees more than it can reliably handle at once.

</td>
<td width="50%" valign="top">

**Grounded, not guessed**
Every value is validated against the document; fields that fail are retried surgically rather
than left wrong. Opt into a hallucination score per value.

</td>
</tr>
<tr>
<td width="50%" valign="top">

**Any model**
Groq, OpenAI, or any OpenAI-compatible endpoint (Together, Fireworks, DeepSeek, vLLM, Ollama,
LM Studio) through one `base_url`. No local weights required.

</td>
<td width="50%" valign="top">

**Typed in, typed out**
Give it a JSON Schema, a Pydantic model, or a dataclass; get back nested JSON in the same
shape. Ships `py.typed` and passes `mypy --strict`.

</td>
</tr>
</table>

## How nfield compares

Different tools live in different lanes. nfield's lane is field count: staying complete when the
schema is wide.

| Tool | Wide schemas (hundreds to thousands of fields) | Grounds values to source | Works with any remote model |
|------|:---:|:---:|:---:|
| **nfield** | ✅ splits + reassembles | ✅ | ✅ |
| Instructor | single call | no | ✅ |
| LangExtract | chunked entities | ✅ | ✅ |
| ContextGem | aspect / concept | ✅ | ✅ |
| LangStruct | single call, auto-prompt | ✅ | ✅ |

nfield's lane is field count: keeping one wide schema complete when it does not fit in a single
response. The table covers that one axis; it is not a quality ranking.

## Working with files

Real schemas and documents live on disk. The `nfield.io` helpers load them and persist results
as JSON Lines, one record per line:

```python
from nfield import nfield, load_document, load_schema, save_results

schema = load_schema("schemas/clinical_trial.json")   # a JSON Schema with hundreds of fields
document = load_document("records/trial_4471.txt")

result = nfield(
    document, schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072,     # the model's real context window
    max_output_tokens=32_768,   # the model's real output limit
)
save_results([result], "out/trials.jsonl")            # round-trips with load_results
```

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

Install the `cli` extra, then:

```bash
# Offline: field count, type breakdown, and a minimum-call (K_min) estimate. No API calls.
nfield inspect schema.json

# Extract one document (JSON to stdout).
nfield extract invoice.txt --schema schema.json --model groq/llama-3.1-8b-instant

# Opt-ins + a run summary (status, quality, grounding) on stderr; JSON Lines output.
nfield extract invoice.txt -s schema.json -m groq/llama-3.1-8b-instant \
    --ground-values --show-metadata --format jsonl

# Extract a whole directory with one reused engine, streaming JSON Lines.
nfield batch ./docs -s schema.json -m groq/llama-3.1-8b-instant -o out.jsonl
```

Every `ExtractionConfig` setting is a flag on `extract`, grouped into panels in
`nfield extract --help`. `--format` is `json`, `jsonl`, or `csv`. Both commands exit non-zero
when an API/call failure leaves the result incomplete. Full reference:
**[docs/cli.md](docs/cli.md)**.

## Grounding and provenance

Turn on grounding to label how well the source supports each value, and provenance to get the
exact character span each value came from:

```python
from nfield import nfield, ExtractionConfig

result = nfield(
    document, schema, "groq/llama-3.1-8b-instant",
    config=ExtractionConfig(ground_values=True, provenance=True),
)
print(result.metadata.hallucination_rate)   # 0.0
print(result.provenance)
# {'vendor': [22, 38], 'invoice_number': [8, 13], 'total': [50, 56], 'currency': [58, 61]}
```

Grounding is non-destructive: a weakly-supported value is reported, never silently dropped,
because a correct value is often not verbatim (units, derived dates, enum choices).

## Configuration

Tune behaviour with `ExtractionConfig`, passed as `config=...`. Everything is opt-in with a
sane default, and each setting is also a CLI flag. The ones you reach for most:

| Setting | Default | What it does |
|---------|---------|--------------|
| `ground_values` | `False` | Score each value's support against the source; report `hallucination_rate`. |
| `provenance` | `False` | Attach `[start, end)` source char offsets per value. |
| `reasoning_model` | `False` | Disable a reasoning model's thinking so it does not eat the output budget. |
| `max_concurrent_calls` | `4` | Leaf calls in flight at once; raise on higher-throughput plans. |
| `fallback_model` | `None` | Stronger model to escalate still-failing fields to after recovery. |
| `knowledge_fallback` | `False` | Fill fields absent from the document from the model's own knowledge. |
| `closed_book` | `False` | Fill the schema from model knowledge with no document at all. |

The complete table (25+ settings) is in **[docs/configuration.md](docs/configuration.md)**.

## Reasoning models

For a reasoning model (Qwen3, DeepSeek-R1, QwQ), set `reasoning_model=True` so its thinking is
turned off per call and does not consume the answer's output budget:

```python
result = nfield(document, schema, "groq/qwen-3.6-27b",
                config=ExtractionConfig(reasoning_model=True))
```

## How it works

A seven-stage pipeline turns one wide request into many small, grounded ones:

```
calibrate -> analyze schema -> group fields -> retrieve text (BM25)
          -> pack to capacity -> extract (key = value) -> validate + retry -> assemble JSON
```

Only the extraction and retry stages call the model; everything else is local. Two ideas carry
it: the output is `field.path = value` lines rather than one JSON blob, so a run cannot fail on a
truncated brace; and the schema is split so every call stays inside the model's real context and
output budget. See **[docs/concepts/pipeline.md](docs/concepts/pipeline.md)** for the full
walkthrough.

## Documentation

| | |
|---|---|
| [Quickstart](docs/quickstart.md) | Run your first extraction. |
| [Configuration](docs/configuration.md) | API keys, and every `ExtractionConfig` setting. |
| [CLI reference](docs/cli.md) | `inspect`, `extract`, `batch`, and all flags. |
| [Examples](docs/examples.md) | Verified, runnable recipes with real output. |
| [Errors and troubleshooting](docs/errors.md) | Every failure, its message, and the fix. |
| [How the pipeline works](docs/concepts/pipeline.md) | The seven stages, end to end. |
| [The SFEP format](docs/concepts/sfep.md) | Why `key = value` beats nested JSON. |
| [API reference](docs/api/nfield.md) | Every public name. |
| [Benchmarks](benchmark/README.md) | ExtractBench, FinTagging, and scaling runs. |

## Contributing

Issues and pull requests are welcome. Adding a provider is a single registry entry, and the
development workflow is `uv` + `ruff` + `mypy --strict` + `pytest`. See
[CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0. See [LICENSE](LICENSE).
