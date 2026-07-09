# nfield

**Pull N structured fields out of a document, reliably.**

N is whatever your schema has, from a handful to thousands. Ask an LLM to fill one big JSON
schema in a single call and the answers get worse as the schema grows: the model spends its
output budget on brackets and commas instead of values, and a wide schema overflows the context
window before it finishes.

nfield is built for the wide case. It splits the schema into groups that fit the model, finds
the part of the document each group needs, extracts plain `key = value` lines instead of nested
JSON, validates every field against the text, retries the ones that fail, and reassembles the
clean nested JSON you asked for.

```python
from nfield import nfield

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
# {'vendor': 'Acme Corporation', 'total': 1284.5, 'currency': 'USD', 'paid': True}
```

On ExtractBench's 369-key SEC filings, every frontier model in the paper scores 0% (the schema
is too wide to emit in one response); nfield reaches 81-85% on a 27B open model. See the
[benchmarks](https://github.com/nfield-labs/nfield/tree/main/benchmark).

## Install

```bash
pip install nfield
pip install "nfield[groq]"      # Groq provider
pip install "nfield[openai]"    # OpenAI / any OpenAI-compatible endpoint
pip install "nfield[cli]"       # command-line interface
pip install "nfield[export]"    # pandas / CSV export
```

## Where to go next

| Guide | What it covers |
|-------|----------------|
| [Quickstart](quickstart.md) | Run your first extraction end to end. |
| [Configuration](configuration.md) | API keys (env, `.env`, explicit), model limits, and every `ExtractionConfig` setting. |
| [CLI reference](cli.md) | `inspect`, `extract`, `batch`, and all flags. |
| [Examples](examples.md) | Verified, runnable recipes with real output. |
| [Errors and troubleshooting](errors.md) | Every failure, its message, and the fix. |
| [The pipeline](concepts/pipeline.md) | How the seven stages fit together. |
| [SFEP](concepts/sfep.md) | Why `key = value` beats nested JSON. |
| [API reference](api/nfield.md) | Every public name. |
