# NField

[![CI](https://github.com/nfield-labs/nfield/actions/workflows/ci.yml/badge.svg)](https://github.com/nfield-labs/nfield/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/nfield.svg)](https://pypi.org/project/nfield/)
[![Python](https://img.shields.io/pypi/pyversions/nfield.svg)](https://pypi.org/project/nfield/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Extract N structured fields from any document with zero format tax.**

## The problem

Asking an LLM for one large nested JSON object degrades as the schema grows - the
model spends attention on brackets, commas, and quotes instead of the values, and
accuracy falls off a cliff on schemas with hundreds of fields. Most structured-output
tools cap out well before that scale.

NField decomposes the schema into capacity-bounded groups, retrieves only the
relevant document spans per group, and extracts a flat `path = value` format (SFEP)
instead of nested JSON - then validates per field, retries failures surgically, and
reassembles valid nested JSON.

## Install

```bash
pip install nfield
pip install "nfield[groq]"   # Groq provider
pip install "nfield[cli]"    # command-line interface
```

## Quickstart

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

result = nfield(
    document_text,
    schema,
    "groq/llama-3.1-8b-instant",
    context_window=131_072,      # the model's real limits - NField's
    max_output_tokens=32_768,    # capacity planning is a function of these
)
print(result.data)
```

The model is a `"provider/model-name"` string (`groq/…` and `openai/…` today).
Unlike single-call libraries, NField **plans** how to split a wide schema across
calls, so it needs the model's real `context_window` (C_eff) and
`max_output_tokens` (M_O). Omit them and a conservative 8192 default applies -
safe, but it under-fills large models (gpt-4o and llama-3.3-70b are ~128K), so
pass the real window for full throughput. Pydantic models and dataclasses work as
schemas too. For many documents, reuse a `NField` (sync) or `AsyncNField` (async)
engine.

## How it works

A seven-stage pipeline (S0–S6): resource calibration → schema analysis → structural
grouping → document pre-pass (BM25) → capacity packing → excerpt finalisation →
SFEP extraction → per-field validation & surgical retry → JSON assembly. Only three
stages call the model. See [docs/concepts/pipeline.md](docs/concepts/pipeline.md).

## Command line

```bash
nfield inspect schema.json
nfield extract doc.txt --schema schema.json --model groq/llama-3.1-8b-instant
```

## Supported providers

- **Groq** - `from_model("groq/<model>")`, install `nfield[groq]`.
- **OpenAI-compatible** - `from_model("openai/<model>")`, install `nfield[openai]`. A
  `base_url` retargets the same provider at any compatible endpoint, hosted
  (Together, Fireworks, OpenRouter, DeepSeek, xAI, Mistral, Azure) or local
  (Ollama, vLLM, LM Studio).

For a reasoning/thinking model (Qwen3, DeepSeek-R1, QwQ), pass
`ExtractionConfig(reasoning_model=True)` so its thinking is disabled per call and
does not consume the answer's output budget.

The provider layer is a small Protocol; adding one is a single registry entry -
see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add providers
and pipeline stages, and the development workflow (`uv`, `ruff`, `mypy --strict`,
`pytest`).

## License

Apache-2.0. See [LICENSE](LICENSE).
