# FormatShield

[![CI](https://github.com/nfield-labs/formatshield/actions/workflows/ci.yml/badge.svg)](https://github.com/nfield-labs/formatshield/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/formatshield.svg)](https://pypi.org/project/formatshield/)
[![Python](https://img.shields.io/pypi/pyversions/formatshield.svg)](https://pypi.org/project/formatshield/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

**Extract N structured fields from any document with zero format tax.**

## The problem

Asking an LLM for one large nested JSON object degrades as the schema grows — the
model spends attention on brackets, commas, and quotes instead of the values, and
accuracy falls off a cliff on schemas with hundreds of fields. Most structured-output
tools cap out well before that scale.

FormatShield decomposes the schema into capacity-bounded groups, retrieves only the
relevant document spans per group, and extracts a flat `path = value` format (SFEP)
instead of nested JSON — then validates per field, retries failures surgically, and
reassembles valid nested JSON.

## Install

```bash
pip install formatshield
pip install "formatshield[groq]"   # Groq provider
pip install "formatshield[cli]"    # command-line interface
```

## Quickstart

```python
from formatshield import nfield

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
    context_window=131_072,      # the model's real limits — FormatShield's
    max_output_tokens=32_768,    # capacity planning is a function of these
)
print(result.data)
```

The model is a `"provider/model-name"` string (`groq/…` today; `openai/…`,
`anthropic/…` as providers are added). Unlike single-call libraries, FormatShield
**plans** how to split a wide schema across calls, so it needs the model's real
`context_window` (C_eff) and `max_output_tokens` (M_O); omit them to fall back to
a conservative default. Pydantic models and dataclasses work as schemas too. For
many documents, reuse a `FormatShield` (sync) or `AsyncFormatShield` (async) engine.

## How it works

A seven-stage pipeline (S0–S6): resource calibration → schema analysis → structural
grouping → document pre-pass (BM25) → capacity packing → excerpt finalisation →
SFEP extraction → per-field validation & surgical retry → JSON assembly. Only three
stages call the model. See [docs/concepts/pipeline.md](docs/concepts/pipeline.md).

## Command line

```bash
formatshield inspect schema.json
formatshield extract doc.txt --schema schema.json --model groq/llama-3.1-8b-instant
```

## Supported providers

Groq (MVP), via `from_model("groq/<model>")`. The provider layer is a small Protocol;
adding one is a single registry entry — see [CONTRIBUTING.md](CONTRIBUTING.md).

## Contributing

Contributions welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add providers
and pipeline stages, and the development workflow (`uv`, `ruff`, `mypy --strict`,
`pytest`).

## License

Apache-2.0. See [LICENSE](LICENSE).
