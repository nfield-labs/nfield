# FormatShield

**N-field structured extraction from documents with LLMs.**

Extract hundreds of structured fields from any document — without the *format tax*.

## The problem

Asking an LLM to emit one large, deeply-nested JSON object degrades sharply as the
field count grows: brackets, commas, and quotes consume attention that should go to
the *values*, and accuracy collapses on schemas with hundreds of fields. Existing
structured-output tools effectively cap out well before that.

## The approach

FormatShield decomposes a large schema into capacity-bounded field groups, retrieves
only the relevant document spans per group, and extracts with a flat
`path = value` intermediate format (SFEP) instead of nested JSON. Each field is
validated against its schema constraint, failures are retried surgically, and the
flat pairs are reassembled into valid nested JSON.

```python
from formatshield import nfield

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
```

## Install

```bash
pip install formatshield
pip install "formatshield[groq]"   # Groq provider
pip install "formatshield[cli]"    # command-line interface
```

See [Quickstart](quickstart.md) to run your first extraction, or
[Pipeline](concepts/pipeline.md) for how the seven stages fit together.
