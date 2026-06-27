# nfield

**Pull hundreds of structured fields out of a document, reliably.**

Ask an LLM to fill one big JSON schema in a single call and the answers get worse as the
schema grows. The model spends its output budget on brackets, commas, and quotes instead of
the values, and a wide schema can overflow the context window before it finishes.

nfield is built for the wide case. It splits the schema into groups that fit the model, finds
the part of the document each group needs, extracts plain `key = value` lines instead of
nested JSON, validates every field against the text, retries the ones that fail, and
reassembles the clean nested JSON you asked for.

```python
from nfield import nfield

result = nfield(document, schema, "groq/llama-3.1-8b-instant")
print(result.data)
```

## Install

```bash
pip install nfield
pip install "nfield[groq]"      # Groq provider
pip install "nfield[openai]"    # OpenAI / any OpenAI-compatible endpoint
pip install "nfield[cli]"       # command-line interface
```

## Next

- [Quickstart](quickstart.md) - run your first extraction.
- [The pipeline](concepts/pipeline.md) - how the seven stages fit together.
- [SFEP](concepts/sfep.md) - why `key = value` beats nested JSON.
- [API reference](api/nfield.md) - every public name.
