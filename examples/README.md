# Examples

Runnable examples for nfield, in two forms: notebooks you can open and step through, and short
scripts you can copy into your own code. Every example calls a real model, so set your API key
first:

```bash
pip install "nfield[groq,cli]"
export GROQ_API_KEY="gsk_..."
```

The model is a `provider/model` string. The examples use `groq/llama-3.3-70b-versatile`; swap it
for any model you have access to.

## Notebooks

A short learning path. Start at the top if you are new. Each notebook ships with its outputs, so
you can read them without running anything.

| Notebook | What it covers |
|----------|----------------|
| [01_schema_injection.ipynb](notebooks/01_schema_injection.ipynb) | Feed a schema as a JSON Schema dict, a Pydantic model, or a dataclass. |
| [02_wide_schema.ipynb](notebooks/02_wide_schema.ipynb) | Estimate the work, then extract a schema with many fields. |
| [03_grounding_provenance.ipynb](notebooks/03_grounding_provenance.ipynb) | Score values against the source and get character spans. |
| [04_agent_tool.ipynb](notebooks/04_agent_tool.ipynb) | Wrap nfield as a tool an agent can call. |

To run them yourself:

```bash
pip install -r examples/requirements.txt
jupyter lab examples/notebooks
```

## Scripts

Single-file examples you can run directly.

| Script | What it shows |
|--------|---------------|
| [scripts/quickstart.py](scripts/quickstart.py) | The shortest path: one document, a dict schema. |
| [scripts/schema_dict_example.py](scripts/schema_dict_example.py) | A dict schema tuned with an `ExtractionConfig`. |
| [scripts/invoice_extraction.py](scripts/invoice_extraction.py) | Reuse one engine across documents with a Pydantic schema. |
| [scripts/wide_schema.py](scripts/wide_schema.py) | A wide schema, with the completeness metadata. |
| [scripts/grounding_provenance.py](scripts/grounding_provenance.py) | Grounding scores and character-level provenance. |
| [scripts/agent_tool.py](scripts/agent_tool.py) | nfield used as a tool an agent can call. |
| [scripts/async_batch.py](scripts/async_batch.py) | Extract many documents concurrently with `AsyncNField`. |
| [scripts/response_cache.py](scripts/response_cache.py) | Cache responses so a repeated extraction is free. |

```bash
python examples/scripts/quickstart.py
```
