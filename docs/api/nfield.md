# API reference

The public surface of the library. Import everything from the top level:

```python
from nfield import nfield, NField, AsyncNField, ExtractionConfig, ExtractionResult
```

For the settings behind these calls, see [Configuration](../configuration.md); for what can go
wrong, see [Errors](../errors.md).

## `nfield`

```python
nfield(
    document, schema, model=None, *,
    config=None,
    context_window=None, max_output_tokens=None,
    api_key=None, base_url=None,
    instructions="",
) -> ExtractionResult
```

Run one document through the pipeline and return the result. `schema` can be a JSON Schema dict,
a Pydantic model, or a dataclass (class or instance). When `model` is omitted it is read from
`$NFIELD_MODEL`, then `config.default_model`.

- `model` is a `"provider/model-name"` string, e.g. `groq/llama-3.1-8b-instant` or `openai/...`.
- `context_window` and `max_output_tokens` are the model's real limits. Pass them for wide
  schemas so planning uses the true window rather than a safe default.
- `api_key` overrides the provider's environment variable; `base_url` points at any
  OpenAI-compatible endpoint.
- `instructions` is your own steering (domain, hints). It is prepended to the built-in SFEP
  prompt, never replacing it, and counts toward each call's budget.

## `nfield_async`

```python
await nfield_async(document, schema, model=None, *,
    config=None, context_window=None, max_output_tokens=None,
    api_key=None, base_url=None, instructions="") -> ExtractionResult
```

The `async` twin of `nfield`, with the same arguments.

## `NField` and `AsyncNField`

```python
NField(model=None, schema=None, *,
    config=None, context_window=None, max_output_tokens=None,
    api_key=None, base_url=None, instructions="")

AsyncNField(...)   # identical arguments
```

Reusable engines. Build one, then extract many documents so the schema is parsed and the model
calibrated only once. Call the instance directly or use `.extract(document, schema=None)`; a
schema passed to `extract` overrides the one given at construction. `NField` runs the async core
on your behalf and is Jupyter-safe (it detects a running event loop). `AsyncNField` is also an
async context manager:

```python
async with AsyncNField("groq/llama-3.1-8b-instant", schema) as engine:
    result = await engine(document)
```

## `extract_batch`

```python
engine.extract_batch(documents, schema=None, *, max_concurrent=None, return_exceptions=False)
```

Extract many documents through one reused engine (`NField` and `AsyncNField` both have it).
Documents run concurrently, bounded by `max_concurrent` (default 4) so a large batch stays under
provider rate limits. You get one result per document, in input order. A provider failure comes
back as a `FAILED`-status result, not an exception; with `return_exceptions=True`, an error that
escapes `extract()` is kept in that document's slot instead.

## `from_model`

```python
from_model(model_string, *,
    context_window=None, max_output_tokens=None,
    api_key=None, base_url=None, reasoning_model=False, max_retries=None,
    cache=None) -> LLMProvider
```

The provider factory, routed by the prefix before `/`. The engines call it for you; you rarely
need it directly. `cache` attaches a `ResponseCache` to the provider (see
[Response caching](../configuration.md#response-caching)).

## `ExtractionResult`

| Attribute | Type | Meaning |
|-----------|------|---------|
| `data` | `dict` | Extracted fields as nested JSON, matching the schema shape. |
| `status` | `ExtractionStatus` | `SUCCESS`, `PARTIAL`, or `FAILED`. |
| `metadata` | `Metadata` | Run statistics (below). |
| `fields` | `tuple[FieldResult, ...]` | Optional flat per-field detail; empty unless populated. |
| `provenance` | `dict[str, list[int]] \| None` | Per-value `[start, end)` char offsets, when `provenance=True`; `None` otherwise. |

`result.to_dict()` and `ExtractionResult.from_dict(...)` convert to and from the plain JSON form
(what `save_results` / `load_results` write and read).

### `Metadata`

Everything on `result.metadata`:

- **Counts** - `fields_extracted`, `fields_total`, `fields_missing`, `fields_conflicted`,
  `fields_needs_revalidation`, `fields_call_failed`.
- **Calls** - `K`, `K_min`, `optimality_gap`, `calls_by_origin`.
- **Spend** - `tokens_prompt`, `tokens_completion` (always tracked, from the provider's
  own counts), and `cost` in USD when `ExtractionConfig.pricing` is set.
- **Quality** - `quality_score`, `confidence_level`, `per_field_confidence`, `retry_rounds`.
- **`error`** - a representative provider error when a call failed, else `None`.
- **Grounding** (when `ground_values` is on) - `fields_grounded`, `fields_ungrounded`,
  `hallucination_rate`.
- **Closed-book** (when `closed_book` is on) - `answer_rate`, `abstain_rate`.

## `ExtractionConfig`

Per-call settings, passed as `config=...`. Every field is keyword-only with a default, so
`ExtractionConfig()` is a valid starting point:

```python
from nfield import ExtractionConfig

config = ExtractionConfig(ground_values=True, provenance=True, max_concurrent_calls=8)
```

The full list of settings, with defaults and what each does, lives in
[Configuration](../configuration.md).

## Filesystem helpers (`nfield.io`)

```python
load_document(path) -> str                     # read a UTF-8 text document
load_schema(path) -> dict                       # parse a JSON Schema file (SchemaError on bad JSON)
save_results(results, path) -> None             # write JSON Lines, one result per line
load_results(path) -> list[ExtractionResult]    # read them back
```

Text and JSON only; converting PDF or DOCX to text is the caller's job.

## Tabular export (`nfield.export`, optional `pandas`)

```python
results_to_dataframe(results, *, include_metadata=False) -> pandas.DataFrame
result_to_dataframe(result, *, include_metadata=False) -> pandas.DataFrame
results_to_csv(results, path, *, include_metadata=False) -> None
```

One row per result; columns are the flat dot-notation field paths. Install with
`pip install "nfield[export]"`. pandas is imported only when you call these.

## Grounding viewer (`nfield.viz`)

```python
save_html(result, document, path=None) -> str
```

Renders a result's provenance spans over the source document as one self-contained
HTML page: each located value highlighted in place with its field path on hover, plus
a table of exact spans. Needs a result produced with `provenance=True`; raises
`ValueError` otherwise. Stdlib only - no extra install.

## Exceptions

All inherit from `NFieldError`, so a single `except NFieldError` catches the library:
`SchemaError`, `ProviderError`, `ExtractionError`, `ValidationError`, `AssemblyError`. See
[Errors](../errors.md) for when each one fires and how to fix it.
