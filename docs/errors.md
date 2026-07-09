# Errors and troubleshooting

When an extraction goes wrong, here is what you will see and how to fix it. The messages below
are the actual ones the library prints.

## The exception hierarchy

Every exception nfield raises inherits from `NFieldError`, so one `except` catches the whole
library:

```
NFieldError
├── SchemaError       - invalid, unsupported, or unsatisfiable schema (or no model)
├── ProviderError     - an LLM provider request failed
├── ExtractionError   - the extraction pipeline failed for a field or overall
├── ValidationError   - a field value failed post-extraction validation
└── AssemblyError     - final assembly / serialization failed
```

```python
from nfield import nfield, NFieldError

try:
    result = nfield(document, schema, "groq/llama-3.1-8b-instant")
except NFieldError as exc:
    print(f"extraction failed: {exc}")
```

Two input errors are raised as the built-in `TypeError` / `ValueError` (not `NFieldError`),
because they are ordinary bad arguments caught at the boundary (see the table below).

## Provider and call failures do not raise

One thing is worth knowing up front. A **provider or call failure is reported, not raised**: a
bad model name, a missing or invalid API key, or an unreachable endpoint gives you back a result
with `status = FAILED`, `data = {}`, `metadata.fields_call_failed > 0`, and the reason on
`metadata.error`. You do not get an exception.

Why not? So `extract_batch` stays predictable (one result per document, in order) and any partial
data survives when only some calls fail. The habit to build: **check `status` before you trust
`data`.**

```python
from nfield import ExtractionStatus

result = nfield(document, schema, "groq/llama-3.1-8b-instant")

if result.status is not ExtractionStatus.SUCCESS and result.metadata.fields_call_failed:
    print("call failed:", result.metadata.error)
    # provider error: Failed to initialize Groq client: The api_key client option must be set ...
```

| Field | Tells you |
|-------|-----------|
| `result.status` | `SUCCESS`, `PARTIAL`, or `FAILED`. |
| `result.metadata.fields_call_failed` | How many fields failed on the model call (vs genuinely absent). |
| `result.metadata.error` | A representative provider error, or `None` when every call succeeded. |

From the CLI, the same situation exits non-zero with the cause on stderr:

```console
$ nfield extract invoice.txt -s schema.json -m groq/wrong-model-name
Warning: 5 field(s) across 1 document(s) were left unextracted by API/call failures;
the result is incomplete. Cause: provider error: Groq API call failed: Error code: 404 ...
$ echo $?
1
```

## Scenario reference

| Scenario | What you get | Message (abridged) | Fix |
|----------|--------------|--------------------|-----|
| No model, and `$NFIELD_MODEL` unset | `SchemaError` | `No model specified for extraction. hint: Pass model=..., set NFIELD_MODEL, or ExtractionConfig(default_model=...)` | Pass a model string, set the env var, or set `default_model`. |
| Unknown provider prefix (`foo/bar`) | `ProviderError` | `Unknown provider: 'foo'. Registered providers: anthropic, cerebras, deepseek, fireworks, google, groq, mistral, ollama, openai, openrouter, perplexity, together, xai.` | Use a registered prefix (native `groq/`, `openai/`, `google/`, `anthropic/`, or a preset like `openrouter/`, `deepseek/`). |
| **Missing / invalid API key** | `FAILED` result | `metadata.error`: `... The api_key client option must be set ... Set GROQ_API_KEY ...` | `export GROQ_API_KEY=...`, or pass `api_key=...`. |
| **Wrong model name** (valid key) | `FAILED` result | `metadata.error`: `Groq API call failed: Error code: 404 - model ... does not exist` | Fix the model name (`provider/model`). |
| Endpoint unreachable / network down | `FAILED` result | `metadata.error`: connection / timeout error | Check `base_url` and connectivity. |
| Document is not text (e.g. `int`, `bytes`) | `TypeError` | `document must be text (str), got int. Read a file first with load_document('path') ...` | Pass a `str`; convert PDF/DOCX to text first. |
| Empty document, not closed-book | `ValueError` | `no document to extract from; pass a document, or set closed_book=True ...` | Provide a document, or set `closed_book=True`. |
| A document passed in closed-book mode | `ValueError` | `closed_book=True fills the schema from the model's knowledge and ignores the document ...` | Pass an empty document, or set `closed_book=False`. |
| Schema is not a dict / model / dataclass | `SchemaError` | `Unsupported schema type: int. hint: Pass a JSON Schema dict, a Pydantic model, or a dataclass.` | Pass a supported schema type. |
| Unsatisfiable schema (`minimum > maximum`, empty `enum`, ...) | `SchemaError` | `minimum (10) must be <= maximum (1) [field=n] hint: set minimum <= maximum` | Fix the contradiction, or set `validate_schema=False` to skip the check. |
| `chars_per_token` set to `<= 0` | `ValueError` | `chars_per_token must be > 0, got -1.0` | Use a positive ratio, or leave it `None`. |
| CSV output without pandas (CLI) | clean CLI error | `CSV output needs the export extra: pip install 'nfield[cli,export]'` | Install the `export` extra. |
| Bad file path / non-UTF-8 file (CLI) | clean CLI error | `Document file not found: ...` / `... is not valid UTF-8 text` | Fix the path, or convert the file to UTF-8 text. |

Errors above the double line raise **before any API call**, so they cost nothing and fail fast.

## Error object fields

Some `NFieldError` subclasses carry structured detail beyond the message:

| Exception | Extra attributes |
|-----------|------------------|
| `SchemaError` | `field`, `hint` |
| `ProviderError` | `status_code`, `retry_after`, `retryable` |
| `ExtractionError` | `field`, `attempt` |
| `ValidationError` | `field`, `value`, `hint` |
| `AssemblyError` | `path` |

```python
from nfield import ProviderError

try:
    ...
except ProviderError as exc:
    if exc.retryable:      # 408 / 429 / 5xx and timeouts
        ...
```

## Seeing the details: logging

nfield uses the standard library `logging` and follows the library convention: it emits to a
per-module logger (`logging.getLogger("nfield...")`) and never configures logging itself, so
your application stays in control. Transient retries, emergency splits, and per-leaf call
failures are logged at `WARNING` / `INFO`. To see them, configure logging in your app:

```python
import logging

logging.basicConfig(level=logging.INFO)     # or DEBUG for the fine detail
```

By default (no configuration) these messages are silent, and nfield never prints to stdout.

## See also

- [Configuration](configuration.md) - the settings, and the result object.
- [CLI reference](cli.md#exit-codes) - CLI exit codes.
