# Configuration

Three things control a run: your **API key**, the model's **limits**, and an
**`ExtractionConfig`**. The defaults are sane, so most of this you can ignore until you need it.
Everything here is also a flag on `nfield extract`.

## Setting your API key

nfield reads the provider's standard environment variable, so the common path needs no code:

```bash
export GROQ_API_KEY="gsk_..."          # Groq
export OPENAI_API_KEY="sk-..."         # OpenAI or any OpenAI-compatible endpoint
export GEMINI_API_KEY="..."            # Google Gemini
export ANTHROPIC_API_KEY="sk-ant-..."  # Anthropic (Claude)
export OPENROUTER_API_KEY="sk-or-..."  # OpenRouter (one key, many model families)
```

Native providers:

| Provider | Environment variable | Model prefix |
|----------|----------------------|--------------|
| Groq | `GROQ_API_KEY` | `groq/...` |
| OpenAI (and compatible) | `OPENAI_API_KEY` | `openai/...` |
| Google (Gemini) | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `google/...` |
| Anthropic (Claude) | `ANTHROPIC_API_KEY` | `anthropic/...` |

The variable name is fixed by the provider: the SDK looks for `GROQ_API_KEY` or `OPENAI_API_KEY`
specifically, so a name like `api_key` or `MY_KEY` is not read. You can set several at once, one
per provider; the model prefix picks which one is used (`groq/...` reads `GROQ_API_KEY`,
`openai/...` reads `OPENAI_API_KEY`), so they never clash.

### OpenAI-compatible presets

Several hosted endpoints speak the OpenAI API, so nfield reaches them with a preset prefix: the
same OpenAI path with the base URL and key variable already set. One key per provider, and the
model name is whatever the endpoint lists.

| Prefix | Environment variable |
|--------|----------------------|
| `openrouter/...` | `OPENROUTER_API_KEY` |
| `deepseek/...` | `DEEPSEEK_API_KEY` |
| `together/...` | `TOGETHER_API_KEY` |
| `fireworks/...` | `FIREWORKS_API_KEY` |
| `mistral/...` | `MISTRAL_API_KEY` |
| `xai/...` | `XAI_API_KEY` |
| `perplexity/...` | `PERPLEXITY_API_KEY` |
| `cerebras/...` | `CEREBRAS_API_KEY` |
| `ollama/...` | none (local server) |

`openrouter/` is the widest: one key reaches Anthropic, Google, Mistral, Meta, DeepSeek and more,
e.g. `openrouter/anthropic/claude-sonnet-4`. Only the first `/` selects the provider, so the rest
stays the model name. For any other OpenAI-compatible endpoint, use `openai/...` with `base_url`.

```python
from nfield import nfield

# one OPENROUTER_API_KEY, three different vendors
nfield(document, schema, "openrouter/anthropic/claude-3-haiku")
nfield(document, schema, "openrouter/deepseek/deepseek-chat")
nfield(document, schema, "openrouter/mistralai/mistral-small-3.2-24b-instruct")
```

Use the exact model slug OpenRouter lists (its `/models` endpoint); an unknown name returns a 404.

!!! note "Reasoning models need one flag"
    If you point at a reasoning model that thinks before answering (DeepSeek-R1 /
    `deepseek-reasoner`, Qwen3, QwQ, or a Gemini/Claude thinking model), set
    `ExtractionConfig(reasoning_model=True)` so its thinking does not eat the output budget.
    nfield leaves this **`False`** by default, so ordinary models (Llama, GPT-4o,
    `deepseek-chat`, Gemini Flash) need nothing. It is one setting you flip only for a
    reasoning model, and it is safe if the model does not support the switch: nfield falls
    back to stripping the thinking from the reply.

### From a `.env` file

nfield reads `os.environ`; it does not parse `.env` itself. Load the file into the environment
first, either with your shell or with `python-dotenv`:

```python
from dotenv import load_dotenv
load_dotenv()                        # reads .env into os.environ

from nfield import nfield
result = nfield(document, schema, "groq/llama-3.1-8b-instant")
```

Keep `.env` out of version control (add it to `.gitignore`).

### Passing the key explicitly

For a secret vault or a multi-tenant service, pass the key directly. It is never logged, and it
takes precedence over the environment variable:

```python
from nfield import NField

engine = NField("groq/llama-3.1-8b-instant", schema, api_key="gsk_...")
result = engine.extract(document)
```

### Custom endpoints

Any OpenAI-compatible server (Together, Fireworks, DeepSeek, vLLM, Ollama, LM Studio) works
through `base_url`:

```python
engine = NField(
    "openai/llama-3.1-8b",
    schema,
    base_url="http://localhost:11434/v1",   # Ollama, for example
    api_key="ollama",                        # some servers accept any non-empty key
)
```

## Model limits

For large schemas on large models, pass the model's real limits so nfield plans across the full
window instead of a conservative default. These are per-call arguments, not part of
`ExtractionConfig`:

| Argument | Meaning | Default |
|----------|---------|---------|
| `context_window` | The model's real context window in tokens (C_eff). | provider default |
| `max_output_tokens` | The model's real output ceiling in tokens (M_O). | provider default |

```python
result = nfield(
    document, schema, "groq/llama-3.3-70b-versatile",
    context_window=131_072,
    max_output_tokens=32_768,
)
```

## ExtractionConfig

Pass an `ExtractionConfig` as `config=...`. Every field is keyword-only and has a default, so
`ExtractionConfig()` with no arguments is a fine starting point; change only what you need.

```python
from nfield import nfield, ExtractionConfig

config = ExtractionConfig(ground_values=True, provenance=True, max_concurrent_calls=8)
result = nfield(document, schema, "groq/llama-3.1-8b-instant", config=config)
```

### Grounding and provenance

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `ground_values` | bool | `False` | Label each groundable value by how well the excerpt supports it, and report `hallucination_rate`. Non-destructive: a weak value is reported, never dropped. |
| `grounding_min_score` | float | `0.5` | Minimum grounding score in `[0, 1]` for a value to count as supported. |
| `provenance` | bool | `False` | Attach `result.provenance`: each value's `[start, end)` char offsets in the source (verbatim values only). |

### Reliability and recovery

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `max_retry_rounds` | int | `2` | Extraction retry rounds for low-confidence or missing fields. |
| `max_api_retries` | int | `10` | Per-call retry budget for transient failures (429 / 5xx / timeout), honoring Retry-After. |
| `max_concurrent_calls` | int | `4` | Leaf extraction calls in flight at once. Raise on higher-throughput plans. |
| `recover_conflicts` | bool | `True` | Re-extract conflicting fields during the recovery pass. |
| `recover_call_failed` | bool | `True` | Retry transiently-failed fields during the recovery pass. |
| `validate_schema` | bool | `True` | Reject a provably-unsatisfiable schema (e.g. `minimum > maximum`) before any API call. |
| `fallback_model` | str \| None | `None` | Stronger model to escalate still-failing fields to once, after recovery. |
| `strict_validation` | bool | `False` | Validate values exactly as extracted (no lenient coercion of formatted numbers/booleans). |

### Reasoning and thinking

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `reasoning_model` | bool | `False` | Treat the model as a reasoning model and disable its thinking per call, so it does not consume the output budget. |
| `think_phase_budget` | (int, int) | `(100, 150)` | `(min, max)` token budget for the thinking phase. |

### Closed-book and knowledge

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `knowledge_fallback` | bool | `False` | Fill fields the document does not state from the model's own knowledge instead of leaving them `None`. |
| `closed_book` | bool | `False` | Fill the schema from the model's knowledge with no document at all (pass an empty document). |
| `self_consistency` | bool | `False` | Sample each closed-book leaf twice and keep a value only if both agree. Doubles calls; no-op unless `closed_book` is set. |

### Retrieval and packing (advanced)

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `context_utilization_ratio` | float | `0.50` | Fraction of the context window used for document chunks, in `(0, 1]`. |
| `z_target` | float | `1.645` | Z-score target for output reservation (95th percentile). |
| `evidence_score_threshold` | float | `0.3` | Minimum evidence score for a chunk to enter extraction context. |
| `max_fields_per_call` | int | `50` | Per-leaf reliability budget in difficulty-weighted units; forces many small, reliable calls. |
| `use_advanced_sfr` | bool | `False` | Enable advanced Semantic Field Routing for large schemas. |
| `chars_per_token` | float \| None | `None` | Pin the characters-per-token ratio; `None` uses a script-aware estimate from `document_language`. |
| `document_language` | str | `"en"` | BCP-47 language tag of the document; sizes the token budget. |
| `confidence_thresholds` | dict | `{"HIGH": 0.9, "MEDIUM": 0.7}` | Tier label to minimum confidence score. |
| `default_model` | str \| None | `None` | Model used when none is passed per call (after the explicit model and `$NFIELD_MODEL`). |

### Dependencies (schemas with cross-field references)

| Setting | Type | Default | What it does |
|---------|------|---------|--------------|
| `inject_dependencies` | bool | `True` | Feed a dependent field's prompt the values its upstream dependencies produced. No-op without cross-leaf dependencies. |
| `cascade_dependency_invalidation` | bool | `False` | When an upstream value changes on retry, flag its dependents `NEEDS_REVALIDATION`. Requires `inject_dependencies`. |

## Response caching

Extracting a document costs a set of model calls. Run the same document again and, by
default, you pay for all of them again. Turn on the cache and nfield remembers each
response, so a repeat of the same extraction returns the saved answer for free. It is off
until you ask for it.

| `cache` value | What you get |
|---------------|--------------|
| `False` (default) | No cache. Every call goes to the model. |
| `True` | A cache in memory, living as long as the engine does. |
| a `ResponseCache` | Whatever store you hand it, e.g. `DiskCache("path")` to keep entries between runs, or your own Redis backend. Pass one instance to share it across engines. |

Caching is **exact-match**: the key is the request itself (model, messages, output ceiling),
so the smallest change to any of them is a new key. A hit is therefore the same text the
model would have given you, never a near-miss from a similar prompt the way a semantic cache
would. That is what makes it safe to leave on while you iterate on the same document and
schema, or re-run a benchmark.

```python
from nfield import NField, ExtractionConfig, DiskCache

# Kept in memory, gone when the process exits.
nf = NField("groq/llama-3.3-70b-versatile", schema, config=ExtractionConfig(cache=True))

# Kept on disk, so the next run reads it back. Re-running the same extraction is free.
nf = NField(
    "groq/llama-3.3-70b-versatile",
    schema,
    config=ExtractionConfig(cache=DiskCache(".nfield_cache")),
)
```

Call `.clear()` on either cache to empty it. The key carries a format version, so a
version of nfield never reads stale entries written by an older one. Keep one `DiskCache`
directory per model setup: if two runs differ in something the request does not spell out,
such as reasoning-model handling, give them separate directories.

On the command line, hand `--cache-dir` a directory to cache on disk:

```bash
nfield extract invoice.txt --schema invoice.json -m groq/llama-3.3-70b-versatile \
  --cache-dir .nfield_cache
```

Any object with a `get` and a `set` is a `ResponseCache`, so a custom backend is two methods:

```python
class RedisCache:
    def get(self, key: str) -> str | None: ...
    def set(self, key: str, value: str) -> None: ...
```

Pass an instance as `cache=...` and nfield uses it.

## The result object

Every call returns an `ExtractionResult`.

| Attribute | Type | What it holds |
|-----------|------|---------------|
| `data` | dict | The extracted fields as nested JSON, matching the schema shape. |
| `metadata` | `Metadata` | Run statistics (see below). |
| `status` | `ExtractionStatus` | `SUCCESS`, `PARTIAL`, or `FAILED`. |
| `fields` | tuple | Optional flat per-field detail (`FieldResult`), empty unless populated. |
| `provenance` | dict \| None | Per-value `[start, end)` char offsets, when `provenance=True`. |

Useful `metadata` fields:

| Field | Meaning |
|-------|---------|
| `fields_extracted` / `fields_total` | How many of the schema's fields were filled. |
| `fields_missing` | Fields genuinely absent from the document. |
| `fields_call_failed` | Fields left unextracted by an API/call failure (distinct from absent). |
| `quality_score` | Aggregate run quality in `[0, 1]`. |
| `K` / `K_min` | Model calls used vs the computed minimum. |
| `hallucination_rate` | Fraction of grounded-checked values the source did not support (`None` if grounding off). |
| `per_field_confidence` | Map of field path to confidence in `[0, 1]`. |

`result.to_dict()` serializes the whole object (round-trips with `ExtractionResult.from_dict`
and the `save_results` / `load_results` helpers).

## See also

- [CLI reference](cli.md) - every setting above is also a flag on `nfield extract`.
- [Examples](examples.md) - verified recipes using these settings.
