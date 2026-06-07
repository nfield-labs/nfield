# API reference

## `nfield`

```python
nfield(
    document, schema, model=None,
    *, context_window=None, max_output_tokens=None,
    system_prompt="", user_prompt="", config=None,
) -> ExtractionResult
```

Synchronous one-shot extraction. `schema` may be a JSON Schema dict, a Pydantic
model (class or instance), or a dataclass (class or instance). If `model` is
omitted, it is resolved from `FORMATSHIELD_MODEL`, then `config.default_model`.

`system_prompt` and `user_prompt` add caller context (domain, task instructions).
They are **prepended** to FormatShield's built-in SFEP prompts — never replacing
them — and are counted in each leaf's overhead, so a large prompt correctly
shrinks the per-call document budget. The same kwargs exist on `FormatShield`,
`AsyncFormatShield`, `nfield_async`, and as `--system-prompt` / `--user-prompt`
on the CLI.

`model` is a `"provider/model-name"` string (`groq/…` today; `openai/…`,
`anthropic/…` as providers are added). `context_window` (C_eff) and
`max_output_tokens` (M_O) are the model's real limits — supply them so capacity
planning uses the true window; otherwise the provider's conservative default
applies.

## `nfield_async`

```python
nfield_async(document, schema, model=None, *, config=None) -> Awaitable[ExtractionResult]
```

Async variant of `nfield`.

## `FormatShield`

```python
FormatShield(model=None, schema=None, *, config=None)
```

Reusable synchronous engine. Call the instance or `.extract(document, schema=None)`.
A schema given at construction is normalised once and reused; a schema passed to
`extract` overrides it for that call. Jupyter-safe (detects a running event loop).

## `AsyncFormatShield`

```python
AsyncFormatShield(model=None, schema=None, *, config=None)
```

Async engine and async context manager. `await engine.extract(document)` or
`await engine(document)`.

## `from_model`

```python
from_model("groq/llama-3.1-8b-instant") -> LLMProvider
```

Provider factory routed by the prefix before `/`.

## `ExtractionResult`

| Attribute | Type | Meaning |
|-----------|------|---------|
| `data` | `dict` | Extracted fields as nested JSON. |
| `status` | `ExtractionStatus` | `SUCCESS` / `PARTIAL` / `FAILED`. |
| `metadata` | `Metadata` | `K`, `K_min`, `optimality_gap`, `quality_score`, `confidence_level`, field counts, `per_field_confidence`, `retry_rounds`. |
| `fields` | `tuple[FieldResult, ...]` | Optional per-field detail. |

## `ExtractionConfig`

Key options: `default_model`, `context_utilization_ratio` (0.50),
`max_retry_rounds` (2), `z_target` (1.645), `confidence_thresholds`,
`document_language`, `evidence_score_threshold`,
`inject_dependencies` (**True** — inject resolved upstream dependency values into
a dependent leaf's prompt, counted in capacity planning; no-op without cross-leaf
dependencies; set `False` for ordering-only handling),
`cascade_dependency_invalidation` (False, requires `inject_dependencies` — flag
dependents `NEEDS_REVALIDATION` when a retry changes an upstream value),
`knowledge_fallback` (False — when True, a field the document does not state may
be filled from the model's own well-established knowledge instead of left `NULL`;
use only for documents about well-known subject matter, as it can produce
confident-but-unsourced values on private documents).

Retrieval note: BMX lexical indexing folds diacritics (Unicode NFKD; Lucene-style ASCII
folding), so an accented document spelling (`Denísov`, `café`) matches an
unaccented query term (`Denisov`, `cafe`) and vice-versa.

Missing-field recovery (architecture engine §5.3) always runs as a core Stage 5
step — fields never produced after surgical retry get one bounded recovery pass
(tree-backtrack absent-ancestor children, then re-extract the missed-only set).
There is no flag; it is a no-op when nothing is missing.

## Exceptions

All inherit from `FormatShieldError`: `SchemaError`, `ProviderError`,
`ExtractionError`, `ValidationError`, `AssemblyError`.
