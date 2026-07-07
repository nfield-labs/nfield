# API reference

## `nfield`

```python
nfield(
    document, schema, model=None,
    *, context_window=None, max_output_tokens=None,
    instructions="", config=None,
) -> ExtractionResult
```

Synchronous one-shot extraction. `schema` may be a JSON Schema dict, a Pydantic
model (class or instance), or a dataclass (class or instance). If `model` is
omitted, it is resolved from `NFIELD_MODEL`, then `config.default_model`.

`instructions` adds caller steering (domain, task hints). It is **prepended** to
NField's built-in SFEP prompt - never replacing it - and is counted in each
leaf's overhead, so a large value correctly shrinks the per-call document budget.
The same kwarg exists on `NField`, `AsyncNField`, `nfield_async`, and
as `--instructions` on the CLI.

`model` is a `"provider/model-name"` string (`groq/…` today; `openai/…`,
`anthropic/…` as providers are added). `context_window` (C_eff) and
`max_output_tokens` (M_O) are the model's real limits - supply them so capacity
planning uses the true window; otherwise the provider's conservative default
applies.

## `nfield_async`

```python
nfield_async(document, schema, model=None, *, config=None) -> Awaitable[ExtractionResult]
```

Async variant of `nfield`.

## `NField`

```python
NField(model=None, schema=None, *, config=None)
```

Reusable synchronous engine. Call the instance or `.extract(document, schema=None)`.
A schema given at construction is normalised once and reused; a schema passed to
`extract` overrides it for that call. Jupyter-safe (detects a running event loop).

## `AsyncNField`

```python
AsyncNField(model=None, schema=None, *, config=None)
```

Async engine and async context manager. `await engine.extract(document)` or
`await engine(document)`.

## `extract_batch`

```python
engine.extract_batch(documents, schema=None, *, max_concurrent=None, return_exceptions=False)
```

Extract many documents through one reused, calibrated engine (`NField` and
`AsyncNField` both expose it). Documents run concurrently, bounded by a semaphore
(`max_concurrent`, default 4) so a large batch stays under provider rate limits. Returns
one result per document, in input order. A provider failure surfaces as a `FAILED`-status
result; with `return_exceptions=True`, an error that escapes `extract()` is kept in that
document's slot instead of being re-raised.

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
| `metadata` | `Metadata` | `K`, `K_min`, `optimality_gap`, `quality_score`, `confidence_level`, field counts, `per_field_confidence`, `retry_rounds`, and (when `ground_values` is on) `fields_grounded`, `fields_ungrounded`, `hallucination_rate`. |
| `fields` | `tuple[FieldResult, ...]` | Optional per-field detail. |
| `provenance` | `dict[str, list[int]] \| None` | Present only when `provenance` is on: each value's dot-path mapped to its `[start, end)` char offsets in the source. `None` otherwise. |

## `ExtractionConfig`

Key options: `default_model`, `context_utilization_ratio` (0.50),
`max_retry_rounds` (2), `z_target` (1.645), `confidence_thresholds`,
`document_language`, `evidence_score_threshold`,
`reasoning_model` (False - set `True` for a reasoning/thinking model, e.g. Qwen3,
DeepSeek-R1, QwQ; its thinking is disabled per call so it does not consume the
answer's output budget. See below),
`inject_dependencies` (**True** - inject resolved upstream dependency values into
a dependent leaf's prompt, counted in capacity planning; no-op without cross-leaf
dependencies; set `False` for ordering-only handling),
`cascade_dependency_invalidation` (False, requires `inject_dependencies` - flag
dependents `NEEDS_REVALIDATION` when a retry changes an upstream value),
`knowledge_fallback` (False - when True, a field the document does not state may
be filled from the model's own well-established knowledge instead of left `NULL`;
use only for documents about well-known subject matter, as it can produce
confident-but-unsourced values on private documents).

Opt-in flags (all default off / behaviour-preserving unless noted):
`ground_values` (False - label each value's source support; see below),
`grounding_min_score` (0.5 - support threshold used by the metric; only read when
`ground_values` is on),
`provenance` (False - attach source char offsets; see below),
`strict_validation` (False - store values exactly as extracted, skipping the
lenient normalisation that accepts `"$1,234"` as `1234`),
`closed_book` (False - fill the schema from the model's own knowledge with **no**
document, answering `NULL` when unsure; reports `answer_rate` / `abstain_rate`;
forces grounding and provenance off),
`self_consistency` (False - sample each closed-book leaf twice and keep a value
only if both agree; doubles calls; no-op unless `closed_book`),
`fallback_model` (None - a stronger model to re-try still-failing fields once
after recovery; `None` keeps the run single-model),
`use_advanced_sfr` (False - enable the targeted re-retrieval path for failed
fields),
`recover_conflicts` (True - re-extract conflicting / revalidation-flagged fields
in the recovery pass instead of reporting them unresolved),
`recover_call_failed` (True - give fields whose Stage 4 call hit a transient
429/timeout one more try in recovery),
`validate_schema` (True - reject a provably-unsatisfiable schema before any API
call). Tuning ints: `max_fields_per_call`, `max_concurrent_calls`,
`max_api_retries`.

### Grounding and provenance

Both are off by default and cost **no extra API calls** (grounding is in-memory
string matching; provenance is a document scan). They are independent.

`ground_values=True` labels every filled value by how well its source excerpt
supports it and reports a `hallucination_rate`. It is **non-destructive**: a value
is never dropped for a weak label, because a correct value is often not verbatim
(a unit written `USD` when the document prints `$`, a derived period like
`FY2025 Q2`). Enum values are `schema_derived` (chosen from the schema, already
validated) and excluded from the metric. Read the rate as a lexical *support*
signal, not a truth verdict.

`provenance=True` adds `result.provenance`, mapping each value's dot-path to its
`[start, end)` char offsets in the document. Only values found verbatim (including
numeric comma/scale forms and currency/unit aliases) get an entry, so a reported
offset always indexes the real source text.

```python
from nfield import nfield
from nfield.config import ExtractionConfig

result = nfield(
    document,
    schema,
    "groq/llama-3.3-70b-versatile",
    config=ExtractionConfig(ground_values=True, provenance=True),
)
result.metadata.hallucination_rate       # e.g. 0.25 (lexical support signal)
start, end = result.provenance["revenue"]  # -> document[start:end] is the value
```

### Reasoning models

A reasoning model emits a thinking pass before its answer. Left on, that pass
shares the per-call output budget and can truncate the answer to nothing. Set
`reasoning_model=True` so each call disables thinking (via the endpoint's own
off-switch); a stray inline `<think>…</think>` block is stripped either way. If
the endpoint does not support the off-switch, it is dropped automatically and the
call still succeeds. This applies to both the `openai/` and `groq/` providers.

```python
from nfield import nfield
from nfield.config import ExtractionConfig

result = nfield(
    document,
    schema,
    "openai/qwen/qwen3.6-27b",          # any reasoning model, openai/ or groq/
    config=ExtractionConfig(reasoning_model=True),
)
```

Retrieval note: BMX lexical indexing folds diacritics (Unicode NFKD; Lucene-style ASCII
folding), so an accented document spelling (`Denísov`, `café`) matches an
unaccented query term (`Denisov`, `cafe`) and vice-versa.

Missing-field recovery always runs as a core Stage 5
step - fields never produced after surgical retry get one bounded recovery pass
(tree-backtrack absent-ancestor children, then re-extract the missed-only set).
There is no flag; it is a no-op when nothing is missing.

## Filesystem helpers (`nfield.io`)

```python
load_document(path) -> str          # read a UTF-8 text document
load_schema(path) -> dict           # read + parse a JSON Schema file (SchemaError on bad JSON / non-object)
save_results(results, path) -> None # write results as JSON Lines (one per line)
load_results(path) -> list[ExtractionResult]   # read them back (round-trips to_dict/from_dict)
```

Text/JSON only - PDF/DOCX/CSV parsing stays the caller's job. `ExtractionResult.to_dict()`
/ `ExtractionResult.from_dict()` give the underlying JSON-serialisable form.

## Tabular export (`nfield.export`, optional `pandas`)

```python
results_to_dataframe(results, *, include_metadata=False) -> pandas.DataFrame
result_to_dataframe(result, *, include_metadata=False) -> pandas.DataFrame
results_to_csv(results, path, *, include_metadata=False) -> None
```

One row per result; columns are the flat dot-notation field paths. Install with
`pip install 'nfield[export]'` - pandas is imported only when these are called.

## Exceptions

All inherit from `NFieldError`: `SchemaError`, `ProviderError`,
`ExtractionError`, `ValidationError`, `AssemblyError`.
