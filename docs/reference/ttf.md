# Reference — TTF Engine

This page documents the `TTFEngine` class, the prompt-building functions, and the `FailureModeDetector` from `formatshield.ttf`.

---

## `TTFEngine`

```python
class TTFEngine:
    def __init__(
        self,
        backend: Backend,
        ttf_fallback: bool = True,
        expose_thinking: bool = False,
    ) -> None: ...
```

Two-pass Think-Then-Format generation engine. Implements the CRANE algorithm (arXiv 2502.09061).

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `Backend` | required | Any object implementing the `Backend` protocol |
| `ttf_fallback` | `bool` | `True` | When `True`, if Pass 2 output fails Pydantic validation the engine retries with direct single-pass generation |
| `expose_thinking` | `bool` | `False` | When `True`, thinking text is retained in streaming events. When `False`, `thinking` events are still yielded internally but callers can filter them |

---

### `TTFEngine.generate()`

```python
async def generate(
    self,
    prompt: str,
    schema: dict[str, Any] | None = None,
    schema_model: type[BaseModel] | None = None,
    kv_cache_prefix: str | None = None,
) -> tuple[str, str]: ...
```

Run two-pass TTF generation and return `(thinking_text, json_output)`.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The original user prompt (unmodified — `TTFEngine` applies the think-prompt formatting internally) |
| `schema` | `dict \| None` | Optional JSON Schema dict. Used as the constraint for Pass 2 |
| `schema_model` | `type[BaseModel] \| None` | Optional Pydantic model class for output validation. When provided, triggers validation and optional fallback |
| `kv_cache_prefix` | `str \| None` | Override the KV-cache prefix passed to the backend. Defaults to the think prompt when the backend supports KV-cache reuse |

**Returns:** `tuple[str, str]` — `(thinking_text, json_output)` where `thinking_text` is extracted from `<think>...</think>` tags and `json_output` is the raw JSON string from Pass 2.

**Raises:** `RuntimeError` re-raised from the backend if both TTF and fallback paths fail.

**Fallback behavior:** When `schema_model` is provided and Pass 2 output fails validation:
- If `ttf_fallback=True`: retry with `generate_direct()` and return `("", direct_output)` (empty thinking string signals a fallback occurred)
- If `ttf_fallback=False`: return the invalid output with a logged warning

---

### `TTFEngine.stream()`

```python
async def stream(
    self,
    prompt: str,
    schema: dict[str, Any] | None = None,
) -> AsyncIterator[StreamEvent]: ...
```

Stream two-pass TTF generation. Yields `StreamEvent` objects in three phases:

1. `type="thinking"` — incremental content chunks from Pass 1 (reasoning phase)
2. `type="output"` — incremental token chunks from Pass 2
3. `type="complete"` — final event with assembled JSON dict

---

### `TTFEngine.generate_direct()`

```python
async def generate_direct(
    self,
    prompt: str,
    schema: dict[str, Any] | None = None,
) -> str: ...
```

Single-pass constrained generation — the fallback path. Sends `prompt` directly to the backend with `constraints="json"`. Used when:

- The oracle routes a request to `"direct"` strategy
- TTF validation fails and `ttf_fallback=True`

---

## Prompt Building Functions

These functions are defined in `formatshield.ttf.prompts`:

### `build_think_prompt()`

```python
def build_think_prompt(prompt: str) -> str: ...
```

Wrap `prompt` with instructions to think in `<think>...</think>` tags before producing output. This is the prompt sent to the backend in Pass 1.

Example output:

```
Think carefully inside <think>...</think> tags before answering.

<think>
</think>

{original_prompt}
```

---

### `build_format_prompt()`

```python
def build_format_prompt(
    think_prompt: str,
    thinking_text: str,
    schema: dict[str, Any] | None = None,
) -> str: ...
```

Build the Pass 2 prompt by appending Pass 1's thinking output to the think prompt, then instructing the model to produce structured JSON.

The resulting prompt structure:

```
{think_prompt}
<think>
{thinking_text}
</think>

Now produce a JSON object that satisfies the following schema:
{schema_json}

Output only valid JSON, no other text:
```

---

### `extract_thinking()`

```python
def extract_thinking(text: str) -> str: ...
```

Extract the content between `<think>` and `</think>` tags from Pass 1 output. Returns the stripped content, or the full text if no tags are found.

---

## `FailureModeDetector`

```python
class FailureModeDetector:
    def detect(
        self,
        features: ComplexityFeatures,
        model_id: str,
        schema: dict[str, Any],
    ) -> list[str]: ...

    def should_override_to_direct(self, failure_modes: list[str]) -> bool: ...
```

Detects patterns that indicate TTF would be counterproductive and should be overridden.

### `detect()`

Returns a list of failure mode label strings. Possible labels:

| Label | Trigger |
|---|---|
| `"native_thinker"` | Model is a known native-thinker (would double-think) |
| `"schema_too_simple"` | Schema has depth 0 or 1 and no constraints — TTF overhead not justified |
| `"prompt_too_short"` | Prompt is in bucket 0 (< 50 tokens) — too short to benefit from reasoning |
| `"no_reasoning_needed"` | Zero reasoning ops and simple schema — pure extraction task |

### `should_override_to_direct()`

Returns `True` if any of the detected failure modes should force the `"direct"` route, overriding the oracle's decision.

Currently overrides to direct when `"native_thinker"` is detected.

---

## Two-Pass Flow Diagram

```
                    User Prompt
                         │
                         ▼
              build_think_prompt(prompt)
                         │
                         ▼
              ┌─ Backend.generate() ─┐
              │   constraints=None   │  ← Pass 1: unconstrained reasoning
              │   schema=None        │
              └──────────────────────┘
                         │
                    raw_thinking
                         │
                  extract_thinking()
                         │
                   thinking_text
                         │
                         ▼
         build_format_prompt(think_prompt, raw_thinking, schema)
                         │
                         ▼
              ┌─ Backend.generate() ─┐
              │   constraints="json" │  ← Pass 2: constrained JSON output
              │   schema=schema_dict │
              │   kv_prefix=...      │  ← KV reuse if backend supports it
              └──────────────────────┘
                         │
                    json_output
                         │
                    Validation
                    (Pydantic)
                         │
             ┌──────────┴──────────┐
             │ valid               │ invalid + ttf_fallback=True
             ▼                     ▼
      (thinking_text,        generate_direct()
       json_output)               │
                             ("", direct_output)
```

---

## KV Cache Reuse

When `backend.supports_kv_cache_reuse is True` (only vLLM with prefix caching enabled):

- Pass 2 receives `kv_cache_prefix = think_prompt` — the full Pass 1 prompt
- The vLLM server reuses the KV activations computed during Pass 1 for the shared prefix
- This avoids re-computing attention for the entire context, reducing Pass 2 latency by ~50-70%
- Net TTF overhead drops from ~30% to ~10%

For all other backends, Pass 2 re-processes the full `format_prompt` (which includes the thinking text as context). This is less efficient but functionally equivalent.
