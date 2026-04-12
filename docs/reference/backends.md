# Reference — Backends

This page documents the `Backend` protocol, all built-in backend implementations, and the retry infrastructure.

---

## `Backend` Protocol

All backends implement this protocol (defined in `formatshield.backends.protocol`):

```python
class Backend(Protocol):
    name: str
    supports_kv_cache_reuse: bool

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str: ...

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]: ...
```

### Protocol Fields

| Field | Type | Description |
|---|---|---|
| `name` | `str` | Backend identifier string (e.g. `"groq"`) |
| `supports_kv_cache_reuse` | `bool` | Whether the backend supports KV-cache prefix reuse for TTF Pass 2. Only `True` for vLLM with prefix caching enabled |

### Protocol Methods

#### `generate()`

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | Full prompt string (already formatted by `TTFEngine` if TTF route) |
| `schema` | `dict \| None` | JSON Schema dict for constrained decoding. `None` for unconstrained (TTF Pass 1) |
| `constraints` | `str \| None` | Constraint type: `"json"` requests JSON mode. `None` for free generation |
| `kv_cache_prefix` | `str \| None` | KV-cache prefix for backends supporting native prefix reuse |

Returns the raw LLM output as a string. For JSON schema requests, this should be valid JSON.

#### `stream()`

Returns an async iterator of `StreamEvent` objects. Each event has `type`, `token`, `content`, `json`, `backend`, and `latency_ms` fields.

---

## `GroqBackend`

```python
class GroqBackend:
    name = "groq"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None: ...
```

Calls the Groq REST API (`api.groq.com/openai/v1`). Uses `response_format={"type": "json_object"}` when `constraints="json"`. Does not support grammar-based schema constraints (no guided JSON on Groq).

**Environment variable:** `GROQ_API_KEY`

**Default model:** `llama-3.1-70b-versatile`

---

## `OpenRouterBackend`

```python
class OpenRouterBackend:
    name = "openrouter"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None: ...
```

Calls the OpenRouter unified API (`openrouter.ai/api/v1`). Supports all models available on OpenRouter. Uses `response_format={"type": "json_object"}` for JSON mode.

**Environment variable:** `OPENROUTER_API_KEY`

---

## `OllamaBackend`

```python
class OllamaBackend:
    name = "ollama"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
    ) -> None: ...
```

Calls a locally running Ollama server. Uses `format="json"` parameter when `constraints="json"`. This is Ollama's native JSON mode — not grammar-based FSM masking.

**No API key required.**

---

## `VLLMBackend`

```python
class VLLMBackend:
    name = "vllm"
    supports_kv_cache_reuse = True

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:8000/v1",
    ) -> None: ...
```

Calls a vLLM server via its OpenAI-compatible REST API. Uses `guided_json` parameter for schema-constrained decoding.

**Key feature:** `supports_kv_cache_reuse = True`. When vLLM is started with `--enable-prefix-caching`, TTF's Pass 2 can reuse the KV activations from Pass 1, reducing overhead from ~30% to ~10%.

---

## `OutlinesBackend`

```python
class OutlinesBackend:
    name = "outlines"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        model: str,
    ) -> None: ...
```

Runs the Outlines library in-process. Uses true FSM-based grammar constraints for JSON schema compliance.

**Requires:** `pip install "formatshield[outlines]"` and a GPU or large RAM machine.

---

## `GuidanceBackend`

```python
class GuidanceBackend:
    name = "guidance"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        model: str,
    ) -> None: ...
```

Runs the Guidance library in-process. Uses Microsoft's interleaved structured generation approach.

**Requires:** `pip install "formatshield[guidance]"`.

---

## `DryRunBackend`

```python
class DryRunBackend:
    name = "dryrun"
    supports_kv_cache_reuse = False

    def __init__(
        self,
        seed: int = 42,
    ) -> None: ...
```

Deterministic zero-dependency backend for testing. Generates valid JSON responses matching the provided schema using type-based generation. Returns `<think>Dry run thinking pass 1</think>` for unconstrained (Pass 1) calls.

**No API key, no network, no GPU required.**

### DryRunBackend Behavior

| Scenario | Response |
|---|---|
| `constraints=None` (TTF Pass 1) | `<think>Dry run thinking pass 1</think>` |
| `constraints="json"` with schema | Valid JSON matching the schema structure |
| `constraints="json"` without schema | `{"result": "dry_run_output"}` |
| `stream()` | Emits one `output` token then `complete` |

---

## Retry Infrastructure

All backends use the `_retry.py` module for automatic exponential backoff. The retry decorator wraps `generate()` and `stream()` calls:

```python
from formatshield._retry import with_retry

@with_retry(max_retries=3, base_delay=1.0, backoff_factor=2.0, max_delay=60.0)
async def generate(self, prompt, schema=None, constraints=None):
    ...
```

### Retry Policy

| Parameter | Default | Description |
|---|---|---|
| `max_retries` | `3` | Maximum number of retry attempts |
| `base_delay` | `1.0` | Initial delay in seconds before first retry |
| `backoff_factor` | `2.0` | Multiply delay by this factor after each retry |
| `max_delay` | `60.0` | Maximum delay between retries in seconds |
| `jitter` | `±10%` | Random jitter added to each delay |

### Retried Errors

- HTTP 429 (Rate Limit)
- HTTP 500, 502, 503 (Server Errors)
- `httpx.ConnectError`
- `httpx.ReadTimeout`
- `asyncio.TimeoutError`

Non-retried errors (raised immediately):

- HTTP 400 (Bad Request — likely a schema error)
- HTTP 401 (Unauthorized — check API key)
- HTTP 404 (Not Found — check model name)
- `pydantic.ValidationError`

---

## Adding a Custom Backend

Implement the `Backend` protocol and register it:

```python
# my_backend.py
from formatshield.scorer.features import StreamEvent
from collections.abc import AsyncIterator
from typing import Any

class MyBackend:
    name = "mybackend"
    supports_kv_cache_reuse = False

    def __init__(self, model: str, api_key: str | None = None):
        self.model = model
        self._api_key = api_key

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        # Make your API call and return JSON string
        ...

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(type="output", token="...", backend=self.name, latency_ms=0.0)
        yield StreamEvent(type="complete", json={}, backend=self.name, latency_ms=100.0)
```

Then inject it directly:

```python
import formatshield as fs
from my_backend import MyBackend

shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")
shield._backend = MyBackend(model="my-model", api_key="my-key")
```
