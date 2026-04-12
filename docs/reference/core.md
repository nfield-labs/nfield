# Reference — Core API

This page documents `formatshield.core` — the `FormatShield` class, `GenerationResult`, and the module-level `generate()` convenience function.

---

## `FormatShield`

```python
class FormatShield:
    def __init__(
        self,
        model: str,
        base_url: str | None = None,
        api_key: str | None = None,
        latency_budget_ms: float | None = None,
        cost_aware: bool = False,
        ttf_fallback: bool = True,
        expose_thinking: bool = False,
        debug: bool = False,
        metrics: MetricsCollector | None = None,
        log_level: str = "WARNING",
    ) -> None: ...
```

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model identifier in `"provider/model"` format (e.g. `"groq/llama-3.1-70b-versatile"`) |
| `base_url` | `str \| None` | `None` | Override the backend's base URL (useful for Ollama remote, custom vLLM endpoint) |
| `api_key` | `str \| None` | `None` | Override the API key read from environment variables |
| `latency_budget_ms` | `float \| None` | `None` | Hard latency cap in milliseconds. When set, TTF is suppressed if estimated overhead would exceed this value |
| `cost_aware` | `bool` | `False` | Apply a small upward bias to the routing threshold to prefer cheaper direct generation (reserved for future token-cost optimisation) |
| `ttf_fallback` | `bool` | `True` | When `True`, if TTF's Pass 2 output fails Pydantic validation, automatically retry with single-pass direct generation |
| `expose_thinking` | `bool` | `False` | When `True`, `result.thinking` is populated with Pass 1 reasoning text. When streaming, thinking events are included |
| `debug` | `bool` | `False` | Print a formatted routing trace to stdout on each `generate()` call |
| `metrics` | `MetricsCollector \| None` | `None` | Shared `MetricsCollector` instance. When `None`, an instance-private collector is created |
| `log_level` | `str` | `"WARNING"` | Python logging level for FormatShield's structured logger |

### Attributes

| Attribute | Type | Description |
|---|---|---|
| `model` | `str` | The model identifier provided at construction |
| `backend_name` | `BackendName` | The inferred backend name (e.g. `"groq"`) |

---

### `FormatShield.generate()`

```python
async def generate(
    self,
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    debug: bool | None = None,
) -> GenerationResult: ...
```

Generate structured output, routing between TTF and direct.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | The user prompt |
| `schema` | `type[BaseModel] \| dict \| None` | Pydantic model class or JSON Schema dict. When `None`, the backend generates free-form text |
| `debug` | `bool \| None` | Override instance-level `debug` setting for this call only |

**Returns:** `GenerationResult`

**Raises:** Backend-specific exceptions on network failure after all retries exhausted.

---

### `FormatShield.generate_sync()`

```python
def generate_sync(
    self,
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    debug: bool | None = None,
) -> GenerationResult: ...
```

Synchronous wrapper around `generate()`. Safe to call from within a running event loop (e.g. pytest-asyncio, Jupyter) by running in a dedicated thread with its own event loop. Raises `TimeoutError` if generation exceeds 120 seconds.

---

### `FormatShield.stream()`

```python
async def stream(
    self,
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
) -> AsyncIterator[StreamEvent]: ...
```

Stream generation events. Yields `StreamEvent` objects of type `"thinking"`, `"output"`, and `"complete"`.

See [Tutorial 05: Streaming](../tutorials/05-streaming.md) for usage examples.

---

### `FormatShield.from_config()`

```python
@classmethod
def from_config(cls, config_path: str) -> FormatShield: ...
```

Load a `FormatShield` instance from a YAML or JSON config file. The file must contain keys matching the constructor parameters.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `config_path` | `str` | Path to a `.yaml`, `.yml`, or `.json` config file |

**Raises:** `FileNotFoundError` if the file doesn't exist. `ImportError` if YAML is requested but `pyyaml` is not installed.

---

## `GenerationResult`

```python
@dataclass
class GenerationResult:
    output: str
    parsed: BaseModel | dict[str, Any] | None
    thinking: str | None
    routing: RoutingDecision
    complexity_score: float
    failure_modes: list[str]
    latency_ms: float
    backend: str
    model: str
    schema_valid: bool
    fallback_triggered: bool
```

### Fields

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Raw JSON string returned by the backend |
| `parsed` | `BaseModel \| dict \| None` | Validated Pydantic model instance. Falls back to a plain `dict` if Pydantic validation failed but JSON parsing succeeded. `None` if both fail |
| `thinking` | `str \| None` | Reasoning text from TTF Pass 1. `None` for direct routes or when `expose_thinking=False` |
| `routing` | `RoutingDecision` | The routing decision made by `ThresholdOracle` |
| `complexity_score` | `float` | Scalar complexity score in [0, 1] computed by `ComplexityScorer` |
| `failure_modes` | `list[str]` | Failure modes detected by `FailureModeDetector` |
| `latency_ms` | `float` | Total wall-clock latency in milliseconds |
| `backend` | `str` | Backend used (e.g. `"groq"`, `"vllm"`) |
| `model` | `str` | Full model identifier (e.g. `"groq/llama-3.1-70b-versatile"`) |
| `schema_valid` | `bool` | `True` if Pydantic schema validation passed; `False` otherwise |
| `fallback_triggered` | `bool` | `True` if TTF failed and fell back to direct generation |

### `GenerationResult.model_dump()`

```python
def model_dump(self) -> dict[str, Any]: ...
```

Returns a JSON-serializable dictionary representation of the result. The `routing` field is expanded into a nested dict.

---

## `RoutingDecision`

```python
@dataclass
class RoutingDecision:
    strategy: str              # "ttf" or "direct"
    expected_accuracy_delta: float
    expected_overhead_pct: float
    confidence: float
    explanation: str
    failure_modes: list[str] = field(default_factory=list)
```

### Fields

| Field | Type | Description |
|---|---|---|
| `strategy` | `str` | `"ttf"` or `"direct"` |
| `expected_accuracy_delta` | `float` | Expected accuracy improvement from TTF. Positive = TTF helps. Typically ~0.17 when TTF is selected |
| `expected_overhead_pct` | `float` | Estimated latency overhead percentage. 0.0 for direct routes |
| `confidence` | `float` | Oracle confidence in [0, 1] |
| `explanation` | `str` | Human-readable routing reason |
| `failure_modes` | `list[str]` | Failure modes that influenced the decision |

### `RoutingDecision.use_ttf`

```python
@property
def use_ttf(self) -> bool: ...
```

Convenience property. Returns `True` when `strategy == "ttf"`.

---

## Module-Level `generate()`

```python
async def generate(
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
    model: str = "groq/llama-3.1-70b-versatile",
    **kwargs: Any,
) -> GenerationResult: ...
```

One-liner convenience wrapper. Creates a `FormatShield` instance internally, runs one call, and returns the result. Extra keyword arguments are forwarded to `FormatShield.__init__()`.

**Example:**

```python
result = await fs.generate(
    "Solve: 2x + 5 = 13",
    schema=MathSolution,
    model="groq/llama-3.1-70b-versatile",
    debug=True,
)
```

!!! tip "Performance note"
    For multiple calls, instantiate `FormatShield` once and reuse it. The module-level `generate()` creates a new instance on every call, which has minor overhead.

---

## `BackendName`

```python
BackendName = Literal["groq", "openrouter", "ollama", "vllm", "outlines", "guidance", "dryrun"]
```

Type alias for valid backend name strings.

---

## `get_backend_name_from_model()`

```python
def get_backend_name_from_model(model: str) -> BackendName: ...
```

Infer the backend from a model string prefix. Returns `"openrouter"` for unrecognised prefixes.
