# Reference — Oracle

This page documents `ThresholdOracle`, `RoutingDecision`, and related constants from `formatshield.oracle`.

---

## `ThresholdOracle`

```python
class ThresholdOracle:
    def __init__(self, model_path: Path | str | None = None) -> None: ...
```

Routes each inference request to either TTF or direct generation.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model_path` | `Path \| str \| None` | `None` | Path to a pre-trained sklearn `LogisticRegression` pickle. When `None`, the default location `oracle_data/threshold_oracle_v1.pkl` (relative to the module) is tried. If the file is missing, the oracle falls back to heuristics |

At construction, the oracle attempts to load a pre-trained model from `model_path`. If no model is found, it uses the heuristic threshold fallback transparently.

---

### `ThresholdOracle.predict()`

```python
def predict(
    self,
    features: ComplexityFeatures,
    backend: str,
    model_id: str,
    latency_budget_ms: float | None = None,
    cost_aware: bool = False,
) -> RoutingDecision: ...
```

Return a `RoutingDecision` for the given request.

| Parameter | Type | Description |
|---|---|---|
| `features` | `ComplexityFeatures` | Feature vector computed by `ComplexityScorer` |
| `backend` | `str` | Inference backend identifier (e.g. `"vllm"`, `"groq"`) |
| `model_id` | `str` | Model identifier string (e.g. `"gpt-4o"`) |
| `latency_budget_ms` | `float \| None` | Optional hard latency cap. When set, TTF is suppressed if estimated overhead exceeds this value |
| `cost_aware` | `bool` | When `True`, applies a +0.03 upward bias to the routing threshold to prefer cheaper direct generation |

Returns `RoutingDecision`. Falls back to `RoutingDecision(strategy="direct", confidence=0.3)` on any error.

---

### Decision Logic (Priority Order)

1. **Native thinker** — If `model_id` matches a native-thinker prefix (o1, o3, deepseek-r1, ...), always return `"direct"` with confidence 0.95.

2. **Latency budget exceeded** — If `latency_budget_ms` is set and the backend's estimated TTF overhead exceeds it, return `"direct"` with confidence 0.85.

3. **sklearn model available** — If a pre-trained `LogisticRegression` bundle is loaded, use it for prediction. Confidence is derived from `predict_proba`. Falls back to heuristic if sklearn prediction fails.

4. **Heuristic threshold** — Compute a weighted score from the feature vector and compare against the per-backend threshold. Confidence is fixed at 0.70.

---

### `ThresholdOracle.from_benchmark_data()`

```python
@classmethod
def from_benchmark_data(
    cls,
    csv_path: str | Path,
    model_path: Path | str | None = None,
    *,
    save: bool = True,
) -> ThresholdOracle: ...
```

Train a `LogisticRegression` oracle from benchmark CSV data.

The CSV must be the output of `BenchmarkHarness.run()` (i.e., `summary.csv`). The target label is `1` (use TTF) when `accuracy_delta > 0`, and `0` (use direct) otherwise.

| Parameter | Type | Description |
|---|---|---|
| `csv_path` | `str \| Path` | Path to the benchmark results CSV file |
| `model_path` | `Path \| str \| None` | Where to save the trained model. Defaults to `oracle_data/threshold_oracle_v1.pkl` |
| `save` | `bool` | If `True` (default), the trained model is persisted to `model_path` |

**Raises:**
- `ImportError` — if `scikit-learn`, `joblib`, or `numpy` are not installed
- `FileNotFoundError` — if `csv_path` does not exist
- `ValueError` — if the CSV contains fewer than 10 valid rows

**Example:**

```python
from formatshield.oracle.threshold_oracle import ThresholdOracle

oracle = ThresholdOracle.from_benchmark_data(
    csv_path="benchmark_results/summary.csv",
    save=True,
)
# oracle._clf is now a dict {"clf": LogisticRegression, "scaler": StandardScaler}
```

---

### `ThresholdOracle.save()`

```python
def save(self, path: Path | str) -> None: ...
```

Persist the current sklearn model bundle to `path`. Parent directories are created automatically.

**Raises:**
- `RuntimeError` — if no trained model is loaded
- `ImportError` — if `joblib` is not installed

---

### `ThresholdOracle.load()`

```python
def load(self, path: Path | str) -> None: ...
```

Load a persisted sklearn model bundle from `path`.

**Raises:**
- `FileNotFoundError` — if `path` does not exist
- `ImportError` — if `joblib` is not installed

---

## Per-Backend Thresholds

Default heuristic thresholds (used when no sklearn model is available):

| Backend | Threshold | Rationale |
|---|---|---|
| `vllm` | 0.60 | Lowest — KV cache reuse makes TTF cheap |
| `outlines` | 0.62 | In-process, moderate overhead |
| `guidance` | 0.63 | In-process, moderate overhead |
| `groq` | 0.65 | API backend, ~30% overhead |
| `ollama` | 0.65 | Local API, ~25% overhead |
| `openrouter` | 0.67 | API backend, highest latency |
| `default` | 0.65 | Fallback for unknown backends |

---

## Per-Backend TTF Overhead Estimates

| Backend | Overhead | Notes |
|---|---|---|
| `vllm` | 10% | With `--enable-prefix-caching` |
| `outlines` | 20% | In-process |
| `guidance` | 22% | In-process |
| `ollama` | 25% | Local server |
| `groq` | 30% | Cloud API, two network round-trips |
| `openrouter` | 35% | Cloud API, routing overhead |

---

## Native Thinker Models

These models have built-in chain-of-thought reasoning. FormatShield always routes them to direct generation to avoid double-thinking:

```python
NATIVE_THINKERS = frozenset({
    "o1",
    "o3",
    "o1-mini",
    "o3-mini",
    "deepseek-r1",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
})
```

Matching is case-insensitive prefix match — any model string starting with one of these entries (e.g. `"openrouter/openai/o1-mini"`) is treated as a native thinker.

---

## `RoutingDecision`

```python
@dataclass
class RoutingDecision:
    strategy: str
    expected_accuracy_delta: float
    expected_overhead_pct: float
    confidence: float
    explanation: str
    failure_modes: list[str] = field(default_factory=list)
```

See [Reference: Core](core.md#routingdecision) for the full field documentation.

---

## Feature Weights (Heuristic Path)

When the sklearn model is not available, the heuristic path uses these weights to compute the routing score from `ComplexityFeatures.to_feature_vector()`:

| Feature | Weight | Cap |
|---|---|---|
| `token_entropy` | 0.20 | 1.0 |
| `schema_depth` | 0.25 | 10.0 |
| `required_reasoning_ops` | 0.20 | 20.0 |
| `instruction_tune_score` | 0.15 | 1.0 |
| `prompt_length_bucket` | 0.10 | 3.0 |
| `schema_constraint_count` | 0.10 | 30.0 |

The `cost_aware=True` flag adds `+0.03` to the effective threshold, requiring a slightly higher score before TTF is triggered.

---

## Example: Direct Oracle Usage

```python
from formatshield.scorer.complexity_scorer import ComplexityScorer
from formatshield.oracle.threshold_oracle import ThresholdOracle

scorer = ComplexityScorer()
oracle = ThresholdOracle()

features = scorer.score(
    "Solve and explain step by step: if 2^x = 32, what is x?",
    schema={"type": "object", "properties": {"x": {"type": "number"}, "steps": {"type": "array"}}},
    model_id="groq/llama-3.1-70b-versatile",
)

decision = oracle.predict(
    features=features,
    backend="groq",
    model_id="groq/llama-3.1-70b-versatile",
    latency_budget_ms=5000,
)

print(f"Strategy:    {decision.strategy}")       # "ttf"
print(f"Confidence:  {decision.confidence:.2f}") # 0.70
print(f"Explanation: {decision.explanation}")
```
