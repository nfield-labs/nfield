# Reference — Oracle

This page documents `ThresholdOracle`, `OracleX`, `RoutingDecision`, and the Φ routing score
from `formatshield.oracle`.

---

## Routing Score Φ(prompt, schema)

FormatShield uses a closed-form, training-free routing score to decide between TTF and direct
generation. No benchmark data or ML model artifacts are required.

```
Φ = 1 − exp(−(A·λ̃₂² + B·τ·λ̃₂ + C·ΔK))
```

| Component | Symbol | Description |
|---|---|---|
| Schema algebraic connectivity | λ̃₂ | Normalized Fiedler value of the schema dependency graph. High value → dense field coupling → prefer TTF |
| Schema constraint tightness | τ | Entropy proxy: 1 − mean(h(v)) / H₀. High value → highly constrained schema |
| NCD alignment gap | ΔK | Normalized Compression Distance between prompt and schema. High value → semantically distant |

**Coefficients** (half-point at each component acting alone):

| Constant | Value | Half-point condition |
|---|---|---|
| A = ln2 / 0.25² | 11.09 | Φ = 0.5 when λ̃₂ = 0.25 and τ = ΔK = 0 |
| B = ln2 / 0.50 | 1.386 | τ·λ̃₂ interaction term |
| C = ln2 / 0.70 | 0.990 | Φ = 0.5 when ΔK = 0.70 and λ̃₂ = τ = 0 |

**Interpretation:** Φ > backend threshold → TTF; Φ ≤ threshold → direct.

```python
from formatshield.oracle.routing_score import compute_routing_score

rs = compute_routing_score(prompt, schema)
print(rs.phi)         # float in [0, 1]
print(rs.lambda2)     # Fiedler value
print(rs.tau)         # constraint tightness
print(rs.delta_k)     # NCD gap
print(rs.explanation) # "Φ=0.712 λ̃₂=0.231 τ=0.161 ΔK=0.847"
```

---

## `ThresholdOracle`

```python
class ThresholdOracle:
    def __init__(self, model_path: Path | str | None = None) -> None: ...
```

Routes each inference request to either TTF or direct generation using the Φ score passed via
`RoutingContext`, falling back to a heuristic weighted score when context is unavailable.

The `model_path` parameter is accepted for API compatibility but ignored — no pkl artifact is
loaded or required.

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
    context: RoutingContext | None = None,
) -> RoutingDecision: ...
```

Return a `RoutingDecision` for the given request.

| Parameter | Type | Description |
|---|---|---|
| `features` | `ComplexityFeatures` | Feature vector computed by `ComplexityScorer` |
| `backend` | `str` | Inference backend identifier (e.g. `"vllm"`, `"groq"`) |
| `model_id` | `str` | Model identifier string (e.g. `"gpt-4o"`) |
| `latency_budget_ms` | `float \| None` | Optional hard latency cap. TTF is suppressed if estimated overhead exceeds this value |
| `cost_aware` | `bool` | When `True`, applies a +0.03 upward bias to the routing threshold |
| `context` | `RoutingContext \| None` | Carries pre-computed Φ score and components from `core.py` |

Returns `RoutingDecision`. Falls back to `RoutingDecision(strategy="direct", confidence=0.3)` on any error.

---

### Decision Logic (Priority Order)

1. **Native thinker** — If `model_id` matches a native-thinker prefix (o1, o3, deepseek-r1, ...), always return `"direct"` with confidence 0.95.

2. **Latency budget exceeded** — If `latency_budget_ms` is set and estimated TTF overhead exceeds it, return `"direct"` with confidence 0.85.

3. **Φ score available** — If `context.phi_score > 0`, use it to compare against the per-backend threshold.

4. **Heuristic fallback** — Compute a weighted score from the feature vector. Confidence is fixed at 0.70.

---

## Per-Backend Thresholds

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

These models have built-in chain-of-thought reasoning. FormatShield always routes them to direct generation:

```python
NATIVE_THINKERS = frozenset({
    "o1", "o3", "o1-mini", "o3-mini",
    "deepseek-r1",
    "deepseek-r1-distill-llama-70b",
    "deepseek-r1-distill-qwen-32b",
})
```

Matching is case-insensitive prefix match.

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

## Φ Math Modules

### `schema_graph.fiedler_value(schema)`

Builds an undirected weighted dependency graph G_σ from the JSON Schema dict and returns the
normalized second Laplacian eigenvalue λ̃₂ = λ₂(L) / (d_max + 1) ∈ [0, 1].

Edge weights: structural nesting = 1.0, `$ref` = 1.5, `allOf`/`anyOf`/`oneOf`/`if` = 2.0,
shared name stems = 0.5.

### `schema_entropy.constraint_tightness(schema)`

Walks the schema type tree and returns τ = 1 − mean(h(v)) / H₀ ∈ [0, 1], where h(v) is
per-leaf Shannon entropy: boolean → 1 bit, enum[k] → log₂(k), int[a,b] → log₂(b−a+1),
string+format → 0.5·H₀, unconstrained → H₀ ≈ 16.97 bits.

### `ncd.prompt_schema_ncd(prompt, schema)`

Returns the Normalized Compression Distance between the prompt string and the schema's flattened
field list (`"field: type\n"` lines) via zlib. Returns 0.5 for inputs shorter than 32 bytes.

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

print(f"Strategy:    {decision.strategy}")
print(f"Confidence:  {decision.confidence:.2f}")
print(f"Explanation: {decision.explanation}")
```

---

## Migration from v0.2

`from_benchmark_data()`, `save()`, and `load()` raise `DeprecationWarning` + `NotImplementedError`
in v0.3. No model artifacts or benchmark CSV files are needed. See
[Oracle v3 Migration](../migration/oracle-v3.md).
