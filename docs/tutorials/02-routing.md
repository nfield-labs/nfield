# Tutorial 02 — Understanding and Controlling Routing

FormatShield automatically decides whether each request should use **direct constrained decoding** or the **Think-Then-Format (TTF)** two-pass strategy. This tutorial explains how that decision is made and how you can observe, influence, and override it.

---

## How Routing Works

Every `generate()` call passes through two components before any backend is contacted:

```
prompt + schema
      │
      ▼
ComplexityScorer   →  ComplexityFeatures + scalar score [0, 1]
      │
      ▼
ThresholdOracle    →  RoutingDecision (strategy="ttf" or "direct")
      │
      ├── "ttf"    →  TTFEngine (Pass 1: think, Pass 2: format)
      └── "direct" →  backend.generate() with JSON constraint
```

The **ComplexityScorer** computes six features from the prompt and schema. The **ThresholdOracle** compares the resulting scalar score against a per-backend threshold to produce a `RoutingDecision`.

---

## 1. Reading the Routing Decision

Every `GenerationResult` contains a `RoutingDecision`:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class Equation(BaseModel):
    expression: str
    solution: float
    steps: list[str]

async def main():
    result = await fs.generate(
        prompt="Solve for x: 3x² - 12x + 9 = 0. Show all steps.",
        schema=Equation,
        model="groq/llama-3.1-70b-versatile",
    )

    rd = result.routing
    print(f"Strategy:           {rd.strategy}")              # "ttf" or "direct"
    print(f"Complexity score:   {result.complexity_score:.3f}")
    print(f"Accuracy delta:     {rd.expected_accuracy_delta:+.3f}")  # e.g. +0.170
    print(f"Overhead estimate:  {rd.expected_overhead_pct:.0f}%")    # e.g. 30%
    print(f"Confidence:         {rd.confidence:.2f}")
    print(f"Explanation:        {rd.explanation}")

asyncio.run(main())
```

---

## 2. Enable Debug Mode to See the Routing Trace

Pass `debug=True` to print a formatted routing trace for each request:

```python
shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    debug=True,
)
result = await shield.generate(
    prompt="What is 2 + 2?",
    schema=MySchema,
)
```

Output:

```
[FormatShield] model=groq/llama-3.1-70b-versatile
[FormatShield] complexity_score=0.183 (schema_depth=1, reasoning_ops=0, length_bucket=0)
[FormatShield] route=direct | expected_delta=+0.000 | estimated_overhead=0%
[FormatShield] confidence=0.70 | explanation='Heuristic score 0.183 ≤ threshold 0.650 for backend 'groq' → direct.'
```

Now try a complex prompt:

```python
result = await shield.generate(
    prompt=(
        "A bacteria colony doubles every 3 hours. Starting with 100 bacteria, "
        "how many will there be after 24 hours? Derive the formula and calculate step by step."
    ),
    schema=Equation,
)
```

Output:

```
[FormatShield] model=groq/llama-3.1-70b-versatile
[FormatShield] complexity_score=0.748 (schema_depth=2, reasoning_ops=3, length_bucket=1)
[FormatShield] route=ttf | expected_delta=+0.170 | estimated_overhead=30%
[FormatShield] confidence=0.70 | explanation='Heuristic score 0.748 > threshold 0.650 for backend 'groq' → TTF.'
```

---

## 3. The Six Complexity Features

The `ComplexityScorer` computes these features and weights them into a scalar:

| Feature | Weight | Description |
|---|---|---|
| `token_entropy` | 20% | Normalised Shannon entropy of the prompt's token distribution |
| `schema_depth` | 25% | Maximum nesting depth of the JSON schema (capped at 10) |
| `required_reasoning_ops` | 20% | Count of CoT keywords (e.g. `solve`, `analyze`, `calculate`) |
| `instruction_tune_score` | 15% | Per-model RLHF strength (o1=1.0, GPT-4=0.8, Llama=0.5) |
| `prompt_length_bucket` | 10% | Token length bucket: 0=short, 1=medium, 2=long, 3=very long |
| `schema_constraint_count` | 10% | Total number of JSON Schema constraints (capped at 30) |

You can access the raw features by using `ComplexityScorer` directly:

```python
from formatshield.scorer.complexity_scorer import ComplexityScorer

scorer = ComplexityScorer()
features = scorer.score(
    prompt="Analyze and compare the two approaches, calculating efficiency.",
    schema={"type": "object", "properties": {"score": {"type": "number"}}, "required": ["score"]},
    model_id="groq/llama-3.1-70b-versatile",
)
scalar = scorer.compute_score(features)

print(f"token_entropy:          {features.token_entropy:.3f}")
print(f"schema_depth:           {features.schema_depth}")
print(f"required_reasoning_ops: {features.required_reasoning_ops}")
print(f"instruction_tune_score: {features.instruction_tune_score}")
print(f"prompt_length_bucket:   {features.prompt_length_bucket}")
print(f"schema_constraint_count:{features.schema_constraint_count}")
print(f"--- composite score: {scalar:.3f}")
```

---

## 4. Per-Backend Thresholds

FormatShield ships with empirically calibrated thresholds per backend. Lower threshold = more TTF:

| Backend | Default Threshold | TTF Overhead |
|---|---|---|
| `vllm` | 0.60 | ~10% (native KV-cache reuse) |
| `outlines` | 0.62 | ~20% |
| `guidance` | 0.63 | ~22% |
| `groq` | 0.65 | ~30% |
| `ollama` | 0.65 | ~25% |
| `openrouter` | 0.67 | ~35% |

vLLM gets a lower threshold because its native prefix caching keeps TTF overhead minimal — it's worth triggering TTF for slightly less complex requests on vLLM.

---

## 5. Controlling Routing with a Latency Budget

If you have a hard latency requirement (e.g. a real-time API with a 2-second SLA), pass `latency_budget_ms`. The oracle will suppress TTF if the estimated overhead would exceed the budget:

```python
shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    latency_budget_ms=1500,   # 1.5 second SLA
)
result = await shield.generate(prompt, schema=MySchema)
# Even if complexity_score is high, TTF will be skipped if its
# ~30% overhead on Groq would exceed 1500ms.
```

---

## 6. Native Thinker Models Always Use Direct

Some models already have built-in chain-of-thought reasoning (o1, o3, DeepSeek-R1). Running TTF on these would cause the model to "double-think" — the oracle automatically routes them to direct regardless of complexity score:

```python
# o1-mini has native thinking — TTF is always suppressed
result = await fs.generate(
    prompt="Complex multi-step reasoning task...",
    schema=MySchema,
    model="openrouter/openai/o1-mini",
)
print(result.routing.explanation)
# "Native thinker model detected – TTF would double-think."
```

The full list of native thinkers:

- `o1`, `o1-mini`, `o1-preview`
- `o3`, `o3-mini`
- `deepseek-r1` (and all distillation variants)

---

## 7. Disabling TTF Fallback

By default, if TTF's Pass 2 output fails Pydantic validation, FormatShield automatically retries with a single-pass direct generation. You can disable this:

```python
shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    ttf_fallback=False,  # raise on validation failure instead of retrying
)
```

When `ttf_fallback=False` and TTF validation fails:
- `result.schema_valid` will be `False`
- `result.parsed` may be `None` or a raw dict
- `result.fallback_triggered` will be `False`

---

## 8. Training the Oracle on Your Own Data

After running benchmarks, you can train a scikit-learn `LogisticRegression` oracle on your own accuracy data for a domain-specific threshold:

```python
from formatshield.oracle.threshold_oracle import ThresholdOracle

oracle = ThresholdOracle.from_benchmark_data(
    csv_path="benchmark_results/summary.csv",
    save=True,  # saves to oracle_data/threshold_oracle_v1.pkl
)

# Use the trained oracle in a FormatShield instance
shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")
shield._oracle = oracle  # inject your trained oracle
```

See [Tutorial 03: Benchmarking](03-benchmarking.md) for how to generate the CSV input.

---

## Next Steps

- [Tutorial 03: Benchmarking](03-benchmarking.md) — measure the format tax on your data
- [Explanation: Routing Algorithm](../explanation/routing-algorithm.md) — deep dive into the routing math
- [Reference: Oracle](../reference/oracle.md) — full `ThresholdOracle` API
