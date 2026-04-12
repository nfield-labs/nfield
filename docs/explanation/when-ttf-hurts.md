# When TTF Hurts

TTF is not always beneficial. For simple tasks, the extra latency of a second pass adds cost without accuracy gain. FormatShield's `FailureModeDetector` identifies these cases automatically.

---

## Failure Modes

### 1. Simple Extraction

**Condition:** Schema depth ≤ 1, no CoT keywords in prompt, high confidence output

**Example:** Extracting a user's name from a sentence.

```python
# TTF is wasteful here — direct is better
result = await fs.generate(
    "Extract the name from: 'Hi, I'm Alice.'",
    schema={"type": "object", "properties": {"name": {"type": "string"}}},
)
# result.routing.strategy == "direct"  — FormatShield routes correctly
```

Direct generation scores 91%+ on simple extraction. TTF adds 200–800ms overhead for 0% gain.

### 2. Over-Constrained Schema

**Condition:** Schema has many `enum`, `const`, or `pattern` constraints with low cardinality output space

**Example:** Classify sentiment into exactly ["positive", "negative", "neutral"].

The output space is so small that constrained decoding is essentially lossless — there's no room for FSM masking to hurt reasoning, because the output IS just a lookup.

### 3. Thinking-Averse Models

**Condition:** Small models (< 7B) that were not instruction-tuned for chain-of-thought

These models often produce poor-quality `<think>` outputs that don't help Pass 2. The thinking is noise rather than signal, and Pass 2 accuracy can actually drop.

**Detection:** `instruction_tune_score < 0.3` triggers this check.

### 4. High-Overhead Low-Gain

**Condition:** `overhead_pct > 80%` and `accuracy_delta < 0.05`

Detected post-run in `BenchmarkHarness`. Logged in `failure_modes_detected`.

---

## FailureModeDetector

```python
from formatshield.ttf.failure_detector import FailureModeDetector

detector = FailureModeDetector()
failure = detector.detect(
    prompt="Extract the name from this text.",
    schema={"type": "object", "properties": {"name": {"type": "string"}}},
    complexity_score=0.12,
)
if failure:
    print(failure.reason)  # "simple_extraction: schema depth 1, no CoT keywords"
```

The detector returns `None` when no failure mode is found (TTF is safe to proceed).

---

## Benchmark-Level Detection

`BenchmarkHarness` records failure modes per result:

```python
results = await harness.run(tasks=["template_fill"], backends=["groq"], ...)
for r in results:
    print(r.failure_modes_detected)
    # ["unnecessary_ttf_overhead"]  — TTF was applied but added no value
```

Failure modes detected:

| Mode | Meaning |
|---|---|
| `ttf_accuracy_regression` | TTF was expected to help but accuracy dropped |
| `unnecessary_ttf_overhead` | TTF not expected to help, but overhead > 30% |
| `high_overhead_low_gain` | Overhead > 80%, accuracy delta < 5% |
| `ttf_routing_error` | Oracle routed to TTF but it hurt accuracy |

---

## Recommendations

1. **Benchmark your own workload** — run `BenchmarkHarness` with your actual tasks and check `failure_modes_detected`
2. **Raise the threshold** for low-reasoning tasks: `ThresholdOracle(threshold=0.65)`
3. **Force direct** for known-simple schemas: `FormatShield(strategy="direct")`
4. **Use DryRunBackend** to test routing logic without API costs

---

## See Also

- [Routing Algorithm](routing-algorithm.md) — how the router works
- [TTF Algorithm](ttf-algorithm.md) — the two-pass approach
- [`FailureModeDetector` API](../reference/ttf.md)
- [`BenchmarkHarness` API](../reference/benchmark.md)
