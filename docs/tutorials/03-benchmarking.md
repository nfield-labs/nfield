# Tutorial 03 — Benchmarking the Format Tax

The `BenchmarkHarness` lets you empirically measure how much accuracy constrained decoding costs on your own tasks. This tutorial shows you how to run benchmarks, interpret the results, and generate paper-ready artifacts.

---

## What the Benchmark Measures

For every (task, backend) combination the harness runs each problem **twice**: once with TTF and once with direct constrained decoding. It computes:

- `direct_accuracy` — score when the model produces JSON directly
- `ttf_accuracy` — score when the model first thinks, then formats
- `accuracy_delta` = `ttf_accuracy - direct_accuracy` (**the Format Tax, sign-reversed**)
- `overhead_pct` — latency added by the TTF two-pass approach

---

## 1. Run a Dry-Run Benchmark (No API Key)

The `DryRunBackend` provides deterministic responses without any API calls, so you can test the full pipeline locally:

```python
import asyncio
from pathlib import Path
from formatshield.benchmark.harness import BenchmarkHarness
from formatshield.backends.dryrun_backend import DryRunBackend

async def main():
    harness = BenchmarkHarness(output_dir=Path("my_benchmark"))

    results = await harness.run(
        tasks=["gsm_symbolic", "medical_ner", "template_fill"],
        backends=["dryrun"],
        models={"dryrun": "dryrun/default"},
        backend_objects={"dryrun": DryRunBackend(seed=42)},
        quick=True,  # use small problem sets for fast testing
    )

    print(f"Total problems run: {len(results)}")
    for r in results[:3]:
        print(
            f"  {r.task:20s} | direct={r.direct_accuracy:.2f} "
            f"| ttf={r.ttf_accuracy:.2f} | delta={r.accuracy_delta:+.2f}"
        )

asyncio.run(main())
```

---

## 2. Run Against a Real Backend

To measure on real model responses, pass your backend objects:

=== "Groq"

    ```python
    import asyncio
    import os
    from pathlib import Path
    from formatshield.benchmark.harness import BenchmarkHarness
    from formatshield.backends.groq_backend import GroqBackend

    async def main():
        harness = BenchmarkHarness(output_dir=Path("benchmark_results"))

        groq = GroqBackend(
            api_key=os.environ["GROQ_API_KEY"],
            model="llama-3.1-70b-versatile",
        )

        results = await harness.run(
            tasks=["gsm_symbolic", "medical_ner"],
            backends=["groq"],
            models={"groq": "groq/llama-3.1-70b-versatile"},
            backend_objects={"groq": groq},
            quick=False,  # full problem set
        )

    asyncio.run(main())
    ```

=== "Ollama"

    ```python
    import asyncio
    from pathlib import Path
    from formatshield.benchmark.harness import BenchmarkHarness
    from formatshield.backends.ollama_backend import OllamaBackend

    async def main():
        harness = BenchmarkHarness(output_dir=Path("benchmark_results"))

        ollama = OllamaBackend(host="http://localhost:11434", model="llama3.1")

        results = await harness.run(
            tasks=["gsm_symbolic", "template_fill"],
            backends=["ollama"],
            models={"ollama": "ollama/llama3.1"},
            backend_objects={"ollama": ollama},
        )

    asyncio.run(main())
    ```

=== "Multiple Backends"

    ```python
    import asyncio
    import os
    from pathlib import Path
    from formatshield.benchmark.harness import BenchmarkHarness
    from formatshield.backends.groq_backend import GroqBackend
    from formatshield.backends.ollama_backend import OllamaBackend

    async def main():
        harness = BenchmarkHarness(output_dir=Path("benchmark_results"))

        results = await harness.run(
            tasks=["gsm_symbolic", "medical_ner", "template_fill"],
            backends=["groq", "ollama"],
            models={
                "groq": "groq/llama-3.1-70b-versatile",
                "ollama": "ollama/llama3.1",
            },
            backend_objects={
                "groq": GroqBackend(api_key=os.environ["GROQ_API_KEY"], model="llama-3.1-70b-versatile"),
                "ollama": OllamaBackend(model="llama3.1"),
            },
        )
        print(f"Ran {len(results)} problem × backend combinations")

    asyncio.run(main())
    ```

---

## 3. Understanding the Three Tasks

| Task | Expected TTF Benefit | Complexity Score | Description |
|---|---|---|---|
| `gsm_symbolic` | **Yes** (high) | 0.82 | Symbolic math word problems requiring multi-step reasoning |
| `medical_ner` | **Yes** (moderate) | 0.68 | Medical named entity recognition from clinical text |
| `template_fill` | **No** (simple) | 0.15 | Template variable substitution — no reasoning required |

`template_fill` is the **control task**: it is intentionally simple enough that constrained decoding should not hurt accuracy. If you see a large accuracy delta on this task, it suggests a confound in your benchmark setup.

---

## 4. Interpreting the Results

Each `BenchmarkResult` contains:

| Field | Type | Description |
|---|---|---|
| `task` | `str` | Task name |
| `backend` | `str` | Backend identifier |
| `model` | `str` | Model string |
| `direct_accuracy` | `float` | Score with direct constrained generation [0, 1] |
| `ttf_accuracy` | `float` | Score with TTF two-pass generation [0, 1] |
| `accuracy_delta` | `float` | `ttf_accuracy - direct_accuracy` |
| `direct_latency_ms` | `float` | Latency for direct path |
| `ttf_latency_ms` | `float` | Latency for TTF path |
| `overhead_pct` | `float` | Percentage latency increase from TTF |
| `complexity_score` | `float` | Task complexity score |
| `failure_modes_detected` | `list[str]` | Any failure modes flagged |

```python
# Aggregate accuracy delta per task
from collections import defaultdict

by_task = defaultdict(list)
for r in results:
    by_task[r.task].append(r.accuracy_delta)

for task, deltas in by_task.items():
    avg = sum(deltas) / len(deltas)
    print(f"{task:20s}: avg accuracy_delta = {avg:+.3f}")
```

---

## 5. Generate Artifacts

After running, call `generate_artifacts()` to produce paper-ready output files:

```python
artifacts = harness.generate_artifacts(results)

for name, path in artifacts.items():
    print(f"{name}: {path}")
```

Output files:

```
table1_accuracy_by_backend  → benchmark_results/artifacts/table1_accuracy_by_backend.csv
table2_failure_modes        → benchmark_results/artifacts/table2_failure_modes.csv
summary_json                → benchmark_results/artifacts/summary.json
table1_latex                → benchmark_results/artifacts/table1_latex.tex
```

The **LaTeX table** can be pasted directly into a paper. Example output:

```latex
\begin{table}[htbp]
\centering
\begin{tabular}{lcccc}
\toprule
Task & Backend & Direct Acc. & TTF Acc. & $\Delta$ \\
\midrule
gsm\_symbolic & groq & 0.61 & 0.78 & +0.17 \\
medical\_ner  & groq & 0.74 & 0.83 & +0.09 \\
template\_fill & groq & 0.95 & 0.94 & -0.01 \\
\bottomrule
\end{tabular}
\caption{Accuracy comparison: direct vs TTF}
\end{table}
```

---

## 6. Train the Oracle on Your Results

After collecting benchmark data, train a data-driven routing oracle:

```python
from formatshield.oracle.threshold_oracle import ThresholdOracle

# summary.csv is written automatically by harness.run()
oracle = ThresholdOracle.from_benchmark_data(
    csv_path="benchmark_results/summary.csv",
    save=True,
)

print("Oracle trained on your data.")
print("Model saved to: oracle_data/threshold_oracle_v1.pkl")
```

The trained oracle uses `LogisticRegression` on your problem's complexity features. On subsequent FormatShield calls, it will use your domain-specific routing boundary instead of the default heuristic thresholds.

---

## 7. Failure Mode Detection

The harness automatically flags four failure patterns:

| Failure Mode | Trigger Condition |
|---|---|
| `ttf_accuracy_regression` | TTF was expected to help but accuracy dropped by >5pp |
| `unnecessary_ttf_overhead` | TTF was not expected to help but added >30% latency |
| `high_overhead_low_gain` | TTF added >80% latency with <5pp accuracy gain |
| `ttf_routing_error` | TTF degraded accuracy on a simple task by >3pp |

```python
# Find all failure modes in your results
failures = [(r.task, r.backend, r.failure_modes_detected) for r in results if r.failure_modes_detected]
for task, backend, modes in failures:
    print(f"  {task}/{backend}: {modes}")
```

---

## 8. Raw JSONL Output

Every run writes timestamped JSONL to `output_dir/raw/`:

```
benchmark_results/raw/results_20260412T143022Z.jsonl
```

Each line is a JSON object from `BenchmarkResult.to_dict()`. You can load these for offline analysis:

```python
import json
from pathlib import Path

results_raw = []
for path in Path("benchmark_results/raw").glob("*.jsonl"):
    with path.open() as f:
        for line in f:
            results_raw.append(json.loads(line))

print(f"Total recorded results: {len(results_raw)}")
```

---

## Next Steps

- [Tutorial 04: Backends](04-backends.md) — run benchmarks on vLLM, Outlines, and Guidance
- [Reference: Benchmark](../reference/benchmark.md) — full `BenchmarkHarness` API
- [Explanation: Format Tax](../explanation/format-tax.md) — the research behind the accuracy numbers
