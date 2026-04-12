# Reference — Benchmark

This page documents `BenchmarkHarness`, `BenchmarkResult`, and the built-in benchmark tasks from `formatshield.benchmark`.

---

## `BenchmarkHarness`

```python
class BenchmarkHarness:
    def __init__(
        self,
        output_dir: Path = Path("benchmark_results"),
        seed: int = 42,
    ) -> None: ...
```

Orchestrates FormatShield benchmarks across multiple tasks and backends.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `output_dir` | `Path` | `Path("benchmark_results")` | Root directory for all benchmark output. Created automatically if it doesn't exist. Also creates `output_dir/raw/` and `output_dir/artifacts/` subdirectories |
| `seed` | `int` | `42` | Seed for reproducible `DryRunBackend` instances |

---

### `BenchmarkHarness.run()`

```python
async def run(
    self,
    tasks: list[str],
    backends: list[str],
    models: dict[str, str],
    quick: bool = False,
    backend_objects: dict[str, Any] | None = None,
) -> list[BenchmarkResult]: ...
```

Run all task × backend combinations and return aggregated results.

| Parameter | Type | Description |
|---|---|---|
| `tasks` | `list[str]` | Task names to run: `"gsm_symbolic"`, `"medical_ner"`, `"template_fill"` |
| `backends` | `list[str]` | Backend names (e.g. `["groq", "ollama"]`) |
| `models` | `dict[str, str]` | Mapping of backend name → model identifier string |
| `quick` | `bool` | When `True`, each task uses its reduced problem set for fast runs |
| `backend_objects` | `dict[str, Any] \| None` | Optional mapping of backend name → backend instance. When a backend is not present, `DryRunBackend` is used automatically |

All (task, backend) pairs run concurrently via `asyncio.gather`. After completion:
- Raw results are written to `output_dir/raw/results_{timestamp}.jsonl`
- Summary CSV is written to `output_dir/summary.csv`

**Returns:** `list[BenchmarkResult]` — all individual problem-level results.

---

### `BenchmarkHarness.run_task_on_backend()`

```python
async def run_task_on_backend(
    self,
    task: Any,
    backend: str,
    model: str,
    quick: bool = False,
    backend_obj: Any | None = None,
) -> list[BenchmarkResult]: ...
```

Run a single task against a single backend. For each problem:

1. Run TTF → score → record TTF accuracy + latency
2. Run direct → score → record direct accuracy + latency
3. Compute `accuracy_delta` and `overhead_pct`
4. Detect failure modes
5. Return `BenchmarkResult`

| Parameter | Type | Description |
|---|---|---|
| `task` | `Any` | Task instance with `name`, `expected_ttf_benefit`, `get_problems(quick)`, and `score_response()` |
| `backend` | `str` | Backend identifier string |
| `model` | `str` | Model string |
| `quick` | `bool` | Use reduced problem set |
| `backend_obj` | `Any \| None` | Backend instance. Falls back to `DryRunBackend` if `None` |

---

### `BenchmarkHarness.generate_artifacts()`

```python
def generate_artifacts(
    self,
    results: list[BenchmarkResult],
) -> dict[str, Path]: ...
```

Generate paper-ready artifact files from a completed result set.

Output files (all under `output_dir/artifacts/`):

| Artifact Key | File | Description |
|---|---|---|
| `table1_accuracy_by_backend` | `table1_accuracy_by_backend.csv` | Backend × task accuracy comparison |
| `table2_failure_modes` | `table2_failure_modes.csv` | Rows where failure modes were detected |
| `summary_json` | `summary.json` | Machine-readable JSON summary |
| `table1_latex` | `table1_latex.tex` | LaTeX table code for papers |

**Returns:** `dict[str, Path]` mapping artifact name → absolute file path.

---

## `BenchmarkResult`

```python
@dataclass
class BenchmarkResult:
    task: str
    backend: str
    model: str
    direct_accuracy: float
    ttf_accuracy: float
    accuracy_delta: float
    direct_latency_ms: float
    ttf_latency_ms: float
    overhead_pct: float
    complexity_score: float
    failure_modes_detected: list[str]
```

### Fields

| Field | Type | Description |
|---|---|---|
| `task` | `str` | Task name (e.g. `"gsm_symbolic"`) |
| `backend` | `str` | Backend identifier |
| `model` | `str` | Model identifier string |
| `direct_accuracy` | `float` | Accuracy score [0, 1] for direct constrained generation |
| `ttf_accuracy` | `float` | Accuracy score [0, 1] for TTF two-pass generation |
| `accuracy_delta` | `float` | `ttf_accuracy - direct_accuracy`. Positive = TTF helped. The Format Tax on direct generation |
| `direct_latency_ms` | `float` | Wall-clock latency for direct path (milliseconds) |
| `ttf_latency_ms` | `float` | Wall-clock latency for TTF path (milliseconds) |
| `overhead_pct` | `float` | `(ttf_latency - direct_latency) / direct_latency × 100` |
| `complexity_score` | `float` | Task complexity score from `_compute_complexity_score()` |
| `failure_modes_detected` | `list[str]` | Failure mode labels detected by `_detect_failure_modes()` |

### `BenchmarkResult.to_dict()`

```python
def to_dict(self) -> dict[str, Any]: ...
```

Returns a JSON-serializable dictionary. Used for JSONL export.

---

## Built-in Tasks

### `GSMSymbolicTask`

```python
class GSMSymbolicTask:
    name = "gsm_symbolic"
    expected_ttf_benefit = True
    schema = GSMOutputSchema  # {steps: list[str], answer: float}
```

Symbolic math word problems. Complexity score: 0.82. These problems require multi-step arithmetic reasoning, which is exactly the case where constrained decoding hurts most (Format Tax up to 27%).

**Quick mode:** 5 problems. **Full mode:** 20 problems.

Scoring: exact match on the numeric `answer` field (within 0.01 tolerance).

---

### `MedicalNERTask`

```python
class MedicalNERTask:
    name = "medical_ner"
    expected_ttf_benefit = True
    schema = MedicalNEROutputSchema  # {entities: list[{text, label, confidence}]}
```

Medical named entity recognition from clinical text. Complexity score: 0.68. Extracting structured entities from domain-specific clinical language benefits from unconstrained reasoning in Pass 1.

**Quick mode:** 5 problems. **Full mode:** 20 problems.

Scoring: entity set F1 score (intersection of predicted vs ground-truth entity texts, case-insensitive).

---

### `TemplateFillTask`

```python
class TemplateFillTask:
    name = "template_fill"
    expected_ttf_benefit = False
    schema = TemplateFillOutputSchema  # {filled_template: str, variables: dict}
```

Template variable substitution. Complexity score: 0.15. A deliberately simple task used as a control. Direct constrained decoding should perform well here; if TTF shows large overhead with no accuracy gain on this task, it signals a routing calibration issue.

**Quick mode:** 5 problems. **Full mode:** 20 problems.

Scoring: exact match on all variable values.

---

## Failure Mode Detection

The harness automatically flags these patterns during benchmark runs:

| Mode | Trigger |
|---|---|
| `ttf_accuracy_regression` | `expected_ttf_benefit=True` AND `accuracy_delta < -0.05` |
| `unnecessary_ttf_overhead` | `expected_ttf_benefit=False` AND `overhead_pct > 30.0` |
| `high_overhead_low_gain` | `overhead_pct > 80.0` AND `accuracy_delta < 0.05` |
| `ttf_routing_error` | `expected_ttf_benefit=False` AND `accuracy_delta < -0.03` |

---

## Output File Formats

### Raw JSONL (`output_dir/raw/results_{timestamp}.jsonl`)

One line per `BenchmarkResult`, as JSON:

```json
{"task": "gsm_symbolic", "backend": "groq", "model": "groq/llama-3.1-70b-versatile", "direct_accuracy": 0.61, "ttf_accuracy": 0.78, "accuracy_delta": 0.17, "direct_latency_ms": 1234.5, "ttf_latency_ms": 1603.9, "overhead_pct": 30.0, "complexity_score": 0.82, "failure_modes_detected": []}
```

### Summary CSV (`output_dir/summary.csv`)

Aggregated statistics by task × backend:

```csv
task,backend,model,direct_accuracy,ttf_accuracy,accuracy_delta,overhead_pct,n_problems
gsm_symbolic,groq,groq/llama-3.1-70b-versatile,0.61,0.78,0.17,30.0,20
medical_ner,groq,groq/llama-3.1-70b-versatile,0.74,0.83,0.09,32.1,20
template_fill,groq,groq/llama-3.1-70b-versatile,0.95,0.94,-0.01,28.4,20
```

### Summary JSON (`output_dir/artifacts/summary.json`)

```json
{
  "run_date": "2026-04-12T14:30:22Z",
  "total_problems": 60,
  "backends": ["groq"],
  "tasks": ["gsm_symbolic", "medical_ner", "template_fill"],
  "aggregate": {
    "mean_accuracy_delta": 0.083,
    "mean_overhead_pct": 30.2,
    "failure_mode_count": 0
  },
  "by_task": {...},
  "by_backend": {...}
}
```
