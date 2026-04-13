# CLAUDE.md — AI Contributor Guide for FormatShield

This file helps AI assistants (Claude Code, Cursor, Copilot) contribute effectively to FormatShield. It covers the architecture, conventions, and workflows needed to make good contributions without asking clarifying questions.

---

## What FormatShield Does

FormatShield fixes the **Format Tax**: grammar-constrained decoding costs LLMs up to 27% accuracy on reasoning tasks. It does this by implementing the **Think-Then-Format (TTF)** algorithm — a two-pass generation approach where the model reasons freely in Pass 1, then formats the output in Pass 2.

**Core insight**: Constrained decoding is not a neutral operation. Every FSM mask applied to the vocabulary distorts the model's output distribution. FormatShield routes around this distortion when the complexity of a task makes it worth the overhead, and measures the benefit empirically.

**Key papers**:
- arXiv 2408.02442 (EMNLP 2024) — quantifies the problem
- arXiv 2502.09061 (ICML 2025, CRANE) — demonstrates TTF as the fix
- arXiv 2604.03616 (Format Tax, 2026) — confirms across 6 models

---

## Development Commands

```bash
# Setup (one time)
pip install uv
uv sync --all-extras
uv run pre-commit install

# Run unit tests (no API keys, no GPU required)
uv run pytest tests/unit/ -v

# Run a single test file
uv run pytest tests/unit/test_core.py -v

# Run a single test
uv run pytest tests/unit/test_scorer.py::test_nested_schema_scores_higher -v

# Run all tests with coverage
uv run pytest tests/ --cov=src/formatshield --cov-report=term-missing

# Lint (must pass before PR)
uv run ruff check src/ tests/
uv run ruff format src/ tests/

# Type check (must pass before PR)
uv run pyright src/

# Security scan
uv run bandit -r src/formatshield/ -ll

# Build docs
uv run mkdocs build --strict

# Serve docs locally
uv run mkdocs serve

# CLI
uv run formatshield --help
uv run formatshield generate --prompt "Hello" --schema '{"type":"object","properties":{"reply":{"type":"string"}}}' --model dryrun/test
uv run formatshield benchmark --tasks gsm --backends dryrun --quick
```

---

## Project Structure

```
formatshield/
├── src/formatshield/
│   ├── __init__.py              # Public API exports
│   ├── core.py                  # FormatShield class — the main entry point
│   ├── cli.py                   # Typer CLI (generate, benchmark, score, stream)
│   ├── _retry.py                # RetryConfig, with_retry() utility
│   │
│   ├── backends/
│   │   ├── protocol.py          # Backend Protocol — the interface all backends implement
│   │   ├── dryrun_backend.py    # DryRunBackend — deterministic, no dependencies (use in all unit tests)
│   │   ├── groq_backend.py      # GroqBackend
│   │   ├── openai_backend.py    # OpenAIBackend
│   │   ├── anthropic_backend.py # AnthropicBackend
│   │   ├── openrouter_backend.py
│   │   ├── ollama_backend.py
│   │   ├── vllm_backend.py      # vLLMBackend — KV cache reuse support
│   │   ├── outlines_backend.py  # OutlinesBackend
│   │   └── guidance_backend.py  # GuidanceBackend
│   │
│   ├── scorer/
│   │   ├── complexity_scorer.py # ComplexityScorer — 6-feature weighted scoring (returns 0.0–1.0)
│   │   ├── features.py          # Feature definitions and StreamEvent type
│   │   └── schema_analyzer.py   # Schema depth, constraint count analysis
│   │
│   ├── oracle/
│   │   ├── threshold_oracle.py  # ThresholdOracle — heuristic or sklearn LogisticRegression routing
│   │   ├── routing_decision.py  # RoutingDecision dataclass
│   │   └── oracle_data/         # Pre-trained oracle data files
│   │
│   ├── ttf/
│   │   ├── engine.py            # TTFEngine — the two-pass generation core
│   │   ├── failure_detector.py  # FailureModeDetector — when NOT to use TTF
│   │   └── prompts.py           # Prompt templates for Pass 1 and Pass 2
│   │
│   ├── benchmark/
│   │   ├── harness.py           # BenchmarkHarness — orchestrates runs, collects results
│   │   ├── cross_backend.py     # CrossBackendBenchmark — measures format tax per backend
│   │   ├── tasks/               # 9 benchmark tasks (gsm_symbolic, medical_ner, etc.)
│   │   └── exporters/           # CSV, LaTeX, PNG exporters for paper artifacts
│   │
│   ├── streaming/
│   │   └── engine.py            # StreamingEngine — async SSE-compatible generator
│   │
│   ├── integrations/
│   │   └── langchain.py         # FormatShieldLLM LangChain wrapper
│   │
│   └── observability/
│       ├── logger.py            # Structured logging (structlog)
│       └── metrics.py           # Metrics collection
│
├── tests/
│   ├── unit/                    # No API keys, no GPU. Use DryRunBackend.
│   └── integration/             # Require API keys. Guarded with @pytest.mark.skipif.
│
├── docs/                        # MkDocs Material documentation
├── examples/                    # Working example scripts
├── .github/workflows/           # CI: ci.yml, docs.yml, security.yml, type-check.yml, benchmarks.yml
└── pyproject.toml               # Project config, dependencies, build system
```

---

## Core Architecture

### Data Flow

```
User calls fs.generate(prompt, schema, model)
    │
    ▼
FormatShield (core.py)
    │
    ├── ComplexityScorer.score(prompt, schema) → float [0.0, 1.0]
    │       6 features: token_entropy, schema_depth, reasoning_ops,
    │                   instruction_tune_score, prompt_length_bucket,
    │                   schema_constraint_count
    │
    ├── FailureModeDetector.detect(prompt, schema, model) → list[FailureMode]
    │       Detects: simple_extraction, over_constrained_schema,
    │                native_thinker, short_prompt, template_fill,
    │                ambiguous_schema
    │
    ├── ThresholdOracle.route(complexity, backend, failure_modes) → RoutingDecision
    │       Returns: strategy ("ttf" | "direct"), expected_accuracy_delta,
    │                expected_overhead_pct, confidence, explanation
    │
    └── [if strategy == "ttf"]
            TTFEngine.generate(prompt, schema, backend) → str
                Pass 1: backend.generate(think_prompt)       ← unconstrained
                Pass 2: backend.generate(format_prompt, schema) ← constrained
        [if strategy == "direct"]
            backend.generate(prompt, schema) → str            ← constrained
    │
    ▼
GenerationResult(output, parsed, thinking, routing, complexity_score,
                 failure_modes, latency_ms, schema_valid, fallback_triggered)
```

### The Backend Protocol

Every backend implements `src/formatshield/backends/protocol.py`:

```python
class Backend(Protocol):
    name: str
    supports_kv_cache_reuse: bool
    accuracy_loss_baseline: float | None

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

`DryRunBackend` is a deterministic implementation that returns a valid JSON blob matching any schema — use it for ALL unit tests. It has zero external dependencies.

---

## Code Style Guidelines

### Types and Annotations

```python
# ALWAYS: use Python 3.11+ syntax
def score(self, prompt: str, schema: dict[str, Any]) -> float: ...
list[str]        # not List[str]
str | None       # not Optional[str]
dict[str, Any]   # not Dict[str, Any]

# ALWAYS: annotate return types, even for simple methods
def is_thinker(self, model: str) -> bool: ...
```

### Error Handling

```python
# BAD: silent failures
try:
    result = call_api()
except Exception:
    pass

# GOOD: structured domain errors with context
try:
    result = call_api()
except httpx.TimeoutException as e:
    logger.warning("Backend timeout", error=str(e), backend=self.name)
    raise BackendTimeoutError(self.name, self.timeout) from e
```

### Constants Over Magic Numbers

```python
# BAD
if complexity > 0.75:
    use_ttf()

# GOOD
DEFAULT_TTF_THRESHOLD = 0.75
if complexity > DEFAULT_TTF_THRESHOLD:
    use_ttf()
```

### Docstrings (Google style on all public methods)

```python
def score(self, prompt: str, schema: dict[str, Any]) -> float:
    """Compute routing complexity score for a generation request.

    Higher scores indicate the request will likely suffer format tax
    under constrained decoding and should be routed to TTF.

    Args:
        prompt: The user prompt. Used for token entropy via tiktoken.
        schema: JSON schema dict for output structure analysis.

    Returns:
        Float in [0.0, 1.0]. 0.0 = trivial, 1.0 = maximum complexity.

    Raises:
        ValueError: If prompt is empty or schema is not a dict.

    Example:
        >>> scorer = ComplexityScorer()
        >>> score = scorer.score("What is 2+2?", {"type": "object"})
        >>> 0.0 <= score <= 1.0
        True
    """
```

### Error Messages (tell users what to do)

```python
# BAD
raise ValueError("invalid model")

# GOOD
raise ValueError(
    "Model string must follow 'provider/model-name' format. "
    f"Got: '{model}'. Examples: 'groq/llama-3.1-70b-versatile', "
    "'openai/gpt-4o-mini', 'ollama/llama3.1'"
)
```

---

## Testing Guidelines

### Unit Tests — Rules

1. **Never use real API backends** in unit tests — always use `DryRunBackend`
2. Every new public function needs at least one meaningful test
3. Tests must pass with **no API keys** and **no GPU**
4. Use `pytest.mark.skipif` for integration tests

```python
# GOOD unit test pattern
from formatshield.backends.dryrun_backend import DryRunBackend
from formatshield.core import FormatShield

def test_ttf_routing_for_complex_prompt():
    shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
    result = shield.generate_sync(
        prompt="Solve step by step: " + "complex reasoning " * 20,
        schema={"type": "object", "properties": {"answer": {"type": "number"}}}
    )
    # Complex prompt + reasoning ops → should route to TTF
    assert result.routing.strategy == "ttf"
    assert result.schema_valid is True
```

### What a Meaningful Test Looks Like

```python
# COVERAGE THEATER (counts toward coverage, proves nothing):
def test_scorer_init():
    scorer = ComplexityScorer()
    assert scorer is not None

# MEANINGFUL TEST (verifies behavior):
def test_nested_schema_scores_higher_than_flat():
    scorer = ComplexityScorer()
    flat = {"type": "string"}
    nested = {
        "type": "object",
        "properties": {
            "a": {"type": "object", "properties": {"b": {"type": "string"}}}
        }
    }
    assert scorer.score("test", nested) > scorer.score("test", flat)
```

### Integration Test Pattern

```python
import pytest

@pytest.mark.skipif(
    not os.environ.get("GROQ_API_KEY"),
    reason="GROQ_API_KEY not set"
)
async def test_groq_backend_live():
    backend = GroqBackend(api_key=os.environ["GROQ_API_KEY"])
    result = await backend.generate("Say hello", schema=None)
    assert len(result) > 0
```

---

## Adding a New Backend

### Step 1: Create the backend file

`src/formatshield/backends/<name>_backend.py`:

```python
from __future__ import annotations

import os
from typing import Any, AsyncIterator

from formatshield.backends.protocol import Backend
from formatshield.streaming.engine import StreamEvent
from formatshield._retry import RetryConfig, with_retry


class MyBackend:
    """Backend adapter for MyProvider.

    Implements the Backend protocol for use with FormatShield routing.

    Args:
        api_key: MyProvider API key. Defaults to MY_API_KEY env var.
        model: Model identifier (without 'myprovider/' prefix).
        retry_config: Retry configuration for transient failures.
    """

    name = "myprovider"
    supports_kv_cache_reuse = False
    accuracy_loss_baseline: float | None = None  # Set after benchmarking

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "my-default-model",
        retry_config: RetryConfig | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("MY_API_KEY", "")
        self._model = model
        self._retry = retry_config or RetryConfig()

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        """Generate a completion from MyProvider.

        Args:
            prompt: The full prompt to send.
            schema: JSON schema for structured output mode.
            constraints: Grammar constraints string (if supported).
            kv_cache_prefix: KV cache prefix for reuse (ignored if unsupported).

        Returns:
            Raw string response from the model.

        Raises:
            BackendError: If the API call fails after retries.
        """
        # Implementation here
        ...

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream completion tokens from MyProvider."""
        # Yield StreamEvent(type="output", content=token) for each token
        # Yield StreamEvent(type="complete", content=full_text) at end
        yield StreamEvent(type="complete", content="")
```

### Step 2: Register in `__init__.py`

`src/formatshield/backends/__init__.py`:

```python
from formatshield.backends.my_backend import MyBackend  # add this line
```

### Step 3: Add to core.py backend resolution

In `src/formatshield/core.py`, add `"myprovider"` to the backend resolution map.

### Step 4: Write unit tests

`tests/unit/test_my_backend.py`:

```python
"""Unit tests for MyBackend — no API keys required."""
import pytest
from formatshield.backends.my_backend import MyBackend


def test_my_backend_name():
    backend = MyBackend(api_key="fake-key")
    assert backend.name == "myprovider"


def test_my_backend_has_required_attributes():
    backend = MyBackend(api_key="fake-key")
    assert hasattr(backend, "supports_kv_cache_reuse")
    assert hasattr(backend, "accuracy_loss_baseline")
```

### Step 5: Write integration test

`tests/integration/test_my_backend_integration.py`:

```python
import os
import pytest

@pytest.mark.skipif(not os.environ.get("MY_API_KEY"), reason="MY_API_KEY not set")
async def test_my_backend_live_generate():
    from formatshield.backends.my_backend import MyBackend
    backend = MyBackend()
    result = await backend.generate("Say hello in one word.")
    assert isinstance(result, str)
    assert len(result) > 0
```

### Step 6: Add optional dependency

`pyproject.toml`:

```toml
[project.optional-dependencies]
myprovider = ["myprovider-sdk>=1.0.0"]
```

---

## Adding a New Benchmark Task

### Step 1: Create the task file

`src/formatshield/benchmark/tasks/<name>.py`:

```python
"""<Name> benchmark task for measuring format tax.

Tests FormatShield's ability to [describe what this measures].
"""
from __future__ import annotations

from typing import Any


def get_problems(quick: bool = False) -> list[dict[str, Any]]:
    """Return benchmark problems.

    Args:
        quick: If True, return a small subset for CI/smoke tests.

    Returns:
        List of dicts with keys: 'prompt', 'ground_truth', 'schema'.
        All data must be embedded — no external API calls allowed.
    """
    problems = [
        {
            "prompt": "Extract the medication name from: Patient takes Aspirin 100mg daily.",
            "ground_truth": {"medication": "Aspirin", "dose": "100mg"},
            "schema": {
                "type": "object",
                "properties": {
                    "medication": {"type": "string"},
                    "dose": {"type": "string"},
                },
                "required": ["medication", "dose"],
            },
        },
        # Add more problems...
    ]
    return problems[:2] if quick else problems


def score_response(predicted: str, ground_truth: Any) -> float:
    """Score a model response against the ground truth.

    Args:
        predicted: Raw string output from the model.
        ground_truth: Ground truth from get_problems().

    Returns:
        Float in [0.0, 1.0] where 1.0 = perfect match.
    """
    import json
    try:
        parsed = json.loads(predicted)
    except json.JSONDecodeError:
        return 0.0

    if not isinstance(ground_truth, dict):
        return 0.0

    correct = sum(
        1 for k, v in ground_truth.items()
        if parsed.get(k, "").lower() == str(v).lower()
    )
    return correct / len(ground_truth)
```

### Step 2: Register in `tasks/__init__.py`

```python
from formatshield.benchmark.tasks.my_task import get_problems, score_response  # add
```

### Step 3: Write tests for the task

```python
def test_my_task_get_problems_quick():
    from formatshield.benchmark.tasks.my_task import get_problems
    problems = get_problems(quick=True)
    assert len(problems) >= 1
    for p in problems:
        assert "prompt" in p
        assert "ground_truth" in p
        assert "schema" in p

def test_my_task_score_perfect():
    from formatshield.benchmark.tasks.my_task import score_response
    gt = {"medication": "Aspirin", "dose": "100mg"}
    predicted = '{"medication": "Aspirin", "dose": "100mg"}'
    assert score_response(predicted, gt) == 1.0

def test_my_task_score_invalid_json():
    from formatshield.benchmark.tasks.my_task import score_response
    assert score_response("not json", {}) == 0.0
```

---

## Commit and PR Guidelines

### Conventional Commits (required)

```
feat: add Cohere backend with JSON-mode support
fix: handle None schema in ComplexityScorer.score()
docs: add tutorial for streaming with TTF
test: add parametrized tests for all backends
bench: update GSM benchmark with Groq results
ci: add Python 3.13 to test matrix
refactor: extract schema validation into SchemaValidator class
```

### PR Rules

1. One PR = one thing (backend, feature, fix, docs)
2. Keep PRs small — reviewable in 30 minutes max
3. All unit tests must pass: `uv run pytest tests/unit/ -v`
4. Ruff must pass: `uv run ruff check src/ tests/`
5. Pyright must pass: `uv run pyright src/`
6. Add a CHANGELOG.md entry under `[Unreleased]`
7. Reference the issue: `Closes #42` in the PR description

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/formatshield/core.py` | Main `FormatShield` class and module-level `generate()` function |
| `src/formatshield/backends/protocol.py` | `Backend` Protocol — the contract all backends must implement |
| `src/formatshield/backends/dryrun_backend.py` | Use this in ALL unit tests |
| `src/formatshield/scorer/complexity_scorer.py` | `ComplexityScorer.score()` — the 6-feature routing score |
| `src/formatshield/oracle/threshold_oracle.py` | `ThresholdOracle.route()` — makes the TTF vs direct decision |
| `src/formatshield/ttf/engine.py` | `TTFEngine.generate()` — the two-pass generation |
| `src/formatshield/ttf/failure_detector.py` | `FailureModeDetector` — detects when NOT to use TTF |
| `src/formatshield/benchmark/harness.py` | `BenchmarkHarness` — orchestrates benchmark runs |
| `src/formatshield/cli.py` | CLI entry points (generate, benchmark, score, stream) |
| `tests/unit/` | All unit tests — must pass without API keys |
| `tests/integration/` | Live API tests — guarded with skipif |
| `.github/workflows/ci.yml` | Main CI: lint, typecheck, test matrix, publish |
| `pyproject.toml` | All project config, dependencies, optional extras |

---

## Development Tips

```python
# Check complexity score without generating:
from formatshield.scorer.complexity_scorer import ComplexityScorer
scorer = ComplexityScorer()
score = scorer.score(prompt="Your prompt here", schema={"type": "object", ...})
print(f"Complexity: {score:.3f}")  # 0.0–1.0

# Use debug mode to see routing decisions:
import formatshield as fs
shield = fs.FormatShield(model="dryrun/test", debug=True)
result = shield.generate_sync("Your prompt", schema=YourSchema)
# Prints: [FormatShield] complexity_score=... route=... expected_delta=...

# Run only failing tests:
uv run pytest tests/unit/ -v -x --tb=short  # -x stops at first failure

# Check what a backend returns with DryRunBackend:
from formatshield.backends.dryrun_backend import DryRunBackend
import asyncio
backend = DryRunBackend()
result = asyncio.run(backend.generate("test", schema={"type": "string"}))
print(result)
```
