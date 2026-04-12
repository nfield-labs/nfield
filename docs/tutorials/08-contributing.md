# Tutorial 08 — Contributing to FormatShield

Thank you for your interest in contributing. This page covers how to set up the development environment, run the test suite, and submit a pull request.

---

## Development Setup

### 1. Fork and Clone

```bash
git clone https://github.com/formatshield/formatshield.git
cd formatshield
```

### 2. Create a Virtual Environment

=== "uv (recommended)"

    ```bash
    uv venv
    source .venv/bin/activate   # Linux/macOS
    .venv\Scripts\activate      # Windows
    uv sync --all-extras
    ```

=== "pip"

    ```bash
    python -m venv .venv
    source .venv/bin/activate
    pip install -e ".[all,dev]"
    ```

### 3. Install Pre-commit Hooks

```bash
pre-commit install
```

This sets up:
- `ruff` — linting and formatting
- `mypy` — type checking
- `pytest` — test suite on staged files

---

## Project Structure

```
formatshield/
├── src/formatshield/
│   ├── __init__.py            # Public API
│   ├── core.py                # FormatShield class + GenerationResult
│   ├── _retry.py              # Exponential backoff retry
│   ├── backends/
│   │   ├── protocol.py        # Backend protocol (ABC)
│   │   ├── groq_backend.py
│   │   ├── openrouter_backend.py
│   │   ├── ollama_backend.py
│   │   ├── vllm_backend.py
│   │   ├── outlines_backend.py
│   │   ├── guidance_backend.py
│   │   └── dryrun_backend.py
│   ├── scorer/
│   │   ├── complexity_scorer.py
│   │   ├── schema_analyzer.py
│   │   └── features.py        # ComplexityFeatures, BenchmarkResult, StreamEvent
│   ├── oracle/
│   │   ├── threshold_oracle.py
│   │   └── routing_decision.py
│   ├── ttf/
│   │   ├── engine.py          # TTFEngine
│   │   ├── prompts.py         # build_think_prompt, build_format_prompt, extract_thinking
│   │   └── failure_detector.py
│   ├── benchmark/
│   │   ├── harness.py
│   │   ├── tasks.py           # GSMSymbolicTask, MedicalNERTask, TemplateFillTask
│   │   └── exporters.py       # CSVExporter
│   ├── observability/
│   │   ├── logger.py          # StructuredLogger
│   │   └── metrics.py         # MetricsCollector
│   └── integrations/
│       └── langchain.py       # FormatShieldRunnable
├── tests/
│   ├── test_core.py
│   ├── test_scorer.py
│   ├── test_oracle.py
│   ├── test_ttf_engine.py
│   ├── test_backends.py
│   ├── test_benchmark.py
│   └── test_streaming.py
├── docs/
├── mkdocs.yml
├── pyproject.toml
└── CHANGELOG.md
```

---

## Running Tests

```bash
# All tests (fast — uses DryRunBackend, no API keys needed)
pytest

# With coverage
pytest --cov=formatshield --cov-report=term-missing

# A specific test file
pytest tests/test_scorer.py -v

# Tests matching a name pattern
pytest -k "test_ttf" -v
```

!!! note "No API keys needed"
    The full test suite uses `DryRunBackend` and mocked HTTP responses. You do not need any API keys to run tests.

---

## Adding a New Backend

1. Create `src/formatshield/backends/mybackend_backend.py`
2. Implement the `Backend` protocol from `formatshield.backends.protocol`:

```python
from formatshield.backends.protocol import Backend
from formatshield.scorer.features import StreamEvent
from collections.abc import AsyncIterator
from typing import Any

class MyBackend:
    name = "mybackend"
    supports_kv_cache_reuse = False

    def __init__(self, model: str, **kwargs):
        self.model = model

    async def generate(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
    ) -> str:
        # Call your backend API here
        # Return raw JSON string
        ...

    async def stream(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        constraints: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        # Yield StreamEvent objects
        yield StreamEvent(type="output", token="...", backend=self.name)
        yield StreamEvent(type="complete", json={}, backend=self.name)
```

3. Register the backend in `core.py`'s `_build_backend()` function
4. Add the backend name to `BackendName` in `backends/protocol.py`
5. Write tests in `tests/test_backends.py`
6. Update the backends documentation in `docs/reference/backends.md`

---

## Adding a New Benchmark Task

1. Create a new task class in `src/formatshield/benchmark/tasks.py`:

```python
from pydantic import BaseModel
from typing import Any

class MyOutputSchema(BaseModel):
    answer: str
    confidence: float

class MyCustomTask:
    name = "my_task"
    expected_ttf_benefit = True  # or False for simple tasks
    schema = MyOutputSchema

    def get_problems(self, quick: bool = False) -> list[dict[str, Any]]:
        problems = [
            {"question": "Hard reasoning question 1?", "answer": "expected_answer_1"},
            {"question": "Hard reasoning question 2?", "answer": "expected_answer_2"},
            # ... more problems
        ]
        return problems[:5] if quick else problems

    def build_prompt(self, question: str) -> str:
        return f"Answer this question carefully: {question}"

    def score_response(self, response: dict[str, Any], ground_truth: Any) -> float:
        """Return a score in [0, 1]."""
        predicted = response.get("answer", "")
        if isinstance(ground_truth, str):
            return 1.0 if predicted.strip().lower() == ground_truth.strip().lower() else 0.0
        return 0.0
```

2. Register it in `BenchmarkHarness.run()`:

```python
_task_registry: dict[str, Any] = {
    "gsm_symbolic": GSMSymbolicTask(),
    "medical_ner": MedicalNERTask(),
    "template_fill": TemplateFillTask(),
    "my_task": MyCustomTask(),   # add your task here
}
```

---

## Code Style

FormatShield uses `ruff` for linting and formatting. The config is in `pyproject.toml`.

Key conventions:

- All public functions and classes must have docstrings
- Type annotations are required on all public APIs
- Use `from __future__ import annotations` in all modules
- Prefer `anyio` over `asyncio` directly where possible
- Error paths must always log via `logger.warning()` or `logger.error()` before returning fallbacks

Run the linter:

```bash
ruff check src/
ruff format src/
mypy src/formatshield/
```

---

## Submitting a Pull Request

1. Create a branch from `main`:
   ```bash
   git checkout -b feat/my-feature
   ```

2. Make your changes and add tests.

3. Run the full test suite:
   ```bash
   pytest --cov=formatshield
   ruff check src/
   mypy src/formatshield/
   ```

4. Update `CHANGELOG.md` under the `[Unreleased]` section.

5. Open a PR on GitHub. Use the provided PR template.

---

## Issue Labels

| Label | Meaning |
|---|---|
| `bug` | Something isn't working |
| `enhancement` | New feature or request |
| `backend:groq` | Groq backend specific |
| `backend:vllm` | vLLM backend specific |
| `ttf` | TTF engine related |
| `routing` | Scoring/oracle related |
| `benchmark` | Benchmark harness related |
| `docs` | Documentation only |
| `good first issue` | Good for newcomers |

---

## Code of Conduct

All contributors are expected to follow the [Contributor Covenant Code of Conduct](../CODE_OF_CONDUCT.md). Please read it before contributing.

---

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
