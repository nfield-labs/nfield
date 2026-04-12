# Contributing to FormatShield

Thank you for considering a contribution to FormatShield. This document explains how to get started.

---

## Quick Start (< 5 minutes)

```bash
# 1. Fork and clone
git clone https://github.com/<you>/formatshield
cd formatshield

# 2. Install dependencies (requires Python 3.11+)
pip install uv
uv sync --all-extras

# 3. Install pre-commit hooks
uv run pre-commit install

# 4. Run tests (no GPU, no API key required)
uv run pytest tests/unit/ -v

# 5. You're ready to contribute
```

---

## Good First Issues

These are tagged `good-first-issue` on GitHub and are genuinely doable in < 1 day:

| Issue | File | Effort |
|-------|------|--------|
| Add Cohere backend | `src/formatshield/backends/cohere_backend.py` | ~50 lines |
| Add SQL extraction task | `src/formatshield/benchmark/tasks/sql_extraction.py` | ~30 lines |
| Non-English ComplexityScorer | `src/formatshield/scorer/complexity_scorer.py` | ~20 lines |
| Benchmark visualization | `src/formatshield/benchmark/exporters/png_exporter.py` | ~40 lines |
| Streaming integration test | `tests/unit/test_streaming.py` | ~30 lines |

---

## Code Standards

- **Python 3.11+** only — use modern type syntax (`list[str]` not `List[str]`)
- **Ruff** for linting: `uv run ruff check src/ tests/`
- **Pyright** for types: `uv run pyright src/`
- **pytest** for tests: every new function needs a test
- **No stubs** — every method must have a real implementation
- All tests in `tests/unit/` must pass without any API key or GPU

### Adding a Backend

1. Create `src/formatshield/backends/<name>_backend.py`
2. Implement the `Backend` protocol from `backends/protocol.py`
3. Add to `backends/__init__.py`
4. Add test in `tests/unit/test_<name>_backend.py` using `MockBackend` pattern
5. Add integration test in `tests/integration/test_<name>_backend.py` with `skipif` guard

Minimum backend implementation:
```python
from formatshield.backends.protocol import Backend
from formatshield.scorer.features import StreamEvent

class MyBackend:
    name = "mybackend"
    supports_kv_cache_reuse = False
    accuracy_loss_baseline = None

    async def generate(self, prompt, schema=None, constraints=None, kv_cache_prefix=None) -> str:
        ...

    async def stream(self, prompt, schema=None, constraints=None):
        yield StreamEvent(type="complete", content="done")
```

### Adding a Benchmark Task

1. Create `src/formatshield/benchmark/tasks/<name>.py`
2. Implement `get_problems(quick=False) -> list[dict]` and `score_response(predicted, ground_truth) -> float`
3. Add to `benchmark/tasks/__init__.py`
4. Tasks must work without external API calls (embed test data directly)

---

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):
- `feat: add Cohere backend`
- `fix: handle None schema in ComplexityScorer`
- `docs: update README with benchmark instructions`
- `test: add streaming integration test`
- `bench: update GSM benchmark results`

---

## Pull Request Checklist

- [ ] All unit tests pass (`uv run pytest tests/unit/ -v`)
- [ ] Ruff passes (`uv run ruff check src/ tests/`)
- [ ] New functions have tests
- [ ] No hardcoded API keys
- [ ] CHANGELOG entry added (conventional commit message is sufficient)

---

## Recognition

Contributors are listed in `CONTRIBUTORS.md`. Sustained contributors (3+ merged PRs) are invited to the maintainer team and acknowledged in the arXiv paper.

"We need someone to write the Guidance backend adapter — it's one class, one interface, and your name goes in the paper acknowledgments."
