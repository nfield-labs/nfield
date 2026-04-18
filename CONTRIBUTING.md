# Contributing to FormatShield

Thank you for considering a contribution to FormatShield. Every bug fix, benchmark result, backend adapter, and documentation improvement makes the Format Tax easier for the world to measure and fix.

Your name goes in the paper acknowledgments. Let's build this together.

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

All unit tests must pass before you open a PR. If they don't on a clean checkout, please file a bug.

---

## Response Commitment

We take contributor time seriously.

| Channel | Response time |
|---------|--------------|
| Issues (bugs) | Acknowledged within 48 hours |
| Issues (features) | Triaged within 5 business days |
| Pull requests | First review within 72 hours |
| Security vulnerabilities | See [SECURITY.md](SECURITY.md) |
| Questions | Use [GitHub Discussions](https://github.com/formatshield/formatshield/discussions) |

---

## Architecture Overview

Understanding the data flow helps you know where to make changes:

```
Request
   │
   ▼
FormatShield.generate(prompt, schema, model)       ← src/formatshield/core.py
   │
   ├──► ComplexityScorer.score()                    ← src/formatshield/scorer/complexity_scorer.py
   │        6 features → float [0.0, 1.0]
   │
   ├──► FailureModeDetector.detect()                ← src/formatshield/ttf/failure_detector.py
   │        Detects when TTF would hurt
   │
   ├──► ThresholdOracle.route()                     ← src/formatshield/oracle/threshold_oracle.py
   │        Returns RoutingDecision (strategy: "ttf" | "direct")
   │
   ├──► [if TTF] TTFEngine.generate()               ← src/formatshield/ttf/engine.py
   │        Pass 1 (unconstrained) + Pass 2 (constrained)
   │
   └──► Backend.generate()                           ← src/formatshield/backends/
            Groq / OpenAI / Anthropic / Ollama / vLLM / Outlines / DryRun
   │
   ▼
GenerationResult(output, parsed, thinking, routing, complexity_score, ...)
```

For the complete AI contributor guide (commands, patterns, code style), see [CLAUDE.md](CLAUDE.md).

---

## Good First Issues

These are tagged [`good-first-issue`](https://github.com/formatshield/formatshield/labels/good-first-issue) and are genuinely doable in under a day:

### Backends (~ 45–60 lines each)

| Issue | File to create | Effort | What to do |
|-------|---------------|--------|------------|
| Add Cohere backend | `src/formatshield/backends/cohere_backend.py` | ~50 lines | Implement `Backend` protocol using the `cohere` SDK. Use `client.chat()` with JSON response format. Add unit test. |
| Add Mistral AI backend | `src/formatshield/backends/mistral_backend.py` | ~50 lines | Implement using `mistralai` SDK. Mistral supports JSON mode via `response_format`. Follow the GroqBackend pattern. |
| Add Together AI backend | `src/formatshield/backends/together_backend.py` | ~45 lines | Together AI has an OpenAI-compatible API. Reuse the OpenAIBackend with a custom `base_url`. |

### Oracle Improvements (~ 20–40 lines each)

| Issue | File | Effort | What to do |
|-------|------|--------|------------|
| Tune Φ formula coefficients | `src/formatshield/oracle/routing_score.py` | ~20 lines | Adjust the A, B, C constants in the Φ = 1 − exp(−(A·λ̃₂² + B·τ·λ̃₂ + C·ΔK)) formula for better routing calibration on your workload. |
| Add routing score unit tests | `tests/unit/test_routing_score.py` | ~35 lines | Verify Φ returns values in [0.0, 1.0], increases with deeper schemas, and that edge cases (empty schema, trivial prompt) return sensible scores. |

### Scorer Improvements (~ 20–25 lines)

| Issue | File | Effort | What to do |
|-------|------|--------|------------|
| Non-English ComplexityScorer test | `tests/unit/test_scorer_multilingual.py` | ~25 lines | Add tests verifying Arabic and Greek prompts score correctly. tiktoken handles Unicode — verify token fragmentation increases the score. |

### CLI (~ 30 lines)

| Issue | File | Effort | What to do |
|-------|------|--------|------------|
| Add `--format table` flag | `src/formatshield/cli.py` | ~30 lines | Add `--format` option to `formatshield generate` that prints the routing trace as a `rich` table to stdout. `rich` is already a dependency. |

### Tests (~ 25–30 lines each)

| Issue | File | Effort | What to do |
|-------|------|--------|------------|
| Streaming integration test | `tests/unit/test_streaming.py` | ~30 lines | Verify `StreamingEngine` yields at least one `output` event and exactly one `complete` event using `DryRunBackend`. |
| Threshold oracle calibration tests | `tests/unit/test_oracle_calibration.py` | ~25 lines | Test that `ThresholdOracle.route()` routes differently at different complexity thresholds. Verify conservative mode and Φ-score-based routing. |

---

## Code Standards

- **Python 3.11+** only — use modern type syntax (`list[str]` not `List[str]`)
- **Ruff** for linting: `uv run ruff check src/ tests/`
- **Pyright** for types: `uv run pyright src/`
- **pytest** for tests: every new function needs a meaningful test
- **No stubs** — every method must have a real implementation
- All tests in `tests/unit/` must pass without any API key or GPU

### Adding a Backend

1. Create `src/formatshield/backends/<name>_backend.py`
2. Implement the `Backend` protocol from `backends/protocol.py`
3. Add to `backends/__init__.py`
4. Add unit test in `tests/unit/test_<name>_backend.py` using `DryRunBackend` pattern
5. Add integration test in `tests/integration/test_<name>_backend.py` with `@pytest.mark.skipif` guard
6. Add optional dependency to `pyproject.toml` under `[project.optional-dependencies]`

Minimum backend implementation:

```python
from formatshield.backends.protocol import Backend
from formatshield.streaming.engine import StreamEvent

class MyBackend:
    name = "mybackend"
    supports_kv_cache_reuse = False
    accuracy_loss_baseline: float | None = None

    async def generate(self, prompt: str, schema: dict | None = None,
                       constraints: str | None = None, kv_cache_prefix: str | None = None,
                       temperature: float | None = None, max_tokens: int | None = None,
                       seed: int | None = None, top_p: float | None = None,
                       top_k: int | None = None, frequency_penalty: float | None = None,
                       presence_penalty: float | None = None, stop: list[str] | None = None) -> str:
        ...  # call your API here

    async def stream(self, prompt: str, schema: dict | None = None, constraints: str | None = None):
        yield StreamEvent(type="complete", content="done")
```

### Adding a Benchmark Task

> **Removed in v0.3.** Benchmark tasks have been removed in v0.3. The oracle now uses the Φ routing score — see `src/formatshield/oracle/routing_score.py`.

---

## Testing Philosophy

**Unit tests** (`tests/unit/`): Zero API keys, zero GPU, zero network. Use `DryRunBackend` for all generation.

**Integration tests** (`tests/integration/`): Require real API keys. Guarded with `@pytest.mark.skipif`.

**Meaningful vs theater**:

```python
# THEATER — proves nothing:
def test_scorer_init():
    scorer = ComplexityScorer()
    assert scorer is not None

# MEANINGFUL — proves behavior:
def test_nested_schema_scores_higher_than_flat():
    scorer = ComplexityScorer()
    flat = {"type": "string"}
    nested = {"type": "object", "properties": {"a": {"type": "object", "properties": {"b": {"type": "string"}}}}}
    assert scorer.score("test", nested) > scorer.score("test", flat)
```

---

## Development Tips

```bash
# Run only one test file
uv run pytest tests/unit/test_core.py -v

# Stop at first failure
uv run pytest tests/unit/ -x --tb=short

# Check complexity score without generating
python -c "
from formatshield.scorer.complexity_scorer import ComplexityScorer
scorer = ComplexityScorer()
print(scorer.score('Extract all medications', {'type': 'object'}))
"

# Use debug routing trace
python -c "
import formatshield as fs
shield = fs.FormatShield(model='dryrun/test', debug=True)
result = shield.generate_sync('test', schema={'type': 'string'})
"
```

---

## Commit Convention

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
feat: add Cohere backend with JSON-mode support
fix: handle None schema in ComplexityScorer
docs: add tutorial for streaming with TTF
test: add streaming integration test with DryRunBackend
bench: update GSM benchmark with Groq results
ci: add Python 3.13 to test matrix
```

---

## Pull Request Checklist

- [ ] All unit tests pass (`uv run pytest tests/unit/ -v`)
- [ ] Ruff passes (`uv run ruff check src/ tests/`)
- [ ] Pyright passes (`uv run pyright src/`)
- [ ] New public functions have docstrings and type annotations
- [ ] New functions have meaningful tests (not coverage theater)
- [ ] No hardcoded API keys
- [ ] CHANGELOG.md entry added under `[Unreleased]`
- [ ] PR description references the issue (`Closes #42`)

---

## Recognition

Contributors are listed in `CONTRIBUTORS.md`. Sustained contributors (3+ merged PRs) are invited to the maintainer team and acknowledged in the arXiv paper.

See [GOVERNANCE.md](GOVERNANCE.md) for the full governance model.

*"We need someone to write the Guidance backend adapter — it's one class, one interface, and your name goes in the paper acknowledgments."*
