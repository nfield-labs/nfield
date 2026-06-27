# Contributing to NField

Thanks for your interest in improving NField. This guide covers the
development workflow and the two most common extensions: adding a provider and
adding a pipeline stage.

## Development setup

NField uses [`uv`](https://github.com/astral-sh/uv).

```bash
uv sync --all-extras --dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ --strict
uv run pytest tests/unit/ -q
```

All four must pass before a PR is merged. Integration tests run only when
`GROQ_API_KEY` is set.

## Conventions

- `src/` layout; every public name is re-exported lazily from a facade `__init__.py`
  (no implementation in `__init__.py`).
- Modern typing (`str | None`, `list[int]`); `mypy --strict` clean, no `# type: ignore`.
- Optional parameters are keyword-only. No magic numbers - name every constant.
- Every raised error inherits from `NFieldError`.
- Comments cite published work or name algorithms; they do not reference internal
  design docs.

## Adding a provider

1. Implement the `LLMProvider` protocol (`providers/_protocol.py`):
   `complete`, `count_tokens`, and the `context_window` / `max_output_tokens` /
   `model_name` properties. Defer the SDK import to the client (keep
   `import nfield` dependency-free).
2. Register it in `providers/_registry.py` with a one-line prefix entry, or call
   `register_provider(prefix, module_path, class_name)`.
3. Add the SDK as an optional dependency in `pyproject.toml`.

## Adding a pipeline stage

1. Add `pipeline/sN_name.py` exposing a `run_stage_N(state, ...)` function over
   `PipelineState`. A field belongs on the state only if the next stage reads it.
2. Wire it into the stage sequence in `engine/_async.py`.
3. Add unit tests in `tests/unit/pipeline/`.

## Pull requests

- One logical change per PR; add a `CHANGELOG.md` entry under `[Unreleased]`.
- Include tests for new behaviour.
