# Contributing to nfield

Thanks for wanting to make nfield better. Issues and pull requests are both
welcome. This guide walks through the development workflow and the two changes
people make most often: adding a provider and adding a pipeline stage.

## Development setup

nfield uses [`uv`](https://github.com/astral-sh/uv). One sync pulls everything,
and these four checks are the same ones CI runs:

```bash
uv sync --all-extras --dev
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run mypy src/ --strict
uv run pytest tests/unit/ -q
```

All four have to pass before a PR is merged. The integration tests that call a real
model run only when `GROQ_API_KEY` is set, so the unit suite works offline.

## Conventions

A few house rules keep the codebase consistent:

- `src/` layout; every public name is re-exported lazily from a facade `__init__.py`
  (no implementation lives in `__init__.py`).
- Modern typing (`str | None`, `list[int]`); `mypy --strict` clean, no `# type: ignore`.
- Optional parameters are keyword-only. No magic numbers - give every constant a name.
- Every error we raise inherits from `NFieldError`.
- Comments cite published work or name the algorithm; they never point at internal
  design docs.

## Adding a provider

Most of the reach comes from one small class. To add another:

1. Implement the `LLMProvider` protocol (`providers/_protocol.py`): the async `complete`
   method plus the `context_window`, `max_output_tokens`, and `model_name` properties.
   Import the SDK inside the client, not at the top of the module, so `import nfield`
   stays dependency-free.
2. Register it in `providers/_registry.py` with a one-line prefix entry, or call
   `register_provider(provider_prefix, module_path, class_name)`.
3. Add the SDK as an optional dependency in `pyproject.toml`.

If the endpoint already speaks the OpenAI API, you may not need a class at all - add a
one-line preset in `providers/_presets.py` (base URL + key variable) and it routes
through the OpenAI provider.

## Adding a pipeline stage

1. Add `pipeline/sN_name.py` exposing a `run_stage_N(state, ...)` function over
   `PipelineState`. A field belongs on the state only if the next stage reads it.
2. Wire it into the stage sequence in `engine/_async.py`.
3. Add unit tests in `tests/unit/pipeline/`.

## Pull requests

- One logical change per PR; add a `CHANGELOG.md` entry under `[Unreleased]`.
- Include tests for new behaviour.
