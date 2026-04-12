## Description

<!-- A clear and concise summary of what this PR does and why. Link any related issues with "Closes #NNN" or "Fixes #NNN". -->

Closes #

---

## Type of Change

<!-- Check all that apply. -->

- [ ] Bug fix (non-breaking change that resolves a reported issue)
- [ ] New feature (non-breaking change that adds functionality)
- [ ] Breaking change (fix or feature that changes existing behaviour in a way that could affect users)
- [ ] Documentation update
- [ ] Refactor / code quality improvement (no functional change)
- [ ] CI / tooling / dependency update
- [ ] New backend adapter

---

## Changes Made

<!-- Bullet-point summary of the key changes. Be specific enough that a reviewer can follow along without reading every line. -->

-
-

---

## Testing

- [ ] New unit tests added that cover the changed code paths
- [ ] All existing tests pass locally (`uv run pytest`)
- [ ] If a new backend was added, integration smoke-test was run manually against the real API

**Test output (paste or summarise):**

```
# uv run pytest -v
```

---

## Documentation

- [ ] Docstrings updated for any changed public API
- [ ] `docs/` updated if user-facing behaviour changed
- [ ] `CHANGELOG.md` entry added under `[Unreleased]`

---

## Code Quality

- [ ] `ruff check src/ tests/` passes with no errors
- [ ] `ruff format --check src/ tests/` passes
- [ ] `pyright src/` reports no new errors (standard mode)

---

## Checklist Before Requesting Review

- [ ] PR title follows the format `type: short description` (e.g. `fix: handle empty response from Cohere adapter`)
- [ ] Branch is up to date with `main` (rebased or merged)
- [ ] No unintentional debug print statements or commented-out code left in
- [ ] Sensitive data (API keys, credentials) is not committed

---

## Screenshots / Benchmarks (if applicable)

<!-- For UI changes or performance-sensitive work, attach screenshots or paste benchmark output here. -->
