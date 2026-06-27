# Changelog

Notable changes to nfield. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-06-27

First public release.

### Added
- Extract hundreds of structured fields from a document. nfield splits a wide schema
  into groups that fit the model, retrieves the relevant part of the document for each,
  validates every value against the text, retries the ones that fail, and reassembles
  nested JSON.
- Schemas as a JSON Schema dict, a Pydantic model, or a dataclass.
- Sync and async API: `nfield`, `nfield_async`, `NField`, `AsyncNField`, plus
  `extract_batch` for running many documents through one reused engine.
- Groq and OpenAI-compatible providers. The OpenAI provider's `base_url` reaches any
  compatible endpoint, hosted (OpenAI, Together, Fireworks, DeepSeek) or local
  (Ollama, vLLM, LM Studio), so documents can stay on your machine.
- Reasoning-model support: `ExtractionConfig(reasoning_model=True)` turns off a model's
  thinking pass per call so it does not eat the answer's output budget.
- Command-line interface (`nfield extract`, `nfield inspect`) under the `[cli]` extra.
- Tabular export to pandas or CSV under the `[export]` extra.
- Fully typed (`py.typed`, `mypy --strict`) with no required dependencies; a provider SDK
  is pulled in only when you use it.
- Documentation site and runnable examples.

[Unreleased]: https://github.com/nfield-labs/nfield/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nfield-labs/nfield/releases/tag/v0.1.0
