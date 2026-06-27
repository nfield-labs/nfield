# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project adheres to
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — 2026-06-27

### Added
- The seven-stage N-field extraction pipeline (S0–S6).
- Public API: `nfield`, `nfield_async`, `NField`, `AsyncNField`,
  `from_model`.
- Schema flattening (dict / Pydantic / dataclass), SFEP extraction, capacity packing,
  BM25 document pre-pass, per-field validation with surgical retry, and radix-trie
  assembly.
- Groq and OpenAI-compatible providers; the OpenAI provider reaches any
  OpenAI-compatible endpoint through `base_url` (OpenAI, Together, Fireworks,
  DeepSeek, vLLM, Ollama, LM Studio).
- Reasoning-model support: `ExtractionConfig(reasoning_model=True)` disables a
  model's thinking pass per call so it does not consume the answer's output budget.
- Command-line interface (`nfield extract`, `nfield inspect`) under the `[cli]` extra.
- Tabular export to pandas / CSV under the `[export]` extra.
- Documentation site (MkDocs Material) and runnable examples.

[Unreleased]: https://github.com/nfield-labs/nfield/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/nfield-labs/nfield/releases/tag/v0.1.0
