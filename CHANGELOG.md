# Changelog

All notable changes to FormatShield are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.1.0] — Stage 1 complete

### Added
- `py.typed` marker — FormatShield is now PEP 561 compliant (typed package)
- `output_type` parameter on `FormatShield.generate()` and `generate_sync()`: pass `int`, `float`, `bool`, `str`, `Enum`, `Literal[...]`, or `list[T]` to get auto-typed results without writing a schema
- Sampling parameter pass-through on all generate calls: `temperature`, `max_tokens`, `seed`, `top_p`, `top_k`, `frequency_penalty`, `presence_penalty`, `stop`
- `FormatShieldGenerator` — reusable synchronous generator object (schema cached at construction, reused across calls)
- `AsyncFormatShieldGenerator` — async generator with `batch()` supporting `max_concurrency` for rate-limit-safe parallel calls
- `FormatShield.generator()` and `FormatShield.async_generator()` factory methods
- `from_provider(model, **kwargs)` unified factory — auto-detects provider from model string prefix

### Added

- **CohereBackend** — optional `cohere>=5.0.0` backend with JSON-mode and streaming support
- **MistralBackend** — optional `mistralai>=1.0.0` backend with structured output and streaming
- **TogetherBackend** — Together AI via OpenAI-compatible client (`openai` SDK, custom `base_url`)
- **SQLExtractionTask** — 15 NL-to-SQL benchmark problems (JOIN, aggregation, window functions, subqueries)
- **CodeExtractionTask** — 15 Python function entity-extraction benchmark problems
- All 11 benchmark tasks now registered in `BenchmarkHarness` and `tasks/__init__.py`
- `BackendName` Literal expanded: `cohere`, `mistral`, `together`, `openai`, `anthropic`, `fireworks`
- `core.py` `_build_backend()` resolves all new backend prefixes with lazy imports
- `examples/streaming_example.py` — four streaming demos (unstructured, Pydantic schema, dict schema, latency comparison)
- 40 new backend unit tests (Cohere, Mistral, Together, BackendName protocol routing)
- 61 new edge-case tests (ComplexityScorer, FailureModeDetector, ThresholdOracle, TTFEngine boundaries)

### Fixed

- `benchmarks.yml` CI: corrected `--dry-run` flag (does not exist) to `--backends dryrun --tasks gsm`
- `pyproject.toml`: `together` extra now correctly declares `openai>=1.0.0` (TogetherBackend uses the OpenAI SDK)
- `ci.yml`: removed duplicate `publish` job (handled by `release.yml`)

---

## [0.0.1] — 2026-04-12

### Added

- Core routing engine: `FormatShield.generate()` / `generate_sync()` / `stream()` with automatic TTF vs. direct routing
- TTF Engine: Two-pass Think-Then-Format from CRANE (arXiv 2502.09061) with Pydantic validation and fallback
- ComplexityScorer: 6-feature weighted scorer (token entropy, schema depth, reasoning ops, instruction-tune score, prompt-length bucket, schema constraint count)
- ThresholdOracle: heuristic per-backend threshold routing + optional sklearn LogisticRegression learned routing
- FailureModeDetector: prevents unnecessary TTF overhead on simple extraction and over-constrained schemas
- Backends: GroqBackend, OpenRouterBackend, OllamaBackend, VLLMBackend, OutlinesBackend, GuidanceBackend
- DryRunBackend: deterministic zero-dependency CI/testing backend with schema-driven JSON generation
- Retry utilities: `RetryConfig` + `with_retry()` with exponential backoff, jitter, and configurable retryable exceptions; applied to all HTTP backends
- BenchmarkHarness: real-backend benchmark orchestrator with CSV, JSONL, JSON, and LaTeX artifact export
- Benchmark tasks: GSMSymbolic, MedicalNER, TemplateFill, Classification, Financial, LegalExtract, AgentState, ToolCall, MATH500
- Streaming engine with `thinking` / `output` / `complete` `StreamEvent` types
- LangChain integration wrapper
- Observability: structured logging + metrics collection
- CLI: `formatshield generate`, `benchmark`, `score`, `stream`
- Full MkDocs Material documentation site
- GitHub Actions: ci.yml, docs.yml, security.yml, type-check.yml, benchmarks.yml
- Community: issue templates, PR template, discussion template, Code of Conduct

### Research Basis

- Format Tax: [arXiv 2408.02442](https://arxiv.org/abs/2408.02442) — up to 27% accuracy loss from constrained decoding
- CRANE / TTF: [arXiv 2502.09061](https://arxiv.org/abs/2502.09061) — unconstrained reasoning recovers accuracy

[0.0.1]: https://github.com/formatshield/formatshield/releases/tag/v0.0.1
