# Changelog

All notable changes to FormatShield are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/).

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
