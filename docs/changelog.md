# Changelog

All notable changes to FormatShield are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) and [Semantic Versioning](https://semver.org/).

---

## [0.0.1] — 2026-04-12

### Added

- **Core routing engine** — `FormatShield.generate()` / `generate_sync()` / `stream()` with automatic TTF vs. direct routing via `ComplexityScorer` + `ThresholdOracle`
- **TTF Engine** — Two-pass Think-Then-Format implementation from CRANE (arXiv 2502.09061); Pass 1 unconstrained reasoning, Pass 2 constrained JSON formatting; optional Pydantic validation + fallback
- **ComplexityScorer** — Six-feature weighted scorer: token entropy (tiktoken), schema depth, reasoning ops (CoT keywords), instruction-tune score, prompt-length bucket, schema constraint count
- **ThresholdOracle** — Heuristic per-backend threshold routing + optional sklearn `LogisticRegression` learned routing
- **FailureModeDetector** — Detects simple-extraction, over-constrained, and thinking-averse failure modes to prevent unnecessary TTF overhead
- **Backend: GroqBackend** — Groq LPU API, JSON-mode support, exponential backoff retry on RateLimitError / InternalServerError
- **Backend: OpenRouterBackend** — OpenRouter unified proxy, OpenAI-compatible JSON-mode, retry on RateLimitError / InternalServerError / APIConnectionError
- **Backend: OllamaBackend** — Local Ollama server, native JSON-format mode, retry on ResponseError
- **Backend: VLLMBackend** — Self-hosted vLLM with KV-cache prefix reuse support, retry on RateLimitError / InternalServerError / APIConnectionError
- **Backend: OutlinesBackend** — Outlines-based constrained decoding (Transformers / vLLM)
- **Backend: GuidanceBackend** — Guidance grammar-based structured generation
- **Backend: DryRunBackend** — Deterministic zero-dependency CI/testing backend; generates structurally valid responses from JSON Schema without any API keys
- **Retry utilities** (`_retry.py`) — `RetryConfig` dataclass + `with_retry()` async function with truncated exponential backoff, optional jitter, configurable retryable exception types; `DEFAULT_RETRY` (3 attempts) and `API_RETRY` (5 attempts) pre-built configs
- **BenchmarkHarness** — Real-backend benchmark orchestrator; drives GSM-Symbolic, Medical NER, Template Fill tasks; generates CSV, JSONL, JSON summary, and LaTeX table artifacts; uses DryRunBackend automatically when no backend object is supplied
- **Benchmark Tasks** — `GSMSymbolicTask` (20 math problems), `MedicalNERTask`, `TemplateFillTask`, `ClassificationTask`, `FinancialTask`, `LegalExtractTask`, `AgentStateTask`, `ToolCallTask`, `MATH500Task`
- **Streaming Engine** — Async streaming via `AsyncIterator[StreamEvent]`; TTF streaming yields `thinking` / `output` / `complete` events
- **CSV / LaTeX / PNG exporters** — Paper-ready artifact generation from benchmark results
- **LangChain integration** — `FormatShieldLangChainLLM` wrapper for drop-in LangChain compatibility
- **Observability** — Structured logging (`ObservabilityLogger`) and metrics tracking (`MetricsCollector`)
- **CLI** — `formatshield` command: `generate`, `benchmark`, `score`, `stream` subcommands
- **Documentation** — Full MkDocs Material site with tutorials, reference API, and explanation pages
- **CI/CD** — GitHub Actions workflows: `ci.yml` (lint + test + publish), `docs.yml` (MkDocs deploy), `security.yml` (Bandit + pip-audit), `type-check.yml` (Pyright), `benchmarks.yml` (weekly benchmark regression)
- **Community** — Issue templates (bug report, feature request, backend request), PR template, discussion template, Code of Conduct

### Research Basis

- **Format Tax** paper: *"Let Me Speak Freely? A Study of LLM Responses to Constrained Output Formats"*, Tam et al., 2024 — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)
- **CRANE / TTF** paper: *"CRANE: Reasoning with constrained LLM generation"*, 2025 — [arXiv 2502.09061](https://arxiv.org/abs/2502.09061)
