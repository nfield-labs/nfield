# Changelog

All notable changes to FormatShield are documented here.
This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and [Semantic Versioning](https://semver.org/).

---

## [Unreleased]

## [0.3.0] — 2026-04-18

### Changed
- **Oracle:** Replaced sklearn `CalibratedClassifierCV` with closed-form Φ(prompt, schema) routing score — zero training data, zero benchmark runs required
- **ThresholdOracle:** `from_benchmark_data()`, `save()`, `load()` removed; now raise `DeprecationWarning` + `NotImplementedError`

### Added
- `oracle/schema_graph.py` — Fiedler value λ̃₂ of JSON schema dependency graph (spectral graph theory)
- `oracle/schema_entropy.py` — Schema constraint tightness τ (information entropy of type tree)
- `oracle/ncd.py` — Normalized Compression Distance ΔK between prompt and schema (via zlib)
- `oracle/routing_score.py` — Φ = 1 − exp(−(A·λ̃₂² + B·τ·λ̃₂ + C·ΔK)) closed-form routing score
- `RoutingContext`: 4 new fields `phi_score`, `phi_lambda2`, `phi_tau`, `phi_delta_k`
- `docs/migration/oracle-v3.md` — migration guide for v0.2 → v0.3

### Removed
- `src/formatshield/benchmark/` — entire benchmark package (harness, tasks, exporters, judge)
- `src/formatshield/oracle/uncertainty.py`, `utility.py`, `adaptive.py`
- `formatshield benchmark` CLI command
- `.github/workflows/benchmarks.yml`
- `scikit-learn` and `joblib` dependencies
- `BenchmarkResult` from public API

---

## [0.2.0] — Stage 4: Production Hardening

### Added
- **TokenUsage dataclass** (`scorer/features.py`) — tracks `input_tokens`, `output_tokens`, `cached_tokens`, `total_tokens` (auto-computed), `ttft_ms`, `forward_passes`; `to_dict()` for serialisation
- **GenerationResult** extended with `token_usage: TokenUsage | None` and `cost_usd: float | None` fields; both included in `model_dump()`
- **Hook system** (`formatshield.hooks`) — `Hooks` class with `on()`, `off()`, `clear()`, `emit()`, `handler_count()`, `events()`; four lifecycle constants: `HOOK_COMPLETION_KWARGS`, `HOOK_COMPLETION_RESPONSE`, `HOOK_COMPLETION_ERROR`, `HOOK_PARSE_ERROR`; sync and async handler support; exceptions in handlers are logged and never propagated
- **FormatShield.hooks** parameter — pass a `Hooks` instance at construction; `completion:kwargs`, `completion:response`, and `completion:error` events are fired automatically on every `generate()` call
- **Reask / retry** (`_retry.py`) — `FailedAttempt` NamedTuple records each failed generation attempt; `FormatShieldRetryException` carries full attempt history; `build_reask_prompt()` constructs corrective prompts that feed back the invalid output and validation error
- **TTFEngine reask** — `max_reasks` parameter (default 2); on validation failure the engine retries up to `max_reasks` times with corrective prompts before falling back to direct generation
- **LLM Judge** (`benchmark/judge.py`) — `build_judge_prompt()` with 8 task-specific rubrics (gsm, medical_ner, legal_extract, financial, classification, gpqa, zebralogic, math500); `parse_verdict()` with reversal detection; `LLMJudge` class with SHA256 cache, optional disk persistence, sync `judge()` and async `ajudge()`
- **PrometheusMetrics** (`observability/metrics.py`) — real `prometheus_client` Counters and Histograms with graceful fallback to `MetricsCollector`; `serve_metrics(port)` and `generate_metrics_text()` module helpers; metric names: `formatshield_routing_decisions_total`, `formatshield_generation_latency_ms`, `formatshield_schema_validation_failures_total`, `formatshield_fallback_activations_total`, `formatshield_accuracy_delta`
- **OpenTelemetry tracing** (`observability/otel.py`) — `FormatShieldTracer` with `generation_span()` context manager and `set_result_attributes()`; `_NoOpSpan` fallback when `opentelemetry-api` is not installed; `get_tracer()` module-level singleton
- **Batch API** (`formatshield.batch`) — `BatchProcessor` with `submit()`, `status()`, `results()`, `cancel()`; `BatchJobInfo`, `BatchSuccess`, `BatchError`, `BatchStatus` types; `asyncio.Semaphore` concurrency control; SHA256 job IDs
- **CLI batch commands** — `formatshield batch submit`, `formatshield batch status`, `formatshield batch results`
- **CLI score command** — `formatshield score` outputs complexity score without generating
- Optional dependency groups: `prometheus = ["prometheus-client>=0.17.0"]`, `otel = ["opentelemetry-api>=1.20.0", "opentelemetry-sdk>=1.20.0"]`, `batch = []`

### Tests
- 151 new unit tests covering all Stage 4 components (Groups F, G, K, M, N)
- Total unit test count: 1730 passing, 3 skipped
- Coverage: 84.07% (above 80% threshold)

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
