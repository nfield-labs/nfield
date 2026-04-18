# FormatShield Roadmap

## v0.0.1 (Released — Day 1-2)

**Goal:** `pip install formatshield` works. Groq + OpenRouter + Ollama + vLLM backends. Real benchmark numbers.

- [x] ComplexityScorer (6 features)
- [x] ThresholdOracle (heuristic v0)
- [x] TTF Engine (two-pass generation)
- [x] FailureModeDetector (6 modes)
- [x] GroqBackend (with exponential backoff retry)
- [x] OpenRouterBackend (with exponential backoff retry)
- [x] OllamaBackend (with exponential backoff retry)
- [x] VLLMBackend (with exponential backoff retry)
- [x] OpenAIBackend (GPT-4o, o1, o3 series)
- [x] AnthropicBackend (Claude 3.5 Sonnet/Haiku/Opus)
- [x] DryRunBackend (deterministic CI testing, no API key)
- [x] StreamingEngine (SSE-compatible)
- [x] GSM-Symbolic + Medical NER benchmark tasks (removed in v0.3)
- [x] `formatshield generate` CLI
- [x] `formatshield benchmark --tasks gsm --backends groq --quick` CLI (removed in v0.3)
- [x] Debug mode with routing trace
- [x] TTF fallback on schema validation failure
- [x] Exponential backoff retry (`RetryConfig` + `with_retry()`)
- [x] MIT license, CONTRIBUTING.md, 5 Good First Issues
- [x] MkDocs documentation (23 pages)
- [x] CI/CD workflows (ci.yml, docs.yml, security.yml, type-check.yml, benchmarks.yml — benchmarks.yml removed in v0.3)

**Good First Issues (open at launch):**
1. `good-first-issue` Add Cohere backend
2. `good-first-issue` Add SQL extraction benchmark task (removed in v0.3 — benchmark module dropped)
3. `good-first-issue` Improve ComplexityScorer for non-English prompts
4. `good-first-issue` Add benchmark visualization (PNG exporter) (removed in v0.3 — benchmark module dropped)
5. `good-first-issue` Write streaming integration test

---

## v0.1.0 (Next — Day 3-7)

**Goal:** Paper companion release. arXiv submission. Show HN post. LangChain integration.

- [ ] Together AI backend (`pip install formatshield[together]`)
- [ ] Fireworks AI backend (`pip install formatshield[fireworks]`)
- [ ] Mistral AI direct backend (`pip install formatshield[mistral]`)
- [ ] OutlinesBackend (Phase 2 — improved schema coverage)
- [ ] GuidanceBackend (Phase 2 — full implementation)
- [x] Full 12-task BenchmarkHarness (removed in v0.3 — benchmark module dropped)
- [x] CrossBackendBenchmark with LaTeX/CSV/PNG export (removed in v0.3 — benchmark module dropped)
- [x] `formatshield benchmark --reproduce-paper` command (removed in v0.3 — benchmark CLI command dropped)
- [x] ThresholdOracle v1 (retrained on real benchmark data) (superseded in v0.3 — replaced by closed-form Φ oracle, zero training required)
- [ ] arXiv paper draft in `/paper/` directory
- [ ] LangChain integration (`FormatShieldLLM` + `FormatShieldLangChain`)
- [ ] Prometheus metrics + structured logger
- [ ] Streaming in TTF mode (Pass 1 streams thinking, Pass 2 streams JSON)
- [ ] Cost tracking for multi-step TTF (Groq + OpenRouter pricing APIs)

---

## v0.2.0 (Community milestone)

**Goal:** First community contributions merged. Oracle trained on real data.

- [x] ThresholdOracle v2 trained on community benchmark contributions (superseded in v0.3 — replaced by closed-form Φ oracle, zero training required)
- [x] `formatshield benchmark upload --anonymized` community data sharing (removed in v0.3 — benchmark module dropped)
- [ ] LangGraph node integration
- [ ] AutoGen agent integration
- [ ] Per-request accuracy delta estimates
- [ ] Multi-tenant serving guide

---

## v0.3.0 (Released — 2026-04-18)

**Goal:** Zero-dependency oracle. Remove benchmark scaffolding. Ship Φ routing score.

- [x] `oracle/schema_graph.py` — Fiedler value λ̃₂ of JSON schema dependency graph
- [x] `oracle/schema_entropy.py` — Schema constraint tightness τ (information entropy of type tree)
- [x] `oracle/ncd.py` — Normalized Compression Distance ΔK between prompt and schema (via zlib)
- [x] `oracle/routing_score.py` — Φ = 1 − exp(−(A·λ̃₂² + B·τ·λ̃₂ + C·ΔK)) closed-form routing score
- [x] `RoutingContext` extended with `phi_score`, `phi_lambda2`, `phi_tau`, `phi_delta_k`
- [x] `docs/migration/oracle-v3.md` — v0.2 → v0.3 migration guide
- [x] Removed entire `src/formatshield/benchmark/` package
- [x] Removed `scikit-learn` and `joblib` dependencies
- [x] Removed `formatshield benchmark` CLI command
- [x] Removed `benchmarks.yml` CI workflow

---

## v1.0.0 (Production)

**Goal:** Production-hardened. Community-driven oracle. Full LangChain ecosystem.

- [ ] LangGraph node integration
- [ ] AutoGen agent integration
- [ ] OpenAI Assistants API integration
- [ ] Per-request accuracy delta estimates with confidence intervals
- [ ] Multi-tenant serving guide with Kubernetes Helm chart
- [ ] NeurIPS 2026 Efficient LLMs Workshop submission

---

## Competitive Gap (What FormatShield Uniquely Owns)

| Feature | Outlines | Instructor | RouteLLM | **FormatShield** |
|---------|---------|-----------|---------|----------------|
| Fixes reasoning accuracy loss | ❌ | ❌ | ❌ | ✅ |
| Routes between generation strategies | ❌ | ❌ | ❌ | ✅ |
| Cross-backend measurement | ❌ | partial | ❌ | ✅ |
| Generates paper benchmark tables | ❌ | ❌ | ❌ | ✅ |
| Failure mode detection | ❌ | ❌ | ❌ | ✅ |
| Works with OpenAI + Anthropic + Groq | partial | ✅ | partial | ✅ |
| Deterministic CI testing backend | ❌ | ❌ | ❌ | ✅ |

No existing library does all seven. This is FormatShield's permanent moat.

---

## Gaps Acknowledged at v0.0.1

These are documented here and on GitHub with `help-wanted` labels:

- No fine-tuned ThresholdOracle (ships with heuristic rules; empirical training requires benchmark data which the harness generates post-install) — **resolved in v0.3**: oracle now uses closed-form Φ score, zero training data required
- No streaming in TTF mode (StreamingEngine handles direct mode; TTF is two sequential calls)
- No cost tracking for multi-step TTF (Groq + OpenRouter pricing APIs; others need manual config)

These are features, not bugs. The library is honest about what it does and doesn't do.
