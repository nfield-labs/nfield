# FormatShield Roadmap

## v0.0.1 (Current — Day 1-2)

**Goal:** `pip install formatshield` works. Groq + OpenRouter + Ollama + vLLM backends. Real benchmark numbers.

- [x] ComplexityScorer (6 features)
- [x] ThresholdOracle (heuristic v0)
- [x] TTF Engine (two-pass generation)
- [x] FailureModeDetector (6 modes)
- [x] GroqBackend
- [x] OpenRouterBackend
- [x] OllamaBackend
- [x] VLLMBackend
- [x] StreamingEngine (SSE-compatible)
- [x] GSM-Symbolic + Medical NER benchmark tasks
- [x] `formatshield generate` CLI
- [x] `formatshield benchmark --tasks gsm --backends groq --quick` CLI
- [x] Debug mode with routing trace
- [x] TTF fallback on schema validation failure
- [x] MIT license, CONTRIBUTING.md, 5 Good First Issues

**Good First Issues (open at launch):**
1. `good-first-issue` Add Cohere backend
2. `good-first-issue` Add SQL extraction benchmark task
3. `good-first-issue` Improve ComplexityScorer for non-English prompts
4. `good-first-issue` Add benchmark visualization (PNG exporter)
5. `good-first-issue` Write streaming integration test

---

## v0.1.0 (Day 3-5)

**Goal:** Paper companion release. arXiv submission. Show HN post.

- [ ] OutlinesBackend (Phase 2)
- [ ] GuidanceBackend (Phase 2)
- [ ] Full 12-task BenchmarkHarness
- [ ] CrossBackendBenchmark with LaTeX/CSV/PNG export
- [ ] `formatshield benchmark --reproduce-paper` command
- [ ] ThresholdOracle v1 (retrained on real benchmark data)
- [ ] arXiv paper draft in `/paper/` directory
- [ ] FastAPI example (`examples/fastapi_server.py`)
- [ ] LangChain integration (`FormatShieldLangChain`)
- [ ] Prometheus metrics + structured logger
- [ ] MkDocs documentation

---

## v1.0.0 (Post-launch)

**Goal:** Production-hardened. Community-driven oracle. LangChain ecosystem.

- [ ] ThresholdOracle trained on community benchmark contributions
- [ ] `formatshield benchmark upload --anonymized` community data sharing
- [ ] LangGraph node integration
- [ ] AutoGen agent integration
- [ ] Cost tracking across all backends
- [ ] Per-request accuracy delta estimates
- [ ] Multi-tenant serving guide
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

No existing library does all five. This is FormatShield's permanent moat.

---

## Gaps Acknowledged at v0.0.1

These are documented here and on GitHub with `help-wanted` labels:

- No fine-tuned ThresholdOracle (ships with heuristic rules; empirical training requires benchmark data which the harness generates post-install)
- No Guidance backend (adapter protocol defined; implementation deferred — `good-first-issue`)
- No streaming in TTF mode (StreamingEngine handles direct mode; TTF is two sequential calls)
- No cost tracking for multi-step TTF (Groq + OpenRouter pricing APIs; others need manual config)

These are features, not bugs. The library is honest about what it does and doesn't do.
