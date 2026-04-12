# FormatShield

**Prior work shows constrained decoding costs LLMs up to 27% accuracy on reasoning tasks (arXiv 2408.02442).
FormatShield routes around it automatically — one import.**

```bash
pip install formatshield
```

```python
import formatshield as fs

result = await fs.generate(
    prompt="Extract all obligations from this contract...",
    schema=ContractObligations,
    model="groq/llama-3.1-70b-versatile"
)
# Automatically detected: complex reasoning task → two-pass TTF generation
# Prior literature: up to 27% accuracy gap from constrained decoding (arXiv 2408.02442)
# FormatShield measures your pipeline's specific delta and routes accordingly.
```

| | Outlines | Instructor | **FormatShield** |
|--|---------|-----------|--------------|
| Fixes invalid JSON | ✅ | ✅ | ✅ |
| Fixes reasoning accuracy loss | ❌ | ❌ | ✅ |
| Works across backends | partial | ✅ | ✅ |
| Routes automatically | ❌ | ❌ | ✅ |
| Generates benchmark tables | ❌ | ❌ | ✅ |

---

## The Problem

Grammar-constrained decoding — the dominant technique for getting LLMs to produce valid JSON — silently degrades model reasoning.

**arXiv 2408.02442** (EMNLP 2024, "Let Me Speak Freely?") found that forcing JSON output reduces GSM8K accuracy by **27.3 percentage points** compared to unconstrained generation.

**arXiv 2604.03616** ("The Format Tax", April 2026) confirmed the pattern across six open-weight models, four API models, and four output formats.

**arXiv 2405.21047** (Grammar-Aligned Decoding, NeurIPS 2024) proved the mechanism: FSM-based token masking distorts the model's output distribution — tokens that are grammatically valid but whose probabilities are not proportional to what the LLM learned.

**arXiv 2502.09061** (CRANE, ICML 2025) demonstrated the fix: separating reasoning from formatting via two-pass Think-Then-Format (TTF) recovers up to +10 percentage points of accuracy.

**FormatShield is the production library that implements this fix across all backends.**

```python
# What happens inside the model when you call guided_json():
# Token 1: "{"          ← constrained — must be JSON
# Token 47: "reasoning" ← constrained — must stay valid JSON
# Token 89: "conclusion"← constrained — even during REASONING tokens
#
# The constraint applies during thinking.
# Your model cannot reason out loud before committing to structure.
# Result: 15–30% accuracy loss on tasks requiring multi-step inference.
```

---

## How FormatShield Fixes It

**Think-Then-Format (TTF):** Two-pass generation following CRANE (arXiv 2502.09061).

```
Pass 1: unconstrained → <think>Step 1: analyze... Step 2: calculate...</think>
Pass 2: constrained  → {"answer": 42, "unit": "dollars", "steps": [...]}
```

The model reasons freely in Pass 1. Only Pass 2 is constrained — after all reasoning is complete.

**Smart routing:** FormatShield doesn't apply TTF blindly. It scores the complexity of each (prompt, schema) pair and routes to TTF only when the benefit exceeds the overhead.

```python
# Debug mode shows every routing decision:
result = await shield.generate(prompt, schema, debug=True)

# Console output:
# [FormatShield] model=groq/llama-3.1-70b-versatile
# [FormatShield] complexity_score=0.82 (schema_depth=4, reasoning_ops=6, tokens=312)
# [FormatShield] route=TTF | expected_delta=+0.18 | estimated_overhead=21%
# [FormatShield] pass1=247 tokens | pass2=89 tokens | total_latency=1.2s
# [FormatShield] schema_valid=True | fallback_triggered=False
```

---

## Installation

```bash
# Base (Groq + Ollama + OpenRouter — works on all platforms)
pip install formatshield

# With vLLM (Linux + NVIDIA GPU only)
pip install formatshield[vllm]

# With Outlines
pip install formatshield[outlines]

# With benchmark tools (for running benchmarks and generating paper figures)
pip install formatshield[benchmark]

# Everything
pip install formatshield[all]
```

---

## Quick Start

### One-liner (any model)

```python
import formatshield as fs
from pydantic import BaseModel

class Analysis(BaseModel):
    key_points: list[str]
    conclusion: str
    confidence: float

result = await fs.generate(
    prompt="Analyze the following contract clause for liability implications...",
    schema=Analysis,
    model="groq/llama-3.1-70b-versatile"
)

print(result.parsed.conclusion)       # typed pydantic model
print(result.routing.strategy)        # "ttf" or "direct"
print(result.complexity_score)        # 0.0–1.0
```

### Full control

```python
import formatshield as fs

shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    latency_budget_ms=500,    # voice agent: never exceed 500ms
    debug=True,               # show routing trace
    ttf_fallback=True,        # auto-retry as direct if TTF output invalid
)

result = await shield.generate(prompt, schema=MySchema)
```

### Sync API (for existing sync codebases)

```python
result = shield.generate_sync(prompt, schema=MySchema)
```

### Streaming

```python
async for event in shield.stream(prompt, schema=MySchema):
    if event.type == "output":
        print(event.token, end="", flush=True)
    elif event.type == "complete":
        parsed = MySchema.model_validate_json(event.json)
```

---

## Supported Backends

| Backend | Model string | Platform | Notes |
|---------|-------------|----------|-------|
| Groq | `groq/llama-3.1-70b-versatile` | All | Free tier available. Fastest API backend. |
| OpenRouter | `openrouter/meta-llama/llama-3.1-70b-instruct` | All | Access 100+ models |
| Ollama | `ollama/llama3.1:70b` | All | Local inference |
| vLLM | `vllm/meta-llama/Llama-3-70b-Instruct` | Linux+GPU | Native KV cache reuse → <10% TTF overhead |
| Outlines | `outlines/mistralai/Mistral-7B-v0.1` | Linux+GPU | `pip install formatshield[outlines]` |

```python
# Switch backends with one parameter change:
result = await fs.generate(prompt, schema, model="groq/llama-3.1-70b-versatile")
result = await fs.generate(prompt, schema, model="ollama/llama3.1")
result = await fs.generate(prompt, schema, model="openrouter/anthropic/claude-3.5-sonnet")
result = await fs.generate(prompt, schema, model="vllm/llama-3-70b", base_url="http://localhost:8000")
```

---

## Benchmark Your Pipeline

```bash
# Quick smoke test (2 minutes, no GPU needed):
export GROQ_API_KEY=your_key_here
formatshield benchmark --tasks gsm --backends groq --quick

# Full cross-backend comparison (generates paper Table 1):
formatshield benchmark --tasks all --backends groq,ollama --output benchmark_results/

# Reproduce paper results:
formatshield benchmark --reproduce-paper
```

Output:
```
benchmark_results/
├── tables/
│   ├── table1_accuracy_by_backend.csv   ← copy into your paper
│   └── table2_failure_modes.csv
└── summary.json
```

Sample output (your numbers will differ by model and task):
```
Backend     | Task         | Direct Acc | TTF Acc | Delta  | Overhead
------------|--------------|------------|---------|--------|----------
groq        | gsm_symbolic | 0.61       | 0.74    | +0.13  | 28%
groq        | medical_ner  | 0.71       | 0.79    | +0.08  | 31%
groq        | template_fill| 0.95       | 0.93    | -0.02  | 29%  ← TTF hurts here
ollama      | gsm_symbolic | 0.58       | 0.72    | +0.14  | 24%
```

---

## Agent Framework Integration

FormatShield is transparent to the agent above and the backend below:

```
┌─────────────────────────────────────┐
│   ANY AGENT FRAMEWORK               │
│   (LangChain / AutoGen / CrewAI)    │
└──────────────────┬──────────────────┘
                   │  prompt + schema
                   ▼
┌─────────────────────────────────────┐
│   FORMATSHIELD                      │
│   Score → Route → TTF/Direct        │
└──────────────────┬──────────────────┘
                   │  structured JSON
                   ▼
┌─────────────────────────────────────┐
│   ANY LLM BACKEND                   │
│   (Groq / Ollama / vLLM / OpenAI)  │
└─────────────────────────────────────┘
```

```python
# LangChain drop-in:
from formatshield.integrations.langchain import FormatShieldLLM
llm = FormatShieldLLM(model="groq/llama-3.1-70b-versatile")
chain = prompt_template | llm | output_parser
```

---

## "Why Not Just..." — The 5 Objections

| Objection | Answer |
|-----------|--------|
| "Why not Instructor?" | Instructor fixes **invalid JSON** via retry. FormatShield fixes **wrong reasoning**. Root cause, not symptom. |
| "Why not just prompt better?" | Prompting doesn't fix constrained decoding — it's architectural. The model can't reason freely while inside JSON. |
| "Why not o1/o3?" | FormatShield detects native thinkers (o1, o3, DeepSeek-R1) and skips TTF automatically. Works on free Groq models for all other cases. |
| "Will this break my code?" | No. Drop-in at the call site. Your agent, schema, and framework don't change. |
| "Why not wait for Outlines to add this?" | Outlines is one backend. FormatShield routes across all backends. Different layers. |

**FormatShield + Instructor = the complete stack:**
```
Instructor: "Your JSON might be invalid — we'll retry until valid."
FormatShield: "Your JSON is valid but 23% less accurate — we'll reason first, then format."
```

---

## Architecture (9 Components)

```
ComplexityScorer → ThresholdOracle → FailureModeDetector → TTFEngine → Backend Adapter
                         ↑
                CrossBackendBenchmark (measures format tax, trains oracle)
                         ↓
                BenchmarkHarness (generates paper artifacts)
```

1. **ComplexityScorer** — 6-feature scoring: token entropy, schema depth, reasoning ops, instruction tuning, prompt length, constraint count
2. **ThresholdOracle** — backend-aware routing classifier (logistic regression trained on benchmark data)
3. **CrossBackendBenchmark** — measures format tax per backend, generates paper Table 1
4. **TTF Engine** — two-pass generation (CRANE pattern, arXiv 2502.09061)
5. **FailureModeDetector** — 6 checks for when TTF would hurt (simple extraction, schema_too_constrained, native_thinker, short_prompt, template_fill, ambiguous_schema)
6. **Backend Adapters** — Groq, OpenRouter, Ollama, vLLM, Outlines (same interface, swappable)
7. **StreamingEngine** — SSE-compatible async generator
8. **BenchmarkHarness** — runs tasks, generates paper artifacts (CSV, LaTeX, PNG)
9. **CLI** — `formatshield generate` + `formatshield benchmark`

---

## Research Background

FormatShield is the production implementation of findings from:

| Paper | Finding | FormatShield component |
|-------|---------|----------------------|
| arXiv 2408.02442 (EMNLP 2024) | Constrained decoding → 27.3pp GSM8K drop | Why we built this |
| arXiv 2502.09061 (ICML 2025, CRANE) | TTF recovers +10pp on reasoning | TTF Engine design |
| arXiv 2405.21047 (NeurIPS 2024, GAD) | FSM masking distorts output distribution | Why direct path exists |
| arXiv 2604.03616 (April 2026, Format Tax) | Confirmed across 6 models, 4 formats | Benchmark design |
| arXiv 2309.06180 (vLLM) | PagedAttention prefix caching | vLLM KV cache reuse |

**The routing gap no paper addresses:** At what complexity score does TTF become beneficial? Does this vary by backend? FormatShield measures this empirically and makes the data public. That's the paper: "When Does Think-Then-Format Help?"

---

## License

MIT — use freely, commercially, academically. No CLA required.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). 5 `good-first-issue` tasks open at launch. Your name in the paper acknowledgments.

## Citation

```bibtex
@software{formatshield2026,
  title={FormatShield: Routing-Based Think-Then-Format for Accurate Structured LLM Generation},
  year={2026},
  url={https://github.com/formatshield/formatshield},
  license={MIT}
}
```
