<div align="center">

<h1>FormatShield</h1>

<p><strong>Stop losing 15–30% LLM accuracy to JSON constraints. Route automatically. Measure everything.</strong></p>

[![PyPI version](https://badge.fury.io/py/formatshield.svg)](https://badge.fury.io/py/formatshield)
[![CI](https://github.com/formatshield/formatshield/workflows/CI/badge.svg)](https://github.com/formatshield/formatshield/actions)
[![Coverage](https://codecov.io/gh/formatshield/formatshield/branch/main/graph/badge.svg)](https://codecov.io/gh/formatshield/formatshield)
[![Docs](https://img.shields.io/badge/docs-formatshield.github.io-blueviolet)](https://formatshield.github.io/formatshield/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

```bash
pip install formatshield
```

</div>

---

**Grammar-constrained decoding costs LLMs up to 27% accuracy on reasoning tasks** (arXiv 2408.02442). FormatShield routes around it — one import, zero schema changes.

```python
import formatshield as fs

result = await fs.generate(
    prompt="Extract all obligations from this contract...",
    schema=ContractObligations,
    model="groq/llama-3.1-70b-versatile"
)
# Complex task detected → two-pass TTF generation (reason free, then format)
# result.routing.strategy  →  "ttf"
# result.complexity_score  →  0.82
# result.parsed            →  ContractObligations(...)  ← typed Pydantic model
```

| | Outlines | Instructor | Guidance | RouteLLM | **FormatShield** |
|--|:-------:|:---------:|:-------:|:-------:|:--------------:|
| Produces valid JSON | ✅ | ✅ | ✅ | ❌ | ✅ |
| Fixes reasoning accuracy loss | ❌ | ❌ | ❌ | ❌ | ✅ |
| Routes direct vs. TTF automatically | ❌ | ❌ | ❌ | ✅ (cost) | ✅ |
| Works across all backends | partial | ✅ | partial | ✅ | ✅ |
| Generates benchmark tables for your pipeline | ❌ | ❌ | ❌ | ❌ | ✅ |
| Failure mode taxonomy | ❌ | ❌ | ❌ | ❌ | ✅ |

---

## News

- **2026/04** — FormatShield v0.0.1 released: 7 backends, 9 benchmark tasks, TTF engine, 567 tests, 81% coverage
- **2026/04** — "The Format Tax" (arXiv 2604.03616) confirms accuracy loss across 6 open-weight models and 4 output formats — the definitive empirical study
- **2025/06** — CRANE (arXiv 2502.09061) accepted at **ICML 2025** — the research basis for TTF. FormatShield productionizes the algorithm across all backends
- **2024/12** — "Let Me Speak Freely?" (arXiv 2408.02442) published at **EMNLP 2024** — quantifies 27.3pp accuracy loss from constrained decoding. The reason FormatShield exists
- **(Upcoming v0.1.0)** — Together AI, Fireworks AI, Mistral AI backends · 12-task benchmark harness · LangChain `FormatShieldLLM` · `formatshield benchmark --reproduce-paper`

---

## Table of Contents

- [Philosophy](#philosophy)
- [The Problem](#the-problem)
- [How FormatShield Fixes It](#how-formatshield-fixes-it)
- [Installation](#installation)
- [Who Is This For?](#who-is-this-for)
- [Quick Start](#quick-start)
- [Supported Backends](#supported-backends)
- [Real-World Use Cases](#real-world-use-cases)
- [Benchmark Your Pipeline](#benchmark-your-pipeline)
- [Ecosystem Integrations](#ecosystem-integrations)
- [Agent Framework Integration](#agent-framework-integration)
- [Architecture](#architecture-9-components)
- [The 5 Objections](#why-not-just----the-5-objections)
- [Research Background](#research-background)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Philosophy

> *"The king who is energetic, who has sharp intellect, who is endowed with prowess — he alone is capable of winning over the world."*
> — Chanakya, Arthashastra (c. 350 BCE)

Most libraries ask: **"How do we get valid JSON?"**

FormatShield asks: **"When does asking for JSON cost you accuracy — and is that cost worth it?"**

This is a different question. Instructor answers the first question brilliantly. Outlines answers it elegantly. FormatShield is the only library that answers the second — and backs the answer with empirical measurement per backend, per task, per schema complexity.

Constrained decoding is not a neutral operation. Every FSM mask applied to the vocabulary during generation is a tax paid in reasoning accuracy. The tax varies by model, task complexity, and schema structure. Sometimes it's 0%. Sometimes it's 27%. FormatShield measures this tax on your specific pipeline and routes around it only when the measurement justifies it.

That's not prompting. That's not retry logic. That's understanding the physics of structured generation.

---

## The Problem

Grammar-constrained decoding — the dominant technique for structured LLM output — silently degrades model reasoning.

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

**Smart routing:** FormatShield scores each (prompt, schema) pair and routes to TTF only when the benefit exceeds the overhead.

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
# Base — Groq + Ollama + OpenRouter (works on all platforms)
pip install formatshield

# With OpenAI
pip install formatshield[openai]

# With Anthropic (Claude models)
pip install formatshield[anthropic]

# With vLLM (Linux + NVIDIA GPU only)
pip install formatshield[vllm]

# With Outlines (local constrained decoding)
pip install formatshield[outlines]

# With benchmark tools
pip install formatshield[benchmark]

# Everything
pip install formatshield[all]
```

---

## Who Is This For?

**ML Engineers** running structured extraction pipelines who need accuracy, not just valid JSON. If your pipeline does RAG, NER, contract parsing, or any multi-step reasoning task — the format tax is silently costing you. FormatShield measures it and routes around it automatically.

**AI Researchers** who want to measure format tax on their models empirically. The benchmark harness generates publication-ready CSV and LaTeX tables. Run `formatshield benchmark --reproduce-paper` to see your numbers.

**AI Agent Developers** who need structured tool call outputs from fast, cheap models (Groq, Ollama). TTF lets models reason through which tool to call before committing to the structured format. This matters when tools have complex parameters.

**LLM Application Developers** who already use Instructor or Outlines and want to drop in accuracy recovery without changing schemas or provider logic. FormatShield wraps your existing stack — it doesn't replace it.

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

print(result.parsed.conclusion)       # typed Pydantic model
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

| Backend | Model string | Install | Notes |
|---------|-------------|---------|-------|
| **Groq** | `groq/llama-3.1-70b-versatile` | `pip install formatshield` | Free tier. Fastest API inference. |
| **OpenAI** | `openai/gpt-4o-mini` | `pip install formatshield[openai]` | GPT-4o, o1, o3 series |
| **Anthropic** | `anthropic/claude-3-5-haiku-20241022` | `pip install formatshield[anthropic]` | Claude 3.5 Sonnet/Haiku/Opus |
| **OpenRouter** | `openrouter/meta-llama/llama-3.1-70b` | `pip install formatshield` | 100+ models via one API |
| **Ollama** | `ollama/llama3.1:70b` | `pip install formatshield` | Local inference, any GGUF model |
| **vLLM** | `vllm/meta-llama/Llama-3-70b-Instruct` | `pip install formatshield[vllm]` | Native KV-cache reuse → <10% TTF overhead |
| **Outlines** | `outlines/mistralai/Mistral-7B-v0.1` | `pip install formatshield[outlines]` | Local constrained decoding |

```python
# Switch backends with one parameter:
result = await fs.generate(prompt, schema, model="groq/llama-3.1-70b-versatile")
result = await fs.generate(prompt, schema, model="openai/gpt-4o-mini")
result = await fs.generate(prompt, schema, model="anthropic/claude-3-5-sonnet-20241022")
result = await fs.generate(prompt, schema, model="ollama/llama3.1")
result = await fs.generate(prompt, schema, model="openrouter/anthropic/claude-3.5-sonnet")
result = await fs.generate(prompt, schema, model="vllm/llama-3-70b", base_url="http://localhost:8000")
```

**Native thinker detection:** o1, o3, DeepSeek-R1, Claude-3.5 with extended thinking, and QwQ models are automatically detected and skip TTF (they already think freely).

---

## Real-World Use Cases

### Customer Support — Ticket Triage

```python
class TicketAnalysis(BaseModel):
    category: str
    priority: Literal["LOW", "MEDIUM", "HIGH", "URGENT"]
    sentiment: float          # 0.0 (angry) – 1.0 (happy)
    required_actions: list[str]
    escalate_to_human: bool

result = await fs.generate(
    prompt=f"Analyze this support ticket: {ticket_text}",
    schema=TicketAnalysis,
    model="groq/llama-3.1-70b-versatile"
)
# FormatShield routes to direct (template fill, low complexity)
# result.routing.strategy → "direct"
```

### RAG Pipeline — Document Fact Extraction

```python
class DocumentFacts(BaseModel):
    key_facts: list[str]
    entities: list[Entity]    # name, type, relevance_score
    summary: str
    confidence: float

# Complex multi-entity reasoning → TTF
result = await fs.generate(prompt=f"Extract facts from: {chunk}", schema=DocumentFacts)
```

### Financial Analysis — Earnings Call Parsing

```python
class EarningsAnalysis(BaseModel):
    revenue_mentioned: bool
    guidance_raised: bool
    key_metrics: list[Metric]
    risks: list[str]
    analyst_recommendation: str

# Stream the analysis as it generates
async for event in shield.stream(earnings_transcript, schema=EarningsAnalysis):
    if event.type == "output":
        print(event.token, end="", flush=True)
```

### Legal — Contract Obligation Extraction

```python
# See examples/contract_extraction.py — full working example
result = await shield.generate(contract_text, schema=ContractAnalysis)
# Complexity: 0.82 → routes to TTF → recovers ~18% accuracy
```

### Medical NER — Clinical Note Parsing

```python
# See examples/medical_ner.py — HIPAA-aware structured extraction
result = await shield.generate(clinical_note, schema=ClinicalEntities)
```

More examples in [`examples/`](examples/):

| Example | Use Case | Routing |
|---------|---------|---------|
| [`contract_extraction.py`](examples/contract_extraction.py) | Legal obligation extraction | TTF (complex) |
| [`medical_ner.py`](examples/medical_ner.py) | Clinical entity recognition | TTF (complex) |
| [`customer_support.py`](examples/customer_support.py) | Ticket triage & routing | Direct (template) |
| [`rag_extraction.py`](examples/rag_extraction.py) | RAG structured fact extraction | TTF (complex) |
| [`financial_analysis.py`](examples/financial_analysis.py) | Earnings call parsing + streaming | TTF (complex) |
| [`document_classification.py`](examples/document_classification.py) | Multi-label legal doc classification | Direct (low complexity) |
| [`agent_tool_calling.py`](examples/agent_tool_calling.py) | Agent tool call extraction loop | TTF (complex) |
| [`fastapi_server.py`](examples/fastapi_server.py) | Production HTTP API server | — |

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

## Ecosystem Integrations

> FormatShield works with every major inference provider and framework. The list grows as the community adds backends.

**Inference Backends**

| Provider | Status | Install |
|----------|--------|---------|
| Groq (LPU) | ✅ Supported | `pip install formatshield` |
| OpenAI | ✅ Supported | `pip install formatshield[openai]` |
| Anthropic | ✅ Supported | `pip install formatshield[anthropic]` |
| OpenRouter (100+ models) | ✅ Supported | `pip install formatshield` |
| Ollama (local) | ✅ Supported | `pip install formatshield` |
| vLLM (self-hosted) | ✅ Supported | `pip install formatshield[vllm]` |
| Outlines (constrained local) | ✅ Supported | `pip install formatshield[outlines]` |
| Together AI | 🔜 v0.1.0 | — |
| Fireworks AI | 🔜 v0.1.0 | — |
| Mistral AI | 🔜 v0.1.0 | — |
| Cohere | 🔜 Community PR welcome | — |

**Agent Frameworks**

| Framework | Status |
|-----------|--------|
| LangChain | 🔜 v0.1.0 (`FormatShieldLLM`) |
| LangGraph | 🔜 v0.1.0 (node integration) |
| AutoGen | 🔜 v1.0.0 |
| CrewAI | Drop-in today (use `FormatShield` as LLM layer) |
| OpenAI Agents SDK | Drop-in today |

**Schema Libraries**

| Library | Notes |
|---------|-------|
| Pydantic v2 | ✅ Native — pass any `BaseModel` subclass directly |
| JSON Schema (dict) | ✅ Native — pass raw schema dict |
| Pydantic v1 | ✅ Compatible |

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
│   (Groq / OpenAI / Anthropic / vLLM)│
└─────────────────────────────────────┘
```

```python
# LangChain drop-in (v0.1.0):
from formatshield.integrations.langchain import FormatShieldLLM
llm = FormatShieldLLM(model="groq/llama-3.1-70b-versatile")
chain = prompt_template | llm | output_parser
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
6. **Backend Adapters** — Groq, OpenAI, Anthropic, OpenRouter, Ollama, vLLM, Outlines (same interface, swappable)
7. **StreamingEngine** — SSE-compatible async generator
8. **BenchmarkHarness** — runs tasks, generates paper artifacts (CSV, LaTeX, PNG)
9. **CLI** — `formatshield generate` + `formatshield benchmark`

---

## "Why Not Just..." — The 5 Objections

| Objection | Answer |
|-----------|--------|
| "Why not Instructor?" | Instructor fixes **invalid JSON** via retry. FormatShield fixes **wrong reasoning**. Root cause, not symptom. |
| "Why not just prompt better?" | Prompting doesn't fix constrained decoding — it's architectural. The model can't reason freely while inside JSON. |
| "Why not o1/o3?" | FormatShield detects native thinkers (o1, o3, DeepSeek-R1, Claude with extended thinking) and skips TTF automatically. Works on free Groq models for all other cases. |
| "Will this break my code?" | No. Drop-in at the call site. Your agent, schema, and framework don't change. |
| "Why not wait for Outlines to add this?" | Outlines is one backend. FormatShield routes across all backends. Different layers. |

**FormatShield + Instructor = the complete stack:**
```
Instructor:    "Your JSON might be invalid — we'll retry until valid."
FormatShield:  "Your JSON is valid but 23% less accurate — we'll reason first, then format."
```

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

**The routing gap no paper addresses:** At what complexity score does TTF become beneficial? Does this vary by backend? FormatShield measures this empirically and makes the data public. That's the paper: *"When Does Think-Then-Format Help?"*

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. **10 `good-first-issue` tasks open at launch.** Your name in the paper acknowledgments.

```bash
git clone https://github.com/formatshield/formatshield
cd formatshield
pip install uv
uv sync --all-extras
uv run pre-commit install
uv run pytest tests/unit/ -v   # all green, no API keys needed
```

**Community:**
- [GitHub Discussions](https://github.com/formatshield/formatshield/discussions) — questions, show and tell, ideas
- [Good First Issues](https://github.com/formatshield/formatshield/labels/good-first-issue) — start contributing today
- [Security Policy](SECURITY.md) — responsible disclosure
- [Governance](GOVERNANCE.md) — how decisions are made

## License

MIT — use freely, commercially, academically. No CLA required.

---

## Citation

```bibtex
@software{formatshield2026,
  title={FormatShield: Routing-Based Think-Then-Format for Accurate Structured LLM Generation},
  year={2026},
  url={https://github.com/formatshield/formatshield},
  license={MIT}
}
```
