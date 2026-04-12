# FormatShield

**Prior work shows constrained decoding costs LLMs up to 27% accuracy on reasoning tasks (arXiv 2408.02442). FormatShield routes around it automatically — one import.**

```python
import formatshield as fs
from pydantic import BaseModel

class Solution(BaseModel):
    steps: list[str]
    answer: float
    confidence: float

result = await fs.generate(
    "Solve step by step: if a train travels 120km at 60km/h, how long does it take?",
    schema=Solution,
    model="groq/llama-3.1-70b-versatile",
)
print(result.parsed.answer)   # 2.0
print(result.routing.strategy)  # "ttf" — routed to Think-Then-Format automatically
```

---

## What Is the Format Tax?

Every time you constrain an LLM to produce JSON, the constrained decoding engine applies a finite-state machine (FSM) mask over the vocabulary at each decoding step. That mask forces the token distribution to stay on valid JSON paths — but it also blocks tokens the model's reasoning process needs. The result: **measurable accuracy loss on any task requiring non-trivial reasoning**, even when the output format itself is simple.

The loss was first documented in [arXiv 2408.02442](https://arxiv.org/abs/2408.02442) and quantified at up to **27%** on standard reasoning benchmarks. FormatShield calls this the **Format Tax**.

FormatShield implements the Think-Then-Format (TTF) algorithm from the [CRANE paper (arXiv 2502.09061)](https://arxiv.org/abs/2502.09061), adds per-backend complexity scoring to decide *when* TTF is worth the overhead, and exposes a unified benchmarking harness so you can **measure the tax on your own workloads**.

---

## How It Works

```
  ┌─────────────────────────────────────────────────┐
  │  FormatShield.generate(prompt, schema, model)    │
  └─────────────────┬───────────────────────────────┘
                    │
           ComplexityScorer
      (token entropy + schema depth +
       reasoning ops + model profile)
                    │
                    ▼
           ThresholdOracle
      (per-backend calibrated threshold
       OR sklearn LogisticRegression)
                    │
          ┌─────────┴──────────┐
          │                    │
       Direct               TTF Engine
    (constrained          Pass 1: think freely
      decoding)           Pass 2: format output
          │                    │
          └────────┬───────────┘
                   │
            GenerationResult
      (output, parsed, routing, thinking,
       complexity_score, failure_modes,
       latency_ms, schema_valid)
```

---

## Comparison

| Feature | **FormatShield** | Outlines | Instructor | Guidance |
|---|---|---|---|---|
| Constrained decoding | Yes (direct route) | Yes | No | Yes |
| Think-Then-Format (TTF) | **Yes** | No | No | No |
| Auto routing (TTF vs direct) | **Yes** | No | No | No |
| Accuracy loss measurement | **Yes** | No | No | No |
| Multi-backend | **6 backends** | vLLM, Transformers | OpenAI-like | Transformers |
| Groq support | **Yes** | No | Yes | No |
| OpenRouter support | **Yes** | No | Partial | No |
| Ollama support | **Yes** | No | Partial | No |
| Streaming | **Yes** | Partial | No | Partial |
| LangChain integration | **Yes** | No | No | No |
| Benchmark harness | **Yes** | No | No | No |
| Python 3.11+, async-first | **Yes** | Partial | Partial | No |

---

## Install

=== "pip"

    ```bash
    pip install formatshield
    ```

=== "uv"

    ```bash
    uv add formatshield
    ```

=== "With extras"

    ```bash
    # vLLM backend (requires CUDA)
    pip install "formatshield[vllm]"

    # Outlines backend
    pip install "formatshield[outlines]"

    # Guidance backend
    pip install "formatshield[guidance]"

    # Benchmarking tools (pandas, matplotlib, seaborn)
    pip install "formatshield[benchmark]"

    # Everything
    pip install "formatshield[all]"
    ```

---

## Quick Start

Set your API key and run your first generation:

```bash
export GROQ_API_KEY=your_key_here
```

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class WeatherReport(BaseModel):
    city: str
    temperature_celsius: float
    condition: str
    humidity_pct: int

async def main():
    result = await fs.generate(
        prompt="What's the weather like in Paris in July?",
        schema=WeatherReport,
        model="groq/llama-3.1-70b-versatile",
    )
    print(result.parsed.city)           # "Paris"
    print(result.routing.strategy)      # "direct" — simple schema, no TTF needed
    print(f"{result.latency_ms:.0f}ms") # e.g. "320ms"

asyncio.run(main())
```

For a full walkthrough, see the [Getting Started guide](getting-started.md).

---

## Key Concepts

**[The Format Tax](explanation/format-tax.md)**
: Why constrained decoding hurts reasoning accuracy, with citations and numbers.

**[Routing Algorithm](explanation/routing-algorithm.md)**
: How ComplexityScorer and ThresholdOracle decide between TTF and direct generation.

**[TTF Algorithm](explanation/ttf-algorithm.md)**
: The two-pass Think-Then-Format algorithm, how KV caching works across backends.

**[When TTF Hurts](explanation/when-ttf-hurts.md)**
: The cases where TTF adds latency without accuracy benefit — and how FormatShield avoids them.

---

## Research Basis

FormatShield is grounded in published research:

- **The Format Tax**: [_"Let Me Speak Freely? A Study of LLM Responses to Constrained Output Formats"_](https://arxiv.org/abs/2408.02442), Tam et al., 2024 (arXiv 2408.02442)
- **CRANE / TTF**: [_"CRANE: Reasoning with constrained LLM generation"_](https://arxiv.org/abs/2502.09061), 2025 (arXiv 2502.09061)

---

## License

MIT. See [LICENSE](https://github.com/formatshield/formatshield/blob/main/LICENSE).
