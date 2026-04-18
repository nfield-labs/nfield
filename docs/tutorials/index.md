# Tutorials

These tutorials take you from your first generation request to advanced backend configuration and observability.

## Learning Path

| Tutorial | What you'll learn | Time |
|----------|-------------------|------|
| [01 · Basic Generation](01-basic-generation.md) | The `generate()` API, Pydantic schemas, sync vs async | 10 min |
| [02 · Smart Routing](02-routing.md) | How ComplexityScorer and ThresholdOracle decide between TTF and direct | 15 min |
| [04 · Backends](04-backends.md) | Configure Groq, OpenAI, Anthropic, Ollama, vLLM, Outlines | 20 min |
| [05 · Streaming](05-streaming.md) | Stream generation tokens with SSE-compatible events | 10 min |
| [06 · LangChain](06-langchain.md) | Drop-in FormatShieldLLM for LangChain pipelines | 10 min |
| [07 · Observability](07-observability.md) | Structured logging, metrics, Prometheus integration | 15 min |
| [08 · Contributing a Backend](08-contributing.md) | Build and test a new backend adapter from scratch | 20 min |

## Prerequisites

- Python 3.11+
- `pip install formatshield` or `uv add formatshield`
- A Groq API key (free at [console.groq.com](https://console.groq.com)) **or** a local Ollama installation

## Quick Navigation

If you already know what you're looking for:

- **Just want structured output?** → [01 · Basic Generation](01-basic-generation.md)
- **Understanding when TTF kicks in?** → [02 · Smart Routing](02-routing.md) and [Routing Algorithm](../explanation/routing-algorithm.md)
- **Adding a new LLM provider?** → [08 · Contributing a Backend](08-contributing.md)
- **Already using LangChain?** → [06 · LangChain](06-langchain.md)
