# Tutorial 04 — Configuring Backends

FormatShield supports six inference backends plus a `DryRunBackend` for testing. Each backend has different install requirements, configuration options, and TTF overhead characteristics. This tutorial walks through each one.

---

## Backend Overview

| Backend | Transport | Install Extra | API Key | TTF Overhead | KV Cache Reuse |
|---|---|---|---|---|---|
| `groq` | REST | base | `GROQ_API_KEY` | ~30% | No |
| `openrouter` | REST | base | `OPENROUTER_API_KEY` | ~35% | No |
| `ollama` | REST | base | None | ~25% | No |
| `vllm` | REST | `[vllm]` | None | ~10% | **Yes** |
| `outlines` | In-process | `[outlines]` | None | ~20% | No |
| `guidance` | In-process | `[guidance]` | None | ~22% | No |
| `dryrun` | In-process | base | None | N/A | No |

---

## 1. Groq

Groq provides ultra-fast LLM inference via their LPU hardware. It is the default backend for FormatShield examples.

```bash
pip install formatshield
export GROQ_API_KEY=gsk_your_key_here
```

```python
import formatshield as fs
from pydantic import BaseModel

class Answer(BaseModel):
    value: float
    explanation: str

shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    # api_key="gsk_..." # or set GROQ_API_KEY env var
)
result = await shield.generate("What is 42 * 17?", schema=Answer)
```

### Available Groq Models

| Model string | Notes |
|---|---|
| `groq/llama-3.1-70b-versatile` | Recommended for most tasks |
| `groq/llama-3.1-8b-instant` | Fastest, for simple schemas |
| `groq/mixtral-8x7b-32768` | Large context window |
| `groq/gemma2-9b-it` | Efficient instruction-tuned |

---

## 2. OpenRouter

OpenRouter provides a unified API to hundreds of models, including GPT-4o, Claude 3.5 Sonnet, Mistral, and many open-source models.

```bash
export OPENROUTER_API_KEY=sk-or-your_key_here
```

```python
shield = fs.FormatShield(
    model="openrouter/anthropic/claude-3.5-sonnet",
)

# Or GPT-4o
shield = fs.FormatShield(
    model="openrouter/openai/gpt-4o",
)

# Or a cheap open-source model
shield = fs.FormatShield(
    model="openrouter/mistralai/mistral-7b-instruct",
)
```

!!! note "Native thinkers via OpenRouter"
    When using `openrouter/openai/o1-mini` or similar native-thinker models,
    FormatShield automatically skips TTF regardless of the complexity score.

---

## 3. Ollama (Local)

Ollama runs models locally. No API key is required.

```bash
# Install Ollama from https://ollama.ai
ollama pull llama3.1           # 8B model, ~5GB
ollama pull llama3.1:70b       # 70B, needs 40GB+ VRAM
ollama serve                   # starts the server on localhost:11434
```

```python
shield = fs.FormatShield(
    model="ollama/llama3.1",
    base_url="http://localhost:11434",  # default; override for remote Ollama
)
```

### JSON Mode with Ollama

FormatShield automatically requests JSON mode from Ollama when a schema is provided. This uses Ollama's native `format: "json"` parameter — not grammar-based constrained decoding, so vocabulary masking is minimal.

---

## 4. vLLM

vLLM is an optimised inference server for production deployments. It supports **native KV-cache prefix reuse**, which means TTF's two-pass overhead is under 10% instead of the ~30% typical of API backends.

```bash
pip install "formatshield[vllm]"
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --port 8000
```

```python
shield = fs.FormatShield(
    model="vllm/meta-llama/Meta-Llama-3.1-8B-Instruct",
    base_url="http://localhost:8000/v1",
)
```

!!! tip "Enable prefix caching on vLLM"
    The `--enable-prefix-caching` flag is required to get the <10% TTF overhead.
    Without it, vLLM falls back to the same ~20-30% overhead as API backends.

### vLLM with Grammar-Based Constrained Decoding

vLLM supports `guided_json` for JSON schema-constrained decoding. FormatShield uses this automatically for direct-route requests:

```python
# FormatShield passes the schema as guided_json to vLLM under the hood
result = await shield.generate(
    "Extract entities from: ...",
    schema=MyNERSchema,
)
# result.routing.strategy will be "direct" for simple tasks
# → FormatShield sends schema via vLLM's guided_json parameter
```

---

## 5. Outlines

Outlines is a Python library for structured generation using finite-state machine (FSM) grammar constraints.

```bash
pip install "formatshield[outlines]"
```

```python
shield = fs.FormatShield(
    model="outlines/meta-llama/Meta-Llama-3.1-8B-Instruct",
    # Outlines loads the model in-process; no separate server needed
)
```

!!! warning "Outlines requires significant VRAM"
    Outlines runs models in-process via Hugging Face Transformers. You need a GPU or a machine with enough RAM to load the model weights (~16GB for 8B models).

The **direct route** with Outlines uses true FSM-constrained decoding — tokens outside valid JSON paths are masked at every step. This produces valid JSON reliably but can hurt accuracy on complex tasks (the Format Tax).

When FormatShield routes to TTF with Outlines:
- Pass 1 runs without FSM constraints (the model reasons freely)
- Pass 2 runs with FSM constraints (the model formats the output)

---

## 6. Guidance

Guidance is Microsoft's structured generation library. It uses a different approach: "interleaved generation" where the generation process is guided by a template.

```bash
pip install "formatshield[guidance]"
```

```python
shield = fs.FormatShield(
    model="guidance/meta-llama/Meta-Llama-3.1-8B-Instruct",
)
```

---

## 7. DryRunBackend (Testing)

`DryRunBackend` is a deterministic zero-dependency backend for unit tests and CI pipelines. It generates valid JSON matching the provided schema using the schema's type information — no LLM calls, no API keys, no latency.

```python
from formatshield.backends.dryrun_backend import DryRunBackend

backend = DryRunBackend(seed=42)

# Use it in FormatShield by setting the internal backend
import formatshield as fs
shield = fs.FormatShield(model="dryrun/default")
# FormatShield does not yet auto-detect "dryrun" from model string,
# so use it directly:
from formatshield.ttf.engine import TTFEngine
engine = TTFEngine(backend=backend)
thinking, output = await engine.generate("What is 2 + 2?", schema={"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]})
```

Or use it via the benchmark harness (the simplest path):

```python
from formatshield.benchmark.harness import BenchmarkHarness
from formatshield.backends.dryrun_backend import DryRunBackend

harness = BenchmarkHarness()
results = await harness.run(
    tasks=["gsm_symbolic"],
    backends=["dryrun"],
    models={"dryrun": "dryrun/default"},
    backend_objects={"dryrun": DryRunBackend(seed=42)},
    quick=True,
)
```

### DryRunBackend Behavior

| Behavior | Details |
|---|---|
| JSON output | Always produces valid JSON matching the schema |
| Thinking text | Returns `<think>Dry run thinking pass 1</think>` |
| Latency | Near-zero (in-process) |
| Determinism | Seeded via `seed` parameter |
| KV cache | `supports_kv_cache_reuse = False` |

---

## 8. Retry Behavior

All backends include automatic exponential backoff retry via `_retry.py`. The default policy is:

| Parameter | Default |
|---|---|
| Max retries | 3 |
| Initial delay | 1.0 second |
| Backoff factor | 2x |
| Max delay | 60 seconds |
| Jitter | ±10% |

Retried errors include: rate limit errors (429), server errors (500, 502, 503), and transient connection errors.

---

## 9. Selecting a Backend by Model String

FormatShield infers the backend from the `model` string prefix:

| Model string prefix | Backend |
|---|---|
| `groq/` | `groq` |
| `openrouter/` | `openrouter` |
| `ollama/` | `ollama` |
| `vllm/` | `vllm` |
| `outlines/` | `outlines` |
| `guidance/` | `guidance` |
| `dryrun/` | `dryrun` |
| anything else | `openrouter` (fallback) |

---

## Next Steps

- [Tutorial 05: Streaming](05-streaming.md) — stream TTF events in real-time
- [Reference: Backends](../reference/backends.md) — full backend protocol and API reference
- [Explanation: TTF Algorithm](../explanation/ttf-algorithm.md) — how KV cache reuse works per backend
