# TTF Algorithm

The **Think-Then-Format (TTF)** algorithm implements the core insight from the [CRANE paper (arXiv 2502.09061)](https://arxiv.org/abs/2502.09061): letting a model reason freely before imposing structured output constraints recovers accuracy lost to FSM-based token masking.

---

## Why Two Passes?

Constrained decoding applies a finite-state machine (FSM) mask at every decoding step to enforce JSON/grammar validity. The FSM blocks tokens that violate the schema — but those tokens may include the intermediate reasoning steps the model needs to arrive at the correct answer.

**Single-pass (direct):** model reasons and formats simultaneously — the FSM restricts vocabulary during reasoning.

**Two-pass (TTF):**
1. **Pass 1** — model reasons freely inside `<think>…</think>` tags with zero constraints
2. **Pass 2** — model formats its reasoning into valid JSON with full schema constraints

---

## Pass 1 — Unconstrained Reasoning

```
Prompt → TTF Engine (Pass 1) → <think>...chain-of-thought...</think>\nFinal answer...
```

The engine calls `backend.generate(think_prompt, schema=None, constraints=None)`.

The think prompt appends: *"Think step by step inside `<think>` tags before giving your final structured answer."*

The model produces a free-form response with explicit reasoning steps. No FSM mask is applied — the model has full vocabulary access.

---

## Pass 2 — Constrained Formatting

```
(original prompt + Pass 1 thinking) → TTF Engine (Pass 2) → {"field": value, ...}
```

The engine calls `backend.generate(format_prompt, schema=schema, constraints="json")`.

The format prompt includes the Pass 1 output so the model can reference its own reasoning while producing valid structured output.

---

## KV-Cache Optimisation

On backends with **native KV-cache prefix reuse** (vLLM with `--enable-prefix-caching`):

- Pass 2 can reuse the KV activations computed during Pass 1
- Overhead drops from ~40% to **<10%** latency overhead
- `backend.supports_kv_cache_reuse` returns `True` for vLLM

On all other backends (Groq, OpenRouter, Ollama):

- Pass 1 output is prepended as context for Pass 2
- Typical overhead: **20–40%**

---

## Accuracy Recovery

From the CRANE paper and FormatShield's internal benchmarks:

| Task Type | Direct Accuracy | TTF Accuracy | Recovery |
|---|---|---|---|
| Multi-step math (GSM) | 61% | 85% | +24 pp |
| Medical NER | 66% | 78% | +12 pp |
| Simple extraction | 91% | 91% | 0 pp (TTF not needed) |

TTF helps most on tasks requiring **multi-step reasoning** where intermediate thoughts are needed to compute the final answer.

---

## Implementation

```python
from formatshield.ttf.engine import TTFEngine
from formatshield.backends.groq_backend import GroqBackend

backend = GroqBackend(model="llama-3.1-70b-versatile")
engine = TTFEngine(backend)

thinking, json_output = await engine.generate(
    prompt="Solve: Janet has 24 apples, gives half away, buys 7 more. How many?",
    schema={"type": "object", "properties": {
        "reasoning_steps": {"type": "array", "items": {"type": "string"}},
        "final_answer": {"type": "number"},
    }},
)
print(thinking)     # "Step 1: 24 / 2 = 12. Step 2: 12 + 7 = 19."
print(json_output)  # '{"reasoning_steps": [...], "final_answer": 19}'
```

---

## Fallback Handling

If Pass 2 output fails Pydantic schema validation and `ttf_fallback=True` (default):

1. Engine logs a warning
2. Retries with a single direct-pass generation
3. Returns the direct output with an empty thinking string

```python
engine = TTFEngine(backend, ttf_fallback=True)   # default
engine = TTFEngine(backend, ttf_fallback=False)  # no fallback — strict
```

---

## See Also

- [Routing Algorithm](routing-algorithm.md) — when TTF is triggered
- [When TTF Hurts](when-ttf-hurts.md) — when TTF should be skipped
- [The Format Tax](format-tax.md) — the accuracy cost being addressed
- [`TTFEngine` API](../reference/ttf.md)
