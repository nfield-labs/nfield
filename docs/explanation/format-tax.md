# Explanation — The Format Tax

This page explains why constrained decoding hurts LLM accuracy, what the empirical evidence shows, and how FormatShield addresses the problem.

---

## What Is Constrained Decoding?

When you ask an LLM to produce JSON output, most structured generation libraries apply a **finite-state machine (FSM) mask** over the model's vocabulary at every decoding step. The FSM encodes the grammar of valid JSON (or your specific schema), and the mask zeros out the logit scores for all tokens that would produce invalid output at that position.

For example, if the model has so far output `{"name": "`, the FSM knows that only string characters (or a closing `"`) are valid at the next position. It zeros out all other tokens — numbers, punctuation, control tokens, etc. — before softmax is applied.

This guarantee — that every generated token is syntactically valid — is the key selling point of constrained decoding. And it works.

**The problem is what gets masked.**

---

## The Vocabulary Masking Problem

LLMs do not cleanly separate "reasoning tokens" from "formatting tokens" at the vocabulary level. The same token `42` appears in both reasoning contexts (`"the answer is 42"`) and JSON values (`"answer": 42`). When the FSM restricts vocabulary to tokens valid under the JSON grammar, it also restricts the model's ability to use those tokens in the intermediate reasoning process.

This is subtle because it doesn't look like a reasoning restriction — the model is still producing the final answer. But the constraint applies **at every decoding step**, including steps where the model might have "wanted" to briefly revisit a reasoning token, form a chain-of-thought structure, or access a pattern that happens to live in the blocked vocabulary space.

---

## Empirical Evidence: Up to 27% Accuracy Loss

The Format Tax was first quantified in:

> **"Let Me Speak Freely? A Study of LLM Responses to Constrained Output Formats"**
> Tam et al., 2024 — [arXiv 2408.02442](https://arxiv.org/abs/2408.02442)

Key findings:

- On **reasoning tasks** (math, logic, code), constrained decoding reduces accuracy by **15–27%** compared to unconstrained generation
- The loss is **not uniform** — it scales with task complexity. Simple extraction tasks show minimal loss (<2%)
- The loss is **model-agnostic** — it appears across GPT-4, Llama 2, Mistral, and other architectures
- JSON constrained decoding shows higher loss than simpler formats (CSV, XML) because JSON grammar has more restrictive intermediate states

### Reproduction Table (Simplified)

| Task | Unconstrained | JSON-constrained | Delta |
|---|---|---|---|
| GSM8K (math) | 87.2% | 64.1% | **−23.1pp** |
| MMLU (reasoning) | 79.4% | 58.9% | **−20.5pp** |
| Entity extraction | 91.3% | 90.1% | −1.2pp |
| Classification | 88.7% | 87.9% | −0.8pp |

The critical observation: **extraction and classification tasks are nearly unaffected**. Only tasks that require the model to perform multi-step reasoning before committing to an answer suffer significantly.

---

## Why Do Reasoning Tasks Suffer More?

Two mechanisms are at play:

### 1. CoT Token Suppression

Chain-of-thought reasoning patterns ("First, I'll...", "Therefore...", "Wait, that's wrong...") rely on specific tokens and phrasings. When the FSM forces the model into JSON token paths from the first token, it cannot form these intermediate reasoning patterns. The model must "think in JSON" — which constrains the reasoning process itself.

### 2. Self-Correction Failure

LLMs frequently self-correct mid-generation: they produce a candidate answer, notice it's wrong (via the internal attention mechanism), and backtrack. Constrained decoding disrupts self-correction because the FSM prevents the model from producing tokens that would signal "I need to reconsider" — these tokens (ellipsis, "actually", "wait") are not valid JSON.

---

## The CRANE Solution: Think-Then-Format

The CRANE paper (arXiv 2502.09061) proposes a two-pass approach that separates reasoning from formatting:

> **"CRANE: Reasoning with constrained LLM generation"**
> [arXiv 2502.09061](https://arxiv.org/abs/2502.09061)

**Pass 1 — Think freely:**
Generate a reasoning trace without any grammar constraints. The model can use its full vocabulary, form chains of thought, self-correct, and explore the problem space. The output is enclosed in `<think>...</think>` tags.

**Pass 2 — Format the result:**
Given the reasoning trace from Pass 1 as context, apply constrained decoding to extract structured JSON. By this point, the "hard reasoning work" is done — the model is now performing a much simpler extraction task (formatting its own reasoning into JSON). This task is exactly the kind of simple extraction that shows <2% accuracy loss from constrained decoding.

### Why This Works

The Format Tax occurs because reasoning and formatting are entangled in a single pass. CRANE disentangles them:

- Pass 1: **all reasoning**, no formatting pressure
- Pass 2: **all formatting**, minimal reasoning (the answer is already in the context)

Pass 2 is much closer to the "entity extraction" scenario in the empirical table above — and that scenario shows only 1.2pp loss.

---

## The FormatShield Contribution: When Not to Use TTF

TTF is not always the right choice. It has real costs:

1. **Latency overhead** — two backend calls instead of one (~10–35% overhead depending on backend)
2. **Token cost** — Pass 1 generates thinking tokens that are not in the final output
3. **Complexity** — more things can go wrong (Pass 1 failure, Pass 2 validation failure)

For simple tasks — extracting a city name, classifying a sentiment label, filling a template — direct constrained decoding works fine. The Format Tax is small (<2pp) and doesn't justify TTF overhead.

FormatShield's `ComplexityScorer` and `ThresholdOracle` exist to answer: **"Is this request complex enough that TTF is worth it?"**

The threshold is calibrated per backend (lower for vLLM with its KV cache reuse, higher for slow API backends) using the Φ routing score — a training-free information-theoretic measure derived from schema algebraic connectivity, constraint tightness, and prompt-schema compression distance. See [Reference: Oracle](../reference/oracle.md) for the Φ formula.

---

## Further Reading

- [arXiv 2408.02442](https://arxiv.org/abs/2408.02442) — "Let Me Speak Freely?" (Tam et al., 2024) — the original Format Tax paper
- [arXiv 2502.09061](https://arxiv.org/abs/2502.09061) — "CRANE" — the Think-Then-Format paper
- [Explanation: Routing Algorithm](routing-algorithm.md) — how FormatShield decides when to use TTF
- [Explanation: TTF Algorithm](ttf-algorithm.md) — how the two-pass engine works in detail
