# Routing Algorithm

FormatShield decides between the **Think-Then-Format (TTF)** strategy and **direct constrained generation** on every inference request.

---

## The Two Strategies

| Strategy | Description | Latency | Accuracy |
|---|---|---|---|
| **Direct** | Single-pass constrained decoding (JSON FSM) | Low | Loses up to 27% on reasoning tasks |
| **TTF** | Two-pass: think freely then format output | Higher (+20–80%) | Recovers accuracy for complex tasks |

---

## Pipeline Overview

```
Request: (prompt, schema, model)
           │
           ▼
    ┌─────────────────┐
    │ ComplexityScorer│  token entropy + schema depth +
    │                 │  reasoning ops + instruction-tune
    └────────┬────────┘
             │ complexity_score in [0, 1]
             ▼
    ┌─────────────────┐
    │ ThresholdOracle │  per-backend threshold OR
    │                 │  sklearn LogisticRegression
    └────────┬────────┘
             │
     ┌───────┴──────┐
     ▼              ▼
  "direct"        "ttf"
```

---

## ComplexityScorer

`ComplexityScorer.score(prompt, schema, model_id)` computes six features and `compute_score(features)` reduces them to a scalar in **[0, 1]**.

### Features and Weights

| Feature | Weight | Description |
|---|---|---|
| `token_entropy` | 0.20 | Shannon entropy of tokenised prompt (normalised) |
| `schema_depth` | 0.25 | Maximum nesting depth of the JSON schema |
| `required_reasoning_ops` | 0.20 | Count of CoT keywords (because, step, analyse, …) |
| `instruction_tune_score` | 0.15 | Model capability — o1 → 1.0, Llama → 0.5 |
| `prompt_length_bucket` | 0.10 | 0–3 bucket by token count |
| `schema_constraint_count` | 0.10 | Number of JSON schema constraints |

**High score (→ 1.0):** complex reasoning prompt + deep schema + capable model
**Low score (→ 0.0):** short extraction prompt + flat schema + basic model

---

## ThresholdOracle

### Heuristic Mode (default)

A per-backend threshold is compared against the complexity score:

```python
if complexity_score >= threshold:
    return RoutingDecision(strategy="ttf")
else:
    return RoutingDecision(strategy="direct")
```

### Learned Mode (sklearn)

```python
oracle = ThresholdOracle(use_learned_threshold=True)
oracle.fit(feature_matrix, labels)  # calibrate on your data
decision = oracle.route(features)
```

---

## Tuning the Router

```python
# Conservative: TTF only for highly complex requests
oracle = ThresholdOracle(backend="groq", threshold=0.65)

# Always direct (lowest latency)
fs = FormatShield(strategy="direct")

# Always TTF (highest accuracy)
fs = FormatShield(strategy="ttf")
```

---

## See Also

- [TTF Algorithm](ttf-algorithm.md)
- [When TTF Hurts](when-ttf-hurts.md)
- [The Format Tax](format-tax.md)
