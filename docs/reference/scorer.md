# Reference — Scorer

This page documents `ComplexityScorer`, `ComplexityFeatures`, `SchemaAnalyzer`, and related types from `formatshield.scorer`.

---

## `ComplexityScorer`

```python
class ComplexityScorer:
    def __init__(self, encoding_name: str = "cl100k_base") -> None: ...
```

Computes a `ComplexityFeatures` object and a scalar complexity score for an inference request.

### Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `encoding_name` | `str` | `"cl100k_base"` | tiktoken encoding to use for tokenisation. `"cl100k_base"` covers GPT-3.5/4 vocabularies and is a good proxy for most modern LLMs |

If tiktoken is not installed or the encoding fails to load, `ComplexityScorer` falls back to character-level ordinal encoding, which still produces a meaningful entropy estimate.

---

### `ComplexityScorer.score()`

```python
def score(
    self,
    prompt: str,
    *,
    schema: dict | None = None,
    model_id: str = "",
) -> ComplexityFeatures: ...
```

Compute and return a `ComplexityFeatures` for the given prompt.

| Parameter | Type | Description |
|---|---|---|
| `prompt` | `str` | Full prompt string (system + user messages concatenated, or just the user message) |
| `schema` | `dict \| None` | Optional target JSON Schema dict. When `None`, schema-based features default to minimal values |
| `model_id` | `str` | Model identifier for instruction-tune score lookup (e.g. `"gpt-4o"`, `"llama-3.1-70b-versatile"`) |

Returns `ComplexityFeatures`. Returns neutral features on any error (does not raise).

---

### `ComplexityScorer.compute_score()`

```python
def compute_score(self, features: ComplexityFeatures) -> float: ...
```

Convert a `ComplexityFeatures` object to a single float in [0, 1].

Higher values indicate higher complexity and a greater likelihood of benefiting from TTF.

The score is a weighted linear combination of normalised features:

```
score = (
    0.20 × token_entropy_normalised +
    0.25 × schema_depth_normalised   +
    0.20 × reasoning_ops_normalised  +
    0.15 × instruction_tune_score    +
    0.10 × length_bucket_normalised  +
    0.10 × constraint_count_normalised
)
```

Returns `0.5` on any error (does not raise).

---

## `ComplexityFeatures`

```python
@dataclass
class ComplexityFeatures:
    token_entropy: float
    schema_depth: int
    required_reasoning_ops: int
    instruction_tune_score: float
    prompt_length_bucket: int
    schema_constraint_count: int
```

### Fields

| Field | Type | Description |
|---|---|---|
| `token_entropy` | `float` | Normalised Shannon entropy of the prompt's token-ID distribution. 0.0 = all same tokens, 1.0 = all unique tokens. In [0, 1] |
| `schema_depth` | `int` | Maximum nesting depth of the JSON Schema. 0 = no schema, 1 = flat object, 2+ = nested objects/arrays |
| `required_reasoning_ops` | `int` | Count of CoT reasoning keyword occurrences in the prompt (case-insensitive whole-word match) |
| `instruction_tune_score` | `float` | Per-model RLHF strength score. Native thinkers (o1, o3) = 1.0; GPT-4/Claude-3 = 0.8; LLaMA/Mistral = 0.5; unknown = 0.4 |
| `prompt_length_bucket` | `int` | Token length bucket: 0 = short (<50 tokens), 1 = medium (50–200), 2 = long (200–1000), 3 = very long (>1000) |
| `schema_constraint_count` | `int` | Total number of JSON Schema constraint keywords (e.g. `minimum`, `maxLength`, `enum`, `pattern`) |

### `ComplexityFeatures.to_feature_vector()`

```python
def to_feature_vector(self) -> list[float]: ...
```

Returns the features as a list of floats in the canonical order used by the sklearn oracle:
`[token_entropy, schema_depth, required_reasoning_ops, instruction_tune_score, prompt_length_bucket, schema_constraint_count]`

---

## CoT Keyword List

The `required_reasoning_ops` feature counts occurrences of these keywords (whole-word, case-insensitive):

```
because, therefore, step, analyze, analyse, calculate, reason,
prove, derive, solve, compare, evaluate, explain
```

---

## Instruction-Tune Score Lookup

| Model prefix | Score | Notes |
|---|---|---|
| `o1-mini`, `o1-preview`, `o1`, `o3`, `o3-mini` | 1.0 | Native thinkers, heavy RLHF |
| `deepseek-r1` | 0.9 | DeepSeek R1 family |
| `gpt-4` | 0.8 | GPT-4 family |
| `claude-3` | 0.8 | Claude-3 family |
| `llama-3` | 0.5 | Open-source instruction-tuned |
| `mistral` | 0.5 | Open-source instruction-tuned |
| (default) | 0.4 | Unknown model |

Matching is longest-prefix first, case-insensitive.

---

## `SchemaAnalyzer`

```python
class SchemaAnalyzer:
    def analyze(self, schema: dict) -> tuple[int, int]: ...
```

Analyzes a JSON Schema dict to extract:
- `schema_depth` — maximum object nesting depth
- `schema_constraint_count` — total number of constraint keywords

### Counted Constraint Keywords

`minimum`, `maximum`, `exclusiveMinimum`, `exclusiveMaximum`, `minLength`, `maxLength`, `pattern`, `enum`, `const`, `minItems`, `maxItems`, `uniqueItems`, `minProperties`, `maxProperties`, `required`, `additionalProperties`, `format`, `allOf`, `anyOf`, `oneOf`, `not`, `if`, `then`, `else`

---

## Feature Normalisation Caps

When computing the scalar score, raw feature values are clipped before weighting:

| Feature | Cap | Effect |
|---|---|---|
| `token_entropy` | 1.0 | Already in [0, 1] |
| `schema_depth` | 10.0 | Schemas deeper than 10 levels all score 1.0 |
| `required_reasoning_ops` | 20.0 | >20 reasoning keywords all score 1.0 |
| `instruction_tune_score` | 1.0 | Already in [0, 1] |
| `prompt_length_bucket` | 3.0 | Bucket 3 (very long) scores 1.0 |
| `schema_constraint_count` | 30.0 | >30 constraints all score 1.0 |

---

## Example: Direct Scoring

```python
from formatshield.scorer.complexity_scorer import ComplexityScorer

scorer = ComplexityScorer()

simple_prompt = "What is 2 + 2?"
simple_schema = {"type": "object", "properties": {"answer": {"type": "number"}}, "required": ["answer"]}

complex_prompt = (
    "Analyze and compare the time complexity of quicksort vs mergesort. "
    "Calculate the expected number of comparisons for n=1000 and derive "
    "the optimal choice for different input distributions. "
    "Explain your reasoning step by step."
)
complex_schema = {
    "type": "object",
    "properties": {
        "quicksort_complexity": {"type": "string"},
        "mergesort_complexity": {"type": "string"},
        "n_1000_quicksort_comparisons": {"type": "number"},
        "n_1000_mergesort_comparisons": {"type": "number"},
        "optimal_choice": {
            "type": "object",
            "properties": {
                "algorithm": {"type": "string"},
                "justification": {"type": "string"},
                "conditions": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["algorithm", "justification", "conditions"],
        },
    },
    "required": ["quicksort_complexity", "mergesort_complexity", "optimal_choice"],
}

for label, prompt, schema in [
    ("simple", simple_prompt, simple_schema),
    ("complex", complex_prompt, complex_schema),
]:
    features = scorer.score(prompt, schema=schema, model_id="groq/llama-3.1-70b-versatile")
    score = scorer.compute_score(features)
    print(f"\n{label.upper()} prompt:")
    print(f"  entropy={features.token_entropy:.3f}  depth={features.schema_depth}  "
          f"ops={features.required_reasoning_ops}  score={score:.3f}")
```

Expected output:

```
SIMPLE prompt:
  entropy=0.721  depth=1  ops=0  score=0.283

COMPLEX prompt:
  entropy=0.891  depth=3  ops=5  score=0.742
```
