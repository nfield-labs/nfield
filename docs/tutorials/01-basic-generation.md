# Tutorial 01 — Basic Generation

This tutorial covers the fundamentals of structured generation with FormatShield: defining schemas, calling `generate`, and interpreting the `GenerationResult`.

---

## Prerequisites

- FormatShield installed (`pip install formatshield`)
- An API key set in the environment (`GROQ_API_KEY` or `OPENROUTER_API_KEY`)

---

## 1. Define a Pydantic Schema

FormatShield accepts any `pydantic.BaseModel` subclass as a schema.  The model's fields become the expected keys in the JSON output.

```python
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str = Field(description="Name of the product being reviewed")
    rating: int = Field(ge=1, le=5, description="Star rating from 1 to 5")
    pros: list[str] = Field(description="List of positive aspects")
    cons: list[str] = Field(description="List of negative aspects")
    summary: str = Field(description="One-sentence summary")
    recommend: bool = Field(description="Would the reviewer recommend this product?")
```

!!! tip "Field annotations improve routing"
    FormatShield's `ComplexityScorer` reads the schema depth and constraint count.
    Annotating fields with `Field(description=...)` makes the schema richer, which
    raises the complexity score slightly — exactly the right signal for a detailed schema.

---

## 2. Run Your First Generation

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel, Field

class ProductReview(BaseModel):
    product_name: str
    rating: int
    pros: list[str]
    cons: list[str]
    summary: str
    recommend: bool

async def main():
    result = await fs.generate(
        prompt=(
            "Review this product: Noise-cancelling wireless headphones. "
            "Great sound quality, 30-hour battery, comfortable fit. "
            "However, they feel a bit plasticky and the app is glitchy."
        ),
        schema=ProductReview,
        model="groq/llama-3.1-70b-versatile",
    )

    # Access the parsed Pydantic model
    review: ProductReview = result.parsed
    print(f"Product: {review.product_name}")
    print(f"Rating: {'★' * review.rating}")
    print(f"Pros: {', '.join(review.pros)}")
    print(f"Cons: {', '.join(review.cons)}")
    print(f"Recommend: {'Yes' if review.recommend else 'No'}")

asyncio.run(main())
```

---

## 3. Inspect the GenerationResult

Every call to `generate()` returns a `GenerationResult` dataclass with rich metadata:

| Field | Type | Description |
|---|---|---|
| `output` | `str` | Raw JSON string from the backend |
| `parsed` | `BaseModel \| dict \| None` | Validated Pydantic model instance |
| `thinking` | `str \| None` | Pass 1 thinking text (TTF routes only) |
| `routing` | `RoutingDecision` | How and why the request was routed |
| `complexity_score` | `float` | Scalar score in [0, 1] |
| `failure_modes` | `list[str]` | Any failure modes detected |
| `latency_ms` | `float` | Total wall-clock time in milliseconds |
| `backend` | `str` | Backend used (e.g. `"groq"`) |
| `model` | `str` | Full model identifier |
| `schema_valid` | `bool` | Whether Pydantic validation passed |
| `fallback_triggered` | `bool` | Whether TTF fell back to direct |

```python
print(result.routing.strategy)         # "direct" or "ttf"
print(result.complexity_score)         # e.g. 0.412
print(result.latency_ms)               # e.g. 341.2
print(result.schema_valid)             # True
print(result.routing.explanation)      # human-readable routing reason
```

---

## 4. Using the FormatShield Class Directly

For multiple calls sharing the same model and config, instantiate `FormatShield` once:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str   # "positive", "negative", "neutral", "mixed"
    score: float # confidence 0.0–1.0
    reasoning: str

async def main():
    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,            # print routing trace on each call
        expose_thinking=True,  # include thinking text in result.thinking
    )

    texts = [
        "This product changed my life. Absolutely love it!",
        "Terrible quality, broke after two days. Very disappointed.",
        "It's fine. Does what it says. Nothing special.",
    ]

    for text in texts:
        result = await shield.generate(
            prompt=f"Analyze the sentiment of this review: {text}",
            schema=Sentiment,
        )
        print(f"{result.parsed.label:10s} ({result.parsed.score:.2f}) — {text[:40]}...")

asyncio.run(main())
```

---

## 5. Synchronous API

If you're not in an async context (e.g. a Flask route, a script), use `generate_sync`:

```python
import formatshield as fs
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    word_count: int

shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")

result = shield.generate_sync(
    prompt="Summarize: The quick brown fox jumps over the lazy dog.",
    schema=Summary,
)
print(result.parsed.title)
print(result.parsed.key_points)
```

!!! note "Thread safety"
    `generate_sync` is safe to call from within a running event loop (e.g. Jupyter notebooks,
    pytest-asyncio). It spins up a dedicated thread with its own event loop.

---

## 6. Using a JSON Schema Dict Instead of Pydantic

You can also pass a raw JSON Schema dict if you don't want to define a Pydantic model:

```python
import asyncio
import formatshield as fs

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "skills": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["name", "age", "skills"],
}

async def main():
    result = await fs.generate(
        prompt="Create a fictional software engineer profile.",
        schema=schema,
        model="groq/llama-3.1-70b-versatile",
    )
    # result.parsed will be a plain dict
    print(result.parsed["name"])
    print(result.parsed["skills"])

asyncio.run(main())
```

---

## 7. Loading Config from File

For production deployments, store your FormatShield config in a YAML or JSON file:

=== "YAML (formatshield.yml)"

    ```yaml
    model: groq/llama-3.1-70b-versatile
    debug: false
    ttf_fallback: true
    expose_thinking: false
    latency_budget_ms: 5000
    ```

=== "JSON (formatshield.json)"

    ```json
    {
      "model": "groq/llama-3.1-70b-versatile",
      "debug": false,
      "ttf_fallback": true,
      "expose_thinking": false,
      "latency_budget_ms": 5000
    }
    ```

Then load it:

```python
import formatshield as fs

shield = fs.FormatShield.from_config("formatshield.yml")
```

---

## Next Steps

- [Tutorial 02: Routing](02-routing.md) — understand and control the TTF vs direct routing decision
- [Tutorial 04: Backends](04-backends.md) — configure different inference backends
- [Reference: Core API](../reference/core.md) — full API reference for `FormatShield` and `GenerationResult`
