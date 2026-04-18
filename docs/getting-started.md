# Getting Started

This guide will get you from zero to a working FormatShield integration in about 5 minutes.

---

## Prerequisites

- Python 3.11 or later
- A Groq API key (free tier available at [console.groq.com](https://console.groq.com)) **or** a locally running Ollama instance

---

## Step 1: Install

```bash
pip install formatshield
```

For Ollama (local, no API key needed):

```bash
pip install formatshield
ollama pull llama3.1
```

---

## Step 2: Set Your API Key

=== "Groq"

    ```bash
    export GROQ_API_KEY=gsk_your_key_here
    ```

=== "OpenRouter"

    ```bash
    export OPENROUTER_API_KEY=sk-or-your_key_here
    ```

=== "Ollama (no key needed)"

    Ollama runs locally. Just make sure the server is running:

    ```bash
    ollama serve
    ```

---

## Step 3: Your First Generation

Create a file `hello_formatshield.py`:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class CapitalCity(BaseModel):
    country: str
    capital: str
    population_millions: float
    fun_fact: str

async def main():
    result = await fs.generate(
        prompt="Tell me about the capital city of France.",
        schema=CapitalCity,
        model="groq/llama-3.1-70b-versatile",
    )

    # The parsed Pydantic model
    city = result.parsed
    print(f"{city.capital} is the capital of {city.country}")
    print(f"Population: ~{city.population_millions}M")
    print(f"Fun fact: {city.fun_fact}")

    # Routing metadata
    print(f"\nRouting strategy: {result.routing.strategy}")
    print(f"Complexity score: {result.complexity_score:.3f}")
    print(f"Latency: {result.latency_ms:.0f}ms")

asyncio.run(main())
```

Run it:

```bash
python hello_formatshield.py
```

Expected output:

```
Paris is the capital of France
Population: ~2.1M
Fun fact: The Eiffel Tower was originally intended to be a temporary structure.

Routing strategy: direct
Complexity score: 0.312
Latency: 287ms
```

The routing strategy is `direct` here because the prompt is simple. FormatShield scores it as low-complexity and skips TTF to save latency.

---

## Step 4: Try a Reasoning Task

Now try a task that actually benefits from Think-Then-Format routing:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class MathSolution(BaseModel):
    problem_restatement: str
    steps: list[str]
    answer: float
    units: str
    confidence: float

async def main():
    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        debug=True,  # prints routing trace
    )

    result = await shield.generate(
        prompt=(
            "A factory produces 240 widgets per hour. "
            "If the factory runs for 8 hours a day, 5 days a week, "
            "how many widgets does it produce in a 4-week month? "
            "Show your steps and give the answer."
        ),
        schema=MathSolution,
    )

    print(f"Answer: {result.parsed.answer} {result.parsed.units}")
    print(f"\nSteps:")
    for i, step in enumerate(result.parsed.steps, 1):
        print(f"  {i}. {step}")

    if result.thinking:
        print(f"\nThinking excerpt (first 200 chars):")
        print(f"  {result.thinking[:200]}...")

asyncio.run(main())
```

Because this prompt contains reasoning keywords (`steps`, `show`) and has a multi-field schema, the ComplexityScorer will assign a higher score. With `debug=True`, you'll see:

```
[FormatShield] model=groq/llama-3.1-70b-versatile
[FormatShield] complexity_score=0.721 (schema_depth=3, reasoning_ops=2, length_bucket=2)
[FormatShield] route=ttf | expected_delta=+0.170 | estimated_overhead=30%
[FormatShield] confidence=0.70 | explanation='Heuristic score 0.721 > threshold 0.650 ...'

Answer: 38400.0 widgets

Steps:
  1. Widgets per hour: 240
  2. Hours per day: 8 → 240 × 8 = 1920 widgets/day
  3. Days per week: 5 → 1920 × 5 = 9600 widgets/week
  4. Weeks per month: 4 → 9600 × 4 = 38400 widgets/month
```

---

## Step 5: Use the Synchronous API

If you are not in an async context, use `generate_sync`:

```python
import formatshield as fs
from pydantic import BaseModel

class Sentiment(BaseModel):
    label: str  # "positive", "negative", or "neutral"
    score: float
    reasoning: str

shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile")

result = shield.generate_sync(
    prompt="The new restaurant downtown has amazing food but terrible service.",
    schema=Sentiment,
)

print(result.parsed.label)   # "mixed" or "negative"
print(result.parsed.score)   # e.g. 0.3
```

---

## Step 6: Use the CLI

FormatShield ships a CLI for quick testing:

```bash
# Simple generation
formatshield generate "What is 17 * 23?" --model groq/llama-3.1-70b-versatile

# With a JSON schema
formatshield generate "Analyze the sentiment: I love this product!" \
  --model groq/llama-3.1-70b-versatile \
  --schema '{"type": "object", "properties": {"label": {"type": "string"}, "score": {"type": "number"}}, "required": ["label", "score"]}' \
  --debug
```

---

## Next Steps

- [Tutorial 01: Basic Generation](tutorials/01-basic-generation.md) — deeper dive into schemas and models
- [Tutorial 02: Routing](tutorials/02-routing.md) — understand and control the routing decision
- [Tutorial 04: Backends](tutorials/04-backends.md) — configure vLLM, Outlines, Guidance, and more
- [Explanation: The Format Tax](explanation/format-tax.md) — the research behind FormatShield

---

## Common Issues

??? question "I get `ImportError: No module named 'groq'`"

    The Groq client is included in the base install. If you see this error, try:
    ```bash
    pip install --upgrade formatshield
    ```

??? question "I get a 401 Unauthorized error"

    Make sure your `GROQ_API_KEY` environment variable is set:
    ```bash
    echo $GROQ_API_KEY
    ```
    If it is empty, export it again or add it to your `.env` file.

??? question "Ollama returns a connection error"

    Make sure `ollama serve` is running in a separate terminal window. By default FormatShield connects to `http://localhost:11434`.

??? question "The route is always `direct`, never `ttf`"

    The ThresholdOracle uses per-backend thresholds (default 0.65). Short or simple prompts will always route to direct. Try a reasoning-heavy prompt with a multi-field schema and `debug=True` to see the complexity score.
