# Tutorial 06 — LangChain Integration

FormatShield ships a `FormatShieldRunnable` that wraps the core `FormatShield` class as a LangChain `Runnable`. This lets you drop FormatShield into any LangChain chain, pipeline, or LangGraph workflow.

---

## Installation

```bash
pip install "formatshield[langchain]" langchain langchain-core
```

---

## 1. Basic Usage as a Runnable

```python
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class SentimentResult(BaseModel):
    label: str   # "positive", "negative", "neutral"
    score: float
    reasoning: str

runnable = FormatShieldRunnable(
    model="groq/llama-3.1-70b-versatile",
    schema=SentimentResult,
)

# Invoke synchronously (LangChain's standard interface)
result = runnable.invoke("The new MacBook Pro is absolutely fantastic. Best laptop I've owned.")
print(result.label)    # "positive"
print(result.score)    # e.g. 0.94
```

The runnable returns the parsed Pydantic model directly (not the full `GenerationResult`). This matches LangChain's convention of returning the "output value" rather than metadata-laden wrapper objects.

---

## 2. Chain Composition

Use FormatShield inside a LangChain `|` chain:

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class EntityList(BaseModel):
    people: list[str]
    organizations: list[str]
    locations: list[str]

# Step 1: prompt → NER extraction
ner_runnable = FormatShieldRunnable(
    model="groq/llama-3.1-70b-versatile",
    schema=EntityList,
    debug=True,
)

# Use with a prompt template
prompt = ChatPromptTemplate.from_template(
    "Extract named entities from this text: {text}"
)

# Build chain: format prompt → FormatShield extraction
chain = prompt | (lambda msg: msg.content) | ner_runnable

result = chain.invoke({"text": "Apple CEO Tim Cook announced the new iPhone at WWDC in San Francisco."})
print(result.people)        # ["Tim Cook"]
print(result.organizations) # ["Apple"]
print(result.locations)     # ["San Francisco"]
```

---

## 3. Async Invocation

```python
import asyncio
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class Summary(BaseModel):
    title: str
    key_points: list[str]
    conclusion: str

async def main():
    runnable = FormatShieldRunnable(
        model="groq/llama-3.1-70b-versatile",
        schema=Summary,
    )

    result = await runnable.ainvoke(
        "Summarize the key contributions of the CRANE paper on constrained LLM generation."
    )
    print(result.title)
    for point in result.key_points:
        print(f"  - {point}")

asyncio.run(main())
```

---

## 4. Batch Processing

```python
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class Classification(BaseModel):
    category: str
    confidence: float

runnable = FormatShieldRunnable(
    model="groq/llama-3.1-70b-versatile",
    schema=Classification,
)

texts = [
    "Classify this email: Your account has been suspended due to suspicious activity.",
    "Classify this email: Don't miss our 50% off sale this weekend only!",
    "Classify this email: Your order #12345 has shipped and will arrive Friday.",
]

# Batch invocation
results = runnable.batch(texts)
for text, result in zip(texts, results):
    print(f"{result.category:20s} ({result.confidence:.2f}) — {text[:50]}...")
```

---

## 5. Streaming in LangChain

```python
import asyncio
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class Story(BaseModel):
    title: str
    opening: str
    conflict: str
    resolution: str

async def main():
    runnable = FormatShieldRunnable(
        model="groq/llama-3.1-70b-versatile",
        schema=Story,
        expose_thinking=True,
    )

    print("Streaming story generation:\n")
    async for chunk in runnable.astream("Write a short sci-fi story about a robot learning to paint."):
        print(chunk, end="", flush=True)
    print()

asyncio.run(main())
```

---

## 6. FormatShieldRunnable Constructor Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | `str` | required | Model string (`"groq/..."`, `"ollama/..."`, etc.) |
| `schema` | `type[BaseModel] \| dict \| None` | `None` | Output schema |
| `debug` | `bool` | `False` | Print routing trace |
| `expose_thinking` | `bool` | `False` | Include thinking in streaming output |
| `ttf_fallback` | `bool` | `True` | Retry with direct on TTF failure |
| `latency_budget_ms` | `float \| None` | `None` | Hard latency cap |
| `api_key` | `str \| None` | `None` | Override env var API key |
| `base_url` | `str \| None` | `None` | Override backend base URL |

---

## 7. Using in a LangGraph Workflow

FormatShield integrates naturally with LangGraph for multi-step agentic workflows:

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict
from formatshield.integrations.langchain import FormatShieldRunnable
from pydantic import BaseModel

class AnalysisState(TypedDict):
    input_text: str
    entities: dict
    sentiment: dict
    final_report: str

class EntityResult(BaseModel):
    people: list[str]
    organizations: list[str]

class SentimentResult(BaseModel):
    label: str
    score: float

entity_extractor = FormatShieldRunnable(
    model="groq/llama-3.1-70b-versatile",
    schema=EntityResult,
)
sentiment_analyzer = FormatShieldRunnable(
    model="groq/llama-3.1-70b-versatile",
    schema=SentimentResult,
)

def extract_entities(state: AnalysisState) -> AnalysisState:
    result = entity_extractor.invoke(f"Extract entities: {state['input_text']}")
    state["entities"] = {"people": result.people, "organizations": result.organizations}
    return state

def analyze_sentiment(state: AnalysisState) -> AnalysisState:
    result = sentiment_analyzer.invoke(f"Analyze sentiment: {state['input_text']}")
    state["sentiment"] = {"label": result.label, "score": result.score}
    return state

def generate_report(state: AnalysisState) -> AnalysisState:
    entities = state["entities"]
    sentiment = state["sentiment"]
    state["final_report"] = (
        f"Entities: {entities['people']} / {entities['organizations']}. "
        f"Sentiment: {sentiment['label']} ({sentiment['score']:.2f})"
    )
    return state

# Build the graph
graph = StateGraph(AnalysisState)
graph.add_node("extract_entities", extract_entities)
graph.add_node("analyze_sentiment", analyze_sentiment)
graph.add_node("generate_report", generate_report)

graph.set_entry_point("extract_entities")
graph.add_edge("extract_entities", "analyze_sentiment")
graph.add_edge("analyze_sentiment", "generate_report")
graph.add_edge("generate_report", END)

app = graph.compile()

result = app.invoke({
    "input_text": "Elon Musk announced that Tesla will open a new factory in Mexico.",
    "entities": {},
    "sentiment": {},
    "final_report": "",
})
print(result["final_report"])
```

---

## 8. Comparing FormatShield vs Direct LangChain Structured Output

| Feature | FormatShield + LangChain | LangChain `.with_structured_output()` |
|---|---|---|
| Auto routing (TTF vs direct) | **Yes** | No |
| Format Tax mitigation | **Yes** | No |
| Accuracy loss measurement | **Yes** | No |
| Multi-backend support | **Yes** | Partial |
| Streaming thinking | **Yes** | No |
| LangChain Runnable interface | Yes | Yes |

---

## Next Steps

- [Tutorial 07: Observability](07-observability.md) — metrics and structured logging
- [Tutorial 08: Contributing](08-contributing.md) — how to contribute to FormatShield
- [Reference: Core](../reference/core.md) — full API reference
