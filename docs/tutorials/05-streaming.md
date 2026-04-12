# Tutorial 05 — Streaming

FormatShield supports streaming generation that emits incremental token events as the model generates, including a distinct `"thinking"` event stream during TTF's reasoning pass.

---

## Why Stream?

Streaming is valuable for:

- **User interfaces** — start rendering output before generation is complete
- **Long responses** — avoid buffering multi-thousand-token outputs in memory
- **TTF transparency** — show users the model's reasoning in real-time alongside the JSON output

---

## 1. Basic Streaming

Use `shield.stream()` to get an async iterator of `StreamEvent` objects:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class TravelPlan(BaseModel):
    destination: str
    duration_days: int
    activities: list[str]
    estimated_budget_usd: float
    best_season: str

async def main():
    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        expose_thinking=True,  # include thinking events in the stream
    )

    print("Generating travel plan...\n")

    async for event in shield.stream(
        prompt="Plan a 7-day trip to Japan for a solo traveler interested in history and food.",
        schema=TravelPlan,
    ):
        if event.type == "thinking":
            print(f"[thinking] {event.content}", end="", flush=True)
        elif event.type == "output":
            print(f"[output]   {event.token}", end="", flush=True)
        elif event.type == "complete":
            print(f"\n\n[complete] Latency: {event.latency_ms:.0f}ms")
            if event.json:
                import json
                print(json.dumps(event.json, indent=2))

asyncio.run(main())
```

---

## 2. StreamEvent Fields

Each `StreamEvent` has the following fields:

| Field | Type | Present When |
|---|---|---|
| `type` | `str` | Always: `"thinking"`, `"output"`, or `"complete"` |
| `content` | `str \| None` | `thinking` events and `complete` events |
| `token` | `str \| None` | `output` events (incremental token chunks) |
| `json` | `dict \| None` | `complete` events (assembled and parsed JSON) |
| `backend` | `str` | Always |
| `latency_ms` | `float` | Always (cumulative since stream start) |

### Event Sequence

For a TTF-routed request the event sequence is:

```
thinking(content="I need to...")
thinking(content="First, let me consider...")
thinking(content="...") × many
output(token="{")
output(token='"destination"')
output(token='":')
output(token='"Japan"')
output(token=",") × many more
complete(json={...}, content='{"destination": "Japan", ...}')
```

For a direct-routed request there are no `thinking` events — only `output` and `complete`.

---

## 3. Streaming Without expose_thinking

By default, `expose_thinking=False`. The model still runs the thinking pass internally, but `"thinking"` events are filtered out of the stream. You only see `"output"` and `"complete"`:

```python
shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    expose_thinking=False,  # default
)

async for event in shield.stream(prompt, schema=MySchema):
    if event.type == "output":
        print(event.token, end="", flush=True)
    elif event.type == "complete":
        print()
        print(event.json)
```

---

## 4. Building a Streaming Web Handler

Here's an example of using FormatShield streaming in a FastAPI endpoint:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import formatshield as fs
from pydantic import BaseModel
import json

app = FastAPI()

class AnalysisSchema(BaseModel):
    sentiment: str
    score: float
    key_phrases: list[str]
    summary: str

shield = fs.FormatShield(
    model="groq/llama-3.1-70b-versatile",
    expose_thinking=False,
)

@app.get("/stream-analyze")
async def stream_analyze(text: str):
    async def generate():
        async for event in shield.stream(
            prompt=f"Analyze this text: {text}",
            schema=AnalysisSchema,
        ):
            if event.type == "output":
                yield event.token or ""
            elif event.type == "complete":
                # Send the final parsed result as a special marker
                yield f"\n[[COMPLETE]]{json.dumps(event.json)}"

    return StreamingResponse(generate(), media_type="text/plain")
```

---

## 5. Streaming with Thinking Visible (Two-Column UI)

A more sophisticated UI can show thinking and JSON output in separate columns:

```python
import asyncio
import formatshield as fs
from pydantic import BaseModel

class SolutionSchema(BaseModel):
    answer: float
    explanation: str
    confidence: float

async def two_column_stream(prompt: str):
    shield = fs.FormatShield(
        model="groq/llama-3.1-70b-versatile",
        expose_thinking=True,
    )

    thinking_buffer = []
    output_buffer = []

    async for event in shield.stream(prompt, schema=SolutionSchema):
        if event.type == "thinking":
            thinking_buffer.append(event.content or "")
            # In a real UI you'd push to a WebSocket or SSE channel
            print(f"\033[34m[THINK]\033[0m {event.content}", end="", flush=True)

        elif event.type == "output":
            output_buffer.append(event.token or "")
            print(f"\033[32m[OUT]\033[0m {event.token}", end="", flush=True)

        elif event.type == "complete":
            print(f"\n\n--- Complete ({event.latency_ms:.0f}ms) ---")
            if event.json:
                print(f"Answer: {event.json.get('answer')}")
                print(f"Confidence: {event.json.get('confidence')}")

asyncio.run(two_column_stream(
    "A train leaves city A at 9am traveling at 80km/h. "
    "Another train leaves city B at 10am traveling at 120km/h toward A. "
    "The cities are 500km apart. When do they meet?"
))
```

---

## 6. Handling Stream Errors

If the backend encounters an error during streaming, a `complete` event is emitted with `json=None`:

```python
async for event in shield.stream(prompt, schema=MySchema):
    if event.type == "complete":
        if event.json is None:
            print("Generation failed — no valid JSON produced")
        else:
            print(f"Success: {event.json}")
```

TTF-specific error events: if Pass 1 (thinking) fails, a `thinking` event with `content="[Pass 1 failed: ...]"` is emitted and generation continues with direct formatting.

---

## 7. Converting Stream to a Full GenerationResult

If you need the streaming interface but also want a full `GenerationResult` at the end, collect the events manually:

```python
async def stream_to_result(shield, prompt, schema):
    thinking_parts = []
    output_parts = []
    final_json = None
    final_latency = 0.0

    async for event in shield.stream(prompt, schema=schema):
        if event.type == "thinking":
            thinking_parts.append(event.content or "")
        elif event.type == "output":
            output_parts.append(event.token or "")
        elif event.type == "complete":
            final_json = event.json
            final_latency = event.latency_ms

    return {
        "thinking": "".join(thinking_parts),
        "output": "".join(output_parts),
        "parsed": final_json,
        "latency_ms": final_latency,
    }
```

---

## Next Steps

- [Tutorial 06: LangChain Integration](06-langchain.md) — use FormatShield as a LangChain runnable
- [Reference: Streaming](../reference/streaming.md) — full `StreamEvent` API reference
- [Reference: Core](../reference/core.md) — `FormatShield.stream()` method signature
