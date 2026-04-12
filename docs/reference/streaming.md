# Reference — Streaming

This page documents the `StreamEvent` dataclass and the streaming API surface across `FormatShield`, `TTFEngine`, and backends.

---

## `StreamEvent`

```python
@dataclass
class StreamEvent:
    type: str
    content: str | None = None
    token: str | None = None
    json: dict[str, Any] | None = None
    backend: str = ""
    latency_ms: float = 0.0
```

Every item yielded from `FormatShield.stream()`, `TTFEngine.stream()`, or a backend's `stream()` method is a `StreamEvent`.

### Fields

| Field | Type | Present When | Description |
|---|---|---|---|
| `type` | `str` | Always | Event type: `"thinking"`, `"output"`, or `"complete"` |
| `content` | `str \| None` | `thinking` and `complete` events | Raw text content |
| `token` | `str \| None` | `output` events | Incremental token chunk |
| `json` | `dict \| None` | `complete` events | Assembled and JSON-parsed final output. `None` if JSON parsing failed |
| `backend` | `str` | Always | Backend that emitted this event (e.g. `"groq"`) |
| `latency_ms` | `float` | Always | Cumulative time since stream start, in milliseconds |

---

## Event Types

### `"thinking"` Events

Emitted during TTF Pass 1 (unconstrained reasoning).

```python
StreamEvent(
    type="thinking",
    content="I need to consider the constraints carefully...",
    backend="groq",
    latency_ms=245.3,
)
```

- Only emitted when `expose_thinking=True` on the `FormatShield` instance (for `FormatShield.stream()`)
- Always emitted by `TTFEngine.stream()` (callers can filter by checking `type == "thinking"`)
- Never emitted for direct-route requests (no Pass 1 occurs)
- If Pass 1 fails, a single `thinking` event with `content="[Pass 1 failed: ...]"` is emitted

### `"output"` Events

Emitted during Pass 2 (or the single pass for direct routes) as token chunks arrive.

```python
StreamEvent(
    type="output",
    token='{"answer":',
    backend="groq",
    latency_ms=891.2,
)
```

- `token` contains an incremental chunk (may be a partial token, full token, or multiple tokens depending on the backend's streaming granularity)
- `content` is `None` for output events

### `"complete"` Events

Emitted once after all tokens have been received.

```python
StreamEvent(
    type="complete",
    json={"answer": 42.0, "steps": ["step 1", "step 2"]},
    content='{"answer": 42.0, "steps": ["step 1", "step 2"]}',
    backend="groq",
    latency_ms=1634.8,
)
```

- `json` is the assembled output parsed as a Python dict. `None` if the assembled output is not valid JSON.
- `content` is the full assembled raw string (joining all `output` tokens)
- There is exactly one `complete` event per stream, always the last event

---

## Stream Sequence Guarantees

### TTF Route (thinking exposed)

```
thinking(content=...) × N  (Pass 1 chunks)
output(token=...)   × M  (Pass 2 chunks)
complete(json=...)         (exactly once, last event)
```

### TTF Route (thinking suppressed)

```
output(token=...)   × M  (Pass 2 chunks)
complete(json=...)         (exactly once, last event)
```

### Direct Route

```
output(token=...)   × M
complete(json=...)
```

### TTF with Pass 1 Failure

```
thinking(content="[Pass 1 failed: ...]")  (error indicator)
output(token=...)   × M                    (direct generation fallback)
complete(json=...)
```

### TTF with Pass 2 Failure

```
thinking(content=...) × N
complete(json=None)                        (no output events)
```

---

## `FormatShield.stream()` API

```python
async def stream(
    self,
    prompt: str,
    schema: type[BaseModel] | dict[str, Any] | None = None,
) -> AsyncIterator[StreamEvent]: ...
```

Internally, `FormatShield.stream()`:

1. Scores the prompt and schema via `ComplexityScorer`
2. Checks for failure modes via `FailureModeDetector`
3. Gets a routing decision from `ThresholdOracle`
4. If TTF: delegates to `TTFEngine._stream_impl()` and filters `"thinking"` events if `expose_thinking=False`
5. If direct: delegates to `backend.stream()` directly

---

## Backend Stream Implementations

Each backend implements `stream()` returning an `AsyncIterator[StreamEvent]`. The expected implementation pattern:

```python
async def stream(
    self,
    prompt: str,
    schema: dict[str, Any] | None = None,
    constraints: str | None = None,
) -> AsyncIterator[StreamEvent]:
    import time
    t0 = time.monotonic()

    # Make streaming API call
    async with httpx.AsyncClient() as client:
        async with client.stream("POST", url, json=payload) as response:
            async for chunk in response.aiter_lines():
                if chunk:
                    token = parse_chunk(chunk)
                    yield StreamEvent(
                        type="output",
                        token=token,
                        backend=self.name,
                        latency_ms=(time.monotonic() - t0) * 1000,
                    )

    # Final complete event
    yield StreamEvent(
        type="complete",
        json=None,  # assembler upstream will handle this
        backend=self.name,
        latency_ms=(time.monotonic() - t0) * 1000,
    )
```

---

## Filtering Events

Common patterns for working with the stream:

```python
async def handle_stream(shield, prompt, schema):
    thinking_text = []
    output_tokens = []

    async for event in shield.stream(prompt, schema=schema):
        match event.type:
            case "thinking":
                thinking_text.append(event.content or "")
                # Update a "thinking" UI element
            case "output":
                output_tokens.append(event.token or "")
                # Append to output buffer / render incrementally
            case "complete":
                # Final result available
                final = event.json
                total_latency = event.latency_ms

    return {
        "thinking": "".join(thinking_text),
        "output": "".join(output_tokens),
        "parsed": final,
        "latency_ms": total_latency,
    }
```

---

## Server-Sent Events (SSE) Serialisation

To send `StreamEvent` objects over HTTP as SSE:

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import json
import formatshield as fs

app = FastAPI()
shield = fs.FormatShield(model="groq/llama-3.1-70b-versatile", expose_thinking=True)

@app.post("/stream")
async def stream_endpoint(request: dict):
    async def event_generator():
        async for event in shield.stream(request["prompt"]):
            data = json.dumps({
                "type": event.type,
                "content": event.content,
                "token": event.token,
                "json": event.json,
                "latency_ms": event.latency_ms,
            })
            yield f"data: {data}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```
