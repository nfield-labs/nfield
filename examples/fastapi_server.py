"""
FastAPI streaming server — FormatShield SSE example.

Run:
    pip install fastapi uvicorn
    uvicorn examples.fastapi_server:app --reload

Then:
    curl http://localhost:8000/generate \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Analyse patient: 45M, T2DM + HTN. List treatment plan.", "model": "groq/llama-3.1-70b-versatile"}'

Streaming endpoint (SSE):
    curl http://localhost:8000/stream \\
      -H "Content-Type: application/json" \\
      -d '{"prompt": "Solve: A train leaves at 60mph...", "model": "groq/llama-3.1-70b-versatile"}'
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from pydantic import BaseModel

import formatshield as fs

# ---------------------------------------------------------------------------
# FastAPI app (imported lazily so the file is importable without fastapi)
# ---------------------------------------------------------------------------

try:
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, StreamingResponse

    app = FastAPI(
        title="FormatShield API",
        description="Intelligent TTF routing for structured LLM generation.",
        version=fs.__version__,
    )
except ImportError as exc:
    raise ImportError("pip install fastapi uvicorn to run this server") from exc


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for /generate and /stream endpoints."""

    prompt: str
    model: str = "groq/llama-3.1-70b-versatile"
    schema_json: dict[str, Any] | None = None
    expose_thinking: bool = False
    debug: bool = False


class GenerateResponse(BaseModel):
    """JSON response from /generate."""

    output: str
    route: str
    complexity_score: float
    latency_ms: float
    schema_valid: bool
    fallback_triggered: bool
    thinking: str | None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run FormatShield structured generation and return the full result as JSON.

    This is the non-streaming endpoint. For token-by-token streaming use /stream.
    """
    shield = fs.FormatShield(
        model=request.model,
        expose_thinking=request.expose_thinking,
        debug=request.debug,
    )
    result = await shield.generate(request.prompt, schema=request.schema_json)
    return GenerateResponse(
        output=result.output,
        route=result.routing.strategy,
        complexity_score=result.complexity_score,
        latency_ms=result.latency_ms,
        schema_valid=result.schema_valid,
        fallback_triggered=result.fallback_triggered,
        thinking=result.thinking,
    )


@app.post("/stream")
async def stream(request: GenerateRequest) -> StreamingResponse:
    """
    Stream FormatShield generation as Server-Sent Events (SSE).

    Each SSE line is a JSON-encoded StreamEvent:
        data: {"type": "thinking", "content": "..."}\\n\\n
        data: {"type": "output",   "token": "..."}\\n\\n
        data: {"type": "complete", "content": "..."}\\n\\n

    Thinking events are suppressed unless expose_thinking=true.
    """
    shield = fs.FormatShield(
        model=request.model,
        expose_thinking=request.expose_thinking,
    )

    async def event_generator() -> AsyncIterator[str]:
        async for event in shield.stream(request.prompt, schema=request.schema_json):
            yield _to_sse(event.__dict__)
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/health")
async def health() -> JSONResponse:
    """Health check endpoint."""
    return JSONResponse({"status": "ok", "version": fs.__version__})


@app.get("/backends")
async def list_backends() -> JSONResponse:
    """List available backends and their capability flags."""
    return JSONResponse(
        {
            "backends": [
                {
                    "name": "groq",
                    "supports_kv_cache_reuse": False,
                    "accuracy_loss_baseline": 0.18,
                    "notes": "Fastest inference via Groq LPU. Free tier at groq.com.",
                },
                {
                    "name": "openrouter",
                    "supports_kv_cache_reuse": False,
                    "accuracy_loss_baseline": 0.20,
                    "notes": "200+ models via OpenRouter unified API.",
                },
                {
                    "name": "ollama",
                    "supports_kv_cache_reuse": False,
                    "accuracy_loss_baseline": 0.22,
                    "notes": "Local inference. No API key required.",
                },
                {
                    "name": "vllm",
                    "supports_kv_cache_reuse": True,
                    "accuracy_loss_baseline": 0.23,
                    "notes": "Local vLLM server with native prefix caching.",
                },
            ]
        }
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_sse(payload: dict[str, Any]) -> str:
    """Format a dict as an SSE data line."""
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


# ---------------------------------------------------------------------------
# Development entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn  # type: ignore[import]

    uvicorn.run("examples.fastapi_server:app", host="0.0.0.0", port=8000, reload=True)  # noqa: S104
