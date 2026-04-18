"""
FormatShield Demo — side-by-side comparison: raw Groq vs FormatShield.

Run:
    cd <repo-root>
    uv run uvicorn demo.app:app --reload --port 8000
Then open http://localhost:8000
"""

from __future__ import annotations

import dataclasses
import json
import os
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

import formatshield as fs
from formatshield.oracle.routing_score import compute_routing_score

app = FastAPI(title="FormatShield Demo")

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

GROQ_MODELS = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/qwen/qwen3-32b",
]


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    prompt: str
    schema_text: str  # raw JSON string of the schema
    model: str = "groq/llama-3.3-70b-versatile"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/models")
async def list_models() -> list[str]:
    return GROQ_MODELS


@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    # -- parse schema --------------------------------------------------------
    try:
        schema_dict: dict = json.loads(req.schema_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {exc}") from exc

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    # -- phi score (pure Python, no API call) --------------------------------
    phi = compute_routing_score(req.prompt, schema_dict)
    phi_info = {
        "phi": round(phi.phi, 4),
        "lambda2": round(phi.lambda2, 4),
        "tau": round(phi.tau, 4),
        "delta_k": round(phi.delta_k, 4),
        "explanation": phi.explanation,
        "recommendation": "TTF" if phi.phi > 0.65 else "Direct",
    }

    # -- FormatShield call ---------------------------------------------------
    fs_result = await _call_with_formatshield(req.prompt, schema_dict, req.model)

    # -- Raw Groq call -------------------------------------------------------
    raw_result = await _call_raw_groq(req.prompt, req.schema_text, req.model)

    return {
        "phi": phi_info,
        "with_formatshield": fs_result,
        "without_formatshield": raw_result,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _call_with_formatshield(prompt: str, schema: dict, model: str) -> dict:
    try:
        shield = fs.FormatShield(model=model)
        t0 = time.perf_counter()
        result: fs.GenerationResult = await shield.generate(prompt=prompt, schema=schema)
        latency = round((time.perf_counter() - t0) * 1000, 1)

        # pretty-print output JSON if possible
        try:
            output_pretty = json.dumps(json.loads(result.output), indent=2)
        except Exception:
            output_pretty = result.output or ""

        token_usage = None
        if result.token_usage is not None:
            try:
                token_usage = dataclasses.asdict(result.token_usage)
            except Exception:
                token_usage = str(result.token_usage)

        raw_payload = {
            "output": result.output,
            "thinking": result.thinking,
            "routing": {
                "strategy": result.routing.strategy,
                "confidence": result.routing.confidence,
                "explanation": result.routing.explanation,
                "expected_accuracy_delta": result.routing.expected_accuracy_delta,
                "expected_overhead_pct": result.routing.expected_overhead_pct,
            },
            "complexity_score": result.complexity_score,
            "failure_modes": result.failure_modes,
            "latency_ms": result.latency_ms,
            "backend": result.backend,
            "model": result.model,
            "schema_valid": result.schema_valid,
            "fallback_triggered": result.fallback_triggered,
            "token_usage": token_usage,
            "cost_usd": result.cost_usd,
        }

        return {
            "ok": True,
            "output": output_pretty,
            "schema_valid": result.schema_valid,
            "routing_strategy": result.routing.strategy,
            "routing_confidence": round(result.routing.confidence, 3),
            "routing_explanation": result.routing.explanation,
            "failure_modes": result.failure_modes or [],
            "thinking": result.thinking or "",
            "latency_ms": result.latency_ms or latency,
            "fallback_triggered": result.fallback_triggered,
            "raw_payload": raw_payload,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _call_raw_groq(prompt: str, schema_json: str, model: str) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GROQ_API_KEY not set"}

    model_id = model.removeprefix("groq/")
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        t0 = time.perf_counter()
        chat = await client.chat.completions.create(
            model=model_id,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a JSON API. Respond ONLY with a valid JSON object "
                        f"that matches this schema:\n{schema_json}"
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        raw_text = chat.choices[0].message.content or ""

        try:
            output_pretty = json.dumps(json.loads(raw_text), indent=2)
            valid = True
        except Exception:
            output_pretty = raw_text
            valid = False

        usage = chat.usage
        tokens = {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        }

        # Full Groq response as raw payload (Pydantic model → dict)
        try:
            raw_payload = chat.model_dump()
        except Exception:
            raw_payload = {"id": chat.id, "model": chat.model, "error": "not serializable"}

        return {
            "ok": True,
            "output": output_pretty,
            "schema_valid": valid,
            "latency_ms": latency,
            "tokens": tokens,
            "raw_payload": raw_payload,
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
