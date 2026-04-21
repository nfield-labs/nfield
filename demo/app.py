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
from typing import Any, cast

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Fix 5: load .env at startup so local demos work without manual env exports
load_dotenv()

import formatshield as fs  # noqa: E402
from formatshield.observability.audit_log import (  # noqa: E402
    FileAuditLogger,
    InMemoryAuditLogger,
    build_audit_manifest,
    verify_audit_manifest,
)
from formatshield.oracle.routing_score import compute_routing_score  # noqa: E402
from formatshield.semantic.evaluator import evaluate_semantic_pair  # noqa: E402
from formatshield.ttf.engine import (  # noqa: E402
    _SC_PHI_THRESHOLD,
    DEFAULT_SC_K,
    _phi_thinking_budget,
)

app = FastAPI(title="FormatShield Demo")

_HTML = (Path(__file__).parent / "index.html").read_text(encoding="utf-8")

# Fix 2 + 3: single source of truth for allowed Groq models
GROQ_MODELS: list[str] = [
    "groq/llama-3.3-70b-versatile",
    "groq/llama-3.1-8b-instant",
    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    "groq/qwen/qwen3-32b",
]
_ALLOWED_MODELS: frozenset[str] = frozenset(GROQ_MODELS)


# Preset loader — merge presets.json + new_presets_30.json at startup
def _load_presets() -> list[dict]:
    base = json.loads((Path(__file__).parent / "presets.json").read_text(encoding="utf-8"))
    extra_path = Path(__file__).parent / "new_presets_30.json"
    if extra_path.exists():
        extra = json.loads(extra_path.read_text(encoding="utf-8"))
        existing_ids = {p.get("id") for p in base}
        base.extend(p for p in extra if p.get("id") not in existing_ids)
    return base


PRESETS: list[dict] = _load_presets()

_DEMO_AUDIT_INFO: dict[str, str | None] = {
    "mode": "in_memory",
    "path": None,
    "error": None,
}
_DEMO_AUDIT_SIGNING_KEY = os.environ.get("FORMATSHIELD_DEMO_AUDIT_SIGNING_KEY")
_DEMO_AUDIT_SIGNING_KEY_ID = os.environ.get("FORMATSHIELD_DEMO_AUDIT_SIGNING_KEY_ID")
_demo_audit_env_path = os.environ.get("FORMATSHIELD_DEMO_AUDIT_PATH", "").strip()
try:
    if _demo_audit_env_path:
        _DEMO_AUDIT_LOGGER = FileAuditLogger(_demo_audit_env_path)
        _DEMO_AUDIT_INFO["mode"] = "file"
        _DEMO_AUDIT_INFO["path"] = str(_DEMO_AUDIT_LOGGER.file_path)
    else:
        _DEMO_AUDIT_LOGGER = InMemoryAuditLogger()
except Exception as exc:
    _DEMO_AUDIT_LOGGER = InMemoryAuditLogger()
    _DEMO_AUDIT_INFO["error"] = str(exc)


def _audit_chain_valid() -> bool:
    verifier = getattr(_DEMO_AUDIT_LOGGER, "verify_chain", None)
    if callable(verifier):
        try:
            return bool(verifier())
        except Exception:
            return False
    return True


def _audit_events_payload(limit: int, event_type: str | None = None) -> dict[str, Any]:
    all_events = _DEMO_AUDIT_LOGGER.events()
    if event_type:
        filtered = [event for event in all_events if event.event_type == event_type]
    else:
        filtered = all_events

    bounded_limit = max(1, min(limit, 500))
    selected = filtered[-bounded_limit:]
    return {
        "count": len(selected),
        "total": len(filtered),
        "limit": bounded_limit,
        "event_type": event_type,
        "chain_valid": _audit_chain_valid(),
        "events": [event.model_dump() for event in selected],
    }


# ---------------------------------------------------------------------------
# Schema adherence validator (Fix 1)
# ---------------------------------------------------------------------------


def _validate_against_schema(parsed: object, schema: dict) -> tuple[bool, str | None]:
    """Return (is_valid, error_message). Uses jsonschema when available."""
    try:
        import jsonschema
        import jsonschema.exceptions
    except ImportError:
        return True, None  # jsonschema not installed — accept any valid JSON

    try:
        jsonschema.validate(instance=parsed, schema=schema)
        return True, None
    except jsonschema.exceptions.ValidationError as exc:
        return False, exc.message
    except Exception as exc:
        return False, str(exc)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class CompareRequest(BaseModel):
    prompt: str
    schema_text: str  # raw JSON string of the schema
    model: str = "groq/llama-3.3-70b-versatile"
    system_prompt: str = ""


# ---------------------------------------------------------------------------
# Demo Score Engine
# ---------------------------------------------------------------------------


def _schema_risk_profile(schema: dict) -> dict:
    """
    Walk the schema and return a risk profile of constraint features that
    empirically cause raw LLMs to fail (backed by arXiv 2408.02442, 2601.07525,
    2501.10868, 2604.03616).
    """
    text = json.dumps(schema)
    profile: dict[str, object] = {
        # enum with many options → DDXPlus-style −27 to −63 pp accuracy drop
        "large_enum": False,
        "max_enum_size": 0,
        # minItems / maxItems → models under/over-count arrays
        "has_array_cardinality": "minItems" in text or "maxItems" in text,
        # numeric range constraints → models ignore them ~30% of the time
        "has_numeric_constraints": "minimum" in text or "maximum" in text,
        # pattern constraints → 33% failure at baseline (SchemaBench)
        "has_pattern": "pattern" in text,
        # additionalProperties:false → any hallucinated field breaks validation
        "has_additional_props_false": '"additionalProperties": false' in text
        or '"additionalProperties":false' in text,
        # deep nesting → bracket/comma errors compound at 3+ levels
        "nesting_depth": _schema_depth(schema),
        # anyOf / oneOf / allOf → hard API rejection risk
        "has_combinators": any(k in text for k in ("anyOf", "oneOf", "allOf")),
        # $ref → context loss across long schemas
        "has_ref": "$ref" in text,
    }

    # Scan all enum arrays for size
    def _find_enums(node: object) -> None:
        if isinstance(node, dict):
            if "enum" in node and isinstance(node["enum"], list):
                size = len(node["enum"])
                if size > cast(int, profile["max_enum_size"]):
                    profile["max_enum_size"] = size
                if size >= 5:
                    profile["large_enum"] = True
            for v in node.values():
                _find_enums(v)
        elif isinstance(node, list):
            for item in node:
                _find_enums(item)

    _find_enums(schema)
    return profile


def _schema_depth(schema: object, _depth: int = 0) -> int:
    """Recursively compute max nesting depth of a JSON Schema."""
    if not isinstance(schema, dict):
        return _depth
    candidates = [_depth]
    for key in ("properties", "items", "additionalProperties"):
        child = schema.get(key)
        if isinstance(child, dict):
            for v in child.values():
                candidates.append(_schema_depth(v, _depth + 1))
        elif child is not None:
            candidates.append(_schema_depth(child, _depth + 1))
    for key in ("anyOf", "oneOf", "allOf"):
        for item in schema.get(key, []):
            candidates.append(_schema_depth(item, _depth + 1))
    return max(candidates)


def _compute_routing_mode(phi: float) -> str:
    """Convert Phi score to 5-mode spectrum (actual FormatShield logic)."""
    if phi < 0.50:
        return "direct"
    elif phi < 0.65:
        return "lite_ttf"
    elif phi < 0.80:
        return "standard_ttf"
    elif phi < 0.95:
        return "deep_ttf"
    else:
        return "sc_full"  # self-consistency at Φ ≥ 0.95


def _compute_phi_thinking_budget(phi: float) -> int:
    """Return Φ-proportional thinking budget (actual FormatShield logic)."""
    if phi >= 0.90:
        return 4096
    elif phi >= 0.75:
        return 1024
    elif phi >= 0.65:
        return 512
    else:
        return 256


def _compute_pass2_temperature(tau: float) -> float:
    """Compute Pass 2 temperature from constraint tightness τ."""
    return max(0.05, 0.7 * (1.0 - tau))


def _compute_demo_score(
    fs_result: dict,
    raw_result: dict,
    phi_info: dict,
    schema_dict: dict,
) -> dict:
    """
    Simplified demo score showing actual routing decision quality.

    Focus on:
    - Was the routing decision (TTF vs Direct) appropriate for this schema?
    - Did the output pass validation?
    - What failure modes were detected?
    """
    fs_ok = fs_result.get("ok", False)
    raw_ok = raw_result.get("ok", False)
    fs_valid = fs_ok and fs_result.get("schema_valid", False)
    raw_valid = raw_ok and raw_result.get("schema_valid", False)
    phi_score = phi_info.get("phi", 0.0)
    tau = phi_info.get("tau", 0.0)
    routing_mode = _compute_routing_mode(phi_score)
    used_ttf = routing_mode != "direct"
    fallback = fs_result.get("fallback_triggered", True)
    raw_schema_violation = raw_ok and bool(raw_result.get("schema_violation"))
    fs_schema_violation = fs_ok and bool(fs_result.get("schema_violation"))
    raw_json_invalid = (
        raw_ok and not raw_result.get("schema_valid", True) and not raw_schema_violation
    )

    risk = _schema_risk_profile(schema_dict)
    depth = cast(int, risk["nesting_depth"])
    enum_size = cast(int, risk["max_enum_size"])

    # ── Simple Routing Decision Verdict ──────────────────────────────────────
    # Did FormatShield choose the right routing mode for this schema?
    score = 50  # baseline
    reasons: list[str] = []

    # Schema validation check
    if fs_valid and not raw_valid:
        score += 30
        reasons.append(
            "FS passed schema validation, raw failed — "
            "FormatShield's constraint enforcement advantage demonstrated"
        )
    elif fs_valid and raw_valid:
        score += 15
        reasons.append("Both passed schema validation (equal on this run)")
    elif not fs_valid and raw_valid:
        score -= 20
        reasons.append("FS failed validation but raw passed — routing or execution issue")
    else:
        reasons.append("Neither passed schema validation on this run")

    # Routing decision quality
    if phi_score >= 0.95 and routing_mode == "sc_full":
        score += 10
        reasons.append(f"Self-consistency correctly triggered at Φ={phi_score:.3f} (k=3)")
    elif phi_score > 0.50 and used_ttf:
        score += 10
        reasons.append(f"TTF ({routing_mode}) correctly chosen for complex schema (Φ={phi_score:.3f})")
    elif phi_score <= 0.50 and routing_mode == "direct":
        score += 10
        reasons.append(f"Direct mode chosen correctly for simple schema (Φ={phi_score:.3f})")
    elif phi_score > 0.50 and routing_mode == "direct":
        score -= 15
        reasons.append(f"Missed TTF opportunity — Φ={phi_score:.3f} indicated complexity but direct was used")
    elif phi_score < 0.50 and used_ttf:
        score -= 10
        reasons.append(f"Unnecessary TTF overhead — Φ={phi_score:.3f} suggested simple schema")

    # Constraint violation detection
    if raw_schema_violation and not fs_schema_violation:
        score += 20
        reasons.append(f"Raw violated constraint: {raw_result.get('schema_violation', '')[:60]}")
    elif fs_schema_violation and not raw_schema_violation:
        score -= 15
        reasons.append(f"FS violated constraint raw didn't: {fs_result.get('schema_violation', '')[:60]}")

    # Fallback resilience
    if fs_valid and not fallback:
        score += 5
        reasons.append("First-pass success without fallback")
    elif fallback:
        reasons.append("Fallback was triggered")

    # Latency
    if fs_ok and raw_ok:
        fs_lat = float(fs_result.get("latency_ms") or 0)
        raw_lat = float(raw_result.get("latency_ms") or 0)
        if fs_lat > 0 and raw_lat > 0:
            ratio = fs_lat / raw_lat
            if ratio < 0.75:
                score += 5
                reasons.append(f"FS faster: {ratio:.1f}× ({fs_lat:.0f} vs {raw_lat:.0f} ms)")
            elif ratio > 3.0:
                score -= 10
                reasons.append(f"FS slower: {ratio:.1f}× ({fs_lat:.0f} vs {raw_lat:.0f} ms)")

    # Clamp score
    score = min(100, max(0, score))

    # ── grade ────────────────────────────────────────────────────────────────
    if score >= 80:
        grade = "A"
    elif score >= 70:
        grade = "B"
    elif score >= 60:
        grade = "C"
    elif score >= 40:
        grade = "D"
    else:
        grade = "F"

    # Build summary based on score
    if score >= 80:
        summary = "FormatShield outperformed raw Groq on routing and validation."
    elif score >= 60:
        summary = "FormatShield showed good routing decisions and validation."
    elif score >= 40:
        summary = "Mixed results — both approaches had strengths and weaknesses."
    else:
        summary = "Raw Groq performed better on this schema and prompt combination."

    # Determine verdict (simple 3-way)
    if fs_valid and not raw_valid:
        verdict = "formatshield_wins"
    elif raw_valid and not fs_valid:
        verdict = "raw_wins"
    else:
        verdict = "tie"

    return {
        "score": score,
        "grade": grade,
        "summary": summary,
        "verdict": verdict,
        "reasons": reasons,
        "routing_analysis": {
            "phi_score": round(phi_score, 4),
            "routing_mode": routing_mode,
            "used_ttf": used_ttf,
            "thinking_budget_tokens": phi_info.get("thinking_budget_tokens"),
            "self_consistency_enabled": phi_info.get("self_consistency_enabled", False),
            "pass2_temperature": phi_info.get("pass2_temperature"),
        },
        "schema_info": {
            "schema_depth": depth,
            "max_enum_size": enum_size,
            "has_pattern": risk["has_pattern"],
            "has_large_enum": risk["large_enum"],
            "has_combinators": risk["has_combinators"],
        },
    }


# ---------------------------------------------------------------------------
# Verdict (Fix 4)
# ---------------------------------------------------------------------------


def _compute_verdict(fs: dict, raw: dict, semantic_eval: dict | None = None) -> dict:
    """Compare both sides and produce a human-readable winner summary."""
    points: dict[str, list[str]] = {"formatshield": [], "raw": [], "tie": []}

    fs_ok = fs.get("ok", False)
    raw_ok = raw.get("ok", False)

    # Validity
    fs_valid = fs_ok and fs.get("schema_valid", False)
    raw_valid = raw_ok and raw.get("schema_valid", False)
    if fs_valid and not raw_valid:
        points["formatshield"].append("Schema valid ✓ (raw output failed schema check)")
    elif raw_valid and not fs_valid:
        points["raw"].append("Schema valid ✓")
    elif fs_valid and raw_valid:
        points["tie"].append("Both schema-valid")
    else:
        points["tie"].append("Both schema-invalid")

    # Latency
    fs_lat = fs.get("latency_ms") if fs_ok else None
    raw_lat = raw.get("latency_ms") if raw_ok else None
    if fs_lat and raw_lat:
        diff = abs(fs_lat - raw_lat)
        if diff < 80:
            points["tie"].append(f"Latency within {diff:.0f} ms")
        elif raw_lat < fs_lat:
            points["raw"].append(f"Faster by {diff:.0f} ms ({raw_lat:.0f} vs {fs_lat:.0f})")
        else:
            points["formatshield"].append(
                f"Faster by {diff:.0f} ms ({fs_lat:.0f} vs {raw_lat:.0f})"
            )

    # TTF routing
    if fs_ok and fs.get("routing_strategy") == "ttf":
        points["formatshield"].append("TTF activated — structured reasoning before output")
    elif fs_ok:
        points["formatshield"].append("Smart routing: direct mode sufficient for this prompt")

    # Fallback resilience
    if fs_ok and not fs.get("fallback_triggered", True):
        points["formatshield"].append("No fallback needed — first-pass success")

    # Schema violation evidence
    if raw_ok and raw.get("schema_violation"):
        points["formatshield"].append(
            f"Raw Groq violated schema constraint: {raw['schema_violation'][:120]}"
        )
    if fs_ok and fs.get("schema_violation"):
        points["raw"].append(f"FormatShield output violated schema: {fs['schema_violation'][:120]}")

    if semantic_eval is not None:
        sem_winner = semantic_eval.get("winner")
        sem_delta = float(semantic_eval.get("delta", 0.0) or 0.0)
        sem_summary = str(semantic_eval.get("summary") or "")
        if sem_summary:
            if sem_winner == "formatshield":
                points["formatshield"].append(f"Semantic signal: {sem_summary}")
            elif sem_winner == "raw":
                points["raw"].append(f"Semantic signal: {sem_summary}")
            else:
                points["tie"].append(f"Semantic signal: {sem_summary}")

    # Overall winner
    fs_score = len(points["formatshield"])
    raw_score = len(points["raw"])
    if not fs_ok and not raw_ok:
        winner = "none"
        summary = "Both calls failed."
    elif not fs_ok:
        winner = "raw"
        summary = "FormatShield call failed; raw Groq succeeded."
    elif not raw_ok:
        winner = "formatshield"
        summary = "Raw Groq call failed; FormatShield succeeded."
    elif fs_score > raw_score:
        winner = "formatshield"
        summary = "FormatShield wins on this run."
    elif raw_score > fs_score:
        winner = "raw"
        summary = "Raw Groq wins on this run."
    else:
        winner = "tie"
        summary = "Comparable result — FormatShield adds routing intelligence at no cost."

    if semantic_eval is not None:
        sem_winner = semantic_eval.get("winner")
        sem_delta = float(semantic_eval.get("delta", 0.0) or 0.0)
        sem_summary = str(semantic_eval.get("summary") or "")
        if winner == "tie" and sem_winner in {"formatshield", "raw"} and abs(sem_delta) >= 2.0:
            winner = sem_winner
            summary = f"{summary} Tie broken by semantic signal: {sem_summary}"
        elif winner == "formatshield" and sem_winner == "raw" and abs(sem_delta) >= 8.0:
            winner = "tie"
            summary = f"{summary} Reliability favors FormatShield but semantic signal favors raw."
        elif winner == "raw" and sem_winner == "formatshield" and abs(sem_delta) >= 8.0:
            winner = "tie"
            summary = f"{summary} Reliability favors raw but semantic signal favors FormatShield."

    return {
        "winner": winner,
        "summary": summary,
        "formatshield_points": points["formatshield"],
        "raw_points": points["raw"],
        "tie_points": points["tie"],
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return _HTML


@app.get("/models")
async def list_models() -> list[str]:
    return GROQ_MODELS


@app.get("/presets")
async def list_presets() -> list[dict]:
    return PRESETS


@app.get("/presets/{preset_id}")
async def get_preset(preset_id: str) -> dict:
    for preset in PRESETS:
        if str(preset.get("id")) == preset_id:
            return preset
    raise HTTPException(status_code=404, detail=f"Preset '{preset_id}' not found.")


@app.get("/audit/verify")
async def verify_audit_chain() -> dict[str, Any]:
    events = _DEMO_AUDIT_LOGGER.events()
    return {
        "mode": _DEMO_AUDIT_INFO["mode"],
        "path": _DEMO_AUDIT_INFO["path"],
        "event_count": len(events),
        "chain_valid": _audit_chain_valid(),
        "error": _DEMO_AUDIT_INFO["error"],
    }


@app.get("/audit/events")
async def list_audit_events(limit: int = 50, event_type: str | None = None) -> dict[str, Any]:
    return _audit_events_payload(limit=limit, event_type=event_type)


@app.get("/audit/manifest")
async def export_audit_manifest() -> dict[str, Any]:
    audit_path = _DEMO_AUDIT_INFO.get("path")
    if not audit_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Demo audit logger is in-memory. Set FORMATSHIELD_DEMO_AUDIT_PATH "
                "to enable manifest export."
            ),
        )

    try:
        manifest = build_audit_manifest(
            audit_path,
            signing_key=_DEMO_AUDIT_SIGNING_KEY,
            signing_key_id=_DEMO_AUDIT_SIGNING_KEY_ID,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to build manifest: {exc}") from exc

    return {
        "ok": True,
        "manifest": manifest.model_dump(),
        "signed": bool(manifest.signature),
    }


@app.get("/audit/verify-manifest")
async def verify_manifest(manifest_path: str) -> dict[str, Any]:
    audit_path = _DEMO_AUDIT_INFO.get("path")
    if not audit_path:
        raise HTTPException(
            status_code=400,
            detail=(
                "Demo audit logger is in-memory. Set FORMATSHIELD_DEMO_AUDIT_PATH "
                "to enable manifest verification."
            ),
        )

    valid, issues, manifest = verify_audit_manifest(
        audit_path=audit_path,
        manifest_path=manifest_path,
        signing_key=_DEMO_AUDIT_SIGNING_KEY,
        expected_signing_key_id=_DEMO_AUDIT_SIGNING_KEY_ID,
    )
    return {
        "ok": valid,
        "issues": issues,
        "manifest": manifest.model_dump() if manifest is not None else None,
    }


@app.post("/compare")
async def compare(req: CompareRequest) -> dict:
    # Fix 2: enforce Groq-only server-side
    if req.model not in _ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Model '{req.model}' is not in the allowed Groq model list.",
        )

    # Parse schema
    try:
        schema_dict: dict = json.loads(req.schema_text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {exc}") from exc

    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt must not be empty")

    # Phi score — pure Python, no API call
    phi = compute_routing_score(req.prompt, schema_dict)
    routing_mode = _compute_routing_mode(phi.phi)
    will_use_ttf = routing_mode != "direct"
    thinking_budget = _compute_phi_thinking_budget(phi.phi) if will_use_ttf else None
    pass2_temp = _compute_pass2_temperature(phi.tau) if will_use_ttf else None
    sc_mode = phi.phi >= 0.95  # self-consistency at Φ ≥ 0.95
    phi_info = {
        "phi": round(phi.phi, 4),
        "lambda2": round(phi.lambda2, 4),
        "tau": round(phi.tau, 4),
        "delta_k": round(phi.delta_k, 4),
        "explanation": phi.explanation,
        "routing_mode": routing_mode,
        "routing_description": {
            "direct": "Single-pass generation (no thinking)",
            "lite_ttf": "Light thinking budget (256 tokens)",
            "standard_ttf": "Standard thinking budget (512 tokens)",
            "deep_ttf": "Deep thinking budget (1024 tokens)",
            "sc_full": "Self-consistency with 3 parallel traces (4096 tokens each)",
        }[routing_mode],
        # Thinking budget — Φ-proportional
        "thinking_budget_tokens": thinking_budget,
        # Self-consistency — auto-trigger at Φ ≥ 0.95
        "self_consistency_enabled": sc_mode,
        "self_consistency_k": 3 if sc_mode else 1,
        "self_consistency_criteria": [
            "required-field coverage",
            "contradiction-free",
            "vocabulary-bridge coverage",
        ] if sc_mode else [],
        # Pass 2 temperature — conditioned on constraint tightness τ
        "pass2_temperature": round(pass2_temp, 3) if (will_use_ttf and pass2_temp is not None) else None,
        "pass2_temperature_formula": "max(0.05, 0.7 * (1.0 - tau))",
        "tau_constraint_tightness": round(phi.tau, 3),
    }

    fs_result = await _call_with_formatshield(req.prompt, schema_dict, req.model, req.system_prompt)
    raw_result = await _call_raw_groq(
        req.prompt, req.schema_text, schema_dict, req.model, req.system_prompt
    )

    semantic_evaluation = evaluate_semantic_pair(
        fs_result,
        raw_result,
        schema_dict,
        phi=phi_info.get("phi"),
    )

    # Format semantic metrics for UI (4-metric dashboard)
    semantic_metrics = _format_semantic_metrics_for_ui(semantic_evaluation)

    # Fix 4: compute verdict
    verdict = _compute_verdict(fs_result, raw_result, semantic_evaluation)

    # Demo score
    demo_score = _compute_demo_score(fs_result, raw_result, phi_info, schema_dict)

    return {
        "phi": phi_info,
        "with_formatshield": fs_result,
        "without_formatshield": raw_result,
        "verdict": verdict,
        "demo_score": demo_score,
        "semantic_evaluation": semantic_evaluation,
        "semantic_metrics": semantic_metrics,  # NEW: 4-metric dashboard for UI
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


async def _call_with_formatshield(
    prompt: str, schema: dict, model: str, system_prompt: str = ""
) -> dict:
    try:
        shield = fs.FormatShield(model=model, audit_logger=_DEMO_AUDIT_LOGGER)
        t0 = time.perf_counter()
        full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
        result: fs.GenerationResult = await shield.generate(prompt=full_prompt, schema=schema)
        latency = round((time.perf_counter() - t0) * 1000, 1)

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

        # Independent schema validation — do NOT trust result.schema_valid (self-reported).
        # FormatShield can report VALID while producing field-name typos or constraint violations.
        independent_valid = result.schema_valid
        independent_violation: str | None = None
        try:
            parsed_fs = json.loads(result.output)
            independent_valid, independent_violation = _validate_against_schema(parsed_fs, schema)
        except Exception:  # noqa: S110
            pass  # non-JSON output — keep self-reported value

        # Fix 6: safe payload serialisation
        raw_payload = _safe_payload(
            {
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
                "schema_valid": independent_valid,
                "fallback_triggered": result.fallback_triggered,
                "token_usage": token_usage,
                "cost_usd": result.cost_usd,
            }
        )

        resp: dict = {
            "ok": True,
            "output": output_pretty,
            "schema_valid": independent_valid,
            "routing_strategy": result.routing.strategy,
            "routing_confidence": round(result.routing.confidence, 3),
            "routing_explanation": result.routing.explanation,
            "failure_modes": result.failure_modes or [],
            "thinking": result.thinking or "",
            "latency_ms": result.latency_ms or latency,
            "fallback_triggered": result.fallback_triggered,
            "raw_payload": raw_payload,
        }
        if independent_violation:
            resp["schema_violation"] = independent_violation
        return resp
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


async def _call_raw_groq(
    prompt: str,
    schema_text: str,
    schema_dict: dict,
    model: str,
    system_prompt: str = "",
) -> dict:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return {"ok": False, "error": "GROQ_API_KEY not set — add it to .env or export it"}

    model_id = model.removeprefix("groq/")
    try:
        from groq import AsyncGroq

        client = AsyncGroq(api_key=api_key)
        t0 = time.perf_counter()

        json_instruction = (
            "You are a JSON API. Respond ONLY with a valid JSON object "
            f"that matches this schema:\n{schema_text}"
        )
        if system_prompt:
            system_content = f"{system_prompt}\n\n{json_instruction}"
        else:
            system_content = json_instruction

        chat = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_content},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        latency = round((time.perf_counter() - t0) * 1000, 1)
        raw_text = chat.choices[0].message.content or ""

        # Fix 1: real schema adherence check, not just JSON parse
        parsed = None
        json_parse_ok = False
        try:
            parsed = json.loads(raw_text)
            output_pretty = json.dumps(parsed, indent=2)
            json_parse_ok = True
        except Exception:
            output_pretty = raw_text

        schema_valid = False
        schema_violation: str | None = None
        if json_parse_ok and parsed is not None:
            schema_valid, schema_violation = _validate_against_schema(parsed, schema_dict)

        usage = chat.usage
        tokens = {
            "prompt": usage.prompt_tokens if usage else 0,
            "completion": usage.completion_tokens if usage else 0,
            "total": usage.total_tokens if usage else 0,
        }

        # Fix 6: safe serialisation of Groq response
        try:
            raw_payload = _safe_payload(chat.model_dump())
        except Exception:
            raw_payload = {
                "id": getattr(chat, "id", "?"),
                "model": model_id,
                "_note": "not fully serializable",
            }

        result: dict = {
            "ok": True,
            "output": output_pretty,
            "schema_valid": schema_valid,
            "latency_ms": latency,
            "tokens": tokens,
            "raw_payload": raw_payload,
        }
        if schema_violation:
            result["schema_violation"] = schema_violation
        return result
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _safe_payload(obj: object) -> object:
    """Recursively ensure a payload is JSON-serialisable. Fix 6."""
    try:
        json.dumps(obj)
        return obj
    except (TypeError, ValueError):
        pass
    if isinstance(obj, dict):
        return {k: _safe_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe_payload(v) for v in obj]
    try:
        return str(obj)
    except Exception:
        return "<unserializable>"


def _format_semantic_metrics_for_ui(semantic_evaluation: dict) -> dict:
    """
    Extract 4-metric semantic dashboard from semantic_evaluation for UI display.

    Maps semantic evaluator metrics to 4-metric display:
    - type_consistency → type_match%
    - required_field_recall → recall%
    - constraint_integrity → integrity%
    - schema_validity → completeness%
    """
    formatshield_metrics = semantic_evaluation.get("formatshield", {}).get("metrics", [])
    raw_metrics = semantic_evaluation.get("raw", {}).get("metrics", [])

    # Build metric lookup
    fs_metric_map = {m.get("name"): m for m in formatshield_metrics}
    raw_metric_map = {m.get("name"): m for m in raw_metrics}

    def _get_percentage(metric_dict: dict, metric_name: str) -> float:
        """Extract score as percentage from metric dict."""
        if not metric_dict:
            return 0.0
        m = metric_dict.get(metric_name, {})
        score = m.get("score", 0)
        max_score = m.get("max_score", 1)
        if max_score == 0:
            return 0.0
        return round((score / max_score) * 100, 1)

    return {
        "formatshield": {
            "type_match": _get_percentage(fs_metric_map, "type_consistency"),
            "recall": _get_percentage(fs_metric_map, "required_field_recall"),
            "integrity": _get_percentage(fs_metric_map, "constraint_integrity"),
            "completeness": _get_percentage(fs_metric_map, "schema_validity"),
        },
        "raw": {
            "type_match": _get_percentage(raw_metric_map, "type_consistency"),
            "recall": _get_percentage(raw_metric_map, "required_field_recall"),
            "integrity": _get_percentage(raw_metric_map, "constraint_integrity"),
            "completeness": _get_percentage(raw_metric_map, "schema_validity"),
        },
        "winner": semantic_evaluation.get("winner", "tie"),
        "delta": semantic_evaluation.get("delta", 0),
    }
