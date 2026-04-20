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


def _compute_demo_score(
    fs_result: dict,
    raw_result: dict,
    phi_info: dict,
    schema_dict: dict,
) -> dict:
    """
    Compute a research-backed 0-100 Demo Score.

    Scoring axes — winner reflects *real output quality*, not process compliance:
    - schema_validity   (0-30): FS valid when raw fails → clear win; both valid → 10 pts only
    - constraint_edge   (0-25): TTF warranted by schema complexity; penalised if TTF ran
                                 on a simple schema (unnecessary overhead)
    - reliability       (0-20): first-pass success, no fallback required
    - constraint_protection (0-25): raw violated a schema constraint FormatShield enforced
    - latency_efficiency (-15 to +5): raw 3× faster on simple schema → −15 pts
    """
    fs_ok = fs_result.get("ok", False)
    raw_ok = raw_result.get("ok", False)
    fs_valid = fs_ok and fs_result.get("schema_valid", False)
    raw_valid = raw_ok and raw_result.get("schema_valid", False)
    phi_score = phi_info.get("phi", 0.0)
    used_ttf = fs_ok and fs_result.get("routing_strategy") == "ttf"
    fallback = fs_result.get("fallback_triggered", True)
    raw_schema_violation = raw_ok and bool(raw_result.get("schema_violation"))
    fs_schema_violation = fs_ok and bool(fs_result.get("schema_violation"))
    raw_json_invalid = (
        raw_ok and not raw_result.get("schema_valid", True) and not raw_schema_violation
    )

    risk = _schema_risk_profile(schema_dict)
    depth = cast(int, risk["nesting_depth"])
    enum_size = cast(int, risk["max_enum_size"])

    # ── schema complexity score (used by multiple axes) ───────────────────────
    # Weighted sum of hard constraint signals — determines if TTF overhead is justified
    _risk_pts = (
        (6 if cast(int, risk["max_enum_size"]) >= 8 else 3 if risk["large_enum"] else 0)
        + (4 if risk["has_pattern"] else 0)
        + (4 if risk["has_additional_props_false"] else 0)
        + (3 if risk["has_array_cardinality"] else 0)
        + (2 if risk["has_numeric_constraints"] else 0)
        + (3 if depth >= 3 else 1 if depth >= 2 else 0)
        + (2 if risk["has_combinators"] else 0)
    )
    schema_is_complex = _risk_pts >= 6  # TTF overhead justified
    schema_is_simple = _risk_pts <= 2  # direct mode is correct choice

    # ── schema_validity 0-30 ─────────────────────────────────────────────────
    # Honest: both valid → 10 pts only (they are equal on this run)
    if fs_valid and not raw_valid and not fs_schema_violation:
        schema_validity = 30
    elif fs_valid and raw_valid and not fs_schema_violation:
        schema_validity = 10  # equal — not a FS win
    elif fs_schema_violation and not raw_schema_violation:
        schema_validity = 0  # FS violated a constraint raw Groq didn't
    else:
        schema_validity = 0

    # ── constraint_edge 0-25 ─────────────────────────────────────────────────
    # Rewards *correct routing decisions*, not just "TTF ran".
    # TTF on a simple schema = overhead without benefit → penalised.
    if used_ttf and schema_is_complex:
        constraint_edge = 25  # TTF warranted and executed
    elif used_ttf and not schema_is_complex and not schema_is_simple:
        constraint_edge = 12  # borderline — plausible TTF
    elif used_ttf and schema_is_simple:
        constraint_edge = 4  # TTF ran but schema didn't need it
    elif not used_ttf and schema_is_simple:
        constraint_edge = 20  # correct: direct mode, no wasted overhead
    elif phi_score > 0.65 and not used_ttf:
        constraint_edge = 10  # Φ flagged complexity but direct was used
    else:
        constraint_edge = 8  # neutral

    # ── reliability 0-20 ─────────────────────────────────────────────────────
    if fs_valid and not fallback:
        reliability = 20
    elif not fallback:
        reliability = 10
    else:
        reliability = 0

    # ── constraint_protection 0-25 ───────────────────────────────────────────
    # Primary signal: raw Groq actually violated a constraint FormatShield enforced.
    # Secondary: latent risk — schema features that raw LLMs fail on at non-trivial rates.
    # No points for "schema looks hard but raw happened to pass this time".
    if raw_schema_violation:
        constraint_protection = 25  # raw broke a real constraint
    elif raw_json_invalid:
        constraint_protection = 15  # raw output wasn't even valid JSON
    elif schema_is_complex and raw_valid:
        # Schema is hard — raw passed *this run* but failure rate is non-trivial
        constraint_protection = 10
    elif _risk_pts >= 3 and raw_valid:
        constraint_protection = 5  # moderate risk, raw OK this time
    else:
        constraint_protection = 0

    # ── latency_efficiency -15 to +5 ─────────────────────────────────────────
    # FormatShield's TTF overhead is a real cost. If raw Groq is much faster AND
    # the schema doesn't justify TTF, that cost hurts the score.
    latency_efficiency = 0
    if fs_ok and raw_ok:
        fs_lat = float(fs_result.get("latency_ms") or 0)
        raw_lat = float(raw_result.get("latency_ms") or 0)
        if fs_lat > 0 and raw_lat > 0:
            ratio = fs_lat / raw_lat  # >1 means FS is slower
            if ratio > 3.0 and schema_is_simple:
                latency_efficiency = -15  # 3× slower, simple schema = unjustified
            elif ratio > 2.0 and schema_is_simple:
                latency_efficiency = -10
            elif ratio > 1.5 and schema_is_simple:
                latency_efficiency = -5
            elif ratio > 2.0 and not schema_is_complex:
                latency_efficiency = -5  # borderline schema, still slow
            elif ratio < 0.75:
                latency_efficiency = 5  # FS actually faster

    score = min(
        100,
        max(
            0,
            schema_validity
            + constraint_edge
            + reliability
            + constraint_protection
            + latency_efficiency,
        ),
    )

    # ── grade ────────────────────────────────────────────────────────────────
    if score >= 82:
        grade = "A"
    elif score >= 66:
        grade = "B"
    elif score >= 50:
        grade = "C"
    elif score >= 35:
        grade = "D"
    else:
        grade = "F"

    # ── latency context strings (used in verdict and reasons) ────────────────
    fs_lat_v = float(fs_result.get("latency_ms") or 0) if fs_ok else 0.0
    raw_lat_v = float(raw_result.get("latency_ms") or 0) if raw_ok else 0.0
    lat_ratio = (fs_lat_v / raw_lat_v) if raw_lat_v > 0 and fs_lat_v > 0 else 1.0
    lat_context = ""
    if fs_lat_v > 0 and raw_lat_v > 0:
        if lat_ratio >= 4.0:
            lat_context = f"{lat_ratio:.0f}× slower ({fs_lat_v:.0f} ms vs {raw_lat_v:.0f} ms)"
        elif lat_ratio >= 1.5:
            lat_context = f"{lat_ratio:.1f}× slower ({fs_lat_v:.0f} ms vs {raw_lat_v:.0f} ms)"
        elif lat_ratio < 0.75:
            lat_context = f"{1 / lat_ratio:.1f}× faster ({fs_lat_v:.0f} ms vs {raw_lat_v:.0f} ms)"
        else:
            lat_context = f"comparable latency ({fs_lat_v:.0f} ms vs {raw_lat_v:.0f} ms)"

    # ── verdict_tier — 4-level scale reflecting real evidence magnitude ───────
    # decisive_win : raw failed schema validation OR raw violated a hard constraint
    # marginal_win : both valid, FS shows measurable advantage (complex schema + TTF)
    # tie          : both valid, no clear evidence advantage, comparable or modest latency
    # loss         : FS has schema violation raw doesn't, OR simple schema + FS much slower
    if fs_schema_violation and not raw_schema_violation:
        verdict_tier = "loss"
    elif raw_valid and not fs_valid:
        verdict_tier = "loss"
    elif latency_efficiency <= -10 and schema_is_simple and not raw_schema_violation:
        verdict_tier = "loss"  # simple schema, both valid, FS much slower — raw wins
    elif fs_valid and not raw_valid and not fs_schema_violation:
        verdict_tier = "decisive_win"
    elif raw_schema_violation and not fs_schema_violation and fs_valid:
        verdict_tier = "decisive_win"
    elif raw_json_invalid and fs_valid:
        verdict_tier = "decisive_win"
    elif (
        schema_is_complex and used_ttf and fs_valid and not fs_schema_violation and lat_ratio < 4.0
    ):
        # Complex schema, TTF warranted, latency not extreme (< 4×) — real marginal win
        verdict_tier = "marginal_win"
    elif (
        schema_is_complex and used_ttf and fs_valid and not fs_schema_violation and lat_ratio >= 4.0
    ):
        # Complex schema + TTF but latency is severe — call it a tie; cost vs benefit unclear
        verdict_tier = "tie"
    elif latency_efficiency <= -5:
        verdict_tier = "loss"
    else:
        verdict_tier = "tie"

    # winner field (kept for backward compat with verdict bar logic)
    if verdict_tier in ("decisive_win", "marginal_win"):
        winner = "formatshield"
    elif verdict_tier == "loss":
        winner = "raw"
    else:
        winner = "tie"

    # ── tier labels and summary ───────────────────────────────────────────────
    _tier_labels = {
        "decisive_win": "Decisive Win",
        "marginal_win": "Marginal Win",
        "tie": "Tie",
        "loss": "Loss",
    }
    tier_label = _tier_labels[verdict_tier]

    if verdict_tier == "decisive_win" and raw_schema_violation:
        summary = (
            f"Decisive Win — raw Groq violated a schema constraint; "
            f"FormatShield enforced it. {lat_context}."
        )
    elif verdict_tier == "decisive_win" and not raw_valid:
        summary = (
            f"Decisive Win — raw Groq produced invalid output; "
            f"FormatShield passed schema validation. {lat_context}."
        )
    elif verdict_tier == "marginal_win":
        summary = (
            f"Marginal Win — both outputs schema-valid, but FormatShield's TTF routing "
            f"added measurable reasoning quality on this complex schema. "
            f"Cost: {lat_context}."
        )
    elif verdict_tier == "tie":
        if not fs_valid and not raw_valid:
            summary = (
                "Tie — both outputs failed schema validation on this run. "
                "This usually means the schema's constraints are fighting natural reasoning "
                "(e.g., enum values that exclude valid intermediate units). "
                "Review the schema design: constraints should enable structured reasoning, "
                "not prevent correct answers."
            )
        elif fs_valid and raw_valid:
            summary = (
                "Tie — both outputs schema-valid with comparable quality on this run. "
                "FormatShield's advantage shows on Hard/Enterprise schemas with large enums, "
                "pattern constraints, or cross-field dependency rules."
            )
        else:
            summary = (
                "Tie — comparable result on this run. "
                "Try a Hard or Enterprise preset to see FormatShield's constraint enforcement advantage."  # noqa: E501
            )
    else:  # loss
        if latency_efficiency <= -10 and schema_is_simple:
            summary = (
                f"Raw Groq wins — simple schema, both outputs valid, "
                f"FormatShield {lat_context}. "
                "Latency overhead not justified here."
            )
        elif fs_schema_violation:
            summary = (
                "Raw Groq wins — independent jsonschema validation caught a constraint "
                "violation in FormatShield's output that raw Groq avoided."
            )
        else:
            summary = "Raw Groq wins on this run."

    # ── reasons — run-specific evidence first, research claims only when applicable ──
    reasons: list[str] = []

    # 1. Schema adherence (run-specific)
    if schema_validity == 30:
        reasons.append(
            "FormatShield passed jsonschema validation; raw Groq failed — "
            "this is the primary constraint enforcement advantage (arXiv 2408.02442)."
        )
    elif schema_validity == 10:
        reasons.append(
            "Both outputs passed jsonschema validation on this run — "
            "schema adherence is equal; verdict driven by latency and routing correctness."
        )
    else:
        reasons.append(
            "Neither output passed schema validation on this run. "
            "When both fail, check if the schema's enum or pattern constraints "
            "are preventing naturally correct outputs — "
            "the schema may be fighting the reasoning task."
        )

    # 2. Routing decision quality (run-specific, no generic research claims unless TTF earned it)
    if used_ttf and schema_is_complex and raw_schema_violation:
        # TTF both warranted AND produced a better constraint outcome
        reasons.append(
            f"TTF routing (Φ={phi_score:.3f}) activated on a complex schema and "
            f"enforced constraints that raw Groq violated. "
            f"This is the core FormatShield value proposition (arXiv 2601.07525)."
        )
    elif used_ttf and schema_is_complex:
        # TTF warranted by schema but both valid this run
        reasons.append(
            f"TTF routing (Φ={phi_score:.3f}) activated on a complex schema — "
            f"both valid this run, but complex schemas fail for raw LLMs at non-trivial rates. "
            f"The 27 pp accuracy recovery (arXiv 2601.07525) shows on repeated runs."
        )
    elif used_ttf and schema_is_simple:
        reasons.append(
            f"TTF activated (Φ={phi_score:.3f}) but schema complexity doesn't justify it "
            f"({_risk_pts} risk pts, threshold is 6). "
            "Use Hard/Enterprise presets to see TTF earn its latency cost."
        )
    elif not used_ttf and schema_is_simple:
        reasons.append(
            f"Φ={phi_score:.3f} — direct mode chosen correctly; "
            "no TTF overhead on a simple schema (smart routing)."
        )
    else:
        reasons.append(
            f"Φ={phi_score:.3f} — routing decision: {'TTF' if used_ttf else 'direct'} mode."
        )

    # 3. Latency (always run-specific — never suppress)
    if lat_context:
        if verdict_tier in ("loss", "tie") and lat_ratio >= 1.5:
            reasons.append(
                f"Latency: FormatShield {lat_context}. "
                "On simple schemas where both outputs are valid, this overhead "
                "shifts the verdict toward raw Groq."
            )
        elif verdict_tier == "marginal_win" and lat_ratio >= 2.0:
            reasons.append(
                f"Latency cost: {lat_context}. "
                "This is the expected TTF tradeoff — acceptable for high-stakes tasks "
                "where schema precision matters more than speed."
            )
        elif lat_ratio < 0.75:
            reasons.append(f"FormatShield was faster on this run: {lat_context}.")

    # 4. Constraint violations (run-specific)
    if raw_schema_violation:
        violation_snip = raw_result.get("schema_violation", "")[:70]
        reasons.append(
            f"Raw Groq violated: '{violation_snip}' — "
            "enum/pattern failures hit 5–36% of raw outputs (arXiv 2501.10868)."
        )
    if fs_schema_violation:
        fs_violation_snip = fs_result.get("schema_violation", "")[:70]
        reasons.append(
            f"FormatShield also violated a constraint: '{fs_violation_snip}' — "
            "independent jsonschema check caught this."
        )

    # 5. Latent risk notice (only when neither actually failed — purely informational)
    if not raw_schema_violation and not fs_schema_violation and len(reasons) < 4:
        if risk["large_enum"]:
            reasons.append(
                f"Schema has enum({enum_size} options) — both passed this run, but "
                "raw LLMs hallucinate outside large enums on repeated runs "
                "(arXiv 2408.02442). Run again to see the failure rate."
            )
        elif risk["has_pattern"]:
            reasons.append(
                "Schema has pattern constraints — both passed this run, but "
                "33% baseline failure rate in SchemaBench (arXiv 2501.10868) "
                "means raw will fail on repeated runs."
            )

    # 6. Redirect for simple/tie cases
    if verdict_tier in ("tie", "loss") and len(reasons) < 4:
        reasons.append(
            "Load a Hard or Enterprise preset with a large enum, pattern constraint, "
            "or strict additionalProperties to see FormatShield's decisive advantage."
        )

    if reliability == 0 and fallback:
        reasons.append(
            "Fallback triggered — routing edge case; indicates schema complexity at the boundary."
        )

    reliability_signal = schema_validity + reliability
    semantic_proxy_signal = constraint_edge + constraint_protection
    reliability_score = round((reliability_signal / 50.0) * 100.0, 1)
    semantic_proxy_score = round((semantic_proxy_signal / 50.0) * 100.0, 1)

    return {
        "score": score,
        "grade": grade,
        "verdict_tier": verdict_tier,
        "tier_label": tier_label,
        "winner": winner,
        "summary": summary,
        "reasons": reasons,
        "risk_profile": {
            "schema_depth": depth,
            "large_enum": risk["large_enum"],
            "max_enum_size": enum_size,
            "has_array_cardinality": risk["has_array_cardinality"],
            "has_numeric_constraints": risk["has_numeric_constraints"],
            "has_pattern": risk["has_pattern"],
            "has_combinators": risk["has_combinators"],
            "schema_complexity_pts": _risk_pts,
            "schema_is_complex": schema_is_complex,
            "schema_is_simple": schema_is_simple,
        },
        "breakdown": {
            "schema_validity": schema_validity,
            "routing_quality": constraint_edge,
            "reliability": reliability,
            "schema_advantage": constraint_protection,
        },
        "score_channels": {
            "reliability_score": reliability_score,
            "semantic_proxy_score": semantic_proxy_score,
            "notes": (
                "reliability_score reflects schema-validity and fallback stability; "
                "semantic_proxy_score reflects routing quality plus constraint advantage"
            ),
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
    will_use_ttf = phi.phi > 0.65
    phi_info = {
        "phi": round(phi.phi, 4),
        "lambda2": round(phi.lambda2, 4),
        "tau": round(phi.tau, 4),
        "delta_k": round(phi.delta_k, 4),
        "explanation": phi.explanation,
        "recommendation": "TTF" if will_use_ttf else "Direct",
        # TTF Stage 1-4 metadata (schema-aware prompting, quality gate, self-consistency)
        "thinking_budget_tokens": _phi_thinking_budget(phi.phi) if will_use_ttf else None,
        "self_consistency_mode": phi.phi >= _SC_PHI_THRESHOLD,
        "self_consistency_k": DEFAULT_SC_K if phi.phi >= _SC_PHI_THRESHOLD else 1,
        "pass2_temperature": round(max(0.05, 0.7 * (1.0 - phi.tau)), 3) if will_use_ttf else None,
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
