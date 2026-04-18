#!/usr/bin/env python3
"""
FormatShield Live Groq API Test Runner.

Runs one-by-one sequential live calls across 8 Groq models, captures full
routing telemetry per call, and produces a Markdown report + CSV log.

This is NOT a benchmark.  It validates that the routing formula and generation
pipeline behave correctly in production, and surfaces formula improvement signals.

Usage
-----
    # Dry run (phi prediction only, no API calls)
    python scripts/live_test_groq.py --dry-run

    # Cycle 1, all 56 cases
    python scripts/live_test_groq.py --cycle 1

    # Resume from call 21 (after a quota error mid-run)
    python scripts/live_test_groq.py --cycle 1 --start-at 21

    # Run only fast models
    python scripts/live_test_groq.py --models 8b,70b

    # RPM-safe run for compound models
    python scripts/live_test_groq.py --models compound,compmini --sleep-ms 400

Requirements
------------
    GROQ_API_KEY environment variable (or .env file in project root)
    pip install -e ".[dev]"   (or uv pip install -e ".[dev]")
"""

from __future__ import annotations

import argparse
import asyncio
import collections
import csv
import dataclasses
import datetime
import json
import os
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).parent.parent
TESTING_DIR = PROJECT_ROOT / "testing"
RAW_DIR = TESTING_DIR / "raw"
CSV_PATH = TESTING_DIR / "api_test_log.csv"

# ---------------------------------------------------------------------------
# Groq routing constants (mirrors threshold_oracle.py)
# ---------------------------------------------------------------------------

GROQ_PHI_THRESHOLD = 0.65   # phi > 0.65 → TTF
GROQ_OVERHEAD_PCT = 30.0
TTF_ACCURACY_DELTA = 0.17

# ---------------------------------------------------------------------------
# Pydantic schemas for test scenarios
# ---------------------------------------------------------------------------


class SimpleTagSchema(BaseModel):
    tag: str
    confidence: float


class NameExtraction(BaseModel):
    name: str
    title: str | None = None


class ProductReview(BaseModel):
    sentiment: str
    key_themes: list[str]
    rating: int
    summary: str


class ReasoningAnalysis(BaseModel):
    conclusion: str
    supporting_points: list[str]
    confidence: float
    reasoning_steps: list[str]
    uncertainty_factors: list[str]


class ContractClause(BaseModel):
    clause_type: str
    obligation: str
    parties: list[str]
    conditions: list[str]
    effective_date: str | None = None


# ---------------------------------------------------------------------------
# Test case dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass
class TestCase:
    case_id: str
    prompt: str
    schema_class: type[BaseModel]
    scenario_type: str      # direct | ttf | borderline | override | latency
    expected_routing: str   # direct | ttf
    model: str              # FormatShield format: "groq/model_id"
    latency_budget_ms: float | None = None
    compound_flag: bool = False
    notes: str = ""


# ---------------------------------------------------------------------------
# Sample prompts per scenario type
# ---------------------------------------------------------------------------

_DIRECT_PROMPTS = [
    "Tag this text as 'positive' or 'negative': 'The product arrived quickly and works great.'",
    "Classify the sentiment of this review: 'Terrible service, waited two hours for a cold meal.'",
    "Is this feedback positive or negative? 'Easy setup, great battery life, highly recommend.'",
    "Label this message: 'I cannot get this software to open on my machine at all.'",
]

_TTF_PROMPTS = [
    (
        "Analyze the following contract clause step by step. "
        "Identify the obligating parties, the specific obligations imposed, "
        "the conditions that must be met for the obligation to apply, "
        "and any temporal constraints. "
        "Clause: 'Either party may terminate this Agreement upon thirty (30) days "
        "written notice to the other party, provided that all outstanding invoices "
        "are settled within that period, and that no active support tickets remain "
        "unresolved at the time of termination.' "
        "Reason through each element carefully before producing your structured output."
    ),
    (
        "Compare and evaluate the trade-offs between synchronous and asynchronous "
        "API design patterns for a high-throughput financial services backend. "
        "Consider latency, consistency guarantees, error handling complexity, "
        "developer experience, and operational overhead. "
        "Step through your reasoning before producing a structured summary because "
        "the decision affects the entire architecture of the system."
    ),
    (
        "A dataset of 10,000 customer records shows: 42% churn within 90 days of "
        "sign-up, with churned customers having a mean session count of 1.8 vs 7.4 "
        "for retained customers, and 78% of churned customers never completed "
        "onboarding. Analyze the root causes, derive supporting evidence from the "
        "data, and reason through each contributing factor step by step to produce "
        "actionable conclusions."
    ),
    (
        "Evaluate the regulatory implications of storing EU customer PII in "
        "US-based cloud infrastructure under GDPR Article 46 transfer mechanisms. "
        "Compare standard contractual clauses, binding corporate rules, and "
        "adequacy decisions. Because each mechanism has different enforcement risk, "
        "step through the analysis systematically and derive a structured "
        "recommendation."
    ),
]

_BORDERLINE_PROMPTS = [
    (
        "Extract the key information from this product review and rate it: "
        "'The laptop battery lasts about 6 hours under normal use. Screen "
        "brightness could be better outdoors but overall it handles video "
        "editing smoothly.'"
    ),
    (
        "Summarize this customer feedback and classify its sentiment: "
        "'Delivery took longer than expected but the packaging was excellent "
        "and the item is exactly as described. Will order again.'"
    ),
    (
        "Analyze this short review and provide a structured summary: "
        "'Great coffee maker, very easy to use. The carafe could be larger "
        "for bigger households. Timer function works perfectly.'"
    ),
]

_OVERRIDE_PROMPTS = [
    "Extract the author name from: 'Written by Dr. Sarah Chen, Ph.D.'",
    "Who signed this document? 'Signed by: James Wilson, CEO'",
    "Find the person's name and title: 'From the desk of Prof. Michael Torres, Head of Research'",
    "Extract contact: 'Please reach out to Amanda Foster, Director of Sales'",
]

_LATENCY_PROMPTS = [
    (
        "Analyze the regulatory compliance requirements for deploying an AI-based fraud detection "
        "system in the European Union financial services sector. Consider PSD2, GDPR, the AI Act, "
        "and EBA guidelines. Step through each regulation systematically and derive a structured "
        "compliance checklist with obligations, parties responsible, and conditions for each item."
    ),
]


# ---------------------------------------------------------------------------
# Build the 56-case test matrix
# ---------------------------------------------------------------------------


def build_test_cases() -> list[TestCase]:
    cases: list[TestCase] = []

    # Model short-key → FormatShield model string
    models = {
        "8b":       "groq/llama-3.1-8b-instant",
        "70b":      "groq/llama-3.3-70b-versatile",
        "gpt120":   "groq/openai/gpt-oss-120b",
        "gpt20":    "groq/openai/gpt-oss-20b",
        "compound": "groq/groq/compound",
        "compmini": "groq/groq/compound-mini",
        "qwen32":   "groq/qwen/qwen3-32b",
        "scout":    "groq/meta-llama/llama-4-scout-17b-16e-instruct",
    }

    # Models that support TTF and override scenarios
    full_models = ["8b", "70b", "qwen32", "scout"]
    # Models that support TTF but not override/latency (no specific override test needed)
    medium_models = ["gpt120", "gpt20"]
    # Compound models: direct + borderline only
    compound_models = ["compound", "compmini"]

    idx = 0

    # --- Full models: direct(2) + ttf(2) + borderline(1) + override(1) + latency(1) = 7 each ---
    for key in full_models:
        m = models[key]

        # 2x direct
        for i in range(2):
            idx += 1
            cases.append(TestCase(
                case_id=f"direct_{key}_{i+1:02d}",
                prompt=_DIRECT_PROMPTS[i % len(_DIRECT_PROMPTS)],
                schema_class=SimpleTagSchema,
                scenario_type="direct",
                expected_routing="direct",
                model=m,
                notes="flat schema + short prompt → phi low → direct",
            ))

        # 2x ttf
        for i in range(2):
            idx += 1
            schema = ReasoningAnalysis if i == 0 else ContractClause
            cases.append(TestCase(
                case_id=f"ttf_{key}_{i+1:02d}",
                prompt=_TTF_PROMPTS[i % len(_TTF_PROMPTS)],
                schema_class=schema,
                scenario_type="ttf",
                expected_routing="ttf",
                model=m,
                notes="nested schema + CoT prompt → phi high → ttf",
            ))

        # 1x borderline
        idx += 1
        cases.append(TestCase(
            case_id=f"borderline_{key}_01",
            prompt=_BORDERLINE_PROMPTS[0],
            schema_class=ProductReview,
            scenario_type="borderline",
            expected_routing="direct",
            model=m,
            notes="medium schema + medium prompt → phi near threshold",
        ))

        # 1x override-prone (phi says ttf but simple_extraction fires → direct)
        idx += 1
        cases.append(TestCase(
            case_id=f"override_{key}_01",
            prompt=_OVERRIDE_PROMPTS[0],
            schema_class=NameExtraction,
            scenario_type="override",
            expected_routing="direct",
            model=m,
            notes="depth=1 schema + short prompt → simple_extraction override → direct",
        ))

        # 1x latency-budget (TTF schema+prompt but budget=50ms → direct)
        idx += 1
        cases.append(TestCase(
            case_id=f"latency_{key}_01",
            prompt=_LATENCY_PROMPTS[0],
            schema_class=ReasoningAnalysis,
            scenario_type="latency",
            expected_routing="direct",
            model=m,
            latency_budget_ms=50.0,
            notes="50ms budget → TTF overhead ~300ms > budget → forced direct",
        ))

    # --- Medium models: direct(2) + ttf(2) + borderline(1) = 5 each ---
    for key in medium_models:
        m = models[key]

        for i in range(2):
            idx += 1
            cases.append(TestCase(
                case_id=f"direct_{key}_{i+1:02d}",
                prompt=_DIRECT_PROMPTS[i % len(_DIRECT_PROMPTS)],
                schema_class=SimpleTagSchema,
                scenario_type="direct",
                expected_routing="direct",
                model=m,
            ))

        for i in range(2):
            idx += 1
            schema = ReasoningAnalysis if i == 0 else ContractClause
            cases.append(TestCase(
                case_id=f"ttf_{key}_{i+1:02d}",
                prompt=_TTF_PROMPTS[i % len(_TTF_PROMPTS)],
                schema_class=schema,
                scenario_type="ttf",
                expected_routing="ttf",
                model=m,
            ))

        idx += 1
        cases.append(TestCase(
            case_id=f"borderline_{key}_01",
            prompt=_BORDERLINE_PROMPTS[1],
            schema_class=ProductReview,
            scenario_type="borderline",
            expected_routing="direct",
            model=m,
        ))

    # --- Compound models: direct(2) + borderline(1) = 3 each ---
    for key in compound_models:
        m = models[key]
        for i in range(2):
            idx += 1
            cases.append(TestCase(
                case_id=f"direct_{key}_{i+1:02d}",
                prompt=_DIRECT_PROMPTS[i % len(_DIRECT_PROMPTS)],
                schema_class=SimpleTagSchema,
                scenario_type="direct",
                expected_routing="direct",
                model=m,
                compound_flag=True,
                notes="compound model — direct only",
            ))
        idx += 1
        cases.append(TestCase(
            case_id=f"borderline_{key}_01",
            prompt=_BORDERLINE_PROMPTS[2],
            schema_class=ProductReview,
            scenario_type="borderline",
            expected_routing="direct",
            model=m,
            compound_flag=True,
            notes="compound model — borderline, expected direct",
        ))

    # Pad to reach 56 by adding variation cases for full models
    variation_prompts = [
        (_DIRECT_PROMPTS[2], SimpleTagSchema, "direct", "direct"),
        (_DIRECT_PROMPTS[3], SimpleTagSchema, "direct", "direct"),
        (_TTF_PROMPTS[2], ReasoningAnalysis, "ttf", "ttf"),
        (_TTF_PROMPTS[3], ContractClause, "ttf", "ttf"),
        (_BORDERLINE_PROMPTS[2], ProductReview, "borderline", "direct"),
        (_OVERRIDE_PROMPTS[1], NameExtraction, "override", "direct"),
        (_OVERRIDE_PROMPTS[2], NameExtraction, "override", "direct"),
        (_OVERRIDE_PROMPTS[3], NameExtraction, "override", "direct"),
        (_BORDERLINE_PROMPTS[0], ProductReview, "borderline", "direct"),
        (_BORDERLINE_PROMPTS[1], ProductReview, "borderline", "direct"),
        (_TTF_PROMPTS[1], ReasoningAnalysis, "ttf", "ttf"),
        (_TTF_PROMPTS[0], ContractClause, "ttf", "ttf"),
    ]

    var_models = [
        "8b", "70b", "qwen32", "scout",
        "8b", "70b", "qwen32", "scout",
        "gpt120", "gpt20", "8b", "70b",
    ]
    for vi, (vp, vs, vscen, vexp) in enumerate(variation_prompts):
        mkey = var_models[vi % len(var_models)]
        if idx >= 56:
            break
        idx += 1
        cases.append(TestCase(
            case_id=f"var_{vi+1:02d}_{mkey}",
            prompt=vp,
            schema_class=vs,
            scenario_type=vscen,
            expected_routing=vexp,
            model=models[mkey],
            notes="variation case",
        ))

    return cases[:56]  # cap at 56


# ---------------------------------------------------------------------------
# Compound model preflight
# ---------------------------------------------------------------------------

COMPOUND_ALIASES: dict[str, list[str]] = {
    "groq/groq/compound":      ["groq/groq/compound", "groq/compound"],
    "groq/groq/compound-mini": ["groq/groq/compound-mini", "groq/compound-mini"],
}


async def resolve_compound_aliases(dry_run: bool) -> dict[str, str | None]:
    """Resolve working compound model aliases before running the test cycle.

    Returns a mapping: FS model string → resolved model string (or None if all
    aliases fail).  In dry-run mode returns an empty dict (no API calls made).
    """
    if dry_run:
        return {}

    import formatshield as fs

    resolved: dict[str, str | None] = {}
    for fs_model, aliases in COMPOUND_ALIASES.items():
        for alias in aliases:
            try:
                shield = fs.FormatShield(model=alias)
                await shield.generate(prompt="Hi", max_tokens=1)
                resolved[fs_model] = alias
                print(f"  [preflight] {fs_model} → resolved as '{alias}'")
                break
            except Exception as exc:
                print(f"  [preflight] {alias} failed: {type(exc).__name__}: {exc!s:.80}")
        if fs_model not in resolved:
            resolved[fs_model] = None
            print(f"  [preflight] {fs_model} → UNRESOLVABLE (all aliases failed)")
    return resolved


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

CSV_COLUMNS = [
    "cycle", "call_num", "timestamp_utc", "model_fs", "model_groq_api",
    "scenario_type", "expected_routing",
    "phi_score", "phi_lambda2", "phi_tau", "phi_delta_k",
    "actual_routing", "routing_match", "routing_confidence", "routing_explanation",
    "complexity_score", "failure_modes", "failure_mode_count",
    "latency_ms", "schema_valid", "fallback_triggered",
    "expected_accuracy_delta", "expected_overhead_pct",
    "schema_depth", "prompt_length_chars", "prompt_preview", "schema_class",
    "error_type", "error_message",
    "token_usage_prompt", "token_usage_completion", "token_usage_total", "cost_usd",
    "thinking_present", "thinking_length_chars", "output_length_chars",
    "compound_flag", "dry_run", "notes",
]


def ensure_csv_header() -> None:
    """Write CSV header row only if the file does not yet exist."""
    if not CSV_PATH.exists():
        TESTING_DIR.mkdir(parents=True, exist_ok=True)
        with CSV_PATH.open("w", newline="", encoding="utf-8") as fh:
            csv.DictWriter(fh, fieldnames=CSV_COLUMNS).writeheader()


def append_csv_row(row: dict[str, Any]) -> None:
    """Append one row to the cumulative CSV log."""
    with CSV_PATH.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writerow(row)


# ---------------------------------------------------------------------------
# Raw payload dump (git-ignored)
# ---------------------------------------------------------------------------


def save_raw_payload(cycle: int, call_num: int, payload: dict[str, Any]) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    path = RAW_DIR / f"cycle_{cycle}_call_{call_num:03d}.json"
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)


# ---------------------------------------------------------------------------
# Schema depth helper
# ---------------------------------------------------------------------------


def _schema_depth(schema: dict[str, Any], depth: int = 0) -> int:
    """Recursively compute the nesting depth of a JSON Schema dict."""
    if not isinstance(schema, dict):
        return depth
    max_d = depth
    for key in ("properties", "$defs", "definitions"):
        if key in schema:
            for sub in schema[key].values():
                max_d = max(max_d, _schema_depth(sub, depth + 1))
    for key in ("items", "additionalProperties"):
        if key in schema:
            max_d = max(max_d, _schema_depth(schema[key], depth + 1))
    return max_d


# ---------------------------------------------------------------------------
# Model API name helper
# ---------------------------------------------------------------------------


def groq_api_name(model_fs: str) -> str:
    """Derive the Groq API model name from a FormatShield model string.

    FormatShield strips the first segment on '/' to get the model name passed
    to the Groq backend, which then strips an extra 'groq/' prefix if present.
    """
    without_backend = model_fs.split("/", 1)[1] if "/" in model_fs else model_fs
    return without_backend.removeprefix("groq/")


# ---------------------------------------------------------------------------
# Single call executor
# ---------------------------------------------------------------------------


async def run_one_call(
    tc: TestCase,
    cycle: int,
    call_num: int,
    dry_run: bool,
    compound_resolved: dict[str, str | None],
) -> dict[str, Any]:
    """Execute one test case and return a fully-populated CSV row dict."""
    from formatshield.oracle.routing_score import compute_routing_score

    timestamp = datetime.datetime.now(datetime.UTC).isoformat()
    schema_dict = tc.schema_class.model_json_schema()

    # --- Always compute phi first (pure Python, no API call) ---
    try:
        phi_result = compute_routing_score(tc.prompt, schema_dict)
        phi_score = round(phi_result.phi, 6)
        phi_lambda2 = round(phi_result.lambda2, 6)
        phi_tau = round(phi_result.tau, 6)
        phi_delta_k = round(phi_result.delta_k, 6)
    except Exception as phi_exc:
        phi_score = phi_lambda2 = phi_tau = phi_delta_k = -1.0
        print(f"  [phi-error] {phi_exc!s:.60}")

    row: dict[str, Any] = {
        "cycle": cycle,
        "call_num": call_num,
        "timestamp_utc": timestamp,
        "model_fs": tc.model,
        "model_groq_api": groq_api_name(tc.model),
        "scenario_type": tc.scenario_type,
        "expected_routing": tc.expected_routing,
        "phi_score": phi_score,
        "phi_lambda2": phi_lambda2,
        "phi_tau": phi_tau,
        "phi_delta_k": phi_delta_k,
        "schema_depth": _schema_depth(schema_dict),
        "prompt_length_chars": len(tc.prompt),
        "prompt_preview": tc.prompt[:80].replace("\n", " "),
        "schema_class": tc.schema_class.__name__,
        "compound_flag": tc.compound_flag,
        "dry_run": dry_run,
        "notes": tc.notes,
    }

    # --- Dry run: predict from phi, no API call ---
    if dry_run:
        predicted = "ttf" if phi_score > GROQ_PHI_THRESHOLD else "direct"
        row.update({
            "actual_routing": predicted,
            "routing_match": predicted == tc.expected_routing,
            "routing_confidence": "",
            "routing_explanation": f"dry-run phi={phi_score:.4f} threshold={GROQ_PHI_THRESHOLD}",
            "complexity_score": "",
            "failure_modes": "",
            "failure_mode_count": 0,
            "latency_ms": 0,
            "schema_valid": "",
            "fallback_triggered": False,
            "expected_accuracy_delta": "",
            "expected_overhead_pct": "",
            "token_usage_prompt": "",
            "token_usage_completion": "",
            "token_usage_total": "",
            "cost_usd": "",
            "thinking_present": False,
            "thinking_length_chars": 0,
            "output_length_chars": 0,
            "error_type": "",
            "error_message": "",
        })
        return row

    # --- Check compound model resolution ---
    if tc.compound_flag and tc.model in compound_resolved:
        if compound_resolved[tc.model] is None:
            row.update({
                "actual_routing": "skipped",
                "routing_match": False,
                "routing_confidence": "",
                "routing_explanation": "",
                "complexity_score": "",
                "failure_modes": "",
                "failure_mode_count": 0,
                "latency_ms": 0,
                "schema_valid": False,
                "fallback_triggered": False,
                "expected_accuracy_delta": "",
                "expected_overhead_pct": "",
                "token_usage_prompt": "",
                "token_usage_completion": "",
                "token_usage_total": "",
                "cost_usd": "",
                "thinking_present": False,
                "thinking_length_chars": 0,
                "output_length_chars": 0,
                "error_type": "CompoundModelUnresolvable",
                "error_message": "All compound model aliases failed preflight",
            })
            return row

    # --- Live API call ---
    import formatshield as fs

    try:
        shield = fs.FormatShield(
            model=tc.model,
            latency_budget_ms=tc.latency_budget_ms,
        )
        result: fs.GenerationResult = await shield.generate(
            prompt=tc.prompt,
            schema=tc.schema_class,
        )

        failure_modes_str = "|".join(result.failure_modes) if result.failure_modes else "none"
        actual_routing = result.routing.strategy
        routing_match = actual_routing == tc.expected_routing

        row.update({
            "actual_routing": actual_routing,
            "routing_match": routing_match,
            "routing_confidence": round(result.routing.confidence, 6),
            "routing_explanation": result.routing.explanation,
            "complexity_score": round(result.complexity_score, 6),
            "failure_modes": failure_modes_str,
            "failure_mode_count": len(result.failure_modes),
            "latency_ms": round(result.latency_ms, 2),
            "schema_valid": result.schema_valid,
            "fallback_triggered": result.fallback_triggered,
            "expected_accuracy_delta": result.routing.expected_accuracy_delta,
            "expected_overhead_pct": result.routing.expected_overhead_pct,
            "token_usage_prompt": (
                result.token_usage.input_tokens
                if result.token_usage and result.token_usage.input_tokens is not None
                else ""
            ),
            "token_usage_completion": (
                result.token_usage.output_tokens
                if result.token_usage and result.token_usage.output_tokens is not None
                else ""
            ),
            "token_usage_total": (
                result.token_usage.total_tokens
                if result.token_usage and result.token_usage.total_tokens is not None
                else ""
            ),
            "cost_usd": result.cost_usd if result.cost_usd is not None else "",
            "thinking_present": result.thinking is not None,
            "thinking_length_chars": len(result.thinking) if result.thinking else 0,
            "output_length_chars": len(result.output) if result.output else 0,
            "error_type": "",
            "error_message": "",
        })

        # Save raw payload (git-ignored directory)
        save_raw_payload(cycle, call_num, {
            "case_id": tc.case_id,
            "model_fs": tc.model,
            "scenario_type": tc.scenario_type,
            "prompt": tc.prompt,
            "schema_class": tc.schema_class.__name__,
            "phi": {
                "phi": phi_score, "lambda2": phi_lambda2,
                "tau": phi_tau, "delta_k": phi_delta_k,
            },
            "result": result.model_dump(),
        })

    except Exception as exc:
        row.update({
            "actual_routing": "error",
            "routing_match": False,
            "routing_confidence": "",
            "routing_explanation": "",
            "complexity_score": "",
            "failure_modes": "",
            "failure_mode_count": 0,
            "latency_ms": "",
            "schema_valid": False,
            "fallback_triggered": False,
            "expected_accuracy_delta": "",
            "expected_overhead_pct": "",
            "token_usage_prompt": "",
            "token_usage_completion": "",
            "token_usage_total": "",
            "cost_usd": "",
            "thinking_present": False,
            "thinking_length_chars": 0,
            "output_length_chars": 0,
            "error_type": type(exc).__name__,
            "error_message": str(exc)[:400],
        })

    return row


# ---------------------------------------------------------------------------
# Console progress printer
# ---------------------------------------------------------------------------


def print_progress(call_num: int, total: int, row: dict[str, Any]) -> None:
    actual = row.get("actual_routing", "?")
    if actual == "error":
        status = "ERROR   "
    elif actual == "skipped":
        status = "SKIPPED "
    elif row.get("routing_match") is True:
        status = "OK      "
    else:
        status = "MISMATCH"

    phi = row.get("phi_score", "?")
    lat = row.get("latency_ms", "")
    lat_str = f"{lat}ms" if lat != "" else "—"
    api_name = row.get("model_groq_api", "?")[:30]

    print(
        f"[{call_num:3d}/{total}] {api_name:<32} "
        f"scen={row.get('scenario_type','?'):<10} "
        f"exp={row.get('expected_routing','?'):<6} "
        f"got={actual:<6} "
        f"phi={phi!s:<6} "
        f"lat={lat_str:<10} "
        f"[{status}]"
    )


# ---------------------------------------------------------------------------
# Report generator
# ---------------------------------------------------------------------------


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pct(count: int, total: int) -> str:
    if total == 0:
        return "0.0%"
    return f"{100.0 * count / total:.1f}%"


def generate_report(cycle: int, rows: list[dict[str, Any]]) -> None:
    """Generate testing/report_cycle_N.md from in-memory rows."""
    TESTING_DIR.mkdir(parents=True, exist_ok=True)
    report_path = TESTING_DIR / f"report_cycle_{cycle}.md"

    live_rows = [r for r in rows if not r.get("dry_run")]
    total = len(live_rows)
    date_str = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

    lines: list[str] = []
    lines.append(f"# FormatShield Live API Test Report — Cycle {cycle}")
    lines.append(
        f"**Date:** {date_str}  **Total calls (live):** {total}"
        "  **Models:** 8  **Dry-run rows excluded**\n"
    )
    lines.append("---\n")

    # ---------- Section 1: Model Coverage Table ----------
    lines.append("## 1. Model Coverage\n")
    hdr = "| model | calls | direct | ttf | error | skipped"
    hdr += " | schema_valid_rate | fallback_rate | avg_latency_ms |"
    lines.append(hdr)
    lines.append("|-------|-------|--------|-----|-------|---------|"
                 "-------------------|---------------|----------------|")

    model_groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in live_rows:
        model_groups[r.get("model_groq_api", "unknown")].append(r)

    for model_api, mrs in sorted(model_groups.items()):
        n = len(mrs)
        n_direct = sum(1 for r in mrs if r.get("actual_routing") == "direct")
        n_ttf = sum(1 for r in mrs if r.get("actual_routing") == "ttf")
        n_error = sum(1 for r in mrs if r.get("actual_routing") == "error")
        n_skip = sum(1 for r in mrs if r.get("actual_routing") == "skipped")
        valid_rates = [
            1 if r.get("schema_valid") is True else 0
            for r in mrs if r.get("schema_valid") != ""
        ]
        schema_valid_rate = (
            f"{_mean([float(v) for v in valid_rates]):.0%}" if valid_rates else "—"
        )
        n_fallback = sum(1 for r in mrs if r.get("fallback_triggered") is True)
        fallback_rate = _pct(n_fallback, n)
        latencies = [
            float(r["latency_ms"])
            for r in mrs
            if isinstance(r.get("latency_ms"), (int, float)) and r["latency_ms"] != ""
        ]
        avg_lat = f"{_mean(latencies):.0f}" if latencies else "—"
        lines.append(
            f"| {model_api} | {n} | {n_direct} | {n_ttf} | {n_error}"
            f" | {n_skip} | {schema_valid_rate} | {fallback_rate} | {avg_lat} |"
        )

    lines.append("")

    # ---------- Section 2: Scenario Routing Table ----------
    lines.append("## 2. Scenario Routing\n")
    lines.append("| scenario_type | total | expected | direct | ttf | error | match_rate |")
    lines.append("|---------------|-------|----------|--------|-----|-------|------------|")

    scen_groups: dict[str, list[dict]] = collections.defaultdict(list)
    for r in live_rows:
        scen_groups[r.get("scenario_type", "unknown")].append(r)

    for scen, srs in sorted(scen_groups.items()):
        n = len(srs)
        exp = srs[0].get("expected_routing", "?") if srs else "?"
        n_direct = sum(1 for r in srs if r.get("actual_routing") == "direct")
        n_ttf = sum(1 for r in srs if r.get("actual_routing") == "ttf")
        n_error = sum(1 for r in srs if r.get("actual_routing") == "error")
        n_match = sum(1 for r in srs if r.get("routing_match") is True)
        lines.append(
            f"| {scen} | {n} | {exp} | {n_direct}"
            f" | {n_ttf} | {n_error} | {_pct(n_match, n)} |"
        )

    lines.append("")

    # ---------- Section 3: Phi Score Analysis ----------
    lines.append("## 3. Phi Score Analysis\n")
    lines.append("| model | phi_min | phi_max | phi_mean | phi_p50 | threshold |")
    lines.append("|-------|---------|---------|----------|---------|-----------|")

    for model_api, mrs in sorted(model_groups.items()):
        phis = [
            float(r["phi_score"])
            for r in mrs
            if isinstance(r.get("phi_score"), (int, float)) and r["phi_score"] >= 0
        ]
        if phis:
            phis_sorted = sorted(phis)
            p50 = phis_sorted[len(phis_sorted) // 2]
            lines.append(
                f"| {model_api} | {min(phis):.4f} | {max(phis):.4f}"
                f" | {_mean(phis):.4f} | {p50:.4f} | {GROQ_PHI_THRESHOLD} |"
            )

    lines.append("")
    lines.append("**Phi distribution (all live calls):**\n")
    all_phis = [
        float(r["phi_score"])
        for r in live_rows
        if isinstance(r.get("phi_score"), (int, float)) and r["phi_score"] >= 0
    ]
    if all_phis:
        buckets = [0] * 10
        for p in all_phis:
            b = min(int(p * 10), 9)
            buckets[b] += 1
        for bi, cnt in enumerate(buckets):
            bar = "█" * cnt
            lo = bi / 10
            hi = (bi + 1) / 10
            marker = " ← threshold" if lo <= GROQ_PHI_THRESHOLD < hi else ""
            lines.append(f"  [{lo:.1f}–{hi:.1f}]: {bar} ({cnt}){marker}")

    lines.append("")

    # ---------- Section 4: Failure Mode Frequency ----------
    lines.append("## 4. Failure Mode Frequency\n")
    lines.append("| failure_mode | count | pct_of_total |")
    lines.append("|--------------|-------|--------------|")

    mode_counter: collections.Counter = collections.Counter()
    for r in live_rows:
        modes_str = r.get("failure_modes", "")
        if modes_str and modes_str not in ("none", ""):
            for m in modes_str.split("|"):
                mode_counter[m.strip()] += 1

    if mode_counter:
        for mode, cnt in mode_counter.most_common():
            lines.append(f"| {mode} | {cnt} | {_pct(cnt, total)} |")
    else:
        lines.append("| — | 0 | 0.0% |")

    lines.append("")

    # ---------- Section 5: Mismatch Analysis ----------
    lines.append("## 5. Routing Mismatch Analysis\n")
    lines.append("| mismatch_type | count | example_prompt_preview |")
    lines.append("|---------------|-------|------------------------|")

    exp_direct_got_ttf = [
        r for r in live_rows
        if r.get("expected_routing") == "direct" and r.get("actual_routing") == "ttf"
    ]
    exp_ttf_got_direct = [
        r for r in live_rows
        if r.get("expected_routing") == "ttf" and r.get("actual_routing") == "direct"
    ]

    def _ex(rs: list[dict]) -> str:
        return rs[0].get("prompt_preview", "")[:60] if rs else "—"

    lines.append(
        f"| expected_direct_got_ttf | {len(exp_direct_got_ttf)} | {_ex(exp_direct_got_ttf)} |"
    )
    lines.append(
        f"| expected_ttf_got_direct | {len(exp_ttf_got_direct)} | {_ex(exp_ttf_got_direct)} |"
    )
    lines.append("")

    # ---------- Section 6: Formula Health Checks ----------
    lines.append("## 6. Formula Health Checks\n")

    if all_phis:
        boundary = [p for p in all_phis if 0.55 <= p <= 0.75]
        low_conf = [
            r for r in live_rows
            if isinstance(r.get("routing_confidence"), (int, float))
            and float(r["routing_confidence"]) < 0.5
        ]
        confs = [
            float(r["routing_confidence"])
            for r in live_rows
            if isinstance(r.get("routing_confidence"), (int, float))
        ]
        lines.append(f"- Groq Φ threshold: **{GROQ_PHI_THRESHOLD}**")
        lines.append(
            f"- Boundary zone [0.55–0.75]: **{len(boundary)} calls"
            f" ({_pct(len(boundary), total)})** — high-risk routing flip zone"
        )
        lines.append(f"- Low confidence (< 0.5): **{len(low_conf)} calls**")
        if confs:
            lines.append(
                f"- Routing confidence: min={min(confs):.3f}"
                f" mean={_mean(confs):.3f} max={max(confs):.3f}"
            )

    lines.append("")

    # ---------- Section 7: Enterprise Decision ----------
    lines.append("## 7. Enterprise Decision\n")
    lines.append("| model | verdict | rationale |")
    lines.append("|-------|---------|-----------|")
    lines.append(
        "| *Criteria* | GO: valid≥95% errors=0 match≥80%"
        " | CAUTION: valid≥85% OR match≥65%"
        " | HOLD: errors present OR valid<85% | |"
    )

    for model_api, mrs in sorted(model_groups.items()):
        n = len(mrs)
        n_error = sum(1 for r in mrs if r.get("actual_routing") == "error")
        valid_vals = [
            1 if r.get("schema_valid") is True else 0
            for r in mrs if r.get("schema_valid") != ""
        ]
        valid_rate = _mean([float(v) for v in valid_vals]) if valid_vals else 0.0
        match_vals = [
            1 if r.get("routing_match") is True else 0
            for r in mrs if r.get("routing_match") != ""
        ]
        match_rate = _mean([float(v) for v in match_vals]) if match_vals else 0.0

        if n_error == 0 and valid_rate >= 0.95 and match_rate >= 0.80:
            verdict = "GO"
            rationale = f"valid={valid_rate:.0%} match={match_rate:.0%} errors=0"
        elif n_error > 0 or valid_rate < 0.85:
            verdict = "HOLD"
            rationale = f"valid={valid_rate:.0%} errors={n_error}"
        else:
            verdict = "CAUTION"
            rationale = f"valid={valid_rate:.0%} match={match_rate:.0%}"

        lines.append(f"| {model_api} | **{verdict}** | {rationale} |")

    lines.append("")

    # ---------- Section 8: Next Cycle Proposals ----------
    lines.append("## 8. Next Cycle Proposals\n")

    proposals: list[str] = []

    if all_phis:
        boundary_count = len([p for p in all_phis if 0.55 <= p <= 0.75])
        if boundary_count >= 5:
            proposals.append(
                f"- **Threshold calibration**: {boundary_count} calls landed in "
                "the boundary zone [0.55–0.75]. Consider narrowing the test to "
                "boundary cases in cycle 2 with denser prompting around phi=0.65."
            )

    if exp_direct_got_ttf:
        proposals.append(
            f"- **Over-routing to TTF**: {len(exp_direct_got_ttf)} direct-expected "
            "calls routed to TTF. Inspect phi components (lambda2/tau/delta_k) for "
            "these cases — schema graph depth may be over-contributing."
        )

    if exp_ttf_got_direct:
        proposals.append(
            f"- **Under-routing to direct**: {len(exp_ttf_got_direct)} TTF-expected "
            "calls routed to direct. Check if failure-mode override fired "
            "unexpectedly, or if phi < 0.65 for prompts expected to be complex."
        )

    hold_models = [
        api for api, mrs in model_groups.items()
        if sum(1 for r in mrs if r.get("actual_routing") == "error") > 0
    ]
    if hold_models:
        proposals.append(
            f"- **Model stability**: {', '.join(hold_models)} had API errors. "
            "Increase `--sleep-ms` or isolate these models in a separate run."
        )

    if not proposals:
        proposals.append(
            "- No critical formula changes needed — all scenarios routed as "
            "expected. Run cycle 2 with different prompt variations to confirm "
            "stability."
        )

    lines.extend(proposals)
    lines.append("")
    lines.append("---")
    lines.append(f"*Report generated by `scripts/live_test_groq.py` — cycle {cycle}*")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report written: {report_path}")


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


async def async_main(args: argparse.Namespace) -> None:
    """Async body — runs inside a single event loop for the full test cycle."""
    # Build test cases
    all_cases = build_test_cases()

    # Optional model filter
    if args.models:
        allowed_keys = set(args.models.split(","))
        model_key_map = {
            "8b": "llama-3.1-8b-instant",
            "70b": "llama-3.3-70b-versatile",
            "gpt120": "openai/gpt-oss-120b",
            "gpt20": "openai/gpt-oss-20b",
            "compound": "groq/compound",
            "compmini": "groq/compound-mini",
            "qwen32": "qwen/qwen3-32b",
            "scout": "meta-llama/llama-4-scout-17b-16e-instruct",
        }
        allowed_api = {model_key_map[k] for k in allowed_keys if k in model_key_map}
        all_cases = [tc for tc in all_cases if groq_api_name(tc.model) in allowed_api]

    # Apply start-at / max-calls
    cases = all_cases[args.start_at - 1:]
    if args.max_calls is not None:
        cases = cases[:args.max_calls]

    total = len(cases)
    print("\nFormatShield Live Groq Test Runner")
    print(f"  cycle={args.cycle}  cases={total}  dry_run={args.dry_run}  sleep_ms={args.sleep_ms}")
    if args.start_at > 1:
        print(f"  resuming from call #{args.start_at}")
    print()

    # Compound model preflight
    has_compound = any(tc.compound_flag for tc in cases)
    compound_resolved: dict[str, str | None] = {}
    if has_compound and not args.dry_run:
        print("Running compound model preflight...")
        compound_resolved = await resolve_compound_aliases(dry_run=False)
        print()

    # Ensure CSV header exists
    ensure_csv_header()

    current_cycle_rows: list[dict[str, Any]] = []
    call_offset = args.start_at - 1  # for display numbering

    for i, tc in enumerate(cases, start=1):
        call_num = call_offset + i
        row = await run_one_call(
            tc=tc,
            cycle=args.cycle,
            call_num=call_num,
            dry_run=args.dry_run,
            compound_resolved=compound_resolved,
        )
        append_csv_row(row)
        current_cycle_rows.append(row)
        print_progress(call_num, call_offset + total, row)

        if args.sleep_ms > 0 and i < total:
            await asyncio.sleep(args.sleep_ms / 1000.0)

    print(f"\nAll {total} calls complete.")

    # Generate report (only from this run's rows)
    print("Generating report...")
    generate_report(args.cycle, current_cycle_rows)

    # Summary stats
    live = [r for r in current_cycle_rows if not r.get("dry_run")]
    if live:
        n_ok = sum(1 for r in live if r.get("routing_match") is True)
        n_err = sum(1 for r in live if r.get("actual_routing") == "error")
        n_valid = sum(1 for r in live if r.get("schema_valid") is True)
        print(f"\nCycle {args.cycle} summary:")
        print(f"  route match:  {n_ok}/{len(live)} ({_pct(n_ok, len(live))})")
        print(f"  schema valid: {n_valid}/{len(live)} ({_pct(n_valid, len(live))})")
        print(f"  errors:       {n_err}")
    print(f"  CSV:    {CSV_PATH}")
    print(f"  Report: {TESTING_DIR / f'report_cycle_{args.cycle}.md'}")


def main() -> None:
    # Load .env from project root if present (does not override existing env vars)
    env_path = PROJECT_ROOT / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(dotenv_path=env_path, override=False)
        except ImportError:
            pass  # dotenv optional; user can export GROQ_API_KEY directly

    parser = argparse.ArgumentParser(
        description="FormatShield live Groq API test runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--cycle",     type=int,   default=1,    help="Cycle number")
    parser.add_argument("--dry-run",   action="store_true",      help="Phi-only, no API")
    parser.add_argument("--models",    type=str,   default=None, help="Model keys, e.g. 8b,70b")
    parser.add_argument("--start-at",  type=int,   default=1,    help="Resume from call N")
    parser.add_argument("--max-calls", type=int,   default=None, help="Stop after N calls")
    parser.add_argument("--sleep-ms",  type=float, default=0,    help="ms between calls")
    args = parser.parse_args()

    if not args.dry_run and not os.environ.get("GROQ_API_KEY"):
        print("ERROR: GROQ_API_KEY not set. Export it or add to .env in project root.")
        sys.exit(1)

    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
