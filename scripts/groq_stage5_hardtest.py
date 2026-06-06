"""Live Groq hard test: ~1000-field schema on llama-3.3-70b, forcing leaf split.

Exercises the full pipeline at scale (structural grouping → capacity packing →
leaf split → SFEP extraction → validation/SFR → always-on missing-field recovery
→ assembly) against a real model, and saves each run's full result to a numbered,
git-ignored file so successive runs never clobber each other.

Run:  uv run --extra groq python scripts/groq_stage5_hardtest.py
Env:  GROQ_API_KEY (read from .env if present)
Knobs: MODEL / CONTEXT_WINDOW / MAX_OUTPUT / N_RECORDS / FIELDS_PER_RECORD below.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# --- Tunable knobs (the user asked for these exact values) -------------------
MODEL = "groq/llama-3.3-70b-versatile"
CONTEXT_WINDOW = 50_000
MAX_OUTPUT = 10_000
N_RECORDS = 50
FIELDS_PER_RECORD = 20  # 50 x 20 = 1000 fields
RESULTS_DIR = Path(__file__).resolve().parent.parent / "test-results"


def _load_env() -> None:
    env = Path(__file__).resolve().parent.parent / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def build_schema_and_document() -> tuple[dict[str, Any], str, dict[str, str]]:
    """Build a ~1000-field nested schema, a document, and the ground-truth map."""
    properties: dict[str, Any] = {}
    lines: list[str] = ["EMPLOYEE RECORDS EXPORT — 2026", ""]
    truth: dict[str, str] = {}
    for r in range(N_RECORDS):
        rec_key = f"record_{r:02d}"
        fields: dict[str, Any] = {}
        lines.append(f"Record {r:02d}:")
        for f in range(FIELDS_PER_RECORD):
            fkey = f"field_{f:02d}"
            value = f"r{r:02d}f{f:02d}val"
            fields[fkey] = {"type": "string", "description": f"value of field {f:02d}"}
            truth[f"{rec_key}.{fkey}"] = value
            lines.append(f"  field {f:02d} = {value}")
        properties[rec_key] = {"type": "object", "properties": fields}
        lines.append("")
    schema = {"type": "object", "properties": properties}
    return schema, "\n".join(lines), truth


def _next_path() -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(RESULTS_DIR.glob("groq_stage5_answer_*.json"))
    n = 1
    if existing:
        nums = [
            int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()
        ]
        n = (max(nums) + 1) if nums else len(existing) + 1
    return RESULTS_DIR / f"groq_stage5_answer_{n}.json"


def _accuracy(data: dict[str, Any], truth: dict[str, str]) -> tuple[int, int]:
    correct = 0
    for dotted, expected in truth.items():
        rec, fld = dotted.split(".")
        got = data.get(rec, {}).get(fld) if isinstance(data.get(rec), dict) else None
        if got == expected:
            correct += 1
    return correct, len(truth)


def main() -> None:
    _load_env()
    if not os.getenv("GROQ_API_KEY"):
        print("SKIP: GROQ_API_KEY not set")
        return

    from formatshield import nfield
    from formatshield.config import ExtractionConfig

    schema, document, truth = build_schema_and_document()
    total_fields = len(truth)
    print(
        f"Schema: {total_fields} fields ({N_RECORDS}x{FIELDS_PER_RECORD}) | "
        f"doc ~{len(document)} chars | model={MODEL} C_W={CONTEXT_WINDOW} M_O={MAX_OUTPUT}"
    )

    cfg = ExtractionConfig(max_retry_rounds=2)
    t0 = time.time()
    result = nfield(
        document,
        schema,
        MODEL,
        context_window=CONTEXT_WINDOW,
        max_output_tokens=MAX_OUTPUT,
        config=cfg,
    )
    elapsed = time.time() - t0

    m = result.metadata
    correct, total = _accuracy(result.data, truth)
    summary = {
        "model": MODEL,
        "context_window": CONTEXT_WINDOW,
        "max_output_tokens": MAX_OUTPUT,
        "fields_total": m.fields_total,
        "fields_extracted": m.fields_extracted,
        "fields_missing": m.fields_missing,
        "fields_needs_revalidation": m.fields_needs_revalidation,
        "K_leaves": m.K,
        "K_min": m.K_min,
        "optimality_gap": m.optimality_gap,
        "quality_score": m.quality_score,
        "status": result.status.value,
        "ground_truth_correct": correct,
        "ground_truth_total": total,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "elapsed_seconds": round(elapsed, 1),
    }

    out_path = _next_path()
    out_path.write_text(
        json.dumps({"summary": summary, "data": result.data}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n=== RESULT ===")
    for k, v in summary.items():
        print(f"  {k:24s}: {v}")
    print(
        f"\nLeaf split: K={m.K} leaves (K_min={m.K_min}) — "
        f"{'SPLIT into multiple leaves [OK]' if m.K > 1 else 'single leaf'}"
    )
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
