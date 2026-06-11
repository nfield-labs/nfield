"""Run the NEW approach (GLEAN typed-fusion retrieval) across all live fixtures.

One by one, with GLEAN enabled (use_typed_fusion_retrieval=True), using each
integration test's own model / context window / paths. Scores against ground truth
where a truth map exists (clinical, factbook US, factbook multi) and by extraction
coverage otherwise (war & peace, medical). Saves every run to test-results/ with the
same numbered scheme the integration tests use, then prints one combined table.

Sequential by design. Requires GROQ_API_KEY. Optionally set GROQ_BASE_URL to
route through a proxy / gateway / self-hosted Groq-compatible endpoint — both the
key and the base URL are passed through nfield(api_key=..., base_url=...), the
same explicit-credential path the competitor libraries expose.

Usage:
    python scripts/run_new_approach.py
    GROQ_BASE_URL=https://my-proxy/v1 python scripts/run_new_approach.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT / "src"))

_env = _ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_MODEL = "groq/llama-3.3-70b-versatile"
# Explicit credentials path (parity with Outlines/Instructor/Guidance): pass the
# key and an optional base URL through nfield instead of relying only on env
# pickup. api_key=None would still fall back to GROQ_API_KEY inside the SDK.
_API_KEY = os.getenv("GROQ_API_KEY")
_BASE_URL = os.getenv("GROQ_BASE_URL")  # None -> SDK default endpoint
_FIX = _ROOT / "tests" / "fixtures"
_SCHEMAS = _FIX / "schemas"
_DOCS = _FIX / "documents"
_RESULTS = _ROOT / "test-results"


@dataclass(frozen=True)
class Fixture:
    name: str
    save_name: str
    schema: Path
    doc: Path
    truth: Path | None
    window: int
    max_output: int


_FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        "war_and_peace",
        "glean_war_and_peace",
        _SCHEMAS / "war_and_peace_200fields.json",
        _DOCS / "_cache" / "war_and_peace.txt",
        None,
        20_000,
        5_000,
    ),
    Fixture(
        "medical_crf",
        "glean_medical_crf",
        _SCHEMAS / "medical_crf_134fields.json",
        _DOCS / "clinical_note.txt",
        None,
        131_072,
        8_192,
    ),
    Fixture(
        "clinicaltrial",
        "glean_clinicaltrial",
        _SCHEMAS / "clinicaltrial.json",
        _DOCS / "_cache" / "clinicaltrial.txt",
        _SCHEMAS / "clinicaltrial_truth.json",
        40_000,
        8_000,
    ),
    Fixture(
        "factbook_us",
        "glean_factbook_us",
        _SCHEMAS / "factbook_us.json",
        _DOCS / "_cache" / "factbook_us.txt",
        _SCHEMAS / "factbook_us_truth.json",
        40_000,
        8_000,
    ),
    Fixture(
        "factbook_multi",
        "glean_factbook_multi",
        _SCHEMAS / "factbook_multi.json",
        _DOCS / "_cache" / "factbook_multi.txt",
        _SCHEMAS / "factbook_multi_truth.json",
        40_000,
        8_000,
    ),
)


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _flat(obj: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
    elif obj not in (None, ""):
        out[prefix] = obj
    return out


def _typed_paths(schema: dict, truth: dict) -> set[str]:
    typed: set[str] = set()

    def walk(node: object, path: str) -> None:
        if not isinstance(node, dict):
            return
        if node.get("type") == "object":
            for k, v in node.get("properties", {}).items():
                walk(v, f"{path}.{k}" if path else k)
            return
        is_typed = (
            node.get("type") in {"boolean", "integer", "number"}
            or "enum" in node
            or "format" in node
        )
        if is_typed and path in truth:
            typed.add(path)

    walk(schema, "")
    return typed


def _save(name: str, payload: dict) -> Path:
    _RESULTS.mkdir(exist_ok=True)
    existing = sorted(_RESULTS.glob(f"{name}_*.json"))
    nums = [
        int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    n = (max(nums) + 1) if nums else 1
    path = _RESULTS / f"{name}_{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _run_one(fx: Fixture) -> dict | None:
    from formatshield import nfield
    from formatshield.config import ExtractionConfig

    if not (fx.schema.exists() and fx.doc.exists()):
        print(f"\n[{fx.name}] fixture missing — skipped", flush=True)
        return None

    schema = json.loads(fx.schema.read_text(encoding="utf-8"))
    document = fx.doc.read_text(encoding="utf-8")
    truth = (
        json.loads(fx.truth.read_text(encoding="utf-8"))
        if fx.truth and fx.truth.exists()
        else None
    )

    # GLEAN only engages when the doc exceeds the usable window (else fast path).
    est_doc_tokens = len(document) // 4
    retrieval_active = est_doc_tokens > fx.window // 2

    print(f"\n########## {fx.name}  (GLEAN ON) ##########", flush=True)
    print(
        f"  doc={len(document):,} chars (~{est_doc_tokens:,} tok)  window={fx.window:,}  "
        f"retrieval_active={retrieval_active}",
        flush=True,
    )
    t0 = time.time()
    result = nfield(
        document,
        schema,
        _MODEL,
        context_window=fx.window,
        max_output_tokens=fx.max_output,
        api_key=_API_KEY,
        base_url=_BASE_URL,
        config=ExtractionConfig(max_retry_rounds=1),
    )
    elapsed = round(time.time() - t0, 1)
    m = result.metadata

    summary: dict[str, object] = {
        "retriever": "GLEAN",
        "retrieval_active": retrieval_active,
        "fields_total": m.fields_total,
        "fields_extracted": m.fields_extracted,
        "pct_extracted": round(100 * m.fields_extracted / m.fields_total, 1),
        "K": m.K,
        "K_min": m.K_min,
        "quality_score": m.quality_score,
        "status": result.status.value,
        "elapsed_seconds": elapsed,
    }
    if truth:
        extracted = _flat(result.data)
        typed = _typed_paths(schema, truth)
        correct = typed_correct = 0
        for path, true_val in truth.items():
            got, want = _norm(extracted.get(path)), _norm(true_val)
            hit = bool(got) and bool(want) and (got == want or want in got or got in want)
            correct += int(hit)
            if path in typed:
                typed_correct += int(hit)
        summary["value_accuracy_pct"] = round(100 * correct / len(truth), 1)
        summary["typed_accuracy_pct"] = (
            round(100 * typed_correct / len(typed), 1) if typed else None
        )
        summary["typed_correct"] = f"{typed_correct}/{len(typed)}"

    saved = _save(fx.save_name, {"summary": summary, "data": result.data})
    print(f"  {summary}\n  saved -> {saved.name}", flush=True)
    return summary


def main() -> None:
    if not os.getenv("GROQ_API_KEY"):
        print("GROQ_API_KEY not set — cannot run")
        return

    wanted = set(sys.argv[1:])
    fixtures = [f for f in _FIXTURES if not wanted or f.name in wanted]
    rows: list[tuple[str, dict]] = []
    for fx in fixtures:
        try:
            summary = _run_one(fx)
        except Exception as exc:  # one fixture failing must not kill the rest
            print(f"  [{fx.name}] ERROR: {exc}", flush=True)
            summary = None
        if summary is not None:
            rows.append((fx.name, summary))

    print("\n\n================ NEW APPROACH (GLEAN) — SUMMARY TABLE ================")
    hdr = (
        f"{'fixture':<16}{'tot':>5}{'extr':>6}{'extr%':>7}{'val%':>7}"
        f"{'typed%':>8}{'K':>5}{'retr':>6}{'status':>9}{'sec':>7}"
    )
    print(hdr)
    print("-" * len(hdr))
    for name, s in rows:
        val = s.get("value_accuracy_pct")
        typ = s.get("typed_accuracy_pct")
        print(
            f"{name:<16}{s['fields_total']:>5}{s['fields_extracted']:>6}{s['pct_extracted']:>6.1f}%"
            f"{(f'{val:.1f}%' if val is not None else '-'):>7}"
            f"{(f'{typ:.1f}%' if typ is not None else '-'):>8}"
            f"{s['K']:>5}{('yes' if s['retrieval_active'] else 'fast'):>6}"
            f"{s['status']:>9}{s['elapsed_seconds']:>7}"
        )


if __name__ == "__main__":
    main()
