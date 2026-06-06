"""Live full-pipeline test: a STRONG typed, deeply-nested clinical-trial schema.

The toughest fixture so far — not all-string like the factbooks. A real
ClinicalTrials.gov study (public domain) flattened into 300+ leaves spanning
strings, integers, numbers, booleans, enums (with constraints), and arrays, nested
many levels deep (indexed arms / outcomes / locations). The document is rendered
from the same data so every value is genuinely present, and a ground-truth map
allows real, type-aware accuracy scoring.

Generate the fixture with scripts/gen_clinicaltrial_fixture.py. Requires GROQ_API_KEY.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

_ROOT = Path(__file__).parent.parent.parent
_env = _ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_MODEL = "groq/llama-3.3-70b-versatile"
_CONTEXT_WINDOW = 40_000  # sweet spot from the budget matrix
_MAX_OUTPUT = 8_000

_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "clinicaltrial.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "clinicaltrial.txt"
_TRUTH_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "clinicaltrial_truth.json"
_RESULTS_DIR = _ROOT / "test-results"


def _require_inputs() -> tuple[dict, str, dict]:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    for p in (_SCHEMA_PATH, _DOC_PATH, _TRUTH_PATH):
        if not p.exists():
            pytest.skip(f"{p.name} not generated (run scripts/gen_clinicaltrial_fixture.py)")
    return (
        json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")),
        _DOC_PATH.read_text(encoding="utf-8"),
        json.loads(_TRUTH_PATH.read_text(encoding="utf-8")),
    )


def _norm(value: object) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _flat(obj: object, prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flat(v, f"{prefix}.{k}" if prefix else k))
    elif obj is not None and obj != "":
        out[prefix] = obj
    return out


def _values_match(got: object, truth: object) -> bool:
    """Type-aware comparison: bools/numbers exact, lists by set, strings by containment."""
    if isinstance(truth, bool):
        return bool(got) == truth if isinstance(got, (bool, int, str)) else False
    if isinstance(truth, (int, float)) and not isinstance(truth, bool):
        try:
            return abs(float(got) - float(truth)) < 1e-6  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return _norm(got) == _norm(truth)
    if isinstance(truth, list):
        # Credit captured content even when the model returns the items joined into
        # one element (a representation difference, not data loss): split any joined
        # string on comma/semicolon before comparing item sets.
        raw = got if isinstance(got, list) else [got]
        got_items: set[str] = set()
        for x in raw:
            got_items.update(_norm(part) for part in re.split(r"[;,]", str(x)) if part.strip())
        truth_set = {_norm(t) for t in truth}
        return len(got_items & truth_set) >= max(1, len(truth_set) // 2)
    g, t = _norm(got), _norm(truth)
    return bool(g) and bool(t) and (g == t or t in g or g in t)


def _save(name: str, payload: dict) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(_RESULTS_DIR.glob(f"{name}_*.json"))
    nums = [
        int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    n = (max(nums) + 1) if nums else 1
    path = _RESULTS_DIR / f"{name}_{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def test_clinicaltrial_full_pipeline() -> None:
    """Run nfield on a 300+ field typed, nested clinical-trial schema."""
    from formatshield import nfield
    from formatshield.config import ExtractionConfig
    from formatshield.types import ExtractionStatus

    schema, document, truth = _require_inputs()

    t0 = time.time()
    result = nfield(
        document,
        schema,
        _MODEL,
        context_window=_CONTEXT_WINDOW,
        max_output_tokens=_MAX_OUTPUT,
        config=ExtractionConfig(max_retry_rounds=1),
    )
    elapsed = round(time.time() - t0, 1)
    m = result.metadata

    extracted = _flat(result.data)
    correct = sum(
        1 for p, tv in truth.items() if p in extracted and _values_match(extracted[p], tv)
    )
    accuracy = round(100 * correct / len(truth), 1)
    summary = {
        "model": _MODEL,
        "document_chars": len(document),
        "fields_total": m.fields_total,
        "fields_extracted": m.fields_extracted,
        "pct_extracted": round(100 * m.fields_extracted / m.fields_total, 1),
        "values_correct_vs_truth": correct,
        "value_accuracy_pct": accuracy,
        "K_leaves": m.K,
        "K_min": m.K_min,
        "quality_score": m.quality_score,
        "status": result.status.value,
        "elapsed_seconds": elapsed,
    }
    saved = _save("groq_clinicaltrial", {"summary": summary, "data": result.data})
    print(f"\n[clinicaltrial strong typed pipeline] {summary}\nsaved -> {saved}")

    assert m.fields_total >= 250, "this is the >250-field strong-schema test"
    assert isinstance(result.data, dict) and result.data
    assert isinstance(result.status, ExtractionStatus)
    assert m.K < 60, f"K={m.K} indicates a retry/recovery storm regression"
    assert correct >= 120, f"only {correct} typed values matched ground truth"
    blob = " ".join(_norm(v) for v in extracted.values())
    assert "interventional" in blob or "vaccine" in blob or "phase" in blob
