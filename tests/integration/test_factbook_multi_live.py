"""Live full-pipeline test: a 1045-field schema over a 3-country Factbook profile.

The >800-field stress test. Three public-domain country profiles (US, China, India)
merged into one namespaced schema and one ~141 KB document, generated from the same
data so every field's value genuinely appears in the text. The document exceeds the
context window, so this also exercises chunked retrieval at scale. Ground truth lets
us score real value accuracy.

Generate the fixture with scripts/gen_factbook_multi.py. Requires GROQ_API_KEY.
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
_CONTEXT_WINDOW = 40_000  # smaller than the doc → forces chunked retrieval
_MAX_OUTPUT = 8_000

_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_multi.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "factbook_multi.txt"
_TRUTH_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_multi_truth.json"
_RESULTS_DIR = _ROOT / "test-results"


def _require_inputs() -> tuple[dict, str, dict]:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    for p in (_SCHEMA_PATH, _DOC_PATH, _TRUTH_PATH):
        if not p.exists():
            pytest.skip(f"{p.name} not generated (run scripts/gen_factbook_multi.py)")
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
    elif obj not in (None, ""):
        out[prefix] = obj
    return out


def _save(name: str, payload: dict) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(_RESULTS_DIR.glob(f"{name}_*.json"))
    nums = [int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()]
    n = (max(nums) + 1) if nums else 1
    path = _RESULTS_DIR / f"{name}_{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def test_factbook_multi_full_pipeline() -> None:
    """Run nfield on a 1045-field, 3-country schema and score accuracy vs truth."""
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
        1
        for path, true_val in truth.items()
        if (g := _norm(extracted.get(path)))
        and (t := _norm(true_val))
        and (g == t or t in g or g in t)
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
    saved = _save("groq_factbook_multi", {"summary": summary, "data": result.data})
    print(f"\n[factbook multi full pipeline] {summary}\nsaved -> {saved}")

    assert m.fields_total >= 800, "this is the >800-field stress test"
    assert isinstance(result.data, dict) and result.data
    assert isinstance(result.status, ExtractionStatus)
    assert m.K < 120, f"K={m.K} indicates a retry/recovery storm regression"
    assert correct >= 300, f"only {correct} values matched ground truth across 3 countries"
    blob = " ".join(_norm(v) for v in extracted.values())
    assert "washington" in blob or "beijing" in blob, "a capital should be extracted"
