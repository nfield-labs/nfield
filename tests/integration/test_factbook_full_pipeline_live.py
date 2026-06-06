"""Live full-pipeline test: a 335-field schema over a CIA World Factbook profile.

A large, fact-dense, **fair** extraction test — unlike a clinical note or raw
iXBRL, every field's value genuinely appears in the document (both are generated
from the same public-domain Factbook data), and a ground-truth map lets us measure
real value *accuracy*, not just whether something was returned.

  - schema:   tests/fixtures/schemas/factbook_us.json (335 fields, nested by category)
  - document: tests/fixtures/documents/_cache/factbook_us.txt (~44 KB readable profile)
  - truth:    tests/fixtures/schemas/factbook_us_truth.json ({path: value})

Generate the fixtures with scripts/gen_factbook_fixture.py. Requires GROQ_API_KEY;
skips otherwise.
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
_CONTEXT_WINDOW = 40_000
_MAX_OUTPUT = 8_000

_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_us.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "factbook_us.txt"
_TRUTH_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "factbook_us_truth.json"
_RESULTS_DIR = _ROOT / "test-results"


def _require_inputs() -> tuple[dict, str, dict]:
    if not os.getenv("GROQ_API_KEY"):
        pytest.skip("GROQ_API_KEY not set")
    for p in (_SCHEMA_PATH, _DOC_PATH, _TRUTH_PATH):
        if not p.exists():
            pytest.skip(f"{p.name} not generated (run scripts/gen_factbook_fixture.py)")
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
    nums = [
        int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    n = (max(nums) + 1) if nums else 1
    path = _RESULTS_DIR / f"{name}_{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def test_factbook_full_pipeline() -> None:
    """Run nfield on a 335-field Factbook schema and score value accuracy vs truth."""
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

    # Value accuracy vs ground truth: a field is correct if the truth value and the
    # extracted value match after normalisation (either contains the other — the
    # model may trim or lightly rephrase a long Factbook value).
    extracted = _flat(result.data)
    correct = 0
    for path, true_val in truth.items():
        got = extracted.get(path)
        if got is None:
            continue
        g, t = _norm(got), _norm(true_val)
        if g and t and (g == t or t in g or g in t):
            correct += 1

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
    saved = _save("groq_factbook", {"summary": summary, "data": result.data})
    print(f"\n[factbook full pipeline] {summary}\nsaved -> {saved}")

    # --- robustness assertions ---
    assert m.fields_total >= 200, "this is the large-schema (>200 fields) test"
    assert isinstance(result.data, dict) and result.data
    assert isinstance(result.status, ExtractionStatus)
    assert m.K < 60, f"K={m.K} indicates a retry/recovery storm regression"
    # Fair test: the values are in the document, so accuracy must be substantial.
    assert correct >= 100, f"only {correct} values matched ground truth"
    # Well-known facts must be right.
    blob = " ".join(_norm(v) for v in extracted.values())
    assert "washington" in blob, "capital (Washington, DC) should be extracted"
