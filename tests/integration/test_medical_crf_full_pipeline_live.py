"""Live full-pipeline test: medical CRF schema on a real clinical note.

Exercises the public API ``nfield`` end-to-end on a REAL clinical document and a
schema whose fields actually occur in that document, so this is both a robustness
test and a genuine accuracy demonstration (unlike a generic schema vs an
arbitrary filing, where most fields are legitimately absent).

  - schema: ``tests/fixtures/schemas/medical_crf_134fields.json`` (84 leaf fields:
    patient demographics, vitals, diagnosis, medications, labs, symptoms, ...)
  - document: a real de-identified clinical transcription (an "Angina - Consult"
    note for a 68-year-old woman) from the public MTSamples corpus, fetched once
    to ``tests/fixtures/documents/clinical_note.txt`` (git-ignored).

Requires GROQ_API_KEY and the cached note; skips otherwise.

Fetch the note once (public MTSamples mirror):
    curl -sL -A "you@example.com" \\
      https://raw.githubusercontent.com/eshza/medicalTranscriptsKaggle/master/mtsamples.csv \\
      -> pick a rich "transcription" -> tests/fixtures/documents/clinical_note.txt
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

# --- Load .env at import time -------------------------------------------------
_ROOT = Path(__file__).parent.parent.parent
_env_file = _ROOT / ".env"
if _env_file.exists():
    for _line in _env_file.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_MODEL = "groq/llama-3.3-70b-versatile"
_CONTEXT_WINDOW = 131_072
_MAX_OUTPUT = 8_192

_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "medical_crf_134fields.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "clinical_note.txt"
_RESULTS_DIR = _ROOT / "test-results"


def _require_inputs() -> tuple[dict, str]:
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")
    if not _SCHEMA_PATH.exists():
        pytest.skip("medical_crf_134fields.json fixture not found")
    if not _DOC_PATH.exists():
        pytest.skip("clinical_note.txt not cached (fetch a MTSamples note; see module docstring)")
    return (
        json.loads(_SCHEMA_PATH.read_text(encoding="utf-8")),
        _DOC_PATH.read_text(encoding="utf-8"),
    )


def _save_result(name: str, payload: dict) -> Path:
    _RESULTS_DIR.mkdir(exist_ok=True)
    existing = sorted(_RESULTS_DIR.glob(f"{name}_*.json"))
    nums = [
        int(p.stem.rsplit("_", 1)[-1]) for p in existing if p.stem.rsplit("_", 1)[-1].isdigit()
    ]
    n = (max(nums) + 1) if nums else 1
    path = _RESULTS_DIR / f"{name}_{n}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _leaf_values(obj: object) -> list[object]:
    """Flatten all non-null leaf values from the nested result."""
    out: list[object] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out.extend(_leaf_values(v))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_leaf_values(v))
    elif obj is not None:
        out.append(obj)
    return out


def test_medical_crf_full_pipeline_on_real_clinical_note() -> None:
    """Run nfield on the medical CRF schema against a real clinical note."""
    from formatshield import nfield
    from formatshield.config import ExtractionConfig
    from formatshield.types import ExtractionStatus

    schema, document = _require_inputs()

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
    summary = {
        "model": _MODEL,
        "document_chars": len(document),
        "fields_total": m.fields_total,
        "fields_extracted": m.fields_extracted,
        "fields_missing": m.fields_missing,
        "K_leaves": m.K,
        "K_min": m.K_min,
        "quality_score": m.quality_score,
        "status": result.status.value,
        "elapsed_seconds": elapsed,
    }
    saved = _save_result("groq_medical_crf", {"summary": summary, "data": result.data})
    print(f"\n[medical CRF full pipeline] {summary}\nsaved -> {saved}")

    # --- Assertions: a matching schema on a real note must extract WELL ---
    assert m.fields_total == 84, "all schema fields must be planned"
    assert isinstance(result.data, dict) and result.data
    assert isinstance(result.status, ExtractionStatus)
    assert m.K >= 1 and m.K_min >= 1
    # STORM GUARD: must not regress into a per-field retry storm.
    assert m.K < 30, f"K={m.K} indicates a retry/recovery storm regression"

    # The note clearly states demographics + an angina diagnosis, so meaningful
    # extraction must happen (far more than the ~3% seen on a mismatched schema).
    assert m.fields_extracted >= 10, f"only {m.fields_extracted}/84 extracted from a matching note"

    # The diagnosis is the centrepiece of this consult; it must be captured.
    all_text = " ".join(str(v).lower() for v in _leaf_values(result.data))
    assert "angina" in all_text, "primary diagnosis (angina) should be extracted"
