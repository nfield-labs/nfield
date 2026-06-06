"""Live full-pipeline test: a 200-field schema over the full text of War & Peace.

A large-narrative stress test of the whole codebase via the public API ``nfield``:

  - schema: ``tests/fixtures/schemas/war_and_peace_200fields.json`` (200 leaf
    fields about the novel — book metadata, 16 major characters, 10 places,
    7 battles/events, all genuinely answerable from the prose).
  - document: the complete Project Gutenberg text of *War and Peace* (~3.3 MB),
    cached at ``tests/fixtures/documents/_cache/war_and_peace.txt`` (git-ignored).

Unlike a financial 10-K (whose XBRL field names do not match the table wording),
a novel's field words ("Napoleon", "Moscow", "Borodino") appear verbatim in the
prose, so lexical BM25 retrieval can actually find the right chunks. This is the
test that shows the full retrieval -> packing -> extraction -> assembly chain
working on a very large document.

Only the model and its real limits are supplied (context_window, max_output_tokens)
— no hand-tuned packing/retrieval numbers. Requires GROQ_API_KEY and the cached
book; skips otherwise.
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

# Only the model and its real limits — no other tuning numbers.
_MODEL = "groq/llama-3.3-70b-versatile"
_CONTEXT_WINDOW = 20_000
_MAX_OUTPUT = 5_000

_SCHEMA_PATH = _ROOT / "tests" / "fixtures" / "schemas" / "war_and_peace_200fields.json"
_DOC_PATH = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "war_and_peace.txt"
_RESULTS_DIR = _ROOT / "test-results"


def _require_inputs() -> tuple[dict, str]:
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")
    if not _SCHEMA_PATH.exists():
        pytest.skip("war_and_peace_200fields.json fixture not found")
    if not _DOC_PATH.exists():
        pytest.skip("war_and_peace.txt not cached")
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


def test_war_and_peace_full_pipeline() -> None:
    """Run nfield on a 200-field schema over the full War & Peace text."""
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
        # Strict grounding (the default) — values must come from the document, as
        # they would for any private document the model has never seen. This makes
        # the score a measure of the pipeline's own capacity (retrieval, packing,
        # extraction), not of the model's memory of a famous novel.
        config=ExtractionConfig(max_retry_rounds=1),
    )
    elapsed = round(time.time() - t0, 1)

    m = result.metadata
    summary = {
        "model": _MODEL,
        "context_window": _CONTEXT_WINDOW,
        "max_output_tokens": _MAX_OUTPUT,
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
    saved = _save_result("groq_war_and_peace", {"summary": summary, "data": result.data})
    print(f"\n[war & peace full pipeline] {summary}\nsaved -> {saved}")

    # --- Robustness + real-accuracy assertions ---
    assert m.fields_total == 200, "all schema fields must be planned"
    assert isinstance(result.data, dict) and result.data
    assert isinstance(result.status, ExtractionStatus)
    # A 3.3 MB book under a 20k window MUST drive heavy multi-leaf packing.
    assert m.K_min >= 2 and m.K >= 2
    # STORM GUARD: must not regress into a per-field retry/recovery storm.
    assert m.K < 80, f"K={m.K} indicates a retry/recovery storm regression"
    # Accent-folded BM25 must retrieve the characters whose prose spelling is
    # accented (Denísov, Kutúzov, ...). Under strict grounding ~170-180 of the 200
    # are stated in the prose; the rest are interpretive fields it never states
    # (a battle outcome, a notable trait). Floor at 150 locks in the retrieval
    # gain (pre-fold this run extracted 143) with margin for LLM variance.
    assert m.fields_extracted >= 150, f"only {m.fields_extracted}/200 grounded from the novel"

    # Well-known facts that appear verbatim in the prose must be captured.
    blob = " ".join(str(v).lower() for v in _leaf_values(result.data))
    assert "tolstoy" in blob or "napoleon" in blob, "should capture author/Napoleon"
