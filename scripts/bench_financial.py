"""Financial 10-K benchmark — runs the current pipeline, SAVES to test-results/.

Reproducible and verifiable (unlike a throwaway print): writes a numbered JSON
record per run. Loads .env for GROQ_API_KEY like the integration tests.

Usage:  uv run python scripts/bench_financial.py
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent

_env = _ROOT / ".env"
if _env.exists():
    for _line in _env.read_text(encoding="utf-8").splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from formatshield import nfield  # noqa: E402
from formatshield.config import ExtractionConfig  # noqa: E402

_SCHEMA = _ROOT / "tests" / "fixtures" / "schemas" / "financial_10k_3yr.json"
_DOC = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "real_10k.txt"


def main() -> None:
    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    document = _DOC.read_text(encoding="utf-8")
    t0 = time.time()
    model = "groq/llama-3.3-70b-versatile"
    result = nfield(
        document,
        schema,
        model,
        context_window=50_000,
        max_output_tokens=5_000,
        config=ExtractionConfig(max_retry_rounds=2),
    )
    elapsed = round(time.time() - t0, 1)
    m = result.metadata
    summary = {
        "model": model,
        "retriever": "bmx",
        "document_chars": len(document),
        "fields_total": m.fields_total,
        "fields_extracted": m.fields_extracted,
        "fields_missing": m.fields_missing,
        "pct_extracted": round(100 * m.fields_extracted / m.fields_total, 1),
        "K": m.K,
        "K_min": m.K_min,
        "quality_score": m.quality_score,
        "status": result.status.value,
        "elapsed_seconds": elapsed,
    }
    out = _ROOT / "test-results"
    out.mkdir(exist_ok=True)
    n = len(list(out.glob("bench_financial_*.json"))) + 1
    path = out / f"bench_financial_{n}.json"
    path.write_text(json.dumps({"summary": summary, "data": result.data}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved -> {path}")


if __name__ == "__main__":
    main()
