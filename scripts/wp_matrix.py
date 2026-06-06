"""Aggressive W&P matrix: vary context_window x max_output, repeat, record.

Shows how the pipeline adapts its decomposition (K leaves) and holds extraction
across very different model budgets, and exposes the run-to-run model variance by
repeating each setting. Pure measurement — no asserts.

    uv run python scripts/wp_matrix.py
"""

from __future__ import annotations

import json
import os
import statistics
import time
from pathlib import Path

_ROOT = Path(__file__).parent.parent
for _line in (_ROOT / ".env").read_text(encoding="utf-8").splitlines():
    if "=" in _line and not _line.startswith("#"):
        _k, _v = _line.split("=", 1)
        os.environ.setdefault(_k.strip(), _v.strip())

_MODEL = "groq/llama-3.3-70b-versatile"
_SCHEMA = _ROOT / "tests" / "fixtures" / "schemas" / "war_and_peace_200fields.json"
_DOC = _ROOT / "tests" / "fixtures" / "documents" / "_cache" / "war_and_peace.txt"

# (context_window, max_output, reps)
_MATRIX = [
    (20_000, 5_000, 3),  # baseline (record the band)
    (40_000, 8_000, 2),  # large window  -> fewer, bigger leaves
    (12_000, 4_000, 2),  # tight         -> more leaves
    (8_000, 3_000, 2),  # very tight    -> most leaves
]


def main() -> None:
    from formatshield import nfield
    from formatshield.config import ExtractionConfig

    schema = json.loads(_SCHEMA.read_text(encoding="utf-8"))
    document = _DOC.read_text(encoding="utf-8")
    cfg = ExtractionConfig(max_retry_rounds=1)

    rows: list[tuple] = []
    print(f"{'c_w':>7} {'m_o':>6} {'rep':>3} {'extract':>8} {'K':>3} {'Kmin':>4} {'sec':>6}")
    print("-" * 44)
    for c_w, m_o, reps in _MATRIX:
        per_setting: list[int] = []
        ks: list[int] = []
        for rep in range(1, reps + 1):
            t0 = time.time()
            r = nfield(
                document,
                schema,
                _MODEL,
                context_window=c_w,
                max_output_tokens=m_o,
                config=cfg,
            )
            dt = round(time.time() - t0, 1)
            m = r.metadata
            per_setting.append(m.fields_extracted)
            ks.append(m.K)
            rows.append((c_w, m_o, rep, m.fields_extracted, m.K, m.K_min, dt))
            print(
                f"{c_w:>7} {m_o:>6} {rep:>3} {m.fields_extracted:>6}/200 {m.K:>3} {m.K_min:>4} {dt:>6}"
            )
        mean = statistics.mean(per_setting)
        print(
            f"  -> {c_w}/{m_o}: mean extract {mean:.0f}/200  range {min(per_setting)}-{max(per_setting)}  K {min(ks)}-{max(ks)}"
        )
        print("-" * 44)

    out = _ROOT / "test-results" / "wp_matrix.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
