"""Live end-to-end test: real ~1000-field 10-K schema against a real SEC filing.

The largest realistic test in the suite. It runs the full S0-S6 pipeline with a
live Groq model on:
  - schema: tests/fixtures/schemas/financial_10k_realistic.json (1074 real
    US-GAAP / SEC-DEI leaf fields across seven fiscal years)
  - document: a real SEC Form 10-K (~9 MB) cached at
    tests/fixtures/documents/_cache/real_10k.txt (downloaded once)

This exercises BM25 chunking on a multi-megabyte document, multi-leaf packing of
~1000 fields, many real extraction calls, validation, and assembly together.

Requires GROQ_API_KEY and the cached 10-K. Skips otherwise.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Load .env at import time
# ---------------------------------------------------------------------------

_env_file = Path(__file__).parent.parent.parent / ".env"
if _env_file.exists():
    with open(_env_file) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ[_k.strip()] = _v.strip()

_GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

_MODEL_8B = "llama-3.1-8b-instant"
_CTX_8B = 131_072
_MAX_OUT_8B = 131_072

_SCHEMA_PATH = (
    Path(__file__).parent.parent / "fixtures" / "schemas" / "financial_10k_realistic.json"
)
_DOC_PATH = Path(__file__).parent.parent / "fixtures" / "documents" / "_cache" / "real_10k.txt"


def _require_inputs() -> tuple[dict, str]:
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")
    if not _SCHEMA_PATH.exists():
        pytest.skip("financial_10k_realistic.json fixture not found")
    if not _DOC_PATH.exists():
        pytest.skip("real_10k.txt not cached (run the fetch step first)")
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    document = _DOC_PATH.read_text(encoding="utf-8")
    return schema, document


def _make_provider():
    from formatshield.providers.groq._provider import GroqProvider

    # Use the model's REAL context window (131072). Decomposition of the
    # 1074-field schema into many focused calls comes from the accuracy-driven
    # per-call field cap (ExtractionConfig.max_fields_per_call), NOT from
    # under-reporting the context window.
    return GroqProvider(_MODEL_8B, context_window=_CTX_8B, max_output_tokens=_MAX_OUT_8B)


async def _run_pipeline(schema, document, provider):
    from formatshield.config import ExtractionConfig
    from formatshield.pipeline.s0_resources import run_stage_0
    from formatshield.pipeline.s1_schema import run_stage_1
    from formatshield.pipeline.s2a_structure import run_stage_2a
    from formatshield.pipeline.s2b_prepass import run_stage_2b
    from formatshield.pipeline.s2c_packing import run_stage_2c
    from formatshield.pipeline.s3_excerpt import run_stage_3
    from formatshield.pipeline.s4_extract import run_stage_4
    from formatshield.pipeline.s5_validate import run_stage_5
    from formatshield.pipeline.s6_assemble import run_stage_6

    # A 1074-field schema over a focused 10-K leaves most fields legitimately
    # absent; retrying ~900 missing fields adds cost without recall, so retry is
    # disabled here. (Validation still runs; the SFR retry loop is covered by the
    # unit tests and the Section 3/4 live tests.)
    cfg = ExtractionConfig(max_retry_rounds=0)
    state = await run_stage_0(provider, cfg)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, cfg)
    state = run_stage_2c(state, cfg)
    state = run_stage_3(state)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, cfg)
    result = run_stage_6(state)
    return result, state


def _assert_mechanically_sound(result, state) -> None:
    from formatshield.assembly._blackboard import FieldState
    from formatshield.validation._type_check import validate_field

    m = result.metadata
    accounted = (
        m.fields_extracted + m.fields_missing + m.fields_conflicted + m.fields_needs_revalidation
    )
    assert accounted == m.fields_total, f"accounted {accounted} != total {m.fields_total}"
    assert m.fields_total == len(state.fields)
    bb = state.blackboard
    pending = [p for p in bb.all_paths() if bb.get_state(p) == FieldState.PENDING]
    assert pending == [], f"{len(pending)} fields stuck PENDING"
    leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
    assert leaf_paths == {f.path for f in state.fields}
    # Every FILLED value must satisfy its schema constraints (validation ran).
    field_map = state.field_by_path
    for path, value in bb.get_filled().items():
        if value is not None:
            valid, err = validate_field(value, field_map[path])
            assert valid, f"FILLED field {path}={value!r} violates constraints: {err}"


class TestReal10KFullPipelineLive:
    """Full S0-S6 on a real 1074-field schema + a real 9 MB 10-K.

    Uses the model's REAL context window and max-output; capacity (how many
    leaves) is whatever the architecture's token math produces from those real
    numbers — not forced by a faked context or a static field cap. The run is
    expensive, so it executes ONCE and asserts every property from that pass.
    """

    @pytest.mark.asyncio
    async def test_real_10k_full_pipeline(self):
        schema, document = _require_inputs()
        result, state = await _run_pipeline(schema, document, _make_provider())

        # 1. Mechanically sound: all fields accounted, none stuck, no invalid FILLED.
        _assert_mechanically_sound(result, state)
        assert result.metadata.fields_total == 1074

        # 2. Large-document path: the 9 MB filing far exceeds the usable context,
        #    so Stage 2.5 must chunk it (BM25), regardless of how few leaves the
        #    capacity math then needs.
        assert state.lexical_index is not None
        assert len(state.segments) > 100

        # 3. Capacity is dynamic: leaf count follows the model's real C_eff / M_O.
        #    On a 131K-context model this wide schema legitimately fits very few
        #    calls — we assert the plan is coherent, not a specific leaf count.
        assert len(state.leaves) >= 1
        assert result.metadata.K >= 1
        assert result.metadata.K_min >= 1

        # 4. Real extraction occurred: genuine financial fields are recovered
        #    from the real 10-K.
        assert result.metadata.fields_extracted >= 5, (
            f"only {result.metadata.fields_extracted} fields extracted from a real 10-K"
        )
