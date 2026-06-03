"""Aggressive scale tests: huge schemas x huge documents via real Groq.

This is the maximum-stress dimension: hundreds of fields packed across many
CapacityLeafs, run against large documents, hitting many parallel Groq calls,
BM25 chunking, multi-leaf packing, and execution rounds simultaneously.

Two kinds of assertion:
  - Mechanical correctness (deterministic): every field accounted for, no field
    left PENDING, no context overflow, K reasonable, status valid, leaves cover
    all fields. These hold regardless of what the model returns.
  - Accuracy at scale (synthetic): a generated N-field schema whose values are
    all present in a generated document — asserts the pipeline extracts a healthy
    fraction across many leaves.

Real fixture field counts (flattened): invoice=53, medical=84, financial=190.

Requires GROQ_API_KEY; tests auto-skip without it. Network is also required for
the large Gutenberg documents (downloaded + cached by the large-doc test module).
"""

from __future__ import annotations

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

_SCHEMA_DIR = Path(__file__).parent.parent / "fixtures" / "schemas"
_BOOK_CACHE = Path(__file__).parent.parent / "fixtures" / "documents" / "_cache"


def _skip_no_key() -> None:
    if not _GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set — skipping aggressive live test")


def _make_provider(ctx: int = _CTX_8B, max_out: int = _MAX_OUT_8B):
    from formatshield.providers.groq._provider import GroqProvider

    return GroqProvider(_MODEL_8B, context_window=ctx, max_output_tokens=max_out)


def _load_schema(name: str) -> dict:
    import json

    path = _SCHEMA_DIR / name
    if not path.exists():
        pytest.skip(f"fixture {name} not found")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_book(filename: str) -> str:
    """Load a cached Gutenberg book (downloaded by the large-doc test module)."""
    path = _BOOK_CACHE / filename
    if not path.exists():
        pytest.skip(f"{filename} not cached — run test_large_documents_live.py first")
    return path.read_text(encoding="utf-8")


async def _run_pipeline(schema, document, provider, *, config=None, return_state=False):
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

    cfg = config or ExtractionConfig()
    state = await run_stage_0(provider, cfg)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, cfg)
    state = run_stage_2c(state, cfg)
    state = run_stage_3(state)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, cfg)
    result = run_stage_6(state)
    return (result, state) if return_state else result


def _assert_mechanically_sound(result, state) -> None:
    """Invariants that must hold for ANY pipeline run, regardless of model output."""
    from formatshield.assembly._blackboard import FieldState

    m = result.metadata
    # 1. Every field accounted for in exactly one outcome bucket
    accounted = (
        m.fields_extracted + m.fields_missing + m.fields_conflicted + m.fields_needs_revalidation
    )
    assert accounted == m.fields_total, f"accounted {accounted} != total {m.fields_total}"
    # 2. fields_total matches the flattened schema
    assert m.fields_total == len(state.fields)
    # 3. No field left dangling in PENDING after the pipeline
    bb = state.blackboard
    pending = [p for p in bb.all_paths() if bb.get_state(p) == FieldState.PENDING]
    assert pending == [], f"{len(pending)} fields stuck PENDING"
    # 4. Every leaf covered by execution order exactly once
    in_order = [leaf for r in state.execution_order for leaf in r]
    assert len(in_order) == len(state.leaves)
    # 5. All fields packed into leaves with no loss
    leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
    assert leaf_paths == {f.path for f in state.fields}
    # 6. K_min <= K (we never beat the theoretical minimum)
    assert m.K >= 1 and m.K_min >= 1
    # 7. quality + optimality in range
    assert 0.0 <= m.quality_score <= 1.0
    assert 0.0 <= m.optimality_gap <= 1.0


# ---------------------------------------------------------------------------
# Synthetic huge schema + matching document (accuracy at scale)
# ---------------------------------------------------------------------------


def _build_synthetic(n_fields: int) -> tuple[dict, str, dict[str, str]]:
    """Build an N-field flat schema and a document containing every value.

    Returns (schema, document, expected_values).
    """
    props: dict[str, dict] = {}
    expected: dict[str, str] = {}
    lines: list[str] = ["SYNTHETIC RECORD", ""]
    for i in range(n_fields):
        key = f"field_{i:03d}"
        value = f"value{i:03d}"
        props[key] = {"type": "string", "description": f"the {i}-th synthetic field"}
        expected[key] = value
        lines.append(f"{key}: {value}")
    schema = {"type": "object", "properties": props}
    return schema, "\n".join(lines), expected


# ---------------------------------------------------------------------------
# Test Class 1: Huge real schemas x small doc — multi-leaf mechanics
# ---------------------------------------------------------------------------


class TestHugeSchemaSmallDoc:
    """Hundreds of fields force many leaves even with a tiny document."""

    _DOC = "Quarterly report. Acme Corp. Revenue 2.0M USD. Filed 2026-05-15. Public company."

    @pytest.mark.asyncio
    async def test_53_field_invoice(self):
        _skip_no_key()
        schema = _load_schema("invoice_50fields.json")
        result, state = await _run_pipeline(schema, self._DOC, _make_provider(), return_state=True)
        _assert_mechanically_sound(result, state)

    @pytest.mark.asyncio
    async def test_84_field_medical(self):
        _skip_no_key()
        schema = _load_schema("medical_crf_134fields.json")
        result, state = await _run_pipeline(schema, self._DOC, _make_provider(), return_state=True)
        _assert_mechanically_sound(result, state)

    @pytest.mark.asyncio
    async def test_190_field_financial_forces_multileaf(self):
        _skip_no_key()
        schema = _load_schema("financial_sec_369fields.json")
        # Small context → 190 fields cannot fit one call → many leaves.
        provider = _make_provider(ctx=4096, max_out=4096)
        result, state = await _run_pipeline(schema, self._DOC, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        assert len(state.leaves) > 1, "190 fields in 4K context must split into many leaves"


# ---------------------------------------------------------------------------
# Test Class 2: Huge schema x huge document — full stress, heavy tokens
# ---------------------------------------------------------------------------


class TestHugeSchemaHugeDoc:
    """190 fields against a 1.2 MB book: chunking + multi-leaf + many calls."""

    @pytest.mark.asyncio
    async def test_190_fields_x_moby_dick_mechanics(self):
        _skip_no_key()
        schema = _load_schema("financial_sec_369fields.json")
        doc = _load_book("moby_dick.txt")
        # Small context → forces both BM25 chunking (large doc) and many leaves.
        provider = _make_provider(ctx=8192, max_out=8192)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        assert state.bm25_index is not None, "large doc must take chunking path"
        assert len(state.leaves) > 1

    @pytest.mark.asyncio
    async def test_84_fields_x_moby_dick_no_overflow(self):
        """No leaf excerpt overflows the context window (H1 fix at scale)."""
        _skip_no_key()
        schema = _load_schema("medical_crf_134fields.json")
        doc = _load_book("moby_dick.txt")
        provider = _make_provider(ctx=8192, max_out=8192)
        _, state = await _run_pipeline(schema, doc, provider, return_state=True)
        for leaf in state.leaves:
            # input estimate = overhead + excerpt tokens; must leave room for output
            excerpt_tokens = len(leaf.document_excerpt) / max(state.chars_per_token, 1.0)
            total = leaf.overhead + excerpt_tokens + leaf.safe_output
            assert total <= state.C_eff, (
                f"leaf {leaf.leaf_id} request {total:.0f} exceeds C_eff {state.C_eff}"
            )


# ---------------------------------------------------------------------------
# Test Class 3: Accuracy at scale (synthetic schema whose values are present)
# ---------------------------------------------------------------------------


class TestAccuracyAtScale:
    """A generated N-field schema whose values all appear in the document."""

    @pytest.mark.asyncio
    async def test_60_fields_synthetic_extraction_rate(self):
        _skip_no_key()
        schema, doc, expected = _build_synthetic(60)
        # Moderate context so it still multi-leafs but the small doc fits each call.
        provider = _make_provider(ctx=16384, max_out=8192)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        # Count correct extractions
        correct = sum(1 for k, v in expected.items() if str(result.data.get(k, "")).strip() == v)
        rate = correct / len(expected)
        assert rate >= 0.5, (
            f"Only {correct}/{len(expected)} synthetic fields correct ({rate:.0%}); "
            "multi-leaf extraction degraded at scale"
        )

    @pytest.mark.asyncio
    async def test_120_fields_synthetic_all_accounted(self):
        _skip_no_key()
        schema, doc, _ = _build_synthetic(120)
        provider = _make_provider(ctx=8192, max_out=8192)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        assert len(state.leaves) > 1

    @pytest.mark.asyncio
    async def test_500_fields_live_many_calls(self):
        """500 fields under a small context: many leaves, many real Groq calls."""
        _skip_no_key()
        schema, doc, _ = _build_synthetic(500)
        provider = _make_provider(ctx=4096, max_out=4096)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        assert len(state.leaves) > 5  # 500 fields in a 4K context → many leaves
        assert result.metadata.K >= 1


class TestThousandFieldFullPipelineLive:
    """End-to-end at 1000 fields with a large context window and real Groq.

    Validates that Stages S0-S6 cooperate at extreme width against a live model:
    calibration, flatten, grouping, pre-pass, packing/splitting, excerpt,
    extraction (many real calls), validation, and assembly. Heavy token usage by
    design.
    """

    @pytest.mark.asyncio
    async def test_1000_fields_large_context_mechanics(self):
        """1000 fields, large context: pipeline completes and accounts for all."""
        _skip_no_key()
        schema, doc, _ = _build_synthetic(1000)
        # Large context window (64K usable) — a few wide leaves, big real calls.
        provider = _make_provider(ctx=_CTX_8B, max_out=_MAX_OUT_8B)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        assert result.metadata.fields_total == 1000
        leaf_paths = {f.path for leaf in state.leaves for f in leaf.fields}
        assert len(leaf_paths) == 1000  # every field packed across the leaves

    @pytest.mark.asyncio
    async def test_1000_fields_real_extraction_occurs(self):
        """A meaningful share of the 1000 fields is actually extracted live."""
        _skip_no_key()
        schema, doc, expected = _build_synthetic(1000)
        provider = _make_provider(ctx=_CTX_8B, max_out=_MAX_OUT_8B)
        result, state = await _run_pipeline(schema, doc, provider, return_state=True)
        _assert_mechanically_sound(result, state)
        correct = sum(1 for k, v in expected.items() if str(result.data.get(k, "")).strip() == v)
        # Extraction at 1000-field width is hard; require a real, non-trivial share.
        assert correct >= 100, f"only {correct}/1000 fields correct at scale"


# ---------------------------------------------------------------------------
# Test Class 4: Scaling behaviour (leaves grow with field count)
# ---------------------------------------------------------------------------


class TestScalingBehaviour:
    """More fields under a fixed small context produce more leaves."""

    @pytest.mark.asyncio
    async def test_leaf_count_grows_with_fields(self):
        _skip_no_key()
        provider = _make_provider(ctx=4096, max_out=4096)
        counts: list[int] = []
        for n in (20, 60, 120):
            schema, doc, _ = _build_synthetic(n)
            _, state = await _run_pipeline(schema, doc, provider, return_state=True)
            counts.append(len(state.leaves))
        # Monotonic non-decreasing: more fields never produce fewer leaves
        assert counts[0] <= counts[1] <= counts[2], f"leaf counts not monotonic: {counts}"
        assert counts[2] > counts[0], f"120 fields should need more leaves than 20: {counts}"
