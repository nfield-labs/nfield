"""Deterministic full-pipeline (S0-S6) validation at 1000 fields.

Exercises every stage end-to-end with a mock provider so the whole pipeline is
verified line-by-line at extreme width with exact assertions and zero tokens:

* S0 calibration → S1 flatten (1000 fields) → S2A grouping → S2.5 pre-pass →
  S2C packing+splitting → S3 excerpt → S4 extraction → S5 validation → S6 assembly.

The mock provider echoes a full SFEP response covering all 1000 fields; each leaf
keeps only the fields it owns, so a correct pipeline reassembles all 1000 values.
"""

from __future__ import annotations

import pytest

from nfield.config import ExtractionConfig
from nfield.pipeline.s0_resources import run_stage_0
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.pipeline.s2b_prepass import run_stage_2b
from nfield.pipeline.s2c_packing import run_stage_2c
from nfield.pipeline.s3_excerpt import run_stage_3
from nfield.pipeline.s4_extract import run_stage_4
from nfield.pipeline.s5_validate import run_stage_5
from nfield.pipeline.s6_assemble import run_stage_6
from nfield.types import ExtractionStatus

_N = 1000


def _schema(n: int) -> dict:
    return {
        "type": "object",
        "properties": {
            f"field_{i:04d}": {"type": "string", "description": f"synthetic field {i}"}
            for i in range(n)
        },
    }


def _full_sfep(n: int) -> str:
    return "\n".join(f"field_{i:04d} = value{i:04d}" for i in range(n))


class _EchoProvider:
    """Mock provider that returns the full SFEP every call (small context).

    A small context window forces Stage 2C to split the single flat group into
    many leaves, so the multi-leaf path is exercised. ``parse_sfep`` keeps only
    each leaf's own fields from the echoed response.
    """

    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/echo"

    def __init__(self, sfep_text: str) -> None:
        self._sfep = sfep_text
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        return self._sfep

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


async def _run_full(schema: dict, document: str, provider: _EchoProvider):
    config = ExtractionConfig()
    state = await run_stage_0(provider, config)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, document, config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, config)
    result = run_stage_6(state)
    return result, state


class TestThousandFieldFullPipeline:
    """All seven stages must stay correct at 1000 fields."""

    @pytest.mark.asyncio
    async def test_stage1_flattens_all_thousand(self):
        provider = _EchoProvider(_full_sfep(_N))
        config = ExtractionConfig()
        state = await run_stage_0(provider, config)
        state = run_stage_1(state, _schema(_N))
        assert len(state.fields) == _N
        assert state.blackboard is not None
        assert len(state.blackboard.all_paths()) == _N

    @pytest.mark.asyncio
    async def test_packing_splits_into_many_leaves(self):
        _, state = await _run_full(_schema(_N), "short document", _EchoProvider(_full_sfep(_N)))
        # 1000 fields cannot fit one 8K-context call → many leaves.
        assert len(state.leaves) > 1
        leaf_paths = [f.path for leaf in state.leaves for f in leaf.fields]
        assert len(leaf_paths) == _N
        assert len(set(leaf_paths)) == _N  # no field duplicated or lost

    @pytest.mark.asyncio
    async def test_execution_order_covers_every_leaf_once(self):
        _, state = await _run_full(_schema(_N), "short document", _EchoProvider(_full_sfep(_N)))
        in_order = [leaf for r in state.execution_order for leaf in r]
        assert len(in_order) == len(state.leaves)

    @pytest.mark.asyncio
    async def test_all_thousand_fields_extracted_and_assembled(self):
        result, _ = await _run_full(_schema(_N), "short document", _EchoProvider(_full_sfep(_N)))
        assert result.metadata.fields_total == _N
        assert result.metadata.fields_extracted == _N
        assert len(result.data) == _N
        assert result.status == ExtractionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_assembled_values_are_correct(self):
        result, _ = await _run_full(_schema(_N), "short document", _EchoProvider(_full_sfep(_N)))
        assert result.data["field_0000"] == "value0000"
        assert result.data["field_0500"] == "value0500"
        assert result.data["field_0999"] == "value0999"

    @pytest.mark.asyncio
    async def test_one_provider_call_per_leaf(self):
        result, state = await _run_full(
            _schema(_N), "short document", _EchoProvider(_full_sfep(_N))
        )
        # No validation failures (echo is always valid) → no retry calls.
        assert len(state.leaves) == result.metadata.K
        assert result.metadata.K_min >= 1
        assert result.metadata.K_min <= result.metadata.K

    @pytest.mark.asyncio
    async def test_no_field_left_pending_or_missing(self):
        from nfield.assembly._blackboard import FieldState

        _, state = await _run_full(_schema(_N), "short document", _EchoProvider(_full_sfep(_N)))
        bb = state.blackboard
        assert bb is not None
        states = [bb.get_state(p) for p in bb.all_paths()]
        assert all(s == FieldState.FILLED for s in states)
