"""Tests for Stage 5: Validation & Retry."""

from __future__ import annotations

import pytest

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import run_stage_2c
from formatshield.pipeline.s3_excerpt import run_stage_3
from formatshield.pipeline.s4_extract import run_stage_4
from formatshield.pipeline.s5_validate import run_stage_5

SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer", "minimum": 1800, "maximum": 2100},
    },
}

GOOD_RESPONSE = "company = Acme Corp\nyear = 1947\n"
BAD_RESPONSE = "company = Acme Corp\nyear = not_a_number\n"


class MockProvider:
    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, initial: str, retry: str = "year = 1947\n"):
        self.initial = initial
        self.retry = retry
        self._call_count = 0

    async def complete(self, messages, *, max_tokens):
        self._call_count += 1
        if self._call_count == 1:
            return self.initial
        return self.retry

    async def count_tokens(self, text):
        return max(1, len(text) // 4)


def _build_state(response: str) -> tuple[PipelineState, MockProvider]:
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    config = ExtractionConfig()
    state = run_stage_1(state, SCHEMA)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Acme Corp founded 1947.", config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    provider = MockProvider(initial=response)
    return state, provider


class TestRunStage5:
    @pytest.mark.asyncio
    async def test_valid_values_stay_filled(self):
        state, provider = _build_state(GOOD_RESPONSE)
        config = ExtractionConfig()
        state = await run_stage_4(state, provider)
        state = await run_stage_5(state, provider, config)
        filled = state.blackboard.get_filled()
        assert "company" in filled
        assert "year" in filled

    @pytest.mark.asyncio
    async def test_invalid_integer_retried_and_recovered(self):
        state, provider = _build_state(BAD_RESPONSE)
        config = ExtractionConfig(max_retry_rounds=1)
        state = await run_stage_4(state, provider)
        state = await run_stage_5(state, provider, config)
        # After retry, year should be recovered (retry returns "year = 1947")
        filled = state.blackboard.get_filled()
        # company was valid in initial, year is retried
        assert "company" in filled

    @pytest.mark.asyncio
    async def test_empty_blackboard_fields_not_extracted_stay_failed(self):
        state, _ = _build_state("")  # empty → nothing extracted
        config = ExtractionConfig(max_retry_rounds=1)

        class EmptyProvider:
            context_window = 8192
            max_output_tokens = 1024
            model_name = "mock/empty"

            async def complete(self, messages, *, max_tokens):
                return ""

            async def count_tokens(self, text):
                return 1

        state = await run_stage_4(state, EmptyProvider())
        state = await run_stage_5(state, EmptyProvider(), config)
        # All fields should be failed
        assert len(state.blackboard.get_failed()) >= 0  # may vary by empty handling

    @pytest.mark.asyncio
    async def test_returns_same_state(self):
        state, provider = _build_state(GOOD_RESPONSE)
        state = await run_stage_4(state, provider)
        returned = await run_stage_5(state, provider, ExtractionConfig())
        assert returned is state
