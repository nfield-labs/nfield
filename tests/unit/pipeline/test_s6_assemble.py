"""Tests for Stage 6: Assembly."""

from __future__ import annotations

import pytest

from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.pipeline.s2b_prepass import run_stage_2b
from nfield.pipeline.s2c_packing import run_stage_2c
from nfield.pipeline.s3_excerpt import run_stage_3
from nfield.pipeline.s4_extract import run_stage_4
from nfield.pipeline.s5_validate import run_stage_5
from nfield.pipeline.s6_assemble import run_stage_6
from nfield.types import ExtractionResult, ExtractionStatus

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "score": {"type": "integer"},
    },
}

FULL_RESPONSE = "name = Alice\nscore = 99\n"
PARTIAL_RESPONSE = "name = Alice\n"
EMPTY_RESPONSE = ""


class MockProvider:
    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock"

    def __init__(self, response: str):
        self.response = response

    async def complete(self, messages, *, max_tokens):
        return self.response


async def _run_pipeline(response: str) -> ExtractionResult:
    config = ExtractionConfig()
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state = run_stage_1(state, SCHEMA)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Alice scored 99 points.", config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    provider = MockProvider(response)
    state = await run_stage_4(state, provider)
    state = await run_stage_5(state, provider, config)
    return run_stage_6(state)


class TestRunStage6:
    @pytest.mark.asyncio
    async def test_returns_extraction_result(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert isinstance(result, ExtractionResult)

    @pytest.mark.asyncio
    async def test_full_extraction_success_status(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert result.status == ExtractionStatus.SUCCESS

    @pytest.mark.asyncio
    async def test_partial_extraction_partial_status(self):
        result = await _run_pipeline(PARTIAL_RESPONSE)
        # score is missing → PARTIAL
        assert result.status in (ExtractionStatus.PARTIAL, ExtractionStatus.FAILED)

    @pytest.mark.asyncio
    async def test_data_contains_extracted_fields(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert result.data.get("name") == "Alice"
        assert result.data.get("score") == 99

    @pytest.mark.asyncio
    async def test_metadata_fields_total_correct(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert result.metadata.fields_total == 2

    @pytest.mark.asyncio
    async def test_metadata_k_min_set(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert result.metadata.K_min >= 1

    @pytest.mark.asyncio
    async def test_quality_score_in_range(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert 0.0 <= result.metadata.quality_score <= 1.0

    @pytest.mark.asyncio
    async def test_confidence_level_string(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert result.metadata.confidence_level in ("HIGH", "MEDIUM", "LOW")

    @pytest.mark.asyncio
    async def test_empty_response_failed_or_partial(self):
        result = await _run_pipeline(EMPTY_RESPONSE)
        assert result.status in (ExtractionStatus.FAILED, ExtractionStatus.PARTIAL)

    @pytest.mark.asyncio
    async def test_per_field_confidence_present(self):
        result = await _run_pipeline(FULL_RESPONSE)
        assert isinstance(result.metadata.per_field_confidence, dict)
