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
from nfield.pipeline.s6_assemble import _fold_open_maps, run_stage_6
from nfield.schema._flatten import flatten_schema
from nfield.types import ExtractionResult, ExtractionStatus


class TestFoldOpenMaps:
    def _map_fields(self):
        schema = {
            "type": "object",
            "properties": {"cfg": {"type": "object", "additionalProperties": {"type": "number"}}},
        }
        return flatten_schema(schema)

    def test_key_value_list_folds_to_dict(self):
        fields = self._map_fields()
        filled = {"cfg": [{"key": "timeout", "value": 30}, {"key": "retries", "value": 5}]}
        assert _fold_open_maps(filled, fields) == {"cfg": {"timeout": 30, "retries": 5}}

    def test_non_map_values_untouched(self):
        fields = self._map_fields()
        filled = {"cfg": [{"key": "a", "value": 1}], "other": "x"}
        assert _fold_open_maps(filled, fields)["other"] == "x"

    def test_no_open_map_fields_is_noop(self):
        from nfield.schema._types import Field

        fields = [Field("name", "string", {}, "", {})]
        filled = {"name": "Alice"}
        assert _fold_open_maps(filled, fields) is filled

    def test_non_string_key_row_is_dropped_not_crash(self):
        # A garbled emission can carry a non-string (unhashable) key; the row must
        # be dropped, never raise TypeError during folding.
        fields = self._map_fields()
        filled = {"cfg": [{"key": ["oops"], "value": 1}, {"key": "ok", "value": 2}]}
        assert _fold_open_maps(filled, fields) == {"cfg": {"ok": 2}}


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


class TestResolveStructuralUnions:
    """The array|object union collapses to whichever branch the document filled."""

    def _skills_fields(self):
        schema = {
            "type": "object",
            "properties": {
                "skills": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {
                            "type": "object",
                            "additionalProperties": {"type": "array", "items": {"type": "string"}},
                        },
                        {"type": "null"},
                    ]
                }
            },
        }
        return flatten_schema(schema)

    def test_object_branch_wins_when_grouped(self):
        from nfield.pipeline.s6_assemble import _resolve_structural_unions

        fields = self._skills_fields()
        filled = {"skills": [{"key": "Backend", "value": ["Python"]}], "skills__uarr": []}
        out = _resolve_structural_unions(filled, fields)
        assert "skills__uarr" not in out
        assert out["skills"] == [{"key": "Backend", "value": ["Python"]}]

    def test_array_branch_wins_when_flat(self):
        from nfield.pipeline.s6_assemble import _resolve_structural_unions

        fields = self._skills_fields()
        filled = {"skills": [], "skills__uarr": ["Python", "SQL"]}
        out = _resolve_structural_unions(filled, fields)
        assert "skills__uarr" not in out
        assert out["skills"] == ["Python", "SQL"]

    def test_array_winner_survives_open_map_fold(self):
        # After the flat branch wins, the base carries a plain list; fold must not
        # mangle it into an empty dict.
        from nfield.pipeline.s6_assemble import _resolve_structural_unions

        fields = self._skills_fields()
        filled = _resolve_structural_unions({"skills": [], "skills__uarr": ["Python"]}, fields)
        folded = _fold_open_maps(filled, fields)
        assert folded["skills"] == ["Python"]

    def test_shadow_dropped_when_both_empty(self):
        from nfield.pipeline.s6_assemble import _resolve_structural_unions

        fields = self._skills_fields()
        out = _resolve_structural_unions({"skills": [], "skills__uarr": []}, fields)
        assert "skills__uarr" not in out
