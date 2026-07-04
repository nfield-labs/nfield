"""Tests for Stage 4: Extraction (mock provider)."""

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

SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer"},
        "active": {"type": "boolean"},
    },
}

SFEP_RESPONSE = "company = Acme Corp\nyear = 1947\nactive = true\n"


class MockProvider:
    """Provider that returns a fixed SFEP response."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, response: str = SFEP_RESPONSE):
        self.response = response
        self.calls: list[str] = []

    async def complete(self, messages, *, max_tokens):
        self.calls.append(self.response)
        return self.response


def _build_state() -> PipelineState:
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    config = ExtractionConfig()
    state = run_stage_1(state, SCHEMA)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Acme Corp founded in 1947. Publicly traded.", config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    return state


class TestRunStage4:
    @pytest.mark.asyncio
    async def test_k_incremented(self):
        state = _build_state()
        provider = MockProvider()
        state = await run_stage_4(state, provider)
        assert state.K >= 1

    @pytest.mark.asyncio
    async def test_company_extracted(self):
        state = _build_state()
        provider = MockProvider()
        state = await run_stage_4(state, provider)
        filled = state.blackboard.get_filled()
        assert "company" in filled
        assert filled["company"] == "Acme Corp"

    @pytest.mark.asyncio
    async def test_integer_field_extracted(self):
        state = _build_state()
        provider = MockProvider()
        state = await run_stage_4(state, provider)
        filled = state.blackboard.get_filled()
        assert "year" in filled
        assert filled["year"] == 1947

    @pytest.mark.asyncio
    async def test_boolean_field_extracted(self):
        state = _build_state()
        provider = MockProvider()
        state = await run_stage_4(state, provider)
        filled = state.blackboard.get_filled()
        assert "active" in filled
        assert filled["active"] is True

    @pytest.mark.asyncio
    async def test_null_response_marks_failed(self):
        state = _build_state()
        provider = MockProvider(response="company = NULL\nyear = NULL\nactive = NULL\n")
        state = await run_stage_4(state, provider)
        assert len(state.blackboard.get_failed()) >= 1

    @pytest.mark.asyncio
    async def test_provider_error_marks_failed(self):
        class ErrorProvider:
            context_window = 8192
            max_output_tokens = 1024
            model_name = "mock/error"

            async def complete(self, messages, *, max_tokens):
                raise RuntimeError("provider down")

        state = _build_state()
        state = await run_stage_4(state, ErrorProvider())
        failed = state.blackboard.get_failed()
        assert len(failed) > 0

    @pytest.mark.asyncio
    async def test_empty_response_leaves_fields_pending_or_empty(self):
        state = _build_state()
        provider = MockProvider(response="")
        state = await run_stage_4(state, provider)
        # K still incremented (the call happened)
        assert state.K >= 1


class TestEmergencySplit:
    """Stage 4 splits an oversized leaf when the provider reports overflow."""

    class _ContextErrorThenOkProvider:
        """Raises a context-length error on the first call, succeeds after."""

        context_window = 8192
        max_output_tokens = 8192
        model_name = "mock/split"

        def __init__(self) -> None:
            self.call_count = 0

        async def complete(self, messages, *, max_tokens):
            self.call_count += 1
            if self.call_count == 1:
                # Groq's actual overflow message
                raise RuntimeError(
                    "Error code: 400 - Please reduce the length of the messages or completion."
                )
            return "company = Acme Corp\nyear = 1947\nactive = true\n"

    @pytest.mark.asyncio
    async def test_overflow_triggers_split_and_recovers(self):
        """A context-overflow error splits the leaf and re-extracts each half."""
        state = _build_state()
        provider = self._ContextErrorThenOkProvider()
        state = await run_stage_4(state, provider)
        # 1 failed full-leaf call + 2 successful half-leaf calls
        assert provider.call_count == 3
        filled = state.blackboard.get_filled()
        assert "company" in filled

    @pytest.mark.asyncio
    async def test_split_increments_k_per_half(self):
        """Each recovered half counts as one API call in K."""
        state = _build_state()
        provider = self._ContextErrorThenOkProvider()
        state = await run_stage_4(state, provider)
        # Both halves succeeded → K counts the two successful calls
        assert state.K >= 2

    @pytest.mark.asyncio
    async def test_non_context_error_does_not_split(self):
        """A non-overflow error marks fields failed without splitting."""

        class _PlainErrorProvider:
            context_window = 8192
            max_output_tokens = 8192
            model_name = "mock/err"

            def __init__(self) -> None:
                self.call_count = 0

            async def complete(self, messages, *, max_tokens):
                self.call_count += 1
                raise RuntimeError("rate limited")

        state = _build_state()
        provider = _PlainErrorProvider()
        state = await run_stage_4(state, provider)
        # Only the single full-leaf call was made - no split retries
        assert provider.call_count == 1
        assert len(state.blackboard.get_failed()) >= 1


class TestCharsPerTokenCalibration:
    """Stage 4 tightens chars_per_token from the model's real usage report."""

    def test_denser_reading_shrinks_estimate(self):
        from nfield.pipeline._state import PipelineState
        from nfield.pipeline.s4_extract import _calibrate_chars_per_token

        state = PipelineState(chars_per_token=4.0, C_eff=131000, M_O=24000, C_usable=65500.0)
        messages = [{"role": "user", "content": "x" * 2000}]

        class _P:
            last_prompt_tokens = 1000  # 2000 chars / 1000 tokens = 2.0 cpt

        _calibrate_chars_per_token(state, messages, _P())
        assert state.chars_per_token == 2.0

    def test_sparser_reading_never_loosens(self):
        from nfield.pipeline._state import PipelineState
        from nfield.pipeline.s4_extract import _calibrate_chars_per_token

        state = PipelineState(chars_per_token=2.0, C_eff=131000, M_O=24000, C_usable=65500.0)
        messages = [{"role": "user", "content": "x" * 4000}]

        class _P:
            last_prompt_tokens = 1000  # 4.0 cpt - looser; must be ignored

        _calibrate_chars_per_token(state, messages, _P())
        assert state.chars_per_token == 2.0

    def test_missing_usage_is_noop(self):
        from nfield.pipeline._state import PipelineState
        from nfield.pipeline.s4_extract import _calibrate_chars_per_token

        state = PipelineState(chars_per_token=4.0, C_eff=131000, M_O=24000, C_usable=65500.0)

        class _P:
            last_prompt_tokens = None

        _calibrate_chars_per_token(state, [{"role": "user", "content": "abc"}], _P())
        assert state.chars_per_token == 4.0
