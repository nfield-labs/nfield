"""Tests for Stage 0: Resource Calibration."""

from __future__ import annotations

import pytest

from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s0_resources import run_stage_0


class MockProvider:
    """Minimal mock provider for Stage 0 tests."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):
        return ""

    async def count_tokens(self, text: str) -> int:
        # Simple approximation: 1 token per 4 chars
        return max(1, len(text) // 4)


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def default_config():
    return ExtractionConfig()


class TestRunStage0:
    @pytest.mark.asyncio
    async def test_returns_pipeline_state(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        assert isinstance(state, PipelineState)

    @pytest.mark.asyncio
    async def test_chars_per_token_positive(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        assert state.chars_per_token > 0.0

    @pytest.mark.asyncio
    async def test_context_window_set(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        assert state.C_eff == mock_provider.context_window

    @pytest.mark.asyncio
    async def test_max_output_set(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        assert mock_provider.max_output_tokens == state.M_O

    @pytest.mark.asyncio
    async def test_c_usable_respects_ratio(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        expected = mock_provider.context_window * default_config.context_utilization_ratio
        assert abs(state.C_usable - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_custom_utilization_ratio(self, mock_provider):
        config = ExtractionConfig(context_utilization_ratio=0.4)
        state = await run_stage_0(mock_provider, config)
        expected = mock_provider.context_window * 0.4
        assert abs(state.C_usable - expected) < 1e-6

    @pytest.mark.asyncio
    async def test_blackboard_not_set_yet(self, mock_provider, default_config):
        # Blackboard is only initialised in Stage 1
        state = await run_stage_0(mock_provider, default_config)
        assert state.blackboard is None

    @pytest.mark.asyncio
    async def test_fields_empty(self, mock_provider, default_config):
        state = await run_stage_0(mock_provider, default_config)
        assert state.fields == []


# ---------------------------------------------------------------------------
# document_language -> calibration bucket wiring
# ---------------------------------------------------------------------------
import nfield.pipeline.s0_resources as _s0  # noqa: E402
from nfield.pipeline.s0_resources import _calibration_bucket  # noqa: E402


class TestCalibrationLanguage:
    def test_bucket_mapping(self):
        assert _calibration_bucket("en") == "en"
        assert _calibration_bucket("en-US") == "en"
        assert _calibration_bucket("ja") == "cjk"
        assert _calibration_bucket("zh-Hans") == "cjk"
        assert _calibration_bucket("ko") == "cjk"
        assert _calibration_bucket("fr") == "mixed"
        assert _calibration_bucket("cjk") == "cjk"
        assert _calibration_bucket("mixed") == "mixed"

    async def test_document_language_drives_calibration(self, mock_provider, monkeypatch):
        captured: dict = {}

        async def fake_measure(provider, *, language):
            captured["language"] = language
            return 4.0

        monkeypatch.setattr(_s0, "measure_chars_per_token", fake_measure)
        await run_stage_0(mock_provider, ExtractionConfig(document_language="ja"))
        assert captured["language"] == "cjk"  # was hardcoded "en" before wiring
