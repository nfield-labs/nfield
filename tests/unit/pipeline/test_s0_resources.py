"""Tests for Stage 0: Resource Calibration."""

from __future__ import annotations

import pytest

from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s0_resources import run_stage_0


class MockProvider:
    """Minimal provider: context specs only, no token logic."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    async def complete(self, messages, *, max_tokens):
        return ""


@pytest.fixture
def mock_provider():
    return MockProvider()


@pytest.fixture
def default_config():
    return ExtractionConfig()


class TestRunStage0:
    def test_returns_pipeline_state(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        assert isinstance(state, PipelineState)

    def test_chars_per_token_positive(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        assert state.chars_per_token > 0.0

    def test_context_window_set(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        assert state.C_eff == mock_provider.context_window

    def test_max_output_set(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        assert mock_provider.max_output_tokens == state.M_O

    def test_c_usable_respects_ratio(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        expected = mock_provider.context_window * default_config.context_utilization_ratio
        assert abs(state.C_usable - expected) < 1e-6

    def test_custom_utilization_ratio(self, mock_provider):
        config = ExtractionConfig(context_utilization_ratio=0.4)
        state = run_stage_0(mock_provider, config)
        expected = mock_provider.context_window * 0.4
        assert abs(state.C_usable - expected) < 1e-6

    def test_blackboard_not_set_yet(self, mock_provider, default_config):
        # Blackboard is only initialised in Stage 1
        state = run_stage_0(mock_provider, default_config)
        assert state.blackboard is None

    def test_fields_empty(self, mock_provider, default_config):
        state = run_stage_0(mock_provider, default_config)
        assert state.fields == []


class TestCharsPerTokenSourcing:
    """chars_per_token = config override when set, else the script estimate."""

    def test_script_estimate_by_document_language(self, mock_provider):
        en = run_stage_0(mock_provider, ExtractionConfig(document_language="en"))
        cjk = run_stage_0(mock_provider, ExtractionConfig(document_language="ja"))
        other = run_stage_0(mock_provider, ExtractionConfig(document_language="fr"))
        assert en.chars_per_token == 4.0
        assert cjk.chars_per_token == 1.5
        assert other.chars_per_token == 3.0

    def test_config_override_is_used(self, mock_provider):
        # An explicit ratio pins the value, ignoring the language estimate.
        state = run_stage_0(
            mock_provider, ExtractionConfig(document_language="en", chars_per_token=2.71)
        )
        assert state.chars_per_token == 2.71

    def test_language_arg_beats_config_language(self, mock_provider):
        state = run_stage_0(mock_provider, ExtractionConfig(document_language="en"), language="ja")
        assert state.chars_per_token == 1.5
