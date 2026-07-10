"""Tests for nfield.config."""

from __future__ import annotations

import pytest

from nfield.config import (
    DEFAULT_CONTEXT_UTILIZATION_RATIO,
    DEFAULT_MAX_API_RETRIES,
    DEFAULT_MAX_RETRY_ROUNDS,
    ExtractionConfig,
)

# ---------------------------------------------------------------------------
# ExtractionConfig defaults
# ---------------------------------------------------------------------------


class TestExtractionConfigDefaults:
    def test_default_context_utilization_ratio(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.context_utilization_ratio == pytest.approx(DEFAULT_CONTEXT_UTILIZATION_RATIO)
        assert cfg.context_utilization_ratio == pytest.approx(0.50)

    def test_default_max_retry_rounds(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.max_retry_rounds == DEFAULT_MAX_RETRY_ROUNDS
        assert cfg.max_retry_rounds == 2

    def test_default_max_api_retries(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.max_api_retries == DEFAULT_MAX_API_RETRIES
        assert cfg.max_api_retries == 10  # outlasts a rolling-window TPM storm

    def test_cache_defaults_off(self) -> None:
        assert ExtractionConfig().cache is False

    def test_cache_accepts_an_instance(self) -> None:
        from nfield.providers._cache import MemoryCache

        cache = MemoryCache()
        assert ExtractionConfig(cache=cache).cache is cache

    def test_default_model_is_none(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.default_model is None

    def test_default_z_target(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.z_target == pytest.approx(1.645)

    def test_default_document_language(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.document_language == "en"

    def test_default_chars_per_token_is_none(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.chars_per_token is None

    def test_default_reasoning_model_is_false(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.reasoning_model is False

    def test_default_think_phase_budget(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.think_phase_budget == (100, 150)

    def test_default_evidence_score_threshold(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.evidence_score_threshold == pytest.approx(0.3)

    def test_default_use_advanced_sfr(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.use_advanced_sfr is False

    def test_default_confidence_thresholds(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.confidence_thresholds["HIGH"] == pytest.approx(0.9)
        assert cfg.confidence_thresholds["MEDIUM"] == pytest.approx(0.7)

    def test_frozen_cannot_reassign(self) -> None:
        cfg = ExtractionConfig()
        with pytest.raises(AttributeError):
            cfg.max_retry_rounds = 99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ExtractionConfig custom values
# ---------------------------------------------------------------------------


class TestExtractionConfigCustom:
    def test_stores_default_model(self) -> None:
        cfg = ExtractionConfig(default_model="groq/llama-3.1-8b")
        assert cfg.default_model == "groq/llama-3.1-8b"

    def test_stores_custom_ratio(self) -> None:
        cfg = ExtractionConfig(context_utilization_ratio=0.75)
        assert cfg.context_utilization_ratio == pytest.approx(0.75)

    def test_stores_custom_retry_rounds(self) -> None:
        cfg = ExtractionConfig(max_retry_rounds=5)
        assert cfg.max_retry_rounds == 5

    def test_stores_custom_language(self) -> None:
        cfg = ExtractionConfig(document_language="fr")
        assert cfg.document_language == "fr"

    def test_stores_use_advanced_sfr(self) -> None:
        cfg = ExtractionConfig(use_advanced_sfr=True)
        assert cfg.use_advanced_sfr is True

    def test_stores_chars_per_token_override(self) -> None:
        cfg = ExtractionConfig(chars_per_token=3.6)
        assert cfg.chars_per_token == pytest.approx(3.6)

    def test_stores_reasoning_model(self) -> None:
        cfg = ExtractionConfig(reasoning_model=True)
        assert cfg.reasoning_model is True


class TestExtractionConfigValidation:
    def test_chars_per_token_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="chars_per_token must be > 0"):
            ExtractionConfig(chars_per_token=0.0)

    def test_chars_per_token_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="chars_per_token must be > 0"):
            ExtractionConfig(chars_per_token=-1.0)
