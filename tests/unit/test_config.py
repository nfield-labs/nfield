"""Tests for formatshield.config."""

from __future__ import annotations

import pytest

from formatshield.config import (
    DEFAULT_CONTEXT_UTILIZATION_RATIO,
    DEFAULT_MAX_RETRY_ROUNDS,
    DomainConfig,
    ExtractionConfig,
    _builtin_domains,
    _domain_registry,
    get_domain_config,
    register_domain,
)
from formatshield.exceptions import SchemaError

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

    def test_default_model_is_none(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.default_model is None

    def test_default_z_target(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.z_target == pytest.approx(1.645)

    def test_default_document_language(self) -> None:
        cfg = ExtractionConfig()
        assert cfg.document_language == "en"

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


# ---------------------------------------------------------------------------
# DomainConfig.from_dict
# ---------------------------------------------------------------------------


class TestDomainConfigFromDict:
    def _valid_dict(self) -> dict:  # type: ignore[type-arg]
        return {
            "domain": "test",
            "p90_string_tokens": 40,
            "expected_array_size": 3,
            "confidence_thresholds": {"HIGH": 0.9, "MEDIUM": 0.7},
        }

    def test_valid_dict_succeeds(self) -> None:
        cfg = DomainConfig.from_dict(self._valid_dict())
        assert cfg.domain == "test"
        assert cfg.p90_string_tokens == 40

    def test_missing_domain_raises_schema_error(self) -> None:
        d = self._valid_dict()
        del d["domain"]
        with pytest.raises(SchemaError):
            DomainConfig.from_dict(d)

    def test_missing_p90_string_tokens_raises_schema_error(self) -> None:
        d = self._valid_dict()
        del d["p90_string_tokens"]
        with pytest.raises(SchemaError):
            DomainConfig.from_dict(d)

    def test_missing_expected_array_size_raises_schema_error(self) -> None:
        d = self._valid_dict()
        del d["expected_array_size"]
        with pytest.raises(SchemaError):
            DomainConfig.from_dict(d)

    def test_missing_confidence_thresholds_raises_schema_error(self) -> None:
        d = self._valid_dict()
        del d["confidence_thresholds"]
        with pytest.raises(SchemaError):
            DomainConfig.from_dict(d)

    def test_empty_dict_raises_schema_error(self) -> None:
        with pytest.raises(SchemaError):
            DomainConfig.from_dict({})

    def test_confidence_thresholds_copied(self) -> None:
        d = self._valid_dict()
        cfg = DomainConfig.from_dict(d)
        # Mutating the original dict must not affect the frozen config
        d["confidence_thresholds"]["HIGH"] = 0.0
        assert cfg.confidence_thresholds["HIGH"] == pytest.approx(0.9)


# ---------------------------------------------------------------------------
# register_domain / get_domain_config
# ---------------------------------------------------------------------------


class TestDomainRegistry:
    def _make_custom(self, domain: str = "custom_test_domain") -> DomainConfig:
        return DomainConfig(
            domain=domain,
            p90_string_tokens=45,
            expected_array_size=3,
            confidence_thresholds={"HIGH": 0.91, "MEDIUM": 0.72},
        )

    def test_register_then_get(self) -> None:
        cfg = self._make_custom("reg_test_domain_1")
        register_domain(cfg)
        result = get_domain_config("reg_test_domain_1")
        assert result.domain == "reg_test_domain_1"
        # cleanup
        _domain_registry.pop("reg_test_domain_1", None)

    def test_registered_domain_overrides_builtin(self) -> None:
        # Override the built-in "general" domain
        override = DomainConfig(
            domain="general",
            p90_string_tokens=999,
            expected_array_size=1,
            confidence_thresholds={"HIGH": 0.5, "MEDIUM": 0.3},
        )
        register_domain(override)
        result = get_domain_config("general")
        assert result.p90_string_tokens == 999
        # cleanup — restore original
        _domain_registry.pop("general", None)

    def test_get_domain_config_general_builtin(self) -> None:
        _domain_registry.pop("general", None)  # ensure no override
        cfg = get_domain_config("general")
        assert cfg.domain == "general"
        assert cfg.p90_string_tokens == 35

    def test_get_domain_config_medical_p90(self) -> None:
        _domain_registry.pop("medical", None)
        cfg = get_domain_config("medical")
        assert cfg.p90_string_tokens == 50

    def test_get_domain_config_unknown_raises_schema_error(self) -> None:
        from formatshield.exceptions import SchemaError

        with pytest.raises(SchemaError, match="unknown_domain_xyz"):
            get_domain_config("unknown_domain_xyz")

    def test_schema_error_message_lists_available_domains(self) -> None:
        from formatshield.exceptions import SchemaError

        with pytest.raises(SchemaError) as exc_info:
            get_domain_config("does_not_exist_abc")
        assert "general" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Built-in domain coverage
# ---------------------------------------------------------------------------


class TestBuiltinDomains:
    def test_general_available(self) -> None:
        assert "general" in _builtin_domains

    def test_medical_available(self) -> None:
        assert "medical" in _builtin_domains

    def test_legal_available(self) -> None:
        assert "legal" in _builtin_domains

    def test_financial_available(self) -> None:
        assert "financial" in _builtin_domains

    def test_all_four_domains_available_via_get(self) -> None:
        for domain in ("general", "medical", "legal", "financial"):
            _domain_registry.pop(domain, None)
            cfg = get_domain_config(domain)
            assert cfg.domain == domain
