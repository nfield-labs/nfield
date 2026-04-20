"""
TTF tests — schema-aware prompting, quality gate, prefix cache,
trace cache, routing spectrum, self-calibration, logit biasing,
self-consistency, and RCOCR protocol.

No API keys or network access required.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.routing_score import RoutingScore
from formatshield.oracle.self_calibrator import (
    DEFAULT_MIN_SAMPLES,
    DEFAULT_TARGET_ACCURACY,
    DEFAULT_WINDOW_SIZE,
    CalibrationRecord,
    SelfCalibrator,
    _fit_logistic,
    _threshold_from_logistic,
)
from formatshield.oracle.threshold_oracle import (
    PHI_MODE_THRESHOLDS,
    phi_spectrum_mode,
)
from formatshield.scorer.features import StreamEvent
from formatshield.ttf.engine import (
    _SC_PHI_THRESHOLD,
    DEFAULT_SC_K,
    TTFEngine,
    _build_schema_logit_bias,
    _phi_thinking_budget,
    _run_self_consistency_pass1,
)
from formatshield.ttf.prompts import (
    _DK_VOCAB_BRIDGE_THRESHOLD,
    _collect_schema_field_info,
    _phi_depth_label,
    _vocabulary_bridge_hints,
    build_cache_prefix_for_format_prompt,
    build_schema_phi_think_prompt,
)
from formatshield.ttf.quality_gate import (
    QUALITY_GATE_PASS_THRESHOLD,
    QualityGateResult,
    score_thinking_trace,
)
from formatshield.ttf.trace_cache import (
    DEFAULT_MAX_SIZE,
    DEFAULT_TTL_SECONDS,
    TraceCache,
    build_schema_cache_key,
)

# ---------------------------------------------------------------------------
# RCOCR standalone package (path-injected — zero-dependency standalone)
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "rcocr", "src"))
from rcocr import RCOCREngine
from rcocr import __version__ as rcocr_version
from rcocr.protocol import RCOCRBackend

# ---------------------------------------------------------------------------
# Shared schemas
# ---------------------------------------------------------------------------

SIMPLE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
    "required": ["name", "age"],
}

ENUM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pending", "active", "closed"]},
        "priority": {"type": "string", "enum": ["low", "medium", "high"]},
        "notes": {"type": "string"},
    },
    "required": ["status"],
}

NESTED_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "shipping_address": {
            "type": "object",
            "properties": {
                "street": {"type": "string"},
                "city": {"type": "string"},
            },
            "required": ["street", "city"],
        },
    },
    "required": ["order_id", "shipping_address"],
}

COMPLEX_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "order_id": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "shipped"]},
        "total": {"type": "number"},
        "customer": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["name", "email"],
        },
    },
    "required": ["order_id", "status", "total"],
}

SIMPLE_SCHEMA_WITH_DESCRIPTIONS: dict[str, Any] = {
    "type": "object",
    "description": "A user record",
    "properties": {
        "name": {"type": "string", "description": "Full name"},
        "age": {"type": "integer", "description": "Age in years"},
        "active": {"type": "boolean", "title": "Is Active"},
    },
    "required": ["name", "age"],
}

ALTERNATE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "value": {"type": "number"},
    },
    "required": ["id"],
}

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_rs(
    phi: float = 0.75,
    tau: float = 0.40,
    delta_k: float = 0.60,
    lambda2: float = 0.30,
) -> RoutingScore:
    return RoutingScore(
        phi=phi,
        lambda2=lambda2,
        tau=tau,
        delta_k=delta_k,
        explanation=f"phi={phi:.3f}",
    )


# ---------------------------------------------------------------------------
# Stream stub mixin — satisfies Backend.stream() without real streaming
# ---------------------------------------------------------------------------


class _StreamStub:
    name: str = "stub"

    async def stream(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
        seed: int | None = None,
        frequency_penalty: float | None = None,
        presence_penalty: float | None = None,
        stop: list[str] | str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        async def _gen() -> AsyncIterator[StreamEvent]:
            yield StreamEvent(type="complete", content="{}", backend=self.name, latency_ms=0.0)

        return _gen()


# ===========================================================================
# 1. Schema-Phi Think Prompt
# ===========================================================================


class TestPhiDepthLabel:
    def test_maximum_at_095(self) -> None:
        assert "MAXIMUM" in _phi_depth_label(0.95)

    def test_maximum_at_099(self) -> None:
        assert "MAXIMUM" in _phi_depth_label(0.99)

    def test_deep_at_085(self) -> None:
        assert "DEEP" in _phi_depth_label(0.85)

    def test_standard_at_070(self) -> None:
        assert "STANDARD" in _phi_depth_label(0.70)

    def test_light_at_040(self) -> None:
        assert "LIGHT" in _phi_depth_label(0.40)

    def test_light_at_zero(self) -> None:
        assert "LIGHT" in _phi_depth_label(0.0)


class TestCollectSchemaFieldInfo:
    def test_flat_schema_field_count(self) -> None:
        fields = _collect_schema_field_info(SIMPLE_SCHEMA)
        paths = [f["path"] for f in fields]
        assert "name" in paths
        assert "age" in paths
        assert "active" in paths

    def test_required_flag_set(self) -> None:
        fields = _collect_schema_field_info(SIMPLE_SCHEMA)
        by_path = {f["path"]: f for f in fields}
        assert by_path["name"]["required"] is True
        assert by_path["age"]["required"] is True

    def test_optional_flag_set(self) -> None:
        fields = _collect_schema_field_info(SIMPLE_SCHEMA)
        by_path = {f["path"]: f for f in fields}
        assert by_path["active"]["required"] is False

    def test_type_captured(self) -> None:
        fields = _collect_schema_field_info(SIMPLE_SCHEMA)
        by_path = {f["path"]: f for f in fields}
        assert by_path["age"]["type"] == "integer"
        assert by_path["active"]["type"] == "boolean"

    def test_enum_captured(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        by_path = {f["path"]: f for f in fields}
        assert by_path["status"]["enum"] == ["pending", "active", "closed"]
        assert by_path["notes"]["enum"] is None

    def test_nested_fields_included(self) -> None:
        paths = [f["path"] for f in _collect_schema_field_info(NESTED_SCHEMA)]
        assert "shipping_address" in paths
        assert "shipping_address.street" in paths
        assert "shipping_address.city" in paths

    def test_parent_before_child_order(self) -> None:
        paths = [f["path"] for f in _collect_schema_field_info(NESTED_SCHEMA)]
        assert paths.index("shipping_address") < paths.index("shipping_address.street")

    def test_empty_schema_returns_empty(self) -> None:
        assert _collect_schema_field_info({}) == []

    def test_non_dict_returns_empty(self) -> None:
        assert _collect_schema_field_info([]) == []  # type: ignore[arg-type]


class TestSchemaPhiThinkPrompt:
    def test_contains_original_prompt(self) -> None:
        p = "Extract user information."
        assert p in build_schema_phi_think_prompt(p, SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4)

    def test_contains_routing_context_header(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4
        )
        assert "Schema-Guided Reasoning Context" in result

    def test_phi_value_in_output(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4
        )
        assert "Φ=0.750" in result

    def test_tau_value_in_output(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.30, delta_k=0.4
        )
        assert "τ=0.300" in result

    def test_delta_k_value_in_output(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.42
        )
        assert "ΔK=0.420" in result

    def test_lambda2_value_in_output(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4, lambda2=0.22
        )
        assert "λ̃₂=0.220" in result

    def test_depth_label_present(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.96, tau=0.3, delta_k=0.4
        )
        assert "MAXIMUM" in result

    def test_schema_field_names_present(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4
        )
        assert "name" in result
        assert "age" in result

    def test_enum_constraints_block_present(self) -> None:
        result = build_schema_phi_think_prompt("test", ENUM_SCHEMA, phi=0.75, tau=0.8, delta_k=0.4)
        assert "Constrained fields" in result
        assert "pending" in result
        assert "closed" in result

    def test_no_json_output_instruction(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4
        )
        assert "Do NOT produce any JSON" in result

    def test_returns_non_empty_string(self) -> None:
        result = build_schema_phi_think_prompt(
            "test", SIMPLE_SCHEMA, phi=0.75, tau=0.3, delta_k=0.4
        )
        assert isinstance(result, str)
        assert len(result) > 100


# ===========================================================================
# 2. Vocabulary Bridge
# ===========================================================================


class TestVocabularyBridgeHints:
    def test_no_hints_when_field_in_prompt(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        hints = _vocabulary_bridge_hints("Check the status and priority values", fields)
        for hint in hints:
            assert "status" not in hint
            assert "priority" not in hint

    def test_hint_when_field_not_in_prompt(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        hints = _vocabulary_bridge_hints("Extract the classification from this document.", fields)
        assert len(hints) > 0

    def test_hint_contains_field_name(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        all_text = " ".join(_vocabulary_bridge_hints("Extract the classification.", fields))
        assert any(n in all_text for n in ("status", "priority", "notes"))

    def test_enum_values_in_hints(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        all_text = " ".join(_vocabulary_bridge_hints("Extract the contract phase.", fields))
        assert any(v in all_text for v in ("pending", "active", "closed", "low", "medium", "high"))

    def test_max_hints_respected(self) -> None:
        fields = _collect_schema_field_info(ENUM_SCHEMA)
        hints = _vocabulary_bridge_hints("Unrelated text.", fields, max_hints=2)
        assert len(hints) <= 2

    def test_no_hints_for_empty_fields(self) -> None:
        assert _vocabulary_bridge_hints("anything", []) == []

    def test_bridge_block_present_when_dk_high(self) -> None:
        result = build_schema_phi_think_prompt(
            "Extract the classification from the supplied document.",
            ENUM_SCHEMA,
            phi=0.75,
            tau=0.8,
            delta_k=0.80,
        )
        assert "Vocabulary bridge" in result

    def test_bridge_block_absent_when_dk_low(self) -> None:
        result = build_schema_phi_think_prompt(
            "Extract the status and priority from this document.",
            ENUM_SCHEMA,
            phi=0.75,
            tau=0.8,
            delta_k=0.30,
        )
        assert "Vocabulary bridge" not in result

    def test_threshold_constant_is_correct(self) -> None:
        assert _DK_VOCAB_BRIDGE_THRESHOLD == 0.50


# ===========================================================================
# 3. τ-Conditioned Temperature
# ===========================================================================


class _TemperatureCapturingBackend(_StreamStub):
    name = "temp_capture"
    last_temperature: float | None = None
    call_count: int = 0

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return 0.20

    @property
    def supports_logit_bias(self) -> bool:
        return False

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> str:
        self.last_temperature = temperature
        self.call_count += 1
        if constraints == "json":
            return json.dumps({"result": "mock"})
        return "<think>reasoning</think>"


class TestTauConditionedTemperature:
    async def test_no_routing_score_temperature_is_none(self) -> None:
        backend = _TemperatureCapturingBackend()
        engine = TTFEngine(backend=backend)
        await engine.generate("test", schema=SIMPLE_SCHEMA, routing_score=None)
        assert backend.last_temperature is None

    async def test_high_tau_gives_low_temperature(self) -> None:
        backend = _TemperatureCapturingBackend()
        await TTFEngine(backend=backend).generate(
            "test", schema=ENUM_SCHEMA, routing_score=_make_rs(phi=0.80, tau=0.90)
        )
        assert backend.last_temperature is not None
        assert backend.last_temperature < 0.15

    async def test_low_tau_gives_higher_temperature(self) -> None:
        backend = _TemperatureCapturingBackend()
        await TTFEngine(backend=backend).generate(
            "test", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.70, tau=0.10)
        )
        assert backend.last_temperature is not None
        assert backend.last_temperature > 0.50

    async def test_zero_tau_temperature_formula(self) -> None:
        backend = _TemperatureCapturingBackend()
        await TTFEngine(backend=backend).generate(
            "test", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.70, tau=0.0)
        )
        assert backend.last_temperature == pytest.approx(0.70, abs=1e-6)

    async def test_full_tau_temperature_floor(self) -> None:
        backend = _TemperatureCapturingBackend()
        await TTFEngine(backend=backend).generate(
            "test", schema=ENUM_SCHEMA, routing_score=_make_rs(phi=0.90, tau=1.0)
        )
        assert backend.last_temperature == pytest.approx(0.05, abs=1e-6)

    def test_temperature_floor_never_below_005(self) -> None:
        for tau in (0.9, 0.95, 1.0):
            assert max(0.05, 0.7 * (1.0 - tau)) >= 0.05


# ===========================================================================
# 4. Φ-Proportional Thinking Budget
# ===========================================================================


class TestThinkingBudget:
    def test_very_low_phi_returns_256(self) -> None:
        assert _phi_thinking_budget(0.0) == 256

    def test_low_phi_boundary_returns_512(self) -> None:
        assert _phi_thinking_budget(0.65) == 512

    def test_mid_phi_boundary_returns_1024(self) -> None:
        assert _phi_thinking_budget(0.75) == 1024

    def test_high_phi_boundary_returns_4096(self) -> None:
        assert _phi_thinking_budget(0.90) == 4096

    def test_max_phi_returns_4096(self) -> None:
        assert _phi_thinking_budget(1.0) == 4096

    def test_negative_phi_clamped_to_256(self) -> None:
        assert _phi_thinking_budget(-0.1) == 256

    def test_budget_is_integer(self) -> None:
        for phi in [0.0, 0.5, 0.7, 0.9]:
            assert isinstance(_phi_thinking_budget(phi), int)

    def test_budget_increases_with_phi(self) -> None:
        budgets = [_phi_thinking_budget(p) for p in [0.1, 0.65, 0.75, 0.90]]
        assert budgets == sorted(budgets)


class TestThinkingBudgetWiredInEngine:
    @pytest.mark.asyncio
    async def test_pass1_receives_max_tokens_from_phi(self) -> None:
        captured: dict[str, Any] = {}

        class CapturingBackend(_StreamStub):
            name = "capturing"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["max_tokens"] = max_tokens
                    return "<think>name age fields.</think> done"
                return json.dumps({"name": "Alice", "age": 30})

        await TTFEngine(backend=CapturingBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.95)
        )
        assert captured.get("max_tokens") == 4096

    @pytest.mark.asyncio
    async def test_low_phi_gives_small_budget(self) -> None:
        captured: dict[str, Any] = {}

        class CapturingBackend(_StreamStub):
            name = "capturing"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["max_tokens"] = max_tokens
                    return "<think>fields.</think> done"
                return json.dumps({"name": "Bob", "age": 25})

        await TTFEngine(backend=CapturingBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.1)
        )
        assert captured.get("max_tokens") == 256

    @pytest.mark.asyncio
    async def test_no_routing_score_omits_max_tokens(self) -> None:
        captured: dict[str, Any] = {}

        class CapturingBackend(_StreamStub):
            name = "capturing"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["max_tokens"] = max_tokens
                    return "<think>thinking.</think> done"
                return json.dumps({"name": "X", "age": 1})

        await TTFEngine(backend=CapturingBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=None
        )
        assert captured.get("max_tokens") is None


# ===========================================================================
# 5. Pass 1 Quality Gate
# ===========================================================================


class TestQualityGateResult:
    def test_passed_true_when_above_threshold(self) -> None:
        r = QualityGateResult(passed=True, score=0.67, failed_checks=[], details={})
        assert r.passed is True

    def test_passed_false_when_below_threshold(self) -> None:
        r = QualityGateResult(
            passed=False, score=0.33, failed_checks=["required_field_coverage"], details={}
        )
        assert r.passed is False

    def test_frozen_dataclass_immutable(self) -> None:
        r = QualityGateResult(passed=True, score=1.0, failed_checks=[], details={})
        with pytest.raises((AttributeError, TypeError)):
            r.passed = False  # type: ignore[misc]


class TestScoreThinkingTrace:
    def test_empty_thinking_fails(self) -> None:
        result = score_thinking_trace("", SIMPLE_SCHEMA)
        assert result.passed is False
        assert result.score == 0.0
        assert "empty_thinking_trace" in result.failed_checks

    def test_whitespace_only_fails(self) -> None:
        assert score_thinking_trace("   \n\t  ", SIMPLE_SCHEMA).passed is False

    def test_good_thinking_passes(self) -> None:
        thinking = (
            "I need to populate the 'name' field with a string value. "
            "The 'age' field must be an integer. "
            "Let me also consider the active boolean. "
            "No contradictions here, everything is consistent."
        )
        result = score_thinking_trace(thinking, SIMPLE_SCHEMA)
        assert result.passed is True
        assert result.score >= QUALITY_GATE_PASS_THRESHOLD

    def test_missing_required_field_lowers_score(self) -> None:
        result = score_thinking_trace(
            "I'll fill in the age field with an integer value.", SIMPLE_SCHEMA
        )
        assert "required_field_coverage" in result.failed_checks

    def test_contradiction_details_present(self) -> None:
        thinking = "The name should be 'Alice'. Wait, no. Let me reconsider. The age is 30."
        result = score_thinking_trace(thinking, SIMPLE_SCHEMA)
        assert "contradiction_free" in result.details

    def test_multiple_contradictions_fail(self) -> None:
        thinking = (
            "The order_id should be X. Wait, no. "
            "Actually that's wrong. I made an error here too. That was incorrect."
        )
        assert "contradiction_free" in score_thinking_trace(thinking, COMPLEX_SCHEMA).failed_checks

    def test_no_schema_passes_trivially(self) -> None:
        assert score_thinking_trace("The answer is clearly 42.", schema=None).passed is True

    def test_threshold_constant_value(self) -> None:
        assert QUALITY_GATE_PASS_THRESHOLD == pytest.approx(0.67)

    def test_score_in_unit_range(self) -> None:
        result = score_thinking_trace("name age covered, no contradictions.", SIMPLE_SCHEMA)
        assert 0.0 <= result.score <= 1.0

    def test_details_dict_has_all_check_keys(self) -> None:
        result = score_thinking_trace("name age covered.", SIMPLE_SCHEMA)
        for key in ("required_field_coverage", "contradiction_free", "vocab_bridge_coverage"):
            assert key in result.details

    def test_vocab_bridge_inactive_when_dk_low(self) -> None:
        rs = _make_rs(delta_k=0.3)
        result = score_thinking_trace("name age", SIMPLE_SCHEMA, routing_score=rs)
        assert result.details["vocab_bridge_coverage"].get("dk_check_active") is False

    def test_vocab_bridge_active_when_dk_high(self) -> None:
        rs = _make_rs(delta_k=0.8)
        thinking = "name age order_id status total customer email"
        result = score_thinking_trace(thinking, COMPLEX_SCHEMA, routing_score=rs)
        assert result.details["vocab_bridge_coverage"].get("dk_check_active") is True


class TestQualityGateWiredInEngine:
    @pytest.mark.asyncio
    async def test_retries_on_bad_thinking(self) -> None:
        call_count = {"n": 0}

        class RetryBackend(_StreamStub):
            name = "retry"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    call_count["n"] += 1
                    if call_count["n"] == 1:
                        return "No useful thinking here."
                    return (
                        "<think>name is required. age must be integer. No contradictions.</think>"
                    )
                return json.dumps({"name": "Alice", "age": 30})

        await TTFEngine(backend=RetryBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.5, delta_k=0.2)
        )
        assert call_count["n"] == 2

    @pytest.mark.asyncio
    async def test_no_retry_on_good_thinking(self) -> None:
        call_count = {"n": 0}

        class CountingBackend(_StreamStub):
            name = "counting"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    call_count["n"] += 1
                    return (
                        "<think>name field is a string. age must be integer."
                        " All required fields covered.</think>"
                    )
                return json.dumps({"name": "Bob", "age": 25})

        await TTFEngine(backend=CountingBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.5)
        )
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_gate_skipped_without_routing_score(self) -> None:
        call_count = {"n": 0}

        class CountingBackend(_StreamStub):
            name = "counting"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    call_count["n"] += 1
                    return "Weak thinking, no field names."
                return json.dumps({"name": "X", "age": 1})

        await TTFEngine(backend=CountingBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=None
        )
        assert call_count["n"] == 1

    @pytest.mark.asyncio
    async def test_continues_after_double_failure(self) -> None:
        class AlwaysWeakBackend(_StreamStub):
            name = "weak"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    return "weak thinking no fields mentioned at all xyz"
                return json.dumps({"name": "fallback", "age": 0})

        _thinking, output = await TTFEngine(backend=AlwaysWeakBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.5)
        )
        assert output


# ===========================================================================
# 6. Prefix Cache
# ===========================================================================


class TestPrefixCachePromptBuilder:
    def test_returns_string(self) -> None:
        assert isinstance(build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA), str)

    def test_contains_json_hint(self) -> None:
        result = build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA)
        assert "JSON" in result or "json" in result.lower()

    def test_contains_schema_field_names(self) -> None:
        result = build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA)
        assert any(f in result for f in ["name", "age", "active"])

    def test_non_empty(self) -> None:
        assert len(build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA).strip()) > 20

    def test_different_schemas_produce_different_prefixes(self) -> None:
        assert build_cache_prefix_for_format_prompt(
            SIMPLE_SCHEMA
        ) != build_cache_prefix_for_format_prompt(COMPLEX_SCHEMA)

    def test_json_constraint_accepted(self) -> None:
        result = build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA, constraints="json")
        assert isinstance(result, str)
        assert len(result) > 0


class TestPrefixCacheWiredInEngine:
    @pytest.mark.asyncio
    async def test_pass2_receives_schema_cache_prefix(self) -> None:
        captured: dict[str, Any] = {}

        class KVBackend(_StreamStub):
            name = "kv"
            supports_kv_cache_reuse = True
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints == "json":
                    captured["kv_prefix"] = kv_cache_prefix
                    return json.dumps({"name": "Alice", "age": 30})
                return "<think>name field is a string. age field is integer.</think>"

        await TTFEngine(backend=KVBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs()
        )
        assert captured.get("kv_prefix") == build_cache_prefix_for_format_prompt(SIMPLE_SCHEMA)

    @pytest.mark.asyncio
    async def test_no_prefix_when_backend_does_not_support_it(self) -> None:
        captured: dict[str, Any] = {}

        class NoKVBackend(_StreamStub):
            name = "no_kv"
            supports_kv_cache_reuse = False
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints == "json":
                    captured["kv_prefix"] = kv_cache_prefix
                    return json.dumps({"name": "Bob", "age": 25})
                return "<think>name age fields.</think>"

        await TTFEngine(backend=NoKVBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs()
        )
        assert captured.get("kv_prefix") is None

    @pytest.mark.asyncio
    async def test_explicit_prefix_takes_precedence(self) -> None:
        captured: dict[str, Any] = {}
        explicit = "CUSTOM_STATIC_PREFIX"

        class KVBackend(_StreamStub):
            name = "kv"
            supports_kv_cache_reuse = True
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints == "json":
                    captured["kv_prefix"] = kv_cache_prefix
                    return json.dumps({"name": "X", "age": 1})
                return "<think>name age covered.</think>"

        await TTFEngine(backend=KVBackend()).generate(
            "Extract.", schema=SIMPLE_SCHEMA, kv_cache_prefix=explicit
        )
        assert captured.get("kv_prefix") == explicit

    @pytest.mark.asyncio
    async def test_no_schema_falls_back_to_think_prompt(self) -> None:
        captured: dict[str, Any] = {}

        class KVBackend(_StreamStub):
            name = "kv"
            supports_kv_cache_reuse = True
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints == "json":
                    captured["kv_prefix"] = kv_cache_prefix
                    return json.dumps({"answer": "42"})
                return "<think>General reasoning.</think>"

        await TTFEngine(backend=KVBackend()).generate("What is the answer?", schema=None)
        assert captured.get("kv_prefix") is not None


# ===========================================================================
# 7. Quality & Cost Integration
# ===========================================================================


class TestQualityCostIntegration:
    @pytest.mark.asyncio
    async def test_thinking_budget_prefix_cache_temperature_cooperate(self) -> None:
        signals: dict[str, Any] = {}

        class FullCaptureBackend(_StreamStub):
            name = "full"
            supports_kv_cache_reuse = True
            accuracy_loss_baseline = 0.1
            supports_logit_bias = False

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    signals["max_tokens"] = max_tokens
                    return (
                        "<think>order_id is required. status should be pending or shipped. "
                        "total is a number. customer contains name and email.</think>"
                    )
                signals["temperature"] = temperature
                signals["kv_prefix"] = kv_cache_prefix
                return json.dumps(
                    {
                        "order_id": "ORD-001",
                        "status": "pending",
                        "total": 99.99,
                        "customer": {"name": "Alice", "email": "a@b.com"},
                    }
                )

        rs = RoutingScore(phi=0.95, lambda2=0.6, tau=0.9, delta_k=0.6, explanation="test")
        _thinking, output = await TTFEngine(backend=FullCaptureBackend()).generate(
            "Create an order for Alice.", schema=COMPLEX_SCHEMA, routing_score=rs
        )
        assert signals.get("max_tokens") == 4096
        assert signals.get("temperature") == pytest.approx(0.07, abs=0.01)
        assert signals.get("kv_prefix") == build_cache_prefix_for_format_prompt(COMPLEX_SCHEMA)
        assert json.loads(output)["order_id"] == "ORD-001"


# ===========================================================================
# 8. Schema Cache Key
# ===========================================================================


class TestSchemaCacheKey:
    def test_returns_12_char_hex(self) -> None:
        key = build_schema_cache_key(SIMPLE_SCHEMA)
        assert len(key) == 12
        assert all(c in "0123456789abcdef" for c in key)

    def test_same_schema_same_key(self) -> None:
        assert build_schema_cache_key(SIMPLE_SCHEMA) == build_schema_cache_key(SIMPLE_SCHEMA)

    def test_different_structures_different_keys(self) -> None:
        assert build_schema_cache_key(SIMPLE_SCHEMA) != build_schema_cache_key(ALTERNATE_SCHEMA)

    def test_descriptions_excluded_from_key(self) -> None:
        assert build_schema_cache_key(SIMPLE_SCHEMA) == build_schema_cache_key(
            SIMPLE_SCHEMA_WITH_DESCRIPTIONS
        )

    def test_enum_values_excluded_from_key(self) -> None:
        a = {
            "type": "object",
            "properties": {"s": {"type": "string", "enum": ["a", "b"]}},
            "required": ["s"],
        }
        b = {
            "type": "object",
            "properties": {"s": {"type": "string", "enum": ["x", "y", "z"]}},
            "required": ["s"],
        }
        assert build_schema_cache_key(a) == build_schema_cache_key(b)

    def test_empty_schema_returns_key(self) -> None:
        key = build_schema_cache_key({})
        assert isinstance(key, str)
        assert len(key) == 12

    def test_required_difference_changes_key(self) -> None:
        a = {
            "type": "object",
            "properties": {"n": {"type": "string"}, "a": {"type": "integer"}},
            "required": ["n"],
        }
        b = {
            "type": "object",
            "properties": {"n": {"type": "string"}, "a": {"type": "integer"}},
            "required": ["n", "a"],
        }
        assert build_schema_cache_key(a) != build_schema_cache_key(b)


# ===========================================================================
# 9. Trace Cache
# ===========================================================================


class TestTraceCacheBasics:
    def test_miss_on_empty_cache(self) -> None:
        assert TraceCache().get("missing") is None

    def test_put_and_get(self) -> None:
        cache = TraceCache()
        cache.put("k1", "reasoning trace")
        assert cache.get("k1") == "reasoning trace"

    def test_hit_increments_counter(self) -> None:
        cache = TraceCache()
        cache.put("k1", "trace")
        cache.get("k1")
        assert cache.total_hits == 1
        assert cache.total_misses == 0

    def test_miss_increments_counter(self) -> None:
        cache = TraceCache()
        cache.get("missing")
        assert cache.total_hits == 0
        assert cache.total_misses == 1

    def test_size_reflects_entries(self) -> None:
        cache = TraceCache()
        assert cache.size == 0
        cache.put("k1", "t1")
        assert cache.size == 1

    def test_empty_trace_not_stored(self) -> None:
        cache = TraceCache()
        cache.put("k1", "")
        assert cache.size == 0
        assert cache.get("k1") is None

    def test_whitespace_trace_not_stored(self) -> None:
        cache = TraceCache()
        cache.put("k1", "   \n  ")
        assert cache.size == 0

    def test_invalidate_removes_entry(self) -> None:
        cache = TraceCache()
        cache.put("k1", "trace")
        assert cache.invalidate("k1") is True
        assert cache.get("k1") is None

    def test_invalidate_missing_returns_false(self) -> None:
        assert TraceCache().invalidate("missing") is False

    def test_clear_empties_cache(self) -> None:
        cache = TraceCache()
        cache.put("k1", "t1")
        cache.put("k2", "t2")
        cache.clear()
        assert cache.size == 0
        assert cache.total_hits == 0
        assert cache.total_misses == 0

    def test_update_existing_key(self) -> None:
        cache = TraceCache()
        cache.put("k1", "old")
        cache.put("k1", "new")
        assert cache.get("k1") == "new"

    def test_stats_dict_has_expected_keys(self) -> None:
        stats = TraceCache().stats()
        for key in ("size", "max_size", "ttl_seconds", "total_hits", "total_misses", "hit_rate"):
            assert key in stats

    def test_hit_rate_zero_initially(self) -> None:
        assert TraceCache().hit_rate == 0.0

    def test_hit_rate_computation(self) -> None:
        cache = TraceCache()
        cache.put("k", "t")
        cache.get("k")
        cache.get("nope")
        assert cache.hit_rate == pytest.approx(0.5)


class TestTraceCacheLRU:
    def test_lru_evicts_oldest_when_full(self) -> None:
        cache = TraceCache(max_size=3)
        for i in range(1, 4):
            cache.put(f"k{i}", f"trace{i}")
        cache.put("k4", "trace4")
        assert cache.size == 3
        assert cache.get("k1") is None
        assert cache.get("k2") is not None

    def test_access_refreshes_lru_position(self) -> None:
        cache = TraceCache(max_size=2)
        cache.put("k1", "t1")
        cache.put("k2", "t2")
        cache.get("k1")
        cache.put("k3", "t3")
        assert cache.get("k1") is not None
        assert cache.get("k2") is None

    def test_zero_max_size_raises(self) -> None:
        with pytest.raises(ValueError, match="max_size"):
            TraceCache(max_size=0)


class TestTraceCacheTTL:
    def test_fresh_entry_returned(self) -> None:
        cache = TraceCache(ttl_seconds=60.0)
        cache.put("k1", "trace")
        assert cache.get("k1") == "trace"

    def test_expired_entry_not_returned(self) -> None:
        cache = TraceCache(ttl_seconds=0.05)
        cache.put("k1", "trace")
        time.sleep(0.1)
        assert cache.get("k1") is None

    def test_expired_entry_removed_from_size(self) -> None:
        cache = TraceCache(ttl_seconds=0.05)
        cache.put("k1", "trace")
        time.sleep(0.1)
        cache.get("k1")
        assert cache.size == 0

    def test_zero_ttl_means_no_expiry(self) -> None:
        cache = TraceCache(ttl_seconds=0)
        cache.put("k1", "trace")
        assert cache.get("k1") == "trace"

    def test_default_constants(self) -> None:
        assert DEFAULT_MAX_SIZE == 256
        assert DEFAULT_TTL_SECONDS == 3600.0


class TestTraceCacheSchemaIntegration:
    def test_schema_key_drives_hit(self) -> None:
        cache = TraceCache()
        key = build_schema_cache_key(SIMPLE_SCHEMA)
        cache.put(key, "thinking about name and age")
        assert cache.get(key) == "thinking about name and age"

    def test_structurally_equivalent_schemas_share_cache(self) -> None:
        cache = TraceCache()
        k1 = build_schema_cache_key(SIMPLE_SCHEMA)
        k2 = build_schema_cache_key(SIMPLE_SCHEMA_WITH_DESCRIPTIONS)
        cache.put(k1, "scaffold trace")
        assert cache.get(k2) == "scaffold trace"

    def test_different_schemas_separate_slots(self) -> None:
        cache = TraceCache()
        k1 = build_schema_cache_key(SIMPLE_SCHEMA)
        k2 = build_schema_cache_key(ALTERNATE_SCHEMA)
        cache.put(k1, "trace for simple")
        cache.put(k2, "trace for alternate")
        assert cache.get(k1) == "trace for simple"
        assert cache.get(k2) == "trace for alternate"


# ===========================================================================
# 10. Routing Spectrum (5-mode ladder)
# ===========================================================================


class TestPhiSpectrumMode:
    def test_below_050_is_direct(self) -> None:
        for phi in (0.0, 0.30, 0.49):
            assert phi_spectrum_mode(phi) == "direct"

    def test_at_050_is_lite_ttf(self) -> None:
        assert phi_spectrum_mode(0.50) == "lite_ttf"

    def test_in_lite_ttf_range(self) -> None:
        assert phi_spectrum_mode(0.55) == "lite_ttf"
        assert phi_spectrum_mode(0.64) == "lite_ttf"

    def test_at_065_is_standard_ttf(self) -> None:
        assert phi_spectrum_mode(0.65) == "standard_ttf"

    def test_in_standard_ttf_range(self) -> None:
        assert phi_spectrum_mode(0.70) == "standard_ttf"
        assert phi_spectrum_mode(0.79) == "standard_ttf"

    def test_at_080_is_deep_ttf(self) -> None:
        assert phi_spectrum_mode(0.80) == "deep_ttf"

    def test_in_deep_ttf_range(self) -> None:
        assert phi_spectrum_mode(0.85) == "deep_ttf"
        assert phi_spectrum_mode(0.94) == "deep_ttf"

    def test_at_095_is_sc_full(self) -> None:
        assert phi_spectrum_mode(0.95) == "sc_full"

    def test_above_095_is_sc_full(self) -> None:
        assert phi_spectrum_mode(0.97) == "sc_full"
        assert phi_spectrum_mode(1.0) == "sc_full"

    def test_thresholds_list_has_4_entries(self) -> None:
        assert len(PHI_MODE_THRESHOLDS) == 4

    def test_thresholds_in_descending_order(self) -> None:
        thresholds = [t for t, _ in PHI_MODE_THRESHOLDS]
        assert thresholds == sorted(thresholds, reverse=True)

    def test_all_five_modes_reachable(self) -> None:
        modes = {phi_spectrum_mode(phi) for phi in [0.3, 0.55, 0.70, 0.85, 0.97]}
        assert modes == {"direct", "lite_ttf", "standard_ttf", "deep_ttf", "sc_full"}


class TestRoutingDecision:
    def test_default_routing_mode_is_direct(self) -> None:
        d = RoutingDecision(
            strategy="direct",
            expected_accuracy_delta=0.0,
            expected_overhead_pct=0.0,
            confidence=0.8,
            explanation="test",
        )
        assert d.routing_mode == "direct"

    def test_routing_mode_lite_ttf(self) -> None:
        d = RoutingDecision(
            strategy="ttf",
            expected_accuracy_delta=0.10,
            expected_overhead_pct=20.0,
            confidence=0.7,
            explanation="test",
            routing_mode="lite_ttf",
        )
        assert d.routing_mode == "lite_ttf"

    def test_routing_mode_standard_ttf(self) -> None:
        d = RoutingDecision(
            strategy="ttf",
            expected_accuracy_delta=0.15,
            expected_overhead_pct=25.0,
            confidence=0.75,
            explanation="test",
            routing_mode="standard_ttf",
        )
        assert d.routing_mode == "standard_ttf"

    def test_routing_mode_sc_full(self) -> None:
        d = RoutingDecision(
            strategy="ttf",
            expected_accuracy_delta=0.20,
            expected_overhead_pct=35.0,
            confidence=0.9,
            explanation="test",
            routing_mode="sc_full",
        )
        assert d.routing_mode == "sc_full"

    def test_str_representation_works(self) -> None:
        d = RoutingDecision(
            strategy="ttf",
            expected_accuracy_delta=0.17,
            expected_overhead_pct=25.0,
            confidence=0.70,
            explanation="test",
            routing_mode="standard_ttf",
        )
        s = str(d)
        assert "ttf" in s
        assert "0.70" in s


# ===========================================================================
# 11. Self-Calibrator
# ===========================================================================


class TestLogisticFitting:
    def test_fit_on_separable_data(self) -> None:
        xs = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
        y = [0, 0, 0, 1, 1, 1]
        w, _b = _fit_logistic(xs, y)
        assert w > 0

    def test_fit_on_empty_data_returns_zeros(self) -> None:
        w, b = _fit_logistic([], [])
        assert w == 0.0
        assert b == 0.0

    def test_threshold_zero_weight_returns_none(self) -> None:
        assert _threshold_from_logistic(0.0, 1.0) is None

    def test_threshold_clamped_to_range(self) -> None:
        result = _threshold_from_logistic(1.0, 10.0)
        assert result is not None
        assert 0.30 <= result <= 0.95

    def test_threshold_simple_case(self) -> None:
        result = _threshold_from_logistic(10.0, -5.0)
        assert result is not None
        assert abs(result - 0.5) < 0.01


class TestSelfCalibrator:
    def test_defaults(self) -> None:
        cal = SelfCalibrator(persist_path=None)
        assert cal.sample_count == 0
        assert cal.calibration_count == 0
        assert 0.30 <= cal.current_threshold <= 0.95

    def test_default_constants(self) -> None:
        assert DEFAULT_WINDOW_SIZE == 200
        assert DEFAULT_MIN_SAMPLES == 20
        assert DEFAULT_TARGET_ACCURACY == 0.80

    def test_non_ttf_record_ignored(self) -> None:
        cal = SelfCalibrator(persist_path=None)
        cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=0.9, used_ttf=False))
        assert cal.sample_count == 0

    def test_none_eval_ignored(self) -> None:
        cal = SelfCalibrator(persist_path=None)
        cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=None))
        assert cal.sample_count == 0

    def test_record_increments_count(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=50)
        cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=0.85))
        assert cal.sample_count == 1

    def test_stats_keys(self) -> None:
        stats = SelfCalibrator(persist_path=None).stats()
        for key in (
            "current_threshold",
            "sample_count",
            "window_size",
            "min_samples",
            "target_accuracy",
            "calibration_count",
        ):
            assert key in stats

    def test_no_calibration_below_min_samples(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=10)
        for _ in range(9):
            cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=0.9))
        assert cal.calibration_count == 0

    def test_calibration_triggered_at_min_samples(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=5)
        for i in range(5):
            score = 0.9 if i % 2 == 0 else 0.5
            cal.record(CalibrationRecord(phi=float(i) / 4.0, semantic_eval_score=score))
        assert cal.calibration_count >= 1

    def test_rolling_window_caps(self) -> None:
        cal = SelfCalibrator(persist_path=None, window_size=5, min_samples=100)
        for _ in range(10):
            cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=0.9))
        assert cal.sample_count <= 5

    def test_threshold_stays_in_valid_range(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=5)
        for _ in range(3):
            cal.record(CalibrationRecord(phi=0.95, semantic_eval_score=0.95))
        for _ in range(2):
            cal.record(CalibrationRecord(phi=0.10, semantic_eval_score=0.10))
        assert 0.30 <= cal.current_threshold <= 0.95

    def test_skips_calibration_when_all_same_label(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=5)
        for _ in range(5):
            cal.record(CalibrationRecord(phi=0.7, semantic_eval_score=0.99))
        assert cal.calibration_count == 0


class TestSelfCalibratorPersistence:
    def test_persist_and_load(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "calibrated.json"
            cal = SelfCalibrator(persist_path=path, min_samples=5)
            for i in range(5):
                score = 0.9 if i % 2 == 0 else 0.5
                cal.record(CalibrationRecord(phi=float(i) / 4.0, semantic_eval_score=score))
            if cal.calibration_count > 0:
                assert path.exists()
                cal2 = SelfCalibrator(persist_path=path)
                assert cal2.current_threshold == pytest.approx(cal.current_threshold, abs=0.001)

    def test_missing_file_uses_initial_threshold(self) -> None:
        cal = SelfCalibrator(persist_path=Path("/nonexistent/dir/x.json"), initial_threshold=0.65)
        assert cal.current_threshold == pytest.approx(0.65)

    def test_persist_path_none_does_not_error(self) -> None:
        cal = SelfCalibrator(persist_path=None, min_samples=5)
        for i in range(5):
            cal.record(
                CalibrationRecord(
                    phi=float(i) / 4.0,
                    semantic_eval_score=0.9 if i % 2 == 0 else 0.5,
                )
            )


class TestCalibrationRecord:
    def test_defaults(self) -> None:
        rec = CalibrationRecord(phi=0.7, semantic_eval_score=0.85)
        assert rec.used_ttf is True
        assert rec.quality_gate_passed is None

    def test_with_quality_gate(self) -> None:
        rec = CalibrationRecord(phi=0.7, semantic_eval_score=0.85, quality_gate_passed=True)
        assert rec.quality_gate_passed is True

    def test_non_ttf_flag(self) -> None:
        rec = CalibrationRecord(phi=0.7, semantic_eval_score=0.85, used_ttf=False)
        assert rec.used_ttf is False


# ===========================================================================
# 12. Soft Logit Biasing
# ===========================================================================


class TestBuildLogitBias:
    def test_returns_dict(self) -> None:
        assert isinstance(_build_schema_logit_bias(["name", "age", "status"]), dict)

    def test_empty_fields_returns_empty(self) -> None:
        assert _build_schema_logit_bias([]) == {}

    def test_graceful_when_tiktoken_missing(self) -> None:
        real_import = __builtins__.__import__ if hasattr(__builtins__, "__import__") else __import__  # type: ignore[union-attr]

        def _mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "tiktoken":
                raise ImportError("tiktoken not available")
            return real_import(name, *args, **kwargs)  # type: ignore[operator]

        with patch("builtins.__import__", side_effect=_mock_import):
            result = _build_schema_logit_bias(["name", "age"])
        assert isinstance(result, dict)

    def test_all_values_positive_floats(self) -> None:
        result = _build_schema_logit_bias(["name", "age"])
        if result:
            for v in result.values():
                assert isinstance(v, float)
                assert v > 0.0

    def test_all_keys_are_ints(self) -> None:
        result = _build_schema_logit_bias(["name", "age"])
        if result:
            for k in result.keys():
                assert isinstance(k, int)

    def test_custom_bias_value(self) -> None:
        result_custom = _build_schema_logit_bias(["name"], bias_value=5.0)
        if result_custom:
            assert next(iter(result_custom.values())) == pytest.approx(5.0)


class TestLogitBiasWiredInEngine:
    @pytest.mark.asyncio
    async def test_bias_passed_when_backend_supports_it(self) -> None:
        captured: dict[str, Any] = {}

        class LogitBackend(_StreamStub):
            name = "logit"
            supports_kv_cache_reuse = False
            supports_logit_bias = True
            accuracy_loss_baseline = 0.15

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["logit_bias"] = logit_bias
                    return "<think>name and age fields.</think>"
                return json.dumps({"name": "Alice", "age": 30})

        await TTFEngine(backend=LogitBackend()).generate("Extract.", schema=SIMPLE_SCHEMA)
        bias = captured.get("logit_bias")
        if bias is not None:
            assert isinstance(bias, dict)
            assert len(bias) > 0

    @pytest.mark.asyncio
    async def test_bias_none_when_backend_does_not_support(self) -> None:
        captured: dict[str, Any] = {}

        class NoLogitBackend(_StreamStub):
            name = "no_logit"
            supports_kv_cache_reuse = False
            supports_logit_bias = False
            accuracy_loss_baseline = 0.15

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["logit_bias"] = logit_bias
                    return "<think>name field.</think>"
                return json.dumps({"name": "Bob", "age": 25})

        await TTFEngine(backend=NoLogitBackend()).generate("Extract.", schema=SIMPLE_SCHEMA)
        assert captured.get("logit_bias") is None

    @pytest.mark.asyncio
    async def test_bias_none_when_no_schema(self) -> None:
        captured: dict[str, Any] = {}

        class LogitBackend(_StreamStub):
            name = "logit"
            supports_kv_cache_reuse = False
            supports_logit_bias = True
            accuracy_loss_baseline = 0.15

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                top_p: float | None = None,
                top_k: int | None = None,
                max_tokens: int | None = None,
                seed: int | None = None,
                frequency_penalty: float | None = None,
                presence_penalty: float | None = None,
                stop: list[str] | str | None = None,
                logit_bias: dict[int, float] | None = None,
            ) -> str:
                if constraints is None:
                    captured["logit_bias"] = logit_bias
                    return "<think>General reasoning.</think>"
                return json.dumps({"answer": "42"})

        await TTFEngine(backend=LogitBackend()).generate("What is 2+2?", schema=None)
        assert captured.get("logit_bias") is None


# ===========================================================================
# 13. Self-Consistency Pass 1
# ===========================================================================


class _CountingBackend(_StreamStub):
    name = "counting"

    def __init__(self) -> None:
        self.pass1_count = 0
        self.pass2_count = 0

    @property
    def supports_logit_bias(self) -> bool:
        return False

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return 0.1

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        logit_bias: dict[int, float] | None = None,
        **kwargs: Any,
    ) -> str:
        if constraints is None:
            self.pass1_count += 1
            return (
                "<think>Step 1: analyse the schema fields: name (string), age (integer),"
                " active (boolean), order_id, status, total. Step 2: extract each field."
                " The name field is a string. The age field is an integer."
                " The active field is boolean. Proceed to format.</think>"
            )
        self.pass2_count += 1
        return json.dumps({"order_id": "ORD-001", "total": 99.99})


class _ScoredTraceBackend(_StreamStub):
    name = "scored"

    def __init__(self) -> None:
        self._call_n = 0

    @property
    def supports_logit_bias(self) -> bool:
        return False

    @property
    def supports_kv_cache_reuse(self) -> bool:
        return False

    @property
    def accuracy_loss_baseline(self) -> float | None:
        return 0.1

    async def generate(
        self,
        prompt: str,
        schema: dict | None = None,
        constraints: str | None = None,
        kv_cache_prefix: str | None = None,
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
        logit_bias: dict[int, float] | None = None,
        **kwargs: Any,
    ) -> str:
        if constraints == "json":
            return json.dumps({"order_id": "X", "total": 1.0})
        self._call_n += 1
        if self._call_n == 1:
            return "<think>short</think>"
        if self._call_n == 2:
            return (
                "<think>Step 1: read carefully. Step 2: identify order_id field. "
                "Step 3: identify total field. Step 4: validate. Step 5: compose JSON.</think>"
            )
        return "<think>medium length reasoning here for pass 1</think>"


class TestSelfConsistencyConstants:
    def test_phi_threshold_is_0_95(self) -> None:
        assert _SC_PHI_THRESHOLD == pytest.approx(0.95)

    def test_default_k_is_3(self) -> None:
        assert DEFAULT_SC_K == 3


class TestSelfConsistencyPass1Function:
    @pytest.mark.asyncio
    async def test_k1_makes_one_call(self) -> None:
        backend = _CountingBackend()
        await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="Analyse.",
            k=1,
            max_tokens=None,
            logit_bias=None,
            schema=None,
            routing_score=None,
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_k3_makes_three_calls(self) -> None:
        backend = _CountingBackend()
        await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="Analyse.",
            k=3,
            max_tokens=None,
            logit_bias=None,
            schema=None,
            routing_score=None,
        )
        assert backend.pass1_count == 3

    @pytest.mark.asyncio
    async def test_k3_picks_longest_trace_without_schema(self) -> None:
        backend = _ScoredTraceBackend()
        thinking, _ = await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="Analyse.",
            k=3,
            max_tokens=None,
            logit_bias=None,
            schema=None,
            routing_score=None,
        )
        assert "Step 5" in thinking

    @pytest.mark.asyncio
    async def test_k3_with_schema_and_routing_score(self) -> None:
        backend = _ScoredTraceBackend()
        thinking, _raw = await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="Extract.",
            k=3,
            max_tokens=512,
            logit_bias=None,
            schema=SIMPLE_SCHEMA,
            routing_score=_make_rs(phi=0.97),
        )
        assert isinstance(thinking, str)
        assert len(thinking) > 0

    @pytest.mark.asyncio
    async def test_k0_treated_as_k1(self) -> None:
        backend = _CountingBackend()
        await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="test",
            k=0,
            max_tokens=None,
            logit_bias=None,
            schema=None,
            routing_score=None,
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_returns_extracted_thinking_without_tags(self) -> None:
        backend = _ScoredTraceBackend()
        thinking, raw = await _run_self_consistency_pass1(
            backend=backend,
            think_prompt="Extract.",
            k=3,
            max_tokens=None,
            logit_bias=None,
            schema=None,
            routing_score=None,
        )
        assert "<think>" not in thinking
        assert "<think>" in raw


class TestSelfConsistencyEngineParam:
    def test_default_k_is_1(self) -> None:
        assert TTFEngine(backend=_CountingBackend())._ttf_self_consistency == 1

    def test_explicit_k3(self) -> None:
        engine = TTFEngine(backend=_CountingBackend(), ttf_self_consistency=3)
        assert engine._ttf_self_consistency == 3

    def test_k0_clamped_to_1(self) -> None:
        engine = TTFEngine(backend=_CountingBackend(), ttf_self_consistency=0)
        assert engine._ttf_self_consistency == 1

    def test_negative_k_clamped_to_1(self) -> None:
        engine = TTFEngine(backend=_CountingBackend(), ttf_self_consistency=-5)
        assert engine._ttf_self_consistency == 1

    @pytest.mark.asyncio
    async def test_k1_makes_single_pass1_call(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_k3_makes_three_pass1_calls(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=3).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        assert backend.pass1_count == 3

    @pytest.mark.asyncio
    async def test_k3_still_calls_pass2(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=3).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        assert backend.pass2_count >= 1

    @pytest.mark.asyncio
    async def test_output_is_json_string_with_k3(self) -> None:
        backend = _CountingBackend()
        _, output = await TTFEngine(backend=backend, ttf_self_consistency=3).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        parsed = json.loads(output)
        assert "order_id" in parsed


class TestAutoTriggerSelfConsistency:
    @pytest.mark.asyncio
    async def test_phi_above_threshold_triggers_k3(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.97)
        )
        assert backend.pass1_count == DEFAULT_SC_K

    @pytest.mark.asyncio
    async def test_phi_exactly_at_threshold_triggers(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.95)
        )
        assert backend.pass1_count == DEFAULT_SC_K

    @pytest.mark.asyncio
    async def test_phi_below_threshold_no_auto_trigger(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.80)
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_explicit_k_beats_auto_trigger(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=5).generate(
            "Extract.", schema=SIMPLE_SCHEMA, routing_score=_make_rs(phi=0.97)
        )
        assert backend.pass1_count == 5

    @pytest.mark.asyncio
    async def test_no_routing_score_no_auto_trigger(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_just_below_threshold_does_not_trigger(self) -> None:
        backend = _CountingBackend()
        await TTFEngine(backend=backend, ttf_self_consistency=1).generate(
            "Extract.",
            schema=SIMPLE_SCHEMA,
            routing_score=_make_rs(phi=_SC_PHI_THRESHOLD - 0.001),
        )
        assert backend.pass1_count == 1

    @pytest.mark.asyncio
    async def test_k3_pass1_then_pass2_ordering(self) -> None:
        call_log: list[str] = []

        class LoggingBackend(_StreamStub):
            name = "logging"

            @property
            def supports_logit_bias(self) -> bool:
                return False

            @property
            def supports_kv_cache_reuse(self) -> bool:
                return False

            @property
            def accuracy_loss_baseline(self) -> float | None:
                return 0.1

            async def generate(
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                *,
                temperature: float | None = None,
                max_tokens: int | None = None,
                logit_bias: dict[int, float] | None = None,
                **kwargs: Any,
            ) -> str:
                if constraints is None:
                    call_log.append("pass1")
                    return "<think>analyse order_id total carefully</think>"
                call_log.append("pass2")
                return json.dumps({"order_id": "L-1", "total": 10.0})

        await TTFEngine(backend=LoggingBackend(), ttf_self_consistency=3).generate(
            "Extract.", schema=SIMPLE_SCHEMA
        )
        assert call_log[:3] == ["pass1", "pass1", "pass1"]
        assert call_log[-1] == "pass2"


# ===========================================================================
# 14. RCOCR Protocol (standalone package)
# ===========================================================================


class _MockRCOCRBackend:
    name = "mock"

    async def generate(self, prompt: str, constraints: str | None = None, **kwargs: Any) -> str:
        if constraints == "json":
            return json.dumps({"status": "extracted", "value": 42})
        return (
            "<think>Step 1: read prompt. Step 2: identify fields. Step 3: compose answer.</think>"
        )


class TestRCOCRProtocol:
    def test_version_is_0_1_0(self) -> None:
        assert rcocr_version == "0.1.0"

    def test_engine_importable(self) -> None:
        assert RCOCREngine is not None

    def test_backend_protocol_satisfied(self) -> None:
        assert isinstance(_MockRCOCRBackend(), RCOCRBackend)

    @pytest.mark.asyncio
    async def test_generate_returns_two_tuple(self) -> None:
        result = await RCOCREngine(backend=_MockRCOCRBackend()).generate("Extract order data.")
        assert isinstance(result, tuple)
        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_generate_thinking_is_string(self) -> None:
        thinking, _ = await RCOCREngine(backend=_MockRCOCRBackend()).generate("Extract order data.")
        assert isinstance(thinking, str)

    @pytest.mark.asyncio
    async def test_generate_output_is_json(self) -> None:
        _, output = await RCOCREngine(backend=_MockRCOCRBackend()).generate("Extract order data.")
        parsed = json.loads(output)
        assert "status" in parsed

    @pytest.mark.asyncio
    async def test_two_pass_call_pattern(self) -> None:
        calls: list[str | None] = []

        class TrackingBackend:
            name = "tracking"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                calls.append(constraints)
                if constraints == "json":
                    return '{"result": "done"}'
                return "<think>some reasoning</think>"

        await RCOCREngine(backend=TrackingBackend()).generate("Extract.")
        assert None in calls
        assert "json" in calls

    @pytest.mark.asyncio
    async def test_schema_appears_in_format_prompt(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        captured_prompts: list[str] = []

        class CaptureBackend:
            name = "capture"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                captured_prompts.append(prompt)
                if constraints == "json":
                    return '{"name": "test"}'
                return "<think>reasoning</think>"

        await RCOCREngine(backend=CaptureBackend()).generate("Extract name.", schema=schema)
        assert "name" in captured_prompts[-1]

    def test_default_k_is_1(self) -> None:
        assert RCOCREngine(backend=_MockRCOCRBackend())._k == 1

    def test_k0_clamped_to_1(self) -> None:
        assert RCOCREngine(backend=_MockRCOCRBackend(), self_consistency_k=0)._k == 1

    @pytest.mark.asyncio
    async def test_sc_k3_makes_three_pass1_calls(self) -> None:
        count = {"n": 0}

        class CountingRCOCR:
            name = "counting"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                if constraints is None:
                    count["n"] += 1
                    return f"<think>trace {count['n']}</think>"
                return '{"ok": true}'

        await RCOCREngine(backend=CountingRCOCR(), self_consistency_k=3).generate("Extract.")
        assert count["n"] == 3

    @pytest.mark.asyncio
    async def test_temperature_forwarded(self) -> None:
        captured: dict[str, Any] = {}

        class TempCapture:
            name = "temp"

            async def generate(
                self, prompt: str, constraints: str | None = None, **kwargs: Any
            ) -> str:
                captured["temperature"] = kwargs.get("temperature")
                if constraints == "json":
                    return '{"x": 1}'
                return "<think>thinking</think>"

        await RCOCREngine(backend=TempCapture()).generate("test", temperature=0.4)
        assert captured.get("temperature") == pytest.approx(0.4)


class TestRCOCRZeroDependency:
    def _read(self, relative: str) -> str:
        path = os.path.join(
            os.path.dirname(__file__), "..", "..", "rcocr", "src", "rcocr", relative
        )
        with open(path) as f:
            return f.read()

    def test_engine_has_no_formatshield_import(self) -> None:
        assert "formatshield" not in self._read("engine.py")

    def test_protocol_has_no_formatshield_import(self) -> None:
        assert "formatshield" not in self._read("protocol.py")

    def test_no_heavy_third_party_deps_in_init(self) -> None:
        source = self._read("__init__.py")
        for dep in ("tiktoken", "pydantic", "httpx", "openai", "anthropic"):
            assert dep not in source, f"rcocr/__init__.py imports {dep}"


# ===========================================================================
# 15. Schema-Aware Engine Integration
# ===========================================================================


class TestSchemaAwareEngineIntegration:
    async def test_schema_aware_prompt_when_routing_score_provided(self) -> None:
        captured_prompts: list[str] = []

        class CapturingBackend(_TemperatureCapturingBackend):
            async def generate(  # type: ignore[override]
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                temperature: float | None = None,
                **kwargs: Any,
            ) -> str:
                captured_prompts.append(prompt)
                return await super().generate(
                    prompt,
                    schema=schema,
                    constraints=constraints,
                    kv_cache_prefix=kv_cache_prefix,
                    temperature=temperature,
                )

        backend = CapturingBackend()
        await TTFEngine(backend=backend).generate(
            "Extract data.", schema=ENUM_SCHEMA, routing_score=_make_rs(phi=0.80)
        )
        assert "Schema-Guided Reasoning Context" in captured_prompts[0]

    async def test_generic_prompt_without_routing_score(self) -> None:
        captured_prompts: list[str] = []

        class CapturingBackend(_TemperatureCapturingBackend):
            async def generate(  # type: ignore[override]
                self,
                prompt: str,
                schema: dict | None = None,
                constraints: str | None = None,
                kv_cache_prefix: str | None = None,
                temperature: float | None = None,
                **kwargs: Any,
            ) -> str:
                captured_prompts.append(prompt)
                return await super().generate(
                    prompt,
                    schema=schema,
                    constraints=constraints,
                    kv_cache_prefix=kv_cache_prefix,
                    temperature=temperature,
                )

        backend = CapturingBackend()
        await TTFEngine(backend=backend).generate(
            "Extract data.", schema=ENUM_SCHEMA, routing_score=None
        )
        assert "Schema-Guided Reasoning Context" not in captured_prompts[0]

    async def test_generate_returns_tuple(self) -> None:
        result = await TTFEngine(backend=_TemperatureCapturingBackend()).generate(
            "test", schema=SIMPLE_SCHEMA, routing_score=_make_rs()
        )
        assert isinstance(result, tuple)
        assert len(result) == 2

    async def test_backward_compat_no_routing_score(self, mock_backend: Any) -> None:
        thinking, output = await TTFEngine(backend=mock_backend).generate(
            "test prompt", schema=SIMPLE_SCHEMA
        )
        assert isinstance(thinking, str)
        assert isinstance(output, str)
