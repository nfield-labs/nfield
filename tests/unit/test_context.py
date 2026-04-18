"""Unit tests for formatshield.oracle.context."""

from __future__ import annotations

import hashlib

from formatshield.oracle.context import RoutingContext, TelemetryRecord


class TestRoutingContext:
    def test_fields_set_correctly(self) -> None:
        ctx = RoutingContext(
            backend_id="groq",
            model_id="llama-3.1-8b-instant",
            task_id="math500",
            schema_family="math",
            prompt_id="abc123def456",
        )
        assert ctx.backend_id == "groq"
        assert ctx.model_id == "llama-3.1-8b-instant"
        assert ctx.task_id == "math500"
        assert ctx.schema_family == "math"
        assert ctx.prompt_id == "abc123def456"

    def test_from_prompt_computes_prompt_id(self) -> None:
        prompt = "What is 2 + 2?"
        ctx = RoutingContext.from_prompt(
            prompt=prompt,
            backend_id="vllm",
            model_id="mistral-7b",
        )
        expected_id = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        assert ctx.prompt_id == expected_id
        assert ctx.backend_id == "vllm"
        assert ctx.task_id == "unknown"
        assert ctx.schema_family == "unknown"

    def test_from_prompt_with_all_fields(self) -> None:
        ctx = RoutingContext.from_prompt(
            prompt="solve x",
            backend_id="groq",
            model_id="llama-3.3-70b",
            task_id="gsm_symbolic",
            schema_family="math",
        )
        assert ctx.task_id == "gsm_symbolic"
        assert ctx.schema_family == "math"

    def test_to_dict_all_keys(self) -> None:
        ctx = RoutingContext(
            backend_id="ollama",
            model_id="phi-3",
            task_id="unknown",
            schema_family="unknown",
            prompt_id="deadbeef0000",
        )
        d = ctx.to_dict()
        assert set(d.keys()) == {
            "backend_id",
            "model_id",
            "task_id",
            "schema_family",
            "prompt_id",
            "phi_score",
            "phi_lambda2",
            "phi_tau",
            "phi_delta_k",
        }
        assert d["backend_id"] == "ollama"


class TestTelemetryRecord:
    def _make_ctx(self) -> RoutingContext:
        return RoutingContext(
            backend_id="groq",
            model_id="llama-3.1-8b-instant",
            task_id="classification",
            schema_family="classification",
            prompt_id="aabbccddeeff",
        )

    def test_to_dict_serializable(self) -> None:
        record = TelemetryRecord(
            features=[0.5, 3.0, 2.0, 0.5, 1.0, 5.0],
            routing_context=self._make_ctx(),
            chosen_action="ttf",
            expected_utility=0.12,
            realized_outcome=0.18,
            latency_ms=1500.0,
            token_cost=300.0,
            schema_validity=True,
            failure_modes=[],
        )
        d = record.to_dict()
        assert d["chosen_action"] == "ttf"
        assert d["realized_outcome"] == 0.18
        assert d["schema_validity"] is True
        assert isinstance(d["routing_context"], dict)
        assert d["routing_context"]["backend_id"] == "groq"

    def test_realized_outcome_can_be_none(self) -> None:
        record = TelemetryRecord(
            features=[0.1] * 6,
            routing_context=self._make_ctx(),
            chosen_action="direct",
            expected_utility=0.0,
            realized_outcome=None,
            latency_ms=500.0,
            token_cost=100.0,
            schema_validity=True,
        )
        d = record.to_dict()
        assert d["realized_outcome"] is None

    def test_failure_modes_default_empty(self) -> None:
        record = TelemetryRecord(
            features=[0.1] * 6,
            routing_context=self._make_ctx(),
            chosen_action="safe-abstain",
            expected_utility=0.05,
            realized_outcome=None,
            latency_ms=200.0,
            token_cost=0.0,
            schema_validity=False,
        )
        assert record.failure_modes == []
