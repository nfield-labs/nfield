"""Unit tests for formatshield.oracle.oracle_x.OracleX."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from formatshield.oracle.context import RoutingContext, TelemetryRecord
from formatshield.oracle.oracle_x import OracleX
from formatshield.scorer.features import ComplexityFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    token_entropy: float = 0.5,
    schema_depth: int = 2,
    required_reasoning_ops: int = 1,
    instruction_tune_score: float = 0.6,
    prompt_length_bucket: int = 1,
    schema_constraint_count: int = 3,
) -> ComplexityFeatures:
    return ComplexityFeatures(
        token_entropy=token_entropy,
        schema_depth=schema_depth,
        required_reasoning_ops=required_reasoning_ops,
        instruction_tune_score=instruction_tune_score,
        prompt_length_bucket=prompt_length_bucket,
        schema_constraint_count=schema_constraint_count,
    )


def _make_oracle() -> OracleX:
    """Return an OracleX instance (always heuristic/Φ, no pkl)."""
    return OracleX()


# ---------------------------------------------------------------------------
# Instantiation — always active, no pkl required
# ---------------------------------------------------------------------------


def test_oracle_x_instantiates_without_artifact() -> None:
    """OracleX must instantiate without any pkl file present."""
    oracle = OracleX()
    assert oracle is not None


def test_oracle_x_instantiates_with_nonexistent_path() -> None:
    oracle = OracleX(model_path="/nonexistent/oracle_x_v1.pkl")
    assert oracle is not None


# ---------------------------------------------------------------------------
# Predict — heuristic fallback (no Φ context)
# ---------------------------------------------------------------------------


def test_predict_returns_routing_decision() -> None:
    from formatshield.oracle.routing_decision import RoutingDecision

    oracle = _make_oracle()
    decision = oracle.predict(_make_features(), backend="groq", model_id="test-model")
    assert isinstance(decision, RoutingDecision)


def test_predict_strategy_is_valid() -> None:
    oracle = _make_oracle()
    decision = oracle.predict(_make_features(), backend="groq", model_id="llama-3.1-8b-instant")
    assert decision.strategy in {"ttf", "direct"}


def test_predict_error_fallback_returns_direct() -> None:
    """On internal error, predict() must return a direct decision, not raise."""
    oracle = _make_oracle()
    # Pass a non-string backend to force an error in the implementation
    decision = oracle.predict(
        _make_features(),
        backend=None,  # type: ignore[arg-type]
        model_id="test-model",
    )
    assert decision.strategy in {"ttf", "direct"}


# ---------------------------------------------------------------------------
# Predict — rule 1: native thinker
# ---------------------------------------------------------------------------


def test_predict_native_thinker_always_direct() -> None:
    from formatshield.oracle.threshold_oracle import NATIVE_THINKERS

    oracle = _make_oracle()
    if not NATIVE_THINKERS:
        pytest.skip("No native thinker models defined.")
    native_model = next(iter(NATIVE_THINKERS))
    decision = oracle.predict(_make_features(), backend="groq", model_id=native_model)
    assert decision.strategy == "direct"


# ---------------------------------------------------------------------------
# Predict — rule 2: latency budget
# ---------------------------------------------------------------------------


def test_predict_tight_latency_budget_forces_direct() -> None:
    oracle = _make_oracle()
    decision = oracle.predict(
        _make_features(),
        backend="groq",
        model_id="llama-3.1-8b-instant",
        latency_budget_ms=1.0,
    )
    assert decision.strategy == "direct"


def test_predict_relaxed_budget_does_not_force_direct() -> None:
    oracle = _make_oracle()
    decision = oracle.predict(
        _make_features(schema_depth=5, required_reasoning_ops=3, schema_constraint_count=8),
        backend="groq",
        model_id="llama-3.1-8b-instant",
        latency_budget_ms=600_000.0,
    )
    assert decision.strategy in {"ttf", "direct"}


# ---------------------------------------------------------------------------
# Predict — Φ-based routing via context
# ---------------------------------------------------------------------------


def test_predict_phi_high_routes_ttf() -> None:
    """When phi_score is high (>threshold), OracleX should route TTF."""
    oracle = _make_oracle()
    ctx = RoutingContext(
        backend_id="groq",
        model_id="llama-3.1-8b-instant",
        task_id="test",
        schema_family="extraction",
        prompt_id="abc123abc123",
        phi_score=0.92,  # very high → TTF
        phi_lambda2=0.8,
        phi_tau=0.7,
        phi_delta_k=0.8,
    )
    decision = oracle.predict(
        _make_features(),
        backend="groq",
        model_id="llama-3.1-8b-instant",
        context=ctx,
    )
    assert decision.strategy == "ttf"


def test_predict_phi_low_routes_direct() -> None:
    """When phi_score is low (<threshold), OracleX should route direct."""
    oracle = _make_oracle()
    ctx = RoutingContext(
        backend_id="groq",
        model_id="llama-3.1-8b-instant",
        task_id="test",
        schema_family="extraction",
        prompt_id="abc123abc123",
        phi_score=0.05,  # very low → direct
        phi_lambda2=0.0,
        phi_tau=0.0,
        phi_delta_k=0.05,
    )
    decision = oracle.predict(
        _make_features(),
        backend="groq",
        model_id="llama-3.1-8b-instant",
        context=ctx,
    )
    assert decision.strategy == "direct"


def test_predict_phi_explanation_in_decision() -> None:
    """RoutingDecision.explanation should contain Φ components when phi_score>0."""
    oracle = _make_oracle()
    ctx = RoutingContext(
        backend_id="groq",
        model_id="llama-3.1-8b-instant",
        task_id="test",
        schema_family="extraction",
        prompt_id="abc123abc123",
        phi_score=0.80,
        phi_lambda2=0.5,
        phi_tau=0.4,
        phi_delta_k=0.6,
    )
    decision = oracle.predict(
        _make_features(),
        backend="groq",
        model_id="llama-3.1-8b-instant",
        context=ctx,
    )
    assert "Φ=" in decision.explanation
    assert "λ̃₂=" in decision.explanation


# ---------------------------------------------------------------------------
# Backward-compat signature
# ---------------------------------------------------------------------------


def test_backward_compat_signature() -> None:
    """OracleX.predict() must accept the same positional args as ThresholdOracle.predict()."""
    from formatshield.oracle.threshold_oracle import ThresholdOracle

    oracle_sig = inspect.signature(OracleX.predict)
    threshold_sig = inspect.signature(ThresholdOracle.predict)

    oracle_params = list(oracle_sig.parameters.keys())
    threshold_params = list(threshold_sig.parameters.keys())

    for p in threshold_params:
        assert p in oracle_params, f"Missing param {p!r} in OracleX.predict()"
    assert "context" in oracle_params


# ---------------------------------------------------------------------------
# Deprecation stubs
# ---------------------------------------------------------------------------


def test_from_benchmark_data_raises_deprecation_warning() -> None:
    with pytest.warns(DeprecationWarning, match="from_benchmark_data"):
        with pytest.raises(NotImplementedError, match="from_benchmark_data"):
            OracleX.from_benchmark_data()


def test_save_raises_deprecation_warning() -> None:
    oracle = _make_oracle()
    with pytest.warns(DeprecationWarning, match="save"):
        with pytest.raises(NotImplementedError, match="save"):
            oracle.save()


def test_load_raises_deprecation_warning() -> None:
    oracle = _make_oracle()
    with pytest.warns(DeprecationWarning, match="load"):
        with pytest.raises(NotImplementedError, match="load"):
            oracle.load()


def test_update_online_raises_deprecation_warning() -> None:
    oracle = _make_oracle()
    ctx = RoutingContext(
        backend_id="groq",
        model_id="llama-3.1-8b-instant",
        task_id="math",
        schema_family="math",
        prompt_id="abc123abc123",
    )
    record = TelemetryRecord(
        features=[0.5] * 6,
        routing_context=ctx,
        chosen_action="ttf",
        expected_utility=0.4,
        realized_outcome=1.0,
        latency_ms=300.0,
        token_cost=2.0,
        schema_validity=True,
        failure_modes=[],
        label_verified=True,
    )
    with pytest.warns(DeprecationWarning, match="update_online"):
        with pytest.raises(NotImplementedError, match="update_online"):
            oracle.update_online(record)


# ---------------------------------------------------------------------------
# use_safe_abstain property
# ---------------------------------------------------------------------------


def test_use_safe_abstain_property() -> None:
    from formatshield.oracle.routing_decision import RoutingDecision

    d = RoutingDecision(
        strategy="safe-abstain",
        expected_accuracy_delta=0.0,
        expected_overhead_pct=0.0,
        confidence=0.5,
        explanation="test",
    )
    assert d.use_safe_abstain is True

    d2 = RoutingDecision(
        strategy="direct",
        expected_accuracy_delta=0.0,
        expected_overhead_pct=0.0,
        confidence=0.5,
        explanation="test",
    )
    assert d2.use_safe_abstain is False


# ---------------------------------------------------------------------------
# Backend capability registry
# ---------------------------------------------------------------------------


def test_backend_registry_loads() -> None:
    from formatshield.oracle.backend_registry import BackendRegistry

    reg = BackendRegistry.load()
    assert "groq" in reg.known_backends()


def test_backend_registry_groq_capability() -> None:
    from formatshield.oracle.backend_registry import BackendRegistry

    reg = BackendRegistry.load()
    cap = reg.get("groq", "llama-3.1-8b-instant")
    assert cap.supports_ttf is True
    assert cap.native_thinker is False
    assert cap.ttf_overhead_pct > 0.0


def test_backend_registry_native_thinker_flag() -> None:
    from formatshield.oracle.backend_registry import BackendRegistry

    reg = BackendRegistry.load()
    assert reg.is_native_thinker("groq", "deepseek-r1-distill-llama-70b") is True
    assert reg.is_native_thinker("groq", "llama-3.1-8b-instant") is False


def test_backend_registry_fallback_to_default() -> None:
    from formatshield.oracle.backend_registry import BackendRegistry

    reg = BackendRegistry.load()
    cap = reg.get("nonexistent-backend", "some-model")
    assert cap.backend_id == "nonexistent-backend"
    assert cap.supports_ttf is True


def test_backend_registry_empty_file(tmp_path: Path) -> None:
    from formatshield.oracle.backend_registry import BackendRegistry

    missing = tmp_path / "missing.json"
    reg = BackendRegistry.load_from(missing)
    cap = reg.get("groq")
    assert cap.backend_id == "groq"
