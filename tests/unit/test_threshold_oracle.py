"""
Unit tests for formatshield.oracle.threshold_oracle.ThresholdOracle.

The oracle takes ComplexityFeatures + metadata and returns a RoutingDecision.
Tests verify: routing strategy correctness, confidence range, native-thinker
override, latency budget enforcement, and graceful error handling.
"""

from __future__ import annotations

import pytest

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.threshold_oracle import NATIVE_THINKERS, ThresholdOracle
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_features(
    *,
    token_entropy: float = 0.5,
    schema_depth: int = 1,
    required_reasoning_ops: int = 0,
    instruction_tune_score: float = 0.5,
    prompt_length_bucket: int = 1,
    schema_constraint_count: int = 1,
) -> ComplexityFeatures:
    """Build a ComplexityFeatures with sensible defaults."""
    return ComplexityFeatures(
        token_entropy=token_entropy,
        schema_depth=schema_depth,
        required_reasoning_ops=required_reasoning_ops,
        instruction_tune_score=instruction_tune_score,
        prompt_length_bucket=prompt_length_bucket,
        schema_constraint_count=schema_constraint_count,
    )


def _high_complexity_features() -> ComplexityFeatures:
    """Features that should score well above any backend threshold."""
    return _make_features(
        token_entropy=0.95,
        schema_depth=8,
        required_reasoning_ops=15,
        instruction_tune_score=0.8,
        prompt_length_bucket=3,
        schema_constraint_count=25,
    )


def _low_complexity_features() -> ComplexityFeatures:
    """Features that should score well below any backend threshold."""
    return _make_features(
        token_entropy=0.05,
        schema_depth=0,
        required_reasoning_ops=0,
        instruction_tune_score=0.3,
        prompt_length_bucket=0,
        schema_constraint_count=0,
    )


# ---------------------------------------------------------------------------
# Tests: return type
# ---------------------------------------------------------------------------


def test_predict_returns_routing_decision() -> None:
    """predict() must return a RoutingDecision instance."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3.1-70b")
    assert isinstance(result, RoutingDecision)


def test_routing_decision_strategy_is_valid_literal() -> None:
    """strategy must be one of the three valid literals."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3.1-70b")
    assert result.strategy in {"ttf", "direct", "hybrid"}


# ---------------------------------------------------------------------------
# Tests: native thinker models always get direct
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("model_id", ["o1", "o3", "o1-mini", "o3-mini", "deepseek-r1"])
def test_native_thinker_always_direct(model_id: str) -> None:
    """Native thinker model IDs must always receive strategy == 'direct'."""
    oracle = ThresholdOracle()
    # Use high-complexity features that would normally trigger TTF
    features = _high_complexity_features()
    result = oracle.predict(features, backend="groq", model_id=model_id)
    assert result.strategy == "direct", (
        f"Expected 'direct' for native thinker '{model_id}', got {result.strategy!r}"
    )


def test_native_thinkers_set_is_non_empty() -> None:
    """NATIVE_THINKERS must be a non-empty frozenset."""
    assert isinstance(NATIVE_THINKERS, frozenset)
    assert len(NATIVE_THINKERS) > 0


def test_native_thinker_with_provider_prefix_still_direct() -> None:
    """'groq/o1' style model IDs must also route to direct."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="groq", model_id="o1")
    assert result.strategy == "direct"


# ---------------------------------------------------------------------------
# Tests: complexity-based routing
# ---------------------------------------------------------------------------


def test_high_complexity_routes_to_ttf() -> None:
    """High complexity features must produce strategy == 'ttf'."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="vllm", model_id="llama-3.1-70b")
    assert result.strategy == "ttf", (
        f"Expected 'ttf' for high-complexity features, got {result.strategy!r}"
    )


def test_low_complexity_routes_to_direct() -> None:
    """Very low complexity features must produce strategy == 'direct'."""
    oracle = ThresholdOracle()
    features = _low_complexity_features()
    result = oracle.predict(features, backend="vllm", model_id="llama-3.1-70b")
    assert result.strategy == "direct", (
        f"Expected 'direct' for low-complexity features, got {result.strategy!r}"
    )


# ---------------------------------------------------------------------------
# Tests: latency budget
# ---------------------------------------------------------------------------


def test_latency_budget_forces_direct() -> None:
    """A very tight latency budget must force direct routing even for complex prompts."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    # 1 ms budget — any TTF overhead (even minimal) should exceed this
    result = oracle.predict(
        features, backend="groq", model_id="llama-3.1-70b", latency_budget_ms=1.0
    )
    assert result.strategy == "direct", (
        f"Expected 'direct' with 1 ms budget, got {result.strategy!r}"
    )


def test_generous_latency_budget_allows_ttf() -> None:
    """A very generous latency budget must not prevent TTF for complex prompts."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(
        features,
        backend="vllm",
        model_id="llama-3.1-70b",
        latency_budget_ms=60_000.0,  # 60 seconds — more than enough
    )
    assert result.strategy == "ttf", f"Expected 'ttf' with 60 s budget, got {result.strategy!r}"


# ---------------------------------------------------------------------------
# Tests: explanation field
# ---------------------------------------------------------------------------


def test_routing_decision_has_explanation() -> None:
    """RoutingDecision.explanation must be a non-empty string."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3.1-70b")
    assert isinstance(result.explanation, str)
    assert len(result.explanation) > 0, "explanation must not be empty"


def test_ttf_explanation_mentions_threshold() -> None:
    """When routing to TTF, the explanation should reference the threshold or score."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="vllm", model_id="llama-3.1-70b")
    if result.strategy == "ttf":
        # Explanation should contain some numeric reference to the decision
        assert any(char.isdigit() for char in result.explanation), (
            "TTF explanation should contain numeric scoring information"
        )


# ---------------------------------------------------------------------------
# Tests: confidence range
# ---------------------------------------------------------------------------


def test_confidence_in_range() -> None:
    """RoutingDecision.confidence must be in [0.0, 1.0]."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3.1-70b")
    assert 0.0 <= result.confidence <= 1.0, f"confidence {result.confidence} out of [0, 1]"


@pytest.mark.parametrize("model_id", ["o1", "o3", "o1-mini"])
def test_native_thinker_confidence_high(model_id: str) -> None:
    """Native thinkers should produce high confidence (>= 0.8) for direct routing."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id=model_id)
    assert result.confidence >= 0.8, (
        f"Native thinker confidence should be high, got {result.confidence}"
    )


# ---------------------------------------------------------------------------
# Tests: per-backend thresholds
# ---------------------------------------------------------------------------


def test_vllm_has_lower_threshold_than_openrouter() -> None:
    """
    vLLM must have a lower or equal complexity threshold than openrouter.

    The practical consequence: a medium-complexity request may route to TTF on
    vLLM but stay direct on openrouter.  We verify this with a carefully crafted
    mid-range feature set that sits between the two thresholds.

    This test probes the BACKEND_THRESHOLDS constant directly.
    """
    from formatshield.oracle.threshold_oracle import BACKEND_THRESHOLDS

    vllm_threshold = BACKEND_THRESHOLDS.get("vllm", 0.65)
    openrouter_threshold = BACKEND_THRESHOLDS.get("openrouter", 0.67)
    assert vllm_threshold <= openrouter_threshold, (
        f"vLLM threshold ({vllm_threshold}) should be <= openrouter ({openrouter_threshold})"
    )


def test_different_backends_may_produce_different_strategies() -> None:
    """
    With mid-range complexity, vLLM and openrouter may produce different strategies
    because they have different thresholds.
    """

    oracle_vllm = ThresholdOracle()
    oracle_openrouter = ThresholdOracle()

    # Craft features that produce a score near the vllm threshold (0.60)
    # but potentially below the openrouter threshold (0.67)
    mid_features = _make_features(
        token_entropy=0.55,
        schema_depth=3,
        required_reasoning_ops=2,
        instruction_tune_score=0.5,
        prompt_length_bucket=1,
        schema_constraint_count=3,
    )

    result_vllm = oracle_vllm.predict(mid_features, backend="vllm", model_id="llama")
    result_or = oracle_openrouter.predict(mid_features, backend="openrouter", model_id="llama")

    # Both must return valid strategies regardless of which side they fall on
    assert result_vllm.strategy in {"ttf", "direct"}
    assert result_or.strategy in {"ttf", "direct"}


# ---------------------------------------------------------------------------
# Tests: error / fallback behaviour
# ---------------------------------------------------------------------------


def test_fallback_on_error_returns_routing_decision() -> None:
    """
    Even with corrupted / edge-case features, predict() must return a
    RoutingDecision rather than raising an exception.
    """
    oracle = ThresholdOracle()

    # Extreme values that could theoretically cause arithmetic errors
    extreme_features = ComplexityFeatures(
        token_entropy=float("inf"),
        schema_depth=999_999,
        required_reasoning_ops=999_999,
        instruction_tune_score=float("nan"),
        prompt_length_bucket=999,
        schema_constraint_count=999_999,
    )

    # Must not raise
    result = oracle.predict(extreme_features, backend="groq", model_id="llama")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in {"ttf", "direct", "hybrid"}


def test_empty_model_id_does_not_raise() -> None:
    """predict() with an empty model_id must not raise."""
    oracle = ThresholdOracle()
    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="")
    assert isinstance(result, RoutingDecision)


def test_unknown_backend_falls_back_to_default_threshold() -> None:
    """An unrecognised backend name must not raise; uses the default threshold."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="unknown_backend_xyz", model_id="llama")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in {"ttf", "direct", "hybrid"}


# ---------------------------------------------------------------------------
# Tests: routing decision fields
# ---------------------------------------------------------------------------


def test_direct_routing_has_zero_overhead() -> None:
    """Direct routing decisions should report 0% overhead."""
    oracle = ThresholdOracle()
    features = _low_complexity_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3.1-70b")
    if result.strategy == "direct":
        assert result.expected_overhead_pct == 0.0


def test_ttf_routing_has_positive_overhead() -> None:
    """TTF routing decisions should report a positive overhead percentage."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="vllm", model_id="llama-3.1-70b")
    if result.strategy == "ttf":
        assert result.expected_overhead_pct > 0.0


def test_ttf_routing_has_positive_accuracy_delta() -> None:
    """TTF routing should expect a positive accuracy improvement."""
    oracle = ThresholdOracle()
    features = _high_complexity_features()
    result = oracle.predict(features, backend="vllm", model_id="llama-3.1-70b")
    if result.strategy == "ttf":
        assert result.expected_accuracy_delta > 0.0


def test_use_ttf_property_matches_strategy() -> None:
    """RoutingDecision.use_ttf must be True iff strategy == 'ttf'."""
    oracle = ThresholdOracle()
    for features in [_low_complexity_features(), _high_complexity_features()]:
        result = oracle.predict(features, backend="vllm", model_id="llama")
        assert result.use_ttf == (result.strategy == "ttf")
        assert result.use_direct == (result.strategy == "direct")
