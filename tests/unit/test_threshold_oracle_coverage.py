"""
Coverage tests for formatshield.oracle.threshold_oracle — targeting uncovered lines.

Uncovered lines targeted:
  predict() except-block fallback (oracle error path)
  _is_native_thinker() return False
  per-backend routing (groq, openrouter, ollama, vllm)
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.threshold_oracle import (
    BACKEND_THRESHOLDS,
    ThresholdOracle,
    _is_native_thinker,
)
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Shared helpers
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
    return ComplexityFeatures(
        token_entropy=token_entropy,
        schema_depth=schema_depth,
        required_reasoning_ops=required_reasoning_ops,
        instruction_tune_score=instruction_tune_score,
        prompt_length_bucket=prompt_length_bucket,
        schema_constraint_count=schema_constraint_count,
    )


def _high_complexity_features() -> ComplexityFeatures:
    return _make_features(
        token_entropy=0.95,
        schema_depth=8,
        required_reasoning_ops=15,
        instruction_tune_score=0.8,
        prompt_length_bucket=3,
        schema_constraint_count=25,
    )


def _low_complexity_features() -> ComplexityFeatures:
    return _make_features(
        token_entropy=0.05,
        schema_depth=0,
        required_reasoning_ops=0,
        instruction_tune_score=0.3,
        prompt_length_bucket=0,
        schema_constraint_count=0,
    )


# ---------------------------------------------------------------------------
# predict() except-block fallback
# ---------------------------------------------------------------------------


def test_predict_exception_fallback_returns_routing_decision() -> None:
    """When _predict_impl raises, predict() must catch it and return a direct RoutingDecision."""
    oracle = ThresholdOracle(model_path="/nonexistent/path/dummy.pkl")

    features = _make_features()

    with patch.object(oracle, "_predict_impl", side_effect=RuntimeError("injected error")):
        result = oracle.predict(features, backend="groq", model_id="llama-3-70b")

    assert isinstance(result, RoutingDecision)
    assert result.strategy == "direct"
    assert result.confidence == pytest.approx(0.30)
    assert "error" in result.explanation.lower() or "oracle" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Deprecation stubs
# ---------------------------------------------------------------------------


def test_from_benchmark_data_raises_not_implemented() -> None:
    """from_benchmark_data() must raise NotImplementedError in v0.3."""
    with pytest.warns(DeprecationWarning, match="removed in v0.3"):
        with pytest.raises(NotImplementedError):
            ThresholdOracle.from_benchmark_data()  # type: ignore[call-arg]


def test_save_raises_not_implemented() -> None:
    """save() must raise NotImplementedError in v0.3."""
    oracle = ThresholdOracle()
    with pytest.warns(DeprecationWarning, match="removed in v0.3"):
        with pytest.raises(NotImplementedError):
            oracle.save()  # type: ignore[call-arg]


def test_load_raises_not_implemented() -> None:
    """load() must raise NotImplementedError in v0.3."""
    oracle = ThresholdOracle()
    with pytest.warns(DeprecationWarning, match="removed in v0.3"):
        with pytest.raises(NotImplementedError):
            oracle.load()  # type: ignore[call-arg]


# ---------------------------------------------------------------------------
# _is_native_thinker()
# ---------------------------------------------------------------------------


def test_is_native_thinker_returns_false_for_non_thinker() -> None:
    assert _is_native_thinker("llama-3.1-70b-versatile") is False
    assert _is_native_thinker("gpt-4o") is False
    assert _is_native_thinker("claude-3-5-sonnet") is False
    assert _is_native_thinker("mistral-7b") is False
    assert _is_native_thinker("gemini-pro") is False


def test_is_native_thinker_returns_true_for_known_thinkers() -> None:
    assert _is_native_thinker("o1") is True
    assert _is_native_thinker("o3") is True
    assert _is_native_thinker("deepseek-r1") is True
    assert _is_native_thinker("o1-mini") is True


def test_is_native_thinker_prefix_match() -> None:
    assert _is_native_thinker("deepseek-r1-distill-llama-70b") is True
    assert _is_native_thinker("deepseek-r1-distill-qwen-32b") is True


def test_is_native_thinker_case_insensitive() -> None:
    assert _is_native_thinker("O1") is True
    assert _is_native_thinker("DEEPSEEK-R1") is True


# ---------------------------------------------------------------------------
# Per-backend routing (groq, openrouter, ollama, vllm)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("backend", ["groq", "openrouter", "ollama", "vllm"])
def test_high_complexity_ttf_for_all_backends(backend: str) -> None:
    """High-complexity features should produce TTF for all known low-threshold backends."""
    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")
    result = oracle.predict(
        _high_complexity_features(),
        backend=backend,
        model_id="llama-3.1-70b",
    )
    assert isinstance(result, RoutingDecision)
    assert result.strategy == "ttf"


@pytest.mark.parametrize("backend", ["groq", "openrouter", "ollama", "vllm"])
def test_low_complexity_direct_for_all_backends(backend: str) -> None:
    """Low-complexity features should produce direct for all known backends."""
    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")
    result = oracle.predict(
        _low_complexity_features(),
        backend=backend,
        model_id="llama-3.1-70b",
    )
    assert isinstance(result, RoutingDecision)
    assert result.strategy == "direct"


@pytest.mark.parametrize(
    ("model_id", "expected_strategy"),
    [
        ("openai/gpt-4o", "direct"),
        ("anthropic/claude-3-5-sonnet", "direct"),
    ],
)
def test_various_model_ids_low_complexity(model_id: str, expected_strategy: str) -> None:
    """Low-complexity requests route to direct regardless of model family."""
    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")
    result = oracle.predict(
        _low_complexity_features(),
        backend="openrouter",
        model_id=model_id,
    )
    assert result.strategy == expected_strategy


def test_backend_thresholds_all_have_valid_values() -> None:
    """All backend threshold values must be in (0, 1)."""
    for backend_name, threshold in BACKEND_THRESHOLDS.items():
        assert 0.0 < threshold < 1.0, (
            f"Threshold for '{backend_name}' must be in (0, 1), got {threshold}"
        )
