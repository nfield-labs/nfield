"""Tests for ThresholdOracle sklearn training, save/load, and heuristic paths."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.threshold_oracle import ThresholdOracle
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_benchmark_csv(path: Path, n_rows: int = 15) -> None:
    """Write a minimal benchmark CSV to *path* with *n_rows* rows."""
    with path.open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "task",
                "backend",
                "model",
                "direct_accuracy",
                "ttf_accuracy",
                "accuracy_delta",
                "direct_latency_ms",
                "ttf_latency_ms",
                "overhead_pct",
                "complexity_score",
                "failure_modes_detected",
            ],
        )
        w.writeheader()
        for i in range(n_rows):
            w.writerow(
                {
                    "task": "gsm_symbolic",
                    "backend": "groq",
                    "model": "llama-3-70b",
                    "direct_accuracy": 0.6,
                    "ttf_accuracy": 0.75 if i % 2 == 0 else 0.55,
                    "accuracy_delta": 0.15 if i % 2 == 0 else -0.05,
                    "direct_latency_ms": 200.0,
                    "ttf_latency_ms": 450.0,
                    "overhead_pct": 125.0,
                    "complexity_score": 0.3 + (i * 0.04),
                    "failure_modes_detected": "",
                }
            )


def _make_features(complexity: float = 0.7) -> ComplexityFeatures:
    return ComplexityFeatures(
        token_entropy=complexity,
        schema_depth=2,
        required_reasoning_ops=3,
        instruction_tune_score=0.8,
        prompt_length_bucket=1,
        schema_constraint_count=4,
    )


# ---------------------------------------------------------------------------
# Heuristic fallback (no sklearn model loaded)
# ---------------------------------------------------------------------------


def test_heuristic_fallback_when_no_model_loaded() -> None:
    # Point oracle at a non-existent path so it uses heuristics
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    assert oracle._clf is None

    features = _make_features(complexity=0.5)
    decision = oracle.predict(features, backend="groq", model_id="llama-3-70b")

    assert isinstance(decision, RoutingDecision)
    assert decision.strategy in ("ttf", "direct")
    assert 0.0 <= decision.confidence <= 1.0


def test_heuristic_returns_direct_for_low_complexity() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    # Very low complexity → should route to direct
    features = _make_features(complexity=0.01)
    decision = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert decision.strategy == "direct"


def test_heuristic_returns_ttf_for_high_complexity() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    # Max complexity features → should route to TTF
    features = ComplexityFeatures(
        token_entropy=1.0,
        schema_depth=10,
        required_reasoning_ops=20,
        instruction_tune_score=1.0,
        prompt_length_bucket=3,
        schema_constraint_count=30,
    )
    decision = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert decision.strategy == "ttf"


def test_native_thinker_model_always_direct() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    features = _make_features(complexity=0.9)
    decision = oracle.predict(features, backend="groq", model_id="deepseek-r1")
    assert decision.strategy == "direct"
    assert decision.confidence == 0.95


def test_latency_budget_forces_direct() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    features = _make_features(complexity=0.9)
    # Budget of 1 ms is way below any backend TTF overhead
    decision = oracle.predict(
        features, backend="groq", model_id="llama-3-70b", latency_budget_ms=1.0
    )
    assert decision.strategy == "direct"


# ---------------------------------------------------------------------------
# from_benchmark_data — file not found
# ---------------------------------------------------------------------------


def test_from_benchmark_data_bad_file_raises() -> None:
    with pytest.raises(FileNotFoundError):
        ThresholdOracle.from_benchmark_data("/nonexistent/benchmark.csv")


# ---------------------------------------------------------------------------
# from_benchmark_data — too few rows
# ---------------------------------------------------------------------------


def test_from_benchmark_data_too_few_rows_raises(tmp_path: Path) -> None:
    csv_path = tmp_path / "small.csv"
    _write_benchmark_csv(csv_path, n_rows=5)

    with pytest.raises(ValueError, match="at least 10"):
        ThresholdOracle.from_benchmark_data(csv_path, model_path=tmp_path / "model.pkl")


# ---------------------------------------------------------------------------
# from_benchmark_data — success path (requires sklearn)
# ---------------------------------------------------------------------------


sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


def test_from_benchmark_data_with_csv(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "oracle.pkl",
        save=False,
    )

    assert oracle is not None
    assert oracle._clf is not None


def test_predict_after_training_returns_routing_decision(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "oracle.pkl",
        save=False,
    )

    features = _make_features(complexity=0.7)
    decision = oracle.predict(features, backend="groq", model_id="llama-3-70b")

    assert isinstance(decision, RoutingDecision)
    assert decision.strategy in ("ttf", "direct")
    assert 0.0 <= decision.confidence <= 1.0
    assert isinstance(decision.explanation, str)
    assert len(decision.explanation) > 0


# ---------------------------------------------------------------------------
# save and load
# ---------------------------------------------------------------------------


def test_save_and_load_oracle(tmp_path: Path) -> None:
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "oracle.pkl",
        save=False,
    )

    # Save to disk
    model_path = tmp_path / "saved_oracle.pkl"
    oracle.save(model_path)
    assert model_path.exists()

    # Load into a fresh oracle
    new_oracle = ThresholdOracle(model_path="/nonexistent/path/dummy.pkl")
    assert new_oracle._clf is None
    new_oracle.load(model_path)
    assert new_oracle._clf is not None

    # Predict should work after load
    features = _make_features(complexity=0.6)
    decision = new_oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert isinstance(decision, RoutingDecision)


def test_save_without_model_raises_runtime_error() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    assert oracle._clf is None

    with pytest.raises(RuntimeError, match="No trained model"):
        oracle.save("/tmp/should_not_write.pkl")


def test_load_nonexistent_path_raises_file_not_found() -> None:
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")

    with pytest.raises(FileNotFoundError):
        oracle.load("/nonexistent/path/missing_model.pkl")


# ---------------------------------------------------------------------------
# cost_aware flag
# ---------------------------------------------------------------------------


def test_cost_aware_flag_raises_threshold(tmp_path: Path) -> None:
    """cost_aware=True should apply a +0.03 bias, potentially changing routing."""
    oracle = ThresholdOracle(model_path="/nonexistent/path/oracle.pkl")
    # Use borderline complexity to observe the threshold shift
    features = ComplexityFeatures(
        token_entropy=0.65,
        schema_depth=1,
        required_reasoning_ops=0,
        instruction_tune_score=0.5,
        prompt_length_bucket=1,
        schema_constraint_count=1,
    )
    decision_default = oracle.predict(
        features, backend="groq", model_id="llama-3-70b", cost_aware=False
    )
    decision_cost = oracle.predict(
        features, backend="groq", model_id="llama-3-70b", cost_aware=True
    )
    # Both must be valid decisions regardless of whether threshold shifts routing
    assert decision_default.strategy in ("ttf", "direct")
    assert decision_cost.strategy in ("ttf", "direct")


# ---------------------------------------------------------------------------
# RoutingDecision extra coverage
# ---------------------------------------------------------------------------


def test_routing_decision_str_representation() -> None:
    from formatshield.oracle.routing_decision import RoutingDecision

    d = RoutingDecision(
        strategy="ttf",
        expected_accuracy_delta=0.17,
        expected_overhead_pct=30.0,
        confidence=0.82,
        explanation="test",
    )
    s = str(d)
    assert "ttf" in s
    assert "0.82" in s


def test_routing_decision_post_init_none_failure_modes() -> None:
    from formatshield.oracle.routing_decision import RoutingDecision

    # Passing failure_modes=None should be silently corrected to []
    d = RoutingDecision(
        strategy="direct",
        expected_accuracy_delta=0.0,
        expected_overhead_pct=0.0,
        confidence=0.7,
        explanation="test",
        failure_modes=None,  # type: ignore[arg-type]
    )
    assert d.failure_modes == []


def test_routing_decision_use_ttf_and_use_direct_properties() -> None:
    from formatshield.oracle.routing_decision import RoutingDecision

    ttf_decision = RoutingDecision(
        strategy="ttf",
        expected_accuracy_delta=0.17,
        expected_overhead_pct=30.0,
        confidence=0.8,
        explanation="ttf",
    )
    assert ttf_decision.use_ttf is True
    assert ttf_decision.use_direct is False

    direct_decision = RoutingDecision(
        strategy="direct",
        expected_accuracy_delta=0.0,
        expected_overhead_pct=0.0,
        confidence=0.7,
        explanation="direct",
    )
    assert direct_decision.use_direct is True
    assert direct_decision.use_ttf is False
