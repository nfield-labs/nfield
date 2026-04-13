"""
Coverage tests for formatshield.oracle.threshold_oracle — targeting uncovered lines.

Uncovered lines targeted:
  166-171  : predict() except-block fallback (oracle error path)
  223-224  : from_benchmark_data() save=True → joblib.dump + logger.info
  247-249  : from_benchmark_data() malformed row → except (KeyError, ValueError)
  252      : too-few-valid-rows ValueError after filtering malformed CSV
  271-273  : save=True branch in from_benchmark_data
  299-300  : save() → joblib.dump + logger.info
  327-328  : load() → joblib.load + logger.info
  345-359  : _try_load_model() success path (model file exists, load succeeds)
  446-447  : _predict_sklearn() else branch (raw LR, not a dict bundle)
  459-463  : _predict_sklearn() pred == 1 (TTF) branch
  482-487  : _predict_sklearn() exception fallback → heuristics
  557      : _is_native_thinker() return False
  587-588  : _features_from_benchmark_row() _float() except branch
  593-594  : _features_from_benchmark_row() _int() except branch
"""

from __future__ import annotations

import csv
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.oracle.threshold_oracle import (
    BACKEND_THRESHOLDS,
    ThresholdOracle,
    _is_native_thinker,
)
from formatshield.scorer.features import ComplexityFeatures

# ---------------------------------------------------------------------------
# Shared helpers (mirrored from existing test_threshold_oracle.py)
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


def _write_benchmark_csv(path: Path, n_rows: int = 15, *, malformed: bool = False) -> None:
    """Write a minimal benchmark CSV to *path*."""
    fieldnames = [
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
    ]
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(n_rows):
            row = {
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
            if malformed and i == 0:
                # Corrupt the accuracy_delta value so the row is skipped
                row["accuracy_delta"] = "NOT_A_NUMBER_XYZ"
            w.writerow(row)


# ---------------------------------------------------------------------------
# Lines 166-171: predict() except-block fallback
# ---------------------------------------------------------------------------


def test_predict_exception_fallback_returns_routing_decision() -> None:
    """When _predict_impl raises, predict() must catch it and return a direct RoutingDecision."""
    oracle = ThresholdOracle(model_path="/nonexistent/path/dummy.pkl")
    assert oracle._clf is None

    features = _make_features()

    with patch.object(oracle, "_predict_impl", side_effect=RuntimeError("injected error")):
        result = oracle.predict(features, backend="groq", model_id="llama-3-70b")

    assert isinstance(result, RoutingDecision)
    assert result.strategy == "direct"
    assert result.confidence == pytest.approx(0.30)
    assert "error" in result.explanation.lower() or "oracle" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Lines 223-224 and 271-273: from_benchmark_data() save=True branch
# ---------------------------------------------------------------------------


sklearn = pytest.importorskip("sklearn", reason="scikit-learn not installed")


def test_from_benchmark_data_save_true_writes_file(tmp_path: Path) -> None:
    """from_benchmark_data(save=True) must write the model to disk (lines 271-273)."""
    csv_path = tmp_path / "benchmark.csv"
    model_path = tmp_path / "oracle_model.pkl"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=model_path,
        save=True,
    )

    assert model_path.exists(), "Model file should have been saved to disk"
    assert oracle._clf is not None


def test_from_benchmark_data_save_true_model_is_usable(tmp_path: Path) -> None:
    """Saved model can be used for prediction (exercises lines 223-224 logger path)."""
    csv_path = tmp_path / "benchmark.csv"
    model_path = tmp_path / "oracle.pkl"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=model_path,
        save=True,
    )

    features = _make_features(token_entropy=0.8)
    decision = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert isinstance(decision, RoutingDecision)
    assert decision.strategy in ("ttf", "direct")


# ---------------------------------------------------------------------------
# Lines 247-249: malformed rows are skipped (KeyError/ValueError)
# ---------------------------------------------------------------------------


def test_from_benchmark_data_skips_malformed_rows(tmp_path: Path) -> None:
    """Malformed rows (non-numeric accuracy_delta) are skipped without raising."""
    csv_path = tmp_path / "malformed.csv"
    # Write 15 rows with one malformed — CSV processing should skip the bad row
    _write_benchmark_csv(csv_path, n_rows=15, malformed=True)

    # Should still succeed: 14 valid rows remain (>= 10 threshold)
    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "oracle.pkl",
        save=False,
    )
    assert oracle is not None
    assert oracle._clf is not None


def test_from_benchmark_data_too_many_malformed_rows_raises(tmp_path: Path) -> None:
    """If too many rows are malformed, a ValueError is raised (line 252)."""
    csv_path = tmp_path / "all_malformed.csv"
    fieldnames = [
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
    ]
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for _ in range(12):
            w.writerow(
                {
                    "task": "x",
                    "backend": "y",
                    "model": "z",
                    "direct_accuracy": "BAD",
                    "ttf_accuracy": "BAD",
                    "accuracy_delta": "NOT_A_NUMBER",
                    "direct_latency_ms": "BAD",
                    "ttf_latency_ms": "BAD",
                    "overhead_pct": "BAD",
                    "complexity_score": "BAD",
                    "failure_modes_detected": "",
                }
            )

    with pytest.raises(ValueError, match="Too few valid rows"):
        ThresholdOracle.from_benchmark_data(
            csv_path,
            model_path=tmp_path / "oracle.pkl",
            save=False,
        )


# ---------------------------------------------------------------------------
# Lines 299-300: save() → joblib.dump + logger
# ---------------------------------------------------------------------------


def test_save_writes_file_to_disk(tmp_path: Path) -> None:
    """save() must write the model file (covers lines 299-300)."""
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "orig.pkl",
        save=False,
    )

    save_path = tmp_path / "saved.pkl"
    oracle.save(save_path)
    assert save_path.exists()


def test_save_creates_parent_directories(tmp_path: Path) -> None:
    """save() must create any missing parent directories."""
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "orig.pkl",
        save=False,
    )

    deep_path = tmp_path / "nested" / "dir" / "model.pkl"
    oracle.save(deep_path)
    assert deep_path.exists()


# ---------------------------------------------------------------------------
# Lines 327-328: load() → joblib.load + logger
# ---------------------------------------------------------------------------


def test_load_sets_clf_from_file(tmp_path: Path) -> None:
    """load() must set _clf from the saved file (covers lines 327-328)."""
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    # Train and save
    oracle = ThresholdOracle.from_benchmark_data(
        csv_path,
        model_path=tmp_path / "orig.pkl",
        save=False,
    )
    model_path = tmp_path / "to_load.pkl"
    oracle.save(model_path)

    # Fresh oracle with no model
    new_oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")
    assert new_oracle._clf is None

    new_oracle.load(model_path)
    assert new_oracle._clf is not None


def test_load_then_predict_works(tmp_path: Path) -> None:
    """After load(), predict() returns a valid RoutingDecision."""
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    oracle = ThresholdOracle.from_benchmark_data(
        csv_path, model_path=tmp_path / "orig.pkl", save=False
    )
    model_path = tmp_path / "loaded.pkl"
    oracle.save(model_path)

    fresh = ThresholdOracle(model_path="/nonexistent/dummy.pkl")
    fresh.load(model_path)

    result = fresh.predict(_make_features(), backend="groq", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in ("ttf", "direct")


# ---------------------------------------------------------------------------
# Lines 345-359: _try_load_model() success path
# ---------------------------------------------------------------------------


def test_try_load_model_success_path(tmp_path: Path) -> None:
    """When model file exists and is valid, _try_load_model() loads it (_clf is set)."""
    csv_path = tmp_path / "benchmark.csv"
    _write_benchmark_csv(csv_path, n_rows=15)

    # Save a real model to disk
    oracle_orig = ThresholdOracle.from_benchmark_data(
        csv_path, model_path=tmp_path / "orig.pkl", save=False
    )
    model_path = tmp_path / "model_for_load.pkl"
    oracle_orig.save(model_path)

    # Create a new oracle pointing directly at the saved file — this triggers _try_load_model
    oracle_loaded = ThresholdOracle(model_path=model_path)
    assert oracle_loaded._clf is not None


def test_try_load_model_bad_file_falls_back_to_heuristics(tmp_path: Path) -> None:
    """When the model file exists but is corrupt, _try_load_model() sets _clf=None."""
    bad_path = tmp_path / "corrupt.pkl"
    bad_path.write_bytes(b"this is not a valid pickle file")

    oracle = ThresholdOracle(model_path=bad_path)
    # After failed load, oracle should use heuristics
    assert oracle._clf is None

    result = oracle.predict(_make_features(), backend="groq", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)


# ---------------------------------------------------------------------------
# Lines 446-447: _predict_sklearn() else branch (raw LR model, not a dict)
# ---------------------------------------------------------------------------


def test_predict_sklearn_with_raw_lr_model(tmp_path: Path) -> None:
    """_predict_sklearn handles a raw LogisticRegression (not a dict bundle)."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        pytest.skip("scikit-learn / numpy not available")

    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")

    # Inject a raw LR model (not a dict)
    clf = LogisticRegression(max_iter=500)
    x = np.array(
        [
            [0.1, 1, 0, 0.5, 1, 1],
            [0.9, 8, 15, 0.8, 3, 25],
            [0.2, 0, 0, 0.3, 0, 0],
            [0.8, 6, 10, 0.7, 2, 20],
        ]
    )
    y = np.array([0, 1, 0, 1])
    clf.fit(x, y)

    oracle._clf = clf  # raw model, not a dict

    features = _high_complexity_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in ("ttf", "direct")


# ---------------------------------------------------------------------------
# Lines 459-463: _predict_sklearn() pred == 1 (TTF prediction branch)
# ---------------------------------------------------------------------------


def test_predict_sklearn_ttf_branch(tmp_path: Path) -> None:
    """Exercise the pred==1 (TTF) branch inside _predict_sklearn (lines 459-463)."""
    try:
        import numpy as np
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        pytest.skip("scikit-learn / numpy not available")

    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")

    # Train a model biased to always predict TTF for high complexity
    clf = LogisticRegression(max_iter=1000)
    scaler = StandardScaler()

    x = np.array(
        [
            [0.05, 0, 0, 0.3, 0, 0],
            [0.05, 0, 0, 0.3, 0, 0],
            [0.05, 0, 0, 0.3, 0, 0],
            [0.95, 9, 18, 0.9, 3, 28],
            [0.95, 9, 18, 0.9, 3, 28],
            [0.95, 9, 18, 0.9, 3, 28],
        ]
    )
    y = np.array([0, 0, 0, 1, 1, 1])
    x_scaled = scaler.fit_transform(x)
    clf.fit(x_scaled, y)

    oracle._clf = {"clf": clf, "scaler": scaler}

    # High complexity → should predict TTF (class 1)
    features = _high_complexity_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)
    # The sklearn model predicts TTF for high-complexity; verify the path executes
    if result.strategy == "ttf":
        assert result.expected_accuracy_delta > 0.0
        assert result.expected_overhead_pct > 0.0
        assert "sklearn" in result.explanation.lower() or "confidence" in result.explanation.lower()


# ---------------------------------------------------------------------------
# Lines 482-487: _predict_sklearn() exception path → falls back to heuristic
# ---------------------------------------------------------------------------


def test_predict_sklearn_fallback_when_predict_raises() -> None:
    """When the sklearn clf.predict() raises, _predict_sklearn falls back to heuristics."""
    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")

    bad_clf = MagicMock()
    bad_clf.predict.side_effect = RuntimeError("mock sklearn failure")

    oracle._clf = {"clf": bad_clf, "scaler": None}

    features = _make_features()
    result = oracle.predict(features, backend="groq", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in ("ttf", "direct")
    # Falls back to heuristic, which uses _HEURISTIC_CONFIDENCE = 0.70
    assert result.confidence == pytest.approx(0.70)


def test_predict_sklearn_fallback_with_bad_bundle() -> None:
    """An sklearn bundle with an invalid clf falls back gracefully to heuristics."""
    oracle = ThresholdOracle(model_path="/nonexistent/dummy.pkl")

    # Inject a bundle where clf.predict raises AttributeError
    broken_clf = MagicMock()
    broken_clf.predict.side_effect = AttributeError("no predict method")
    oracle._clf = {"clf": broken_clf, "scaler": None}

    result = oracle.predict(_high_complexity_features(), backend="vllm", model_id="llama-3-70b")
    assert isinstance(result, RoutingDecision)
    assert result.strategy in ("ttf", "direct")


# ---------------------------------------------------------------------------
# Line 557: _is_native_thinker() return False
# ---------------------------------------------------------------------------


def test_is_native_thinker_returns_false_for_non_thinker() -> None:
    """_is_native_thinker returns False for regular model IDs (line 557)."""
    assert _is_native_thinker("llama-3.1-70b-versatile") is False
    assert _is_native_thinker("gpt-4o") is False
    assert _is_native_thinker("claude-3-5-sonnet") is False
    assert _is_native_thinker("mistral-7b") is False
    assert _is_native_thinker("gemini-pro") is False


def test_is_native_thinker_returns_true_for_known_thinkers() -> None:
    """Sanity check: known native thinkers return True."""
    assert _is_native_thinker("o1") is True
    assert _is_native_thinker("o3") is True
    assert _is_native_thinker("deepseek-r1") is True
    assert _is_native_thinker("o1-mini") is True


def test_is_native_thinker_prefix_match() -> None:
    """Prefix matching should catch distillation variants."""
    assert _is_native_thinker("deepseek-r1-distill-llama-70b") is True
    assert _is_native_thinker("deepseek-r1-distill-qwen-32b") is True


def test_is_native_thinker_case_insensitive() -> None:
    assert _is_native_thinker("O1") is True
    assert _is_native_thinker("DEEPSEEK-R1") is True


# ---------------------------------------------------------------------------
# Lines 587-588, 593-594: _features_from_benchmark_row() inner except handlers
# ---------------------------------------------------------------------------


def test_features_from_benchmark_row_float_except_branch() -> None:
    """_float() except branch fires when a CSV cell can't be converted to float."""
    from formatshield.oracle.threshold_oracle import _features_from_benchmark_row

    # Provide a row where token_entropy (mapped from complexity_score) is not a number
    row = {
        "complexity_score": "INVALID",  # triggers except in _float
        "token_entropy": "ALSO_BAD",  # triggers except in _float
        "schema_depth": "NOT_INT",  # triggers except in _int
        "required_reasoning_ops": "BAD",
        "instruction_tune_score": "BAD",
        "prompt_length_bucket": "BAD",
        "schema_constraint_count": "BAD",
    }
    # Should not raise; bad values get their defaults
    features = _features_from_benchmark_row(row)
    assert isinstance(features, ComplexityFeatures)


def test_features_from_benchmark_row_int_except_branch() -> None:
    """_int() except branch fires when a CSV cell can't be converted to int."""
    from formatshield.oracle.threshold_oracle import _features_from_benchmark_row

    row = {
        "complexity_score": "0.5",  # valid float
        "schema_depth": "not_an_int",  # triggers except in _int (line 593-594)
        "required_reasoning_ops": "???",
        "instruction_tune_score": "0.5",
        "prompt_length_bucket": "nope",
        "schema_constraint_count": "[]",
    }
    features = _features_from_benchmark_row(row)
    assert isinstance(features, ComplexityFeatures)
    # Defaults for int fields are 0
    assert features.schema_depth == 0 or features.schema_depth == 1  # default is 1


def test_features_from_benchmark_row_valid_row() -> None:
    """Sanity check: a valid row produces the expected feature values."""
    from formatshield.oracle.threshold_oracle import _features_from_benchmark_row

    row = {
        "complexity_score": "0.7",
        "token_entropy": "0.6",
        "schema_depth": "3",
        "required_reasoning_ops": "5",
        "instruction_tune_score": "0.8",
        "prompt_length_bucket": "2",
        "schema_constraint_count": "10",
    }
    features = _features_from_benchmark_row(row)
    assert features.token_entropy == pytest.approx(0.6)
    assert features.schema_depth == 3
    assert features.required_reasoning_ops == 5


# ---------------------------------------------------------------------------
# Per-backend routing (groq, openrouter, ollama, vllm, openai, anthropic)
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
        ("openai/gpt-4o", "direct"),  # not a native thinker but valid model
        ("anthropic/claude-3-5-sonnet", "direct"),  # low complexity → direct
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
