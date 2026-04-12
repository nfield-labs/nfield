"""Tests for formatshield.benchmark.cross_backend.CrossBackendBenchmark.

Also contains targeted coverage tests for small uncovered paths in
observability and langchain integrations.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from formatshield.benchmark.cross_backend import _DEFAULT_MODELS, CrossBackendBenchmark
from formatshield.observability.metrics import MetricsCollector, PrometheusMetrics
from formatshield.scorer.features import BenchmarkResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_result(backend: str = "groq", delta: float = 0.15) -> BenchmarkResult:
    return BenchmarkResult(
        task="gsm",
        backend=backend,
        model=f"{backend}/llama",
        direct_accuracy=0.6,
        ttf_accuracy=0.6 + delta,
        accuracy_delta=delta,
        direct_latency_ms=200.0,
        ttf_latency_ms=450.0,
        overhead_pct=125.0,
        complexity_score=0.8,
        failure_modes_detected=[],
    )


# ---------------------------------------------------------------------------
# __init__
# ---------------------------------------------------------------------------


def test_init_stores_backends() -> None:
    bench = CrossBackendBenchmark(backends=["groq", "ollama"])
    assert bench.backends == ["groq", "ollama"]


def test_init_uses_default_models_when_not_provided() -> None:
    bench = CrossBackendBenchmark(backends=["groq", "ollama"])
    assert bench.models["groq"] == _DEFAULT_MODELS["groq"]
    assert bench.models["ollama"] == _DEFAULT_MODELS["ollama"]


def test_init_uses_custom_models_when_provided() -> None:
    custom = {"groq": "groq/custom-model"}
    bench = CrossBackendBenchmark(backends=["groq"], models=custom)
    assert bench.models["groq"] == "groq/custom-model"


def test_init_unknown_backend_gets_default_suffix() -> None:
    bench = CrossBackendBenchmark(backends=["mybackend"])
    assert bench.models["mybackend"] == "mybackend/default"


# ---------------------------------------------------------------------------
# get_format_tax_table
# ---------------------------------------------------------------------------


def test_get_format_tax_table_returns_mean_per_backend() -> None:
    bench = CrossBackendBenchmark(backends=["groq", "ollama"])
    results = [
        _make_result(backend="groq", delta=0.10),
        _make_result(backend="groq", delta=0.20),
        _make_result(backend="ollama", delta=0.05),
    ]
    tax = bench.get_format_tax_table(results)

    assert abs(tax["groq"] - 0.15) < 1e-9
    assert abs(tax["ollama"] - 0.05) < 1e-9


def test_get_format_tax_table_empty_results() -> None:
    bench = CrossBackendBenchmark(backends=["groq"])
    tax = bench.get_format_tax_table([])
    assert tax == {}


def test_get_format_tax_table_negative_delta() -> None:
    bench = CrossBackendBenchmark(backends=["groq"])
    results = [_make_result(backend="groq", delta=-0.05)]
    tax = bench.get_format_tax_table(results)
    assert tax["groq"] == pytest.approx(-0.05)


def test_get_format_tax_table_single_backend_multiple_results() -> None:
    bench = CrossBackendBenchmark(backends=["groq"])
    results = [
        _make_result(backend="groq", delta=0.10),
        _make_result(backend="groq", delta=0.30),
        _make_result(backend="groq", delta=0.20),
    ]
    tax = bench.get_format_tax_table(results)
    assert tax["groq"] == pytest.approx(0.20)


# ---------------------------------------------------------------------------
# run — async, mocked harness
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_returns_flat_list_of_results() -> None:
    bench = CrossBackendBenchmark(backends=["groq", "ollama"])

    groq_results = [_make_result(backend="groq")]
    ollama_results = [_make_result(backend="ollama")]

    mock_harness = MagicMock()
    mock_harness.run_task_on_backend = AsyncMock(
        side_effect=lambda task, backend, model, quick: (
            groq_results if backend == "groq" else ollama_results
        )
    )

    with patch(
        "formatshield.benchmark.harness.BenchmarkHarness",
        return_value=mock_harness,
    ):
        results = await bench.run(task="gsm", quick=True)

    assert len(results) == 2
    backends = {r.backend for r in results}
    assert "groq" in backends
    assert "ollama" in backends


@pytest.mark.asyncio
async def test_run_handles_backend_failure_gracefully() -> None:
    bench = CrossBackendBenchmark(backends=["groq", "ollama"])

    async def _failing_backend(task, backend, model, quick):
        if backend == "ollama":
            raise RuntimeError("Connection refused")
        return [_make_result(backend="groq")]

    mock_harness = MagicMock()
    mock_harness.run_task_on_backend = AsyncMock(side_effect=_failing_backend)

    with patch(
        "formatshield.benchmark.harness.BenchmarkHarness",
        return_value=mock_harness,
    ):
        results = await bench.run(task="gsm", quick=True)

    # groq results should still be present; ollama failure returns []
    assert all(r.backend == "groq" for r in results)


@pytest.mark.asyncio
async def test_run_with_single_backend() -> None:
    bench = CrossBackendBenchmark(backends=["groq"])
    groq_results = [_make_result(backend="groq"), _make_result(backend="groq")]

    mock_harness = MagicMock()
    mock_harness.run_task_on_backend = AsyncMock(return_value=groq_results)

    with patch(
        "formatshield.benchmark.harness.BenchmarkHarness",
        return_value=mock_harness,
    ):
        results = await bench.run(task="gsm")

    assert len(results) == 2


@pytest.mark.asyncio
async def test_run_empty_backends_returns_empty() -> None:
    bench = CrossBackendBenchmark(backends=[])

    mock_harness = MagicMock()
    mock_harness.run_task_on_backend = AsyncMock(return_value=[])

    with patch(
        "formatshield.benchmark.harness.BenchmarkHarness",
        return_value=mock_harness,
    ):
        results = await bench.run(task="gsm")

    assert results == []


# ---------------------------------------------------------------------------
# train_oracle — async, mocked ThresholdOracle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_train_oracle_returns_oracle_instance() -> None:
    bench = CrossBackendBenchmark(backends=["groq"])

    # Create enough results to satisfy the 10-row minimum
    results = [
        _make_result(backend="groq", delta=(0.1 if i % 2 == 0 else -0.05)) for i in range(15)
    ]

    mock_oracle = MagicMock()
    with patch(
        "formatshield.oracle.threshold_oracle.ThresholdOracle.from_benchmark_data",
        return_value=mock_oracle,
    ):
        oracle = await bench.train_oracle(results)

    assert oracle is mock_oracle


# ---------------------------------------------------------------------------
# PrometheusMetrics — uncovered delegation paths
# ---------------------------------------------------------------------------


def test_prometheus_metrics_record_schema_validation_failure() -> None:
    prom = PrometheusMetrics()
    # Should not raise and should delegate to MetricsCollector
    prom.record_schema_validation_failure()
    summary = prom._collector.get_summary()
    assert summary["schema_validation_failures"] >= 1


def test_prometheus_metrics_record_fallback() -> None:
    prom = PrometheusMetrics()
    prom.record_fallback()
    summary = prom._collector.get_summary()
    assert summary["fallback_count"] >= 1


def test_prometheus_metrics_uses_provided_collector() -> None:
    collector = MetricsCollector()
    prom = PrometheusMetrics(collector=collector)
    prom.record_routing(strategy="ttf", backend="groq")
    assert prom._collector is collector


# ---------------------------------------------------------------------------
# StructuredLogger — exc_info path (line 73)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# ttf/prompts — build_extraction_think_prompt (line 184)
# ---------------------------------------------------------------------------


def test_build_extraction_think_prompt_returns_string() -> None:
    from formatshield.ttf.prompts import build_extraction_think_prompt

    result = build_extraction_think_prompt("Extract entities from this text.")
    assert isinstance(result, str)
    assert len(result) > 0
    # The prompt should embed the original text
    assert "Extract entities from this text." in result


# ---------------------------------------------------------------------------
# streaming engine — output_parts fallback path (line 149)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_streaming_engine_collect_uses_output_parts_when_no_complete_content() -> None:
    """When the complete event has no content, the engine joins output_parts (line 149)."""
    from formatshield.scorer.features import StreamEvent
    from formatshield.streaming.engine import StreamingEngine

    async def _stream():
        yield StreamEvent(type="output", token="hello ", backend="mock")
        yield StreamEvent(type="output", token="world", backend="mock")
        # complete event with no content — forces the else branch on line 149
        yield StreamEvent(type="complete", content=None, backend="mock")

    engine = StreamingEngine()
    _thinking, output = await engine.collect(_stream())
    assert output == "hello world"


def test_structured_logger_log_error_with_exception_at_debug_level() -> None:
    """At DEBUG level, log_error passes exc_info=True to the formatter's exc_info branch."""
    import io

    from formatshield.observability.logger import StructuredLogger

    # Use a StringIO handler to capture output without printing to stdout
    stream = io.StringIO()
    logger_obj = StructuredLogger(name="formatshield.test.exc_info_coverage", level="DEBUG")
    # Replace handler stream with our buffer
    for handler in logger_obj._logger.handlers:
        if hasattr(handler, "stream"):
            handler.stream = stream

    try:
        raise ValueError("coverage test error")
    except ValueError as exc:
        # DEBUG level + Exception argument → exc_info=True → line 73 is covered
        logger_obj.log_error(exc)

    output = stream.getvalue()
    assert output  # something was logged
