"""
Unit tests for formatshield.observability.logger and
formatshield.observability.metrics.

Tests StructuredLogger and MetricsCollector behaviour without
requiring any external services or API keys.
"""

from __future__ import annotations

import pytest

from formatshield.observability.logger import StructuredLogger
from formatshield.observability.metrics import MetricsCollector, PrometheusMetrics

# ===========================================================================
# StructuredLogger tests
# ===========================================================================


class TestStructuredLoggerInit:
    def test_default_instantiation_does_not_raise(self) -> None:
        """StructuredLogger() must construct without raising."""
        logger = StructuredLogger()
        assert logger is not None

    def test_custom_name_does_not_raise(self) -> None:
        """StructuredLogger with a custom name must construct without raising."""
        logger = StructuredLogger(name="formatshield.test.custom")
        assert logger is not None

    def test_debug_level_does_not_raise(self) -> None:
        """StructuredLogger with level='DEBUG' must construct without raising."""
        logger = StructuredLogger(level="DEBUG")
        assert logger is not None

    def test_warning_level_does_not_raise(self) -> None:
        """StructuredLogger with level='WARNING' must construct without raising."""
        logger = StructuredLogger(level="WARNING")
        assert logger is not None

    def test_info_level_does_not_raise(self) -> None:
        """StructuredLogger with level='INFO' must construct without raising."""
        logger = StructuredLogger(level="INFO")
        assert logger is not None


class TestStructuredLoggerLogGeneration:
    def test_log_generation_produces_output(self, capfd: pytest.CaptureFixture[str]) -> None:
        """log_generation() must emit at least one line of output to stdout."""
        logger = StructuredLogger(name="formatshield.test.gen", level="INFO")
        logger.log_generation(
            model="groq/llama3",
            backend="groq",
            route="ttf",
            latency_ms=342.1,
            schema_valid=True,
            fallback=False,
        )
        captured = capfd.readouterr()
        assert len(captured.out) > 0

    def test_log_generation_output_contains_model(self, capfd: pytest.CaptureFixture[str]) -> None:
        """log_generation() output must mention the model identifier."""
        logger = StructuredLogger(name="formatshield.test.gen2", level="INFO")
        logger.log_generation(
            model="my-special-model",
            backend="vllm",
            route="direct",
            latency_ms=100.0,
            schema_valid=True,
            fallback=False,
        )
        captured = capfd.readouterr()
        assert "my-special-model" in captured.out

    def test_log_generation_empty_model_does_not_raise(self) -> None:
        """log_generation() must not raise when model is an empty string."""
        logger = StructuredLogger(name="formatshield.test.edge1", level="INFO")
        logger.log_generation(
            model="",
            backend="groq",
            route="ttf",
            latency_ms=0.0,
            schema_valid=False,
            fallback=False,
        )

    def test_log_generation_zero_latency_does_not_raise(self) -> None:
        """log_generation() must not raise when latency_ms is zero."""
        logger = StructuredLogger(name="formatshield.test.edge2", level="INFO")
        logger.log_generation(
            model="groq/llama3",
            backend="groq",
            route="direct",
            latency_ms=0.0,
            schema_valid=True,
            fallback=False,
        )

    def test_log_generation_with_fallback_true_does_not_raise(self) -> None:
        """log_generation() with fallback=True must not raise."""
        logger = StructuredLogger(name="formatshield.test.fallback", level="INFO")
        logger.log_generation(
            model="ollama/llama3",
            backend="ollama",
            route="ttf",
            latency_ms=800.0,
            schema_valid=False,
            fallback=True,
        )


class TestStructuredLoggerRoutingDecision:
    def test_log_routing_decision_does_not_raise(self) -> None:
        """log_routing_decision() must not raise."""
        logger = StructuredLogger(name="formatshield.test.routing", level="INFO")
        logger.log_routing_decision(
            model="groq/llama3",
            complexity=0.72,
            decision="ttf",
            latency_ms=1.4,
        )

    def test_log_routing_decision_produces_output(self, capfd: pytest.CaptureFixture[str]) -> None:
        """log_routing_decision() must produce some stdout output."""
        logger = StructuredLogger(name="formatshield.test.routing2", level="INFO")
        logger.log_routing_decision(
            model="vllm/mistral",
            complexity=0.3,
            decision="direct",
            latency_ms=0.8,
        )
        captured = capfd.readouterr()
        assert len(captured.out) > 0


class TestStructuredLoggerError:
    def test_log_error_with_exception_does_not_raise(self) -> None:
        """log_error() with an Exception must not raise."""
        logger = StructuredLogger(name="formatshield.test.err1", level="INFO")
        logger.log_error(ValueError("something went wrong"))

    def test_log_error_with_string_does_not_raise(self) -> None:
        """log_error() with a plain string must not raise."""
        logger = StructuredLogger(name="formatshield.test.err2", level="INFO")
        logger.log_error("something went wrong")

    def test_log_error_with_context_does_not_raise(self) -> None:
        """log_error() with a context dict must not raise."""
        logger = StructuredLogger(name="formatshield.test.err3", level="INFO")
        logger.log_error(
            RuntimeError("backend timed out"),
            context={"model": "groq/llama3", "prompt_len": 512},
        )


class TestStructuredLoggerEnableDisable:
    def test_disable_does_not_raise(self) -> None:
        """disable() must not raise."""
        logger = StructuredLogger(name="formatshield.test.disable")
        logger.disable()

    def test_enable_after_disable_does_not_raise(self) -> None:
        """enable() called after disable() must not raise."""
        logger = StructuredLogger(name="formatshield.test.enable")
        logger.disable()
        logger.enable()

    def test_disabled_logger_produces_no_output(self, capfd: pytest.CaptureFixture[str]) -> None:
        """After disable(), log_generation() must produce no stdout output."""
        logger = StructuredLogger(name="formatshield.test.silent", level="INFO")
        logger.disable()
        logger.log_generation(
            model="groq/llama3",
            backend="groq",
            route="ttf",
            latency_ms=100.0,
            schema_valid=True,
            fallback=False,
        )
        captured = capfd.readouterr()
        assert captured.out == ""

    def test_re_enabled_logger_produces_output(self, capfd: pytest.CaptureFixture[str]) -> None:
        """After enable(), log_generation() must produce output again."""
        logger = StructuredLogger(name="formatshield.test.reenable", level="INFO")
        logger.disable()
        logger.enable()
        logger.log_generation(
            model="groq/llama3",
            backend="groq",
            route="direct",
            latency_ms=50.0,
            schema_valid=True,
            fallback=False,
        )
        captured = capfd.readouterr()
        assert len(captured.out) > 0


# ===========================================================================
# MetricsCollector tests
# ===========================================================================


class TestMetricsCollectorInit:
    def test_instantiation_does_not_raise(self) -> None:
        """MetricsCollector() must construct without raising."""
        collector = MetricsCollector()
        assert collector is not None

    def test_initial_summary_routing_total_is_zero(self) -> None:
        """A fresh collector must report routing total of 0."""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["routing"]["total"] == 0

    def test_initial_latency_count_is_zero(self) -> None:
        """A fresh collector must report latency count of 0."""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert summary["latency"]["count"] == 0

    def test_initial_fallback_count_is_zero(self) -> None:
        """A fresh collector must report fallback_count of 0."""
        collector = MetricsCollector()
        assert collector.get_summary()["fallback_count"] == 0

    def test_initial_schema_validation_failures_zero(self) -> None:
        """A fresh collector must report schema_validation_failures of 0."""
        collector = MetricsCollector()
        assert collector.get_summary()["schema_validation_failures"] == 0


class TestMetricsCollectorRecordRouting:
    def test_record_routing_increments_strategy_counter(self) -> None:
        """record_routing increments the strategy counter by one."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        summary = collector.get_summary()
        assert summary["routing"]["by_strategy"]["ttf"] == 1

    def test_record_routing_increments_backend_counter(self) -> None:
        """record_routing increments the backend counter by one."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        summary = collector.get_summary()
        assert summary["routing"]["by_backend"]["groq"] == 1

    def test_record_routing_multiple_calls_accumulate(self) -> None:
        """Multiple record_routing calls accumulate correctly."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        collector.record_routing(strategy="ttf", backend="groq")
        collector.record_routing(strategy="direct", backend="groq")
        summary = collector.get_summary()
        assert summary["routing"]["by_strategy"]["ttf"] == 2
        assert summary["routing"]["by_strategy"]["direct"] == 1

    def test_record_routing_total_reflects_all_calls(self) -> None:
        """routing total must equal the number of record_routing calls."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        collector.record_routing(strategy="direct", backend="vllm")
        assert collector.get_summary()["routing"]["total"] == 2

    def test_get_routing_summary_returns_dict(self) -> None:
        """get_summary()['routing'] must be a dict."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        routing = collector.get_summary()["routing"]
        assert isinstance(routing, dict)


class TestMetricsCollectorRecordLatency:
    def test_record_latency_stores_value(self) -> None:
        """record_latency must store the given ms value for the backend."""
        collector = MetricsCollector()
        collector.record_latency(ms=350.0, backend="groq")
        summary = collector.get_summary()
        assert 350.0 in summary["latency"]["by_backend"]["groq"]

    def test_record_latency_increments_count(self) -> None:
        """record_latency must increment the global latency count."""
        collector = MetricsCollector()
        collector.record_latency(ms=100.0, backend="groq")
        collector.record_latency(ms=200.0, backend="ollama")
        assert collector.get_summary()["latency"]["count"] == 2

    def test_record_latency_computes_mean(self) -> None:
        """get_summary() must compute mean_ms when latencies exist."""
        collector = MetricsCollector()
        collector.record_latency(ms=100.0, backend="groq")
        collector.record_latency(ms=300.0, backend="groq")
        summary = collector.get_summary()
        assert summary["latency"]["mean_ms"] == pytest.approx(200.0)

    def test_record_latency_computes_median(self) -> None:
        """get_summary() must compute median_ms when latencies exist."""
        collector = MetricsCollector()
        collector.record_latency(ms=100.0, backend="groq")
        collector.record_latency(ms=200.0, backend="groq")
        collector.record_latency(ms=300.0, backend="groq")
        summary = collector.get_summary()
        assert summary["latency"]["median_ms"] == pytest.approx(200.0)

    def test_get_latency_summary_returns_dict(self) -> None:
        """get_summary()['latency'] must be a dict."""
        collector = MetricsCollector()
        latency = collector.get_summary()["latency"]
        assert isinstance(latency, dict)


class TestMetricsCollectorRecordFallback:
    def test_record_fallback_increments_counter(self) -> None:
        """record_fallback() must increment fallback_count by one each call."""
        collector = MetricsCollector()
        collector.record_fallback()
        assert collector.get_summary()["fallback_count"] == 1

    def test_record_fallback_multiple_increments(self) -> None:
        """Three record_fallback() calls → fallback_count == 3."""
        collector = MetricsCollector()
        collector.record_fallback()
        collector.record_fallback()
        collector.record_fallback()
        assert collector.get_summary()["fallback_count"] == 3


class TestMetricsCollectorRecordAccuracyDelta:
    def test_record_accuracy_delta_stores_value(self) -> None:
        """record_accuracy_delta must add the delta to observations."""
        collector = MetricsCollector()
        collector.record_accuracy_delta(delta=0.12)
        summary = collector.get_summary()
        assert 0.12 in summary["accuracy_deltas"]["observations"]

    def test_record_accuracy_delta_computes_mean(self) -> None:
        """get_summary() must compute mean_delta when observations exist."""
        collector = MetricsCollector()
        collector.record_accuracy_delta(delta=0.10)
        collector.record_accuracy_delta(delta=0.20)
        summary = collector.get_summary()
        assert summary["accuracy_deltas"]["mean_delta"] == pytest.approx(0.15)

    def test_record_accuracy_delta_counts_positive(self) -> None:
        """get_summary() must count positive deltas correctly."""
        collector = MetricsCollector()
        collector.record_accuracy_delta(delta=0.10)
        collector.record_accuracy_delta(delta=-0.05)
        collector.record_accuracy_delta(delta=0.20)
        summary = collector.get_summary()
        assert summary["accuracy_deltas"]["positive_count"] == 2

    def test_record_accuracy_delta_counts_negative(self) -> None:
        """get_summary() must count negative deltas correctly."""
        collector = MetricsCollector()
        collector.record_accuracy_delta(delta=0.10)
        collector.record_accuracy_delta(delta=-0.05)
        summary = collector.get_summary()
        assert summary["accuracy_deltas"]["negative_count"] == 1

    def test_get_accuracy_delta_summary_returns_dict(self) -> None:
        """get_summary()['accuracy_deltas'] must be a dict."""
        collector = MetricsCollector()
        delta_summary = collector.get_summary()["accuracy_deltas"]
        assert isinstance(delta_summary, dict)


class TestMetricsCollectorReset:
    def test_reset_clears_routing(self) -> None:
        """reset() must clear routing counters."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        collector.reset()
        assert collector.get_summary()["routing"]["total"] == 0

    def test_reset_clears_latency(self) -> None:
        """reset() must clear latency observations."""
        collector = MetricsCollector()
        collector.record_latency(ms=100.0, backend="groq")
        collector.reset()
        assert collector.get_summary()["latency"]["count"] == 0

    def test_reset_clears_fallback_count(self) -> None:
        """reset() must reset fallback_count to 0."""
        collector = MetricsCollector()
        collector.record_fallback()
        collector.reset()
        assert collector.get_summary()["fallback_count"] == 0

    def test_reset_clears_accuracy_deltas(self) -> None:
        """reset() must clear accuracy_delta observations."""
        collector = MetricsCollector()
        collector.record_accuracy_delta(delta=0.15)
        collector.reset()
        summary = collector.get_summary()
        assert summary["accuracy_deltas"]["observations"] == []

    def test_reset_clears_schema_validation_failures(self) -> None:
        """reset() must reset schema_validation_failures to 0."""
        collector = MetricsCollector()
        collector.record_schema_validation_failure()
        collector.reset()
        assert collector.get_summary()["schema_validation_failures"] == 0

    def test_reset_allows_fresh_recording(self) -> None:
        """After reset(), new recordings accumulate correctly from zero."""
        collector = MetricsCollector()
        collector.record_routing(strategy="ttf", backend="groq")
        collector.reset()
        collector.record_routing(strategy="direct", backend="vllm")
        assert collector.get_summary()["routing"]["total"] == 1


class TestMetricsCollectorSchemValidationFailure:
    def test_record_schema_validation_failure_increments(self) -> None:
        """record_schema_validation_failure() must increment the counter."""
        collector = MetricsCollector()
        collector.record_schema_validation_failure()
        assert collector.get_summary()["schema_validation_failures"] == 1


class TestMetricsCollectorGetSummary:
    def test_get_summary_returns_dict(self) -> None:
        """get_summary() must return a dict."""
        collector = MetricsCollector()
        summary = collector.get_summary()
        assert isinstance(summary, dict)

    def test_get_summary_has_routing_key(self) -> None:
        """get_summary() must include 'routing' key."""
        collector = MetricsCollector()
        assert "routing" in collector.get_summary()

    def test_get_summary_has_latency_key(self) -> None:
        """get_summary() must include 'latency' key."""
        collector = MetricsCollector()
        assert "latency" in collector.get_summary()

    def test_get_summary_has_accuracy_deltas_key(self) -> None:
        """get_summary() must include 'accuracy_deltas' key."""
        collector = MetricsCollector()
        assert "accuracy_deltas" in collector.get_summary()


# ===========================================================================
# PrometheusMetrics delegation tests
# ===========================================================================


class TestPrometheusMetrics:
    def test_instantiation_without_collector_does_not_raise(self) -> None:
        """PrometheusMetrics() must construct without raising."""
        prom = PrometheusMetrics()
        assert prom is not None

    def test_record_routing_delegates_to_collector(self) -> None:
        """PrometheusMetrics.record_routing must update the underlying collector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        prom.record_routing(strategy="ttf", backend="groq")
        assert collector.get_summary()["routing"]["total"] == 1

    def test_record_latency_delegates_to_collector(self) -> None:
        """PrometheusMetrics.record_latency must update the underlying collector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        prom.record_latency(ms=250.0, backend="groq")
        assert collector.get_summary()["latency"]["count"] == 1

    def test_record_fallback_delegates_to_collector(self) -> None:
        """PrometheusMetrics.record_fallback must update the underlying collector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        prom.record_fallback()
        assert collector.get_summary()["fallback_count"] == 1

    def test_record_accuracy_delta_delegates_to_collector(self) -> None:
        """PrometheusMetrics.record_accuracy_delta must update the collector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        prom.record_accuracy_delta(delta=0.05)
        assert 0.05 in collector.get_summary()["accuracy_deltas"]["observations"]

    def test_get_summary_returns_same_as_collector(self) -> None:
        """PrometheusMetrics.get_summary must return the collector's summary."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        collector.record_routing(strategy="direct", backend="vllm")
        assert prom.get_summary() == collector.get_summary()

    def test_reset_delegates_to_collector(self) -> None:
        """PrometheusMetrics.reset must clear the underlying collector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        prom.record_routing(strategy="ttf", backend="groq")
        prom.reset()
        assert collector.get_summary()["routing"]["total"] == 0

    def test_collector_property_returns_collector(self) -> None:
        """PrometheusMetrics.collector property must return the MetricsCollector."""
        collector = MetricsCollector()
        prom = PrometheusMetrics(collector=collector)
        assert prom.collector is collector
