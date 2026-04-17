"""Unit tests for PrometheusMetrics and MetricsCollector (GROUP N — Stage 4)."""

from __future__ import annotations

from formatshield.observability.metrics import MetricsCollector, PrometheusMetrics


class TestMetricsCollector:
    def test_initial_summary_has_zero_counts(self) -> None:
        m = MetricsCollector()
        s = m.get_summary()
        assert s["routing"]["total"] == 0
        assert s["schema_validation_failures"] == 0
        assert s["fallback_count"] == 0
        assert s["latency"]["count"] == 0

    def test_record_routing_increments_strategy(self) -> None:
        m = MetricsCollector()
        m.record_routing(strategy="ttf", backend="dryrun")
        m.record_routing(strategy="ttf", backend="dryrun")
        m.record_routing(strategy="direct", backend="dryrun")
        s = m.get_summary()
        assert s["routing"]["by_strategy"]["ttf"] == 2
        assert s["routing"]["by_strategy"]["direct"] == 1
        assert s["routing"]["total"] == 3

    def test_record_routing_by_backend(self) -> None:
        m = MetricsCollector()
        m.record_routing(strategy="ttf", backend="groq")
        m.record_routing(strategy="direct", backend="openai")
        s = m.get_summary()
        assert s["routing"]["by_backend"]["groq"] == 1
        assert s["routing"]["by_backend"]["openai"] == 1

    def test_record_latency_stats(self) -> None:
        m = MetricsCollector()
        m.record_latency(ms=100.0, backend="dryrun")
        m.record_latency(ms=200.0, backend="dryrun")
        s = m.get_summary()
        assert s["latency"]["count"] == 2
        assert s["latency"]["mean_ms"] == 150.0

    def test_record_schema_validation_failure(self) -> None:
        m = MetricsCollector()
        m.record_schema_validation_failure()
        m.record_schema_validation_failure()
        s = m.get_summary()
        assert s["schema_validation_failures"] == 2

    def test_record_fallback(self) -> None:
        m = MetricsCollector()
        m.record_fallback()
        s = m.get_summary()
        assert s["fallback_count"] == 1

    def test_record_accuracy_delta(self) -> None:
        m = MetricsCollector()
        m.record_accuracy_delta(delta=0.15)
        m.record_accuracy_delta(delta=-0.05)
        s = m.get_summary()
        assert s["accuracy_deltas"]["count"] == 2
        assert s["accuracy_deltas"]["positive_count"] == 1
        assert s["accuracy_deltas"]["negative_count"] == 1

    def test_reset_clears_all(self) -> None:
        m = MetricsCollector()
        m.record_routing(strategy="ttf", backend="dryrun")
        m.record_latency(ms=100.0, backend="dryrun")
        m.record_schema_validation_failure()
        m.reset()
        s = m.get_summary()
        assert s["routing"]["total"] == 0
        assert s["latency"]["count"] == 0
        assert s["schema_validation_failures"] == 0

    def test_get_summary_is_json_serialisable(self) -> None:
        import json

        m = MetricsCollector()
        m.record_routing(strategy="ttf", backend="groq")
        m.record_latency(ms=250.0, backend="groq")
        m.record_accuracy_delta(delta=0.10)
        s = m.get_summary()
        # Should not raise
        json.dumps(s)


class TestPrometheusMetricsInit:
    def test_instantiation_no_raise(self) -> None:
        m = PrometheusMetrics()
        assert m is not None

    def test_uses_provided_collector(self) -> None:
        collector = MetricsCollector()
        m = PrometheusMetrics(collector=collector)
        assert m.collector is collector

    def test_creates_collector_when_none(self) -> None:
        m = PrometheusMetrics()
        assert isinstance(m.collector, MetricsCollector)

    def test_prometheus_available_is_bool(self) -> None:
        m = PrometheusMetrics()
        assert isinstance(m._prometheus_available, bool)


class TestPrometheusMetricsRecord:
    def test_record_routing_no_raise(self) -> None:
        m = PrometheusMetrics()
        m.record_routing(strategy="ttf", backend="dryrun")

    def test_record_routing_reflects_in_summary(self) -> None:
        m = PrometheusMetrics()
        m.record_routing(strategy="ttf", backend="dryrun")
        s = m.get_summary()
        assert s["routing"]["by_strategy"]["ttf"] == 1

    def test_record_latency_no_raise(self) -> None:
        m = PrometheusMetrics()
        m.record_latency(ms=123.4, backend="dryrun")

    def test_record_schema_validation_failure_no_raise(self) -> None:
        m = PrometheusMetrics()
        m.record_schema_validation_failure()
        s = m.get_summary()
        assert s["schema_validation_failures"] == 1

    def test_record_fallback_no_raise(self) -> None:
        m = PrometheusMetrics()
        m.record_fallback()
        s = m.get_summary()
        assert s["fallback_count"] == 1

    def test_record_accuracy_delta_no_raise(self) -> None:
        m = PrometheusMetrics()
        m.record_accuracy_delta(delta=0.15)

    def test_reset_clears_collector(self) -> None:
        m = PrometheusMetrics()
        m.record_routing(strategy="ttf", backend="dryrun")
        m.reset()
        s = m.get_summary()
        assert s["routing"]["total"] == 0


class TestPrometheusMetricsModuleFunctions:
    def test_generate_metrics_text_returns_string(self) -> None:
        from formatshield.observability.metrics import generate_metrics_text

        text = generate_metrics_text()
        assert isinstance(text, str)

    def test_serve_metrics_is_callable(self) -> None:
        from formatshield.observability.metrics import serve_metrics

        assert callable(serve_metrics)


class TestPrometheusMetricsIntegration:
    def test_record_from_generation_result(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
        result = shield.generate_sync("What is 2+2?")

        m = PrometheusMetrics()
        m.record_routing(strategy=result.routing.strategy, backend=str(result.backend))
        m.record_latency(ms=result.latency_ms, backend=str(result.backend))
        if not result.schema_valid:
            m.record_schema_validation_failure()
        if result.fallback_triggered:
            m.record_fallback()

        s = m.get_summary()
        assert s["routing"]["total"] == 1
