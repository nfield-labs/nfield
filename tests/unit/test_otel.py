"""Unit tests for OpenTelemetry integration (GROUP N — Stage 4).

These tests verify no-op behaviour when opentelemetry-api is not installed,
and the public API of FormatShieldTracer.
"""

from __future__ import annotations

from formatshield.observability.otel import (
    FormatShieldTracer,
    _NoOpSpan,
    get_tracer,
)


class TestNoOpSpan:
    def test_set_attribute_no_raise(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("key", "value")  # should not raise

    def test_set_status_no_raise(self) -> None:
        span = _NoOpSpan()
        span.set_status("ok")  # should not raise

    def test_record_exception_no_raise(self) -> None:
        span = _NoOpSpan()
        span.record_exception(RuntimeError("boom"))  # should not raise

    def test_set_attribute_accepts_various_types(self) -> None:
        span = _NoOpSpan()
        span.set_attribute("int_key", 42)
        span.set_attribute("float_key", 3.14)
        span.set_attribute("bool_key", True)
        span.set_attribute("none_key", None)  # type: ignore[arg-type]


class TestFormatShieldTracerInit:
    def test_default_service_name(self) -> None:
        tracer = FormatShieldTracer()
        assert tracer._service_name == "formatshield"

    def test_custom_service_name(self) -> None:
        tracer = FormatShieldTracer(service_name="my-app")
        assert tracer._service_name == "my-app"

    def test_custom_tracer_name(self) -> None:
        tracer = FormatShieldTracer(tracer_name="my.tracer")
        assert tracer._tracer_name == "my.tracer"

    def test_is_available_is_bool(self) -> None:
        tracer = FormatShieldTracer()
        assert isinstance(tracer.is_available, bool)


class TestFormatShieldTracerGenerationSpan:
    def test_generation_span_yields_span_like_object(self) -> None:
        tracer = FormatShieldTracer()
        with tracer.generation_span("test prompt") as span:
            assert span is not None

    def test_generation_span_noop_has_set_attribute(self) -> None:
        tracer = FormatShieldTracer()
        with tracer.generation_span("test prompt") as span:
            # _NoOpSpan has these methods; real OTel span also has them
            assert hasattr(span, "set_attribute")
            assert hasattr(span, "set_status")
            assert hasattr(span, "record_exception")

    def test_generation_span_with_schema(self) -> None:
        tracer = FormatShieldTracer()
        schema = {"type": "object", "properties": {"answer": {"type": "string"}}}
        with tracer.generation_span("prompt", schema=schema, model="dryrun/test") as span:
            span.set_attribute("test", True)

    def test_generation_span_no_raise_inside(self) -> None:
        tracer = FormatShieldTracer()
        with tracer.generation_span("prompt") as span:
            span.set_attribute("formatshield.strategy", "ttf")
            span.set_attribute("formatshield.latency_ms", 42.5)

    def test_generation_span_exception_propagates(self) -> None:
        import pytest

        tracer = FormatShieldTracer()
        with pytest.raises(RuntimeError, match="test error"):
            with tracer.generation_span("prompt"):
                raise RuntimeError("test error")


class TestFormatShieldTracerSetResultAttributes:
    def test_set_result_attributes_with_noop_span(self) -> None:
        from formatshield.backends.dryrun_backend import DryRunBackend
        from formatshield.core import FormatShield

        tracer = FormatShieldTracer()
        shield = FormatShield(model="dryrun/test", backend=DryRunBackend())
        result = shield.generate_sync("What is 2+2?")

        span = _NoOpSpan()
        tracer.set_result_attributes(span, result)  # should not raise

    def test_set_result_attributes_never_raises(self) -> None:
        tracer = FormatShieldTracer()

        class BadResult:
            @property
            def routing(self) -> object:
                raise AttributeError("no routing")

        span = _NoOpSpan()
        tracer.set_result_attributes(span, BadResult())  # should not raise


class TestGetTracer:
    def test_get_tracer_returns_instance(self) -> None:
        tracer = get_tracer()
        assert isinstance(tracer, FormatShieldTracer)

    def test_get_tracer_returns_same_instance(self) -> None:
        t1 = get_tracer()
        t2 = get_tracer()
        assert t1 is t2

    def test_get_tracer_custom_service_name(self) -> None:
        # Default instance already created — returns cached regardless
        tracer = get_tracer(service_name="override")
        assert isinstance(tracer, FormatShieldTracer)
