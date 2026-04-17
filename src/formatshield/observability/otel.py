"""OpenTelemetry tracing integration for FormatShield.

Provides span-based tracing around generate() calls. When opentelemetry-api
is not installed, all operations gracefully degrade to no-ops.

Usage::

    from formatshield.observability.otel import FormatShieldTracer

    tracer = FormatShieldTracer(service_name="my-app")
    with tracer.generation_span("my-prompt", schema={"type": "object"}) as span:
        result = await shield.generate(...)
        tracer.set_result_attributes(span, result)
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    pass

# ---------------------------------------------------------------------------
# Optional OTel import
# ---------------------------------------------------------------------------

try:
    from opentelemetry import trace as _otel_trace  # pyright: ignore[reportMissingImports]
    from opentelemetry.trace import (  # pyright: ignore[reportMissingImports]
        StatusCode as _StatusCode,
    )

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover
    _OTEL_AVAILABLE = False
    _otel_trace = None  # type: ignore[assignment]
    _StatusCode = None  # type: ignore[assignment]


# ---------------------------------------------------------------------------
# _NoOpSpan
# ---------------------------------------------------------------------------


class _NoOpSpan:
    """Placeholder span used when opentelemetry-api is not installed.

    All methods are no-ops so callers can use the span API without
    conditionally checking for OTel availability.
    """

    def set_attribute(self, key: str, value: Any) -> None:
        """No-op set_attribute.

        Args:
            key: Attribute key (ignored).
            value: Attribute value (ignored).
        """

    def set_status(self, status: Any) -> None:
        """No-op set_status.

        Args:
            status: Status object (ignored).
        """

    def record_exception(self, exc: BaseException) -> None:
        """No-op record_exception.

        Args:
            exc: Exception to record (ignored).
        """


# ---------------------------------------------------------------------------
# FormatShieldTracer
# ---------------------------------------------------------------------------


def _schema_depth(schema: dict[str, Any], _depth: int = 0) -> int:
    """Recursively compute nesting depth of a JSON schema.

    Args:
        schema: JSON schema dict.
        _depth: Current recursion depth (used internally).

    Returns:
        Integer depth of the deepest nested object/array.
    """
    if not isinstance(schema, dict):
        return _depth
    max_child = _depth
    for value in schema.values():
        if isinstance(value, dict):
            max_child = max(max_child, _schema_depth(value, _depth + 1))
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    max_child = max(max_child, _schema_depth(item, _depth + 1))
    return max_child


class FormatShieldTracer:
    """OpenTelemetry tracer for FormatShield generation calls.

    Wraps opentelemetry-api spans around each generate() call. When
    opentelemetry-api is not installed, all methods are no-ops.

    Args:
        service_name: Logical service name attached to spans.
        tracer_name: OTel tracer/instrumentation name.

    Usage::

        tracer = FormatShieldTracer(service_name="my-app")
        with tracer.generation_span("my-prompt", schema={"type": "object"}) as span:
            result = await shield.generate(...)
            tracer.set_result_attributes(span, result)
    """

    def __init__(
        self,
        service_name: str = "formatshield",
        tracer_name: str = "formatshield.tracer",
    ) -> None:
        self._service_name = service_name
        self._tracer_name = tracer_name
        self._tracer: Any = None

        if _OTEL_AVAILABLE:
            self._tracer = _otel_trace.get_tracer(tracer_name)  # type: ignore[union-attr]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @contextmanager
    def generation_span(
        self,
        prompt: str,
        schema: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> Generator[_NoOpSpan | Any, None, None]:
        """Context manager that wraps a generation call in a span.

        When OTel is available, starts a new span named
        ``"formatshield.generate"`` and sets input attributes. When OTel
        is not installed, yields a :class:`_NoOpSpan` instead.

        Span attributes set at entry:

        - ``formatshield.prompt_length``: ``len(prompt)``
        - ``formatshield.model``: model identifier (if provided)
        - ``formatshield.schema_depth``: nesting depth of schema
        - ``formatshield.has_schema``: whether a schema was supplied

        Args:
            prompt: The user prompt passed to generate().
            schema: JSON schema dict for the generation request.
            model: Model identifier string (e.g. ``"groq/llama-3.3-70b"``).

        Yields:
            An OTel ``Span`` when available, otherwise a :class:`_NoOpSpan`.
        """
        if not _OTEL_AVAILABLE or self._tracer is None:
            yield _NoOpSpan()
            return

        with self._tracer.start_as_current_span("formatshield.generate") as span:
            span.set_attribute("formatshield.prompt_length", len(prompt))
            span.set_attribute("formatshield.has_schema", schema is not None)
            if model is not None:
                span.set_attribute("formatshield.model", model)
            if schema is not None:
                span.set_attribute("formatshield.schema_depth", _schema_depth(schema))
            try:
                yield span
            except Exception as exc:
                span.record_exception(exc)
                if _StatusCode is not None:
                    span.set_status(_StatusCode.ERROR, str(exc))
                raise

    def set_result_attributes(self, span: Any, result: Any) -> None:
        """Set span attributes from a GenerationResult.

        Safe to call with a :class:`_NoOpSpan` — all writes are no-ops when
        OTel is unavailable or the span is not recording.

        Attributes set:

        - ``formatshield.strategy``: routing strategy (``"ttf"`` or ``"direct"``)
        - ``formatshield.complexity_score``: float complexity score
        - ``formatshield.latency_ms``: wall-clock latency in milliseconds
        - ``formatshield.schema_valid``: whether output passed schema validation
        - ``formatshield.fallback_triggered``: whether TTF fell back to direct
        - ``formatshield.backend``: backend used for this request
        - ``formatshield.failure_modes``: comma-joined failure mode names

        Args:
            span: OTel ``Span`` or :class:`_NoOpSpan` from :meth:`generation_span`.
            result: :class:`~formatshield.core.GenerationResult` from generate().
        """
        try:
            span.set_attribute("formatshield.strategy", result.routing.strategy)
            span.set_attribute("formatshield.complexity_score", float(result.complexity_score))
            span.set_attribute("formatshield.latency_ms", float(result.latency_ms))
            span.set_attribute("formatshield.schema_valid", bool(result.schema_valid))
            span.set_attribute("formatshield.fallback_triggered", bool(result.fallback_triggered))
            span.set_attribute("formatshield.backend", str(result.backend))
            failure_modes_str = ",".join(result.failure_modes) if result.failure_modes else ""
            span.set_attribute("formatshield.failure_modes", failure_modes_str)
        except Exception:  # noqa: S110
            # Never let tracing errors surface to callers
            pass

    @property
    def is_available(self) -> bool:
        """Return True when opentelemetry-api is installed.

        Returns:
            ``True`` if opentelemetry-api is importable and a tracer was
            successfully initialised, ``False`` otherwise.
        """
        return _OTEL_AVAILABLE and self._tracer is not None


# ---------------------------------------------------------------------------
# Module-level default tracer
# ---------------------------------------------------------------------------

_default_tracer: FormatShieldTracer | None = None


def get_tracer(service_name: str = "formatshield") -> FormatShieldTracer:
    """Get or create the module-level default tracer.

    Subsequent calls with the same ``service_name`` return the same instance.
    Calling with a different ``service_name`` after the first call will return
    the already-created tracer (the first service_name wins).

    Args:
        service_name: Logical service name for the tracer. Only used on the
            first call — subsequent calls return the cached instance.

    Returns:
        The module-level :class:`FormatShieldTracer` instance.
    """
    global _default_tracer
    if _default_tracer is None:
        _default_tracer = FormatShieldTracer(service_name=service_name)
    return _default_tracer
