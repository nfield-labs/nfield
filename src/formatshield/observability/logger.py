"""Structured JSON logger for FormatShield observability."""

from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any


class _JsonFormatter(logging.Formatter):
    """
    Custom :class:`logging.Formatter` that serialises each log record as a
    single-line JSON object.

    Every log line contains at minimum:

    * ``timestamp`` — ISO-8601 UTC timestamp derived from ``record.created``
    * ``level`` — log level name (``"INFO"``, ``"WARNING"``, …)
    * ``logger`` — the logger name
    * ``message`` — the formatted log message

    Additional structured fields can be attached by passing a ``extra``
    dict to any logging call.  Fields stored in ``record.__dict__`` that are
    not part of the standard :class:`logging.LogRecord` interface are
    automatically included in the JSON output.
    """

    _STDLIB_ATTRS: frozenset[str] = frozenset(
        {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
            "taskName",
        }
    )

    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()

        payload: dict[str, Any] = {
            "timestamp": self._iso_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
        }

        # Attach any extra fields added via the ``extra=`` kwarg.
        for key, value in record.__dict__.items():
            if key not in self._STDLIB_ATTRS and not key.startswith("_"):
                payload[key] = value

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str, ensure_ascii=False)

    @staticmethod
    def _iso_timestamp(created: float) -> str:
        """Return an ISO-8601 UTC timestamp string for *created* (epoch seconds)."""
        # time.gmtime gives us a struct_time in UTC.
        t = time.gmtime(created)
        # Include milliseconds for higher precision.
        ms = int((created - int(created)) * 1000)
        return (
            f"{t.tm_year:04d}-{t.tm_mon:02d}-{t.tm_mday:02d}T"
            f"{t.tm_hour:02d}:{t.tm_min:02d}:{t.tm_sec:02d}.{ms:03d}Z"
        )


class StructuredLogger:
    """
    JSON-formatted structured logger for the FormatShield library.

    All log output is emitted as newline-delimited JSON (NDJSON), making it
    trivially parseable by log aggregation systems such as Elasticsearch,
    Loki, or Datadog.

    Parameters
    ----------
    name:
        Logger name.  Defaults to ``"formatshield"``.
    level:
        Initial log level name (e.g. ``"DEBUG"``, ``"INFO"``, ``"WARNING"``).
        Defaults to ``"INFO"``.

    Examples
    --------
    >>> logger = StructuredLogger(name="formatshield.router", level="DEBUG")
    >>> logger.log_routing_decision(
    ...     model="groq/llama-3.3-70b-versatile",
    ...     complexity=0.72,
    ...     decision="ttf",
    ...     latency_ms=1.4,
    ... )
    """

    def __init__(self, name: str = "formatshield", level: str = "INFO") -> None:
        self._logger = logging.getLogger(name)
        self._handler = logging.StreamHandler(sys.stdout)
        self._handler.setFormatter(_JsonFormatter())

        # Avoid adding duplicate handlers when the class is instantiated
        # multiple times with the same name (e.g. in tests).
        if not any(
            isinstance(h, logging.StreamHandler) and h.stream is sys.stdout
            for h in self._logger.handlers
        ):
            self._logger.addHandler(self._handler)

        self._logger.setLevel(getattr(logging, level.upper(), logging.INFO))
        # Do not propagate to the root logger to avoid double-printing.
        self._logger.propagate = False

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------

    def enable(self) -> None:
        """
        Re-enable logging output after a :meth:`disable` call.

        Sets the effective log level back to ``"INFO"`` (or whatever level
        was configured at construction time).
        """
        self._logger.disabled = False

    def disable(self) -> None:
        """
        Suppress all logging output from this logger without removing its
        handlers.  Call :meth:`enable` to restore output.
        """
        self._logger.disabled = True

    # ------------------------------------------------------------------
    # Structured log methods
    # ------------------------------------------------------------------

    def log_routing_decision(
        self,
        model: str,
        complexity: float,
        decision: str,
        latency_ms: float,
    ) -> None:
        """
        Emit a structured log line describing a routing decision.

        Parameters
        ----------
        model:
            Model identifier that was selected for this request.
        complexity:
            Complexity score (0.0–1.0) computed by
            :class:`~formatshield.scorer.ComplexityScorer`.
        decision:
            Routing decision label, typically ``"ttf"`` or ``"direct"``.
        latency_ms:
            Wall-clock time in milliseconds taken to reach the routing
            decision.
        """
        self._logger.info(
            "routing_decision",
            extra={
                "event": "routing_decision",
                "model": model,
                "complexity": complexity,
                "decision": decision,
                "latency_ms": latency_ms,
            },
        )

    def log_generation(
        self,
        model: str,
        backend: str,
        route: str,
        latency_ms: float,
        schema_valid: bool,
        fallback: bool,
    ) -> None:
        """
        Emit a structured log line describing a completed generation request.

        Parameters
        ----------
        model:
            Model identifier used for the request.
        backend:
            Backend identifier (e.g. ``"groq"``, ``"vllm"``).
        route:
            Routing strategy used (e.g. ``"ttf"``, ``"direct"``).
        latency_ms:
            End-to-end wall-clock latency in milliseconds.
        schema_valid:
            ``True`` if the generated output passed JSON schema validation.
        fallback:
            ``True`` if a fallback strategy was activated (e.g. retrying with
            a different backend or prompt).
        """
        self._logger.info(
            "generation",
            extra={
                "event": "generation",
                "model": model,
                "backend": backend,
                "route": route,
                "latency_ms": latency_ms,
                "schema_valid": schema_valid,
                "fallback": fallback,
            },
        )

    def log_error(
        self,
        error: Exception | str,
        context: dict[str, Any] | None = None,
    ) -> None:
        """
        Emit a structured ERROR log line.

        Parameters
        ----------
        error:
            The exception or error message to log.  If an
            :class:`Exception` is passed, its type name and string
            representation are included; the traceback is appended when the
            logger's effective level is ``DEBUG``.
        context:
            Optional mapping of additional key-value pairs to include in the
            log record (e.g. ``{"model": "groq/llama-3.1-70b", "prompt_len": 512}``).
        """
        extra: dict[str, Any] = {"event": "error"}

        if isinstance(error, Exception):
            extra["error_type"] = type(error).__name__
            extra["error_message"] = str(error)
        else:
            extra["error_message"] = str(error)

        if context:
            extra.update(context)

        self._logger.error(
            str(error),
            exc_info=isinstance(error, Exception) and self._logger.isEnabledFor(logging.DEBUG),
            extra=extra,
        )
