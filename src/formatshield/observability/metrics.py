"""In-memory metrics collector for FormatShield observability."""

from __future__ import annotations

import json
import statistics
import threading
from collections import defaultdict
from typing import Any


class MetricsCollector:
    """
    Thread-safe, in-memory metrics collector for FormatShield.

    Tracks routing decisions, generation latencies, schema-validation failures,
    fallback activations, and accuracy deltas.  All data is stored in plain
    Python collections — no external dependencies are required.

    This class is designed to be instantiated once per process (or per test
    run) and shared across threads via its internal :class:`threading.Lock`.

    Examples
    --------
    >>> collector = MetricsCollector()
    >>> collector.record_routing(strategy="ttf", backend="groq")
    >>> collector.record_latency(ms=342.1, backend="groq")
    >>> collector.record_accuracy_delta(delta=0.12)
    >>> summary = collector.get_summary()
    >>> summary["routing"]["ttf"]
    1
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal state helpers
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        """Initialise (or re-initialise) all metric containers."""
        # routing_decisions: strategy → count  (e.g. "ttf" → 42, "direct" → 18)
        self._routing_decisions: dict[str, int] = defaultdict(int)
        # per-backend routing counts  (e.g. "groq" → 10, "vllm" → 5)
        self._routing_by_backend: dict[str, int] = defaultdict(int)

        # generation_latency: backend → list of millisecond values
        self._generation_latency: dict[str, list[float]] = defaultdict(list)
        # all latencies combined (for global percentiles)
        self._all_latencies: list[float] = []

        self._schema_validation_failures: int = 0
        self._fallback_count: int = 0

        # accuracy_deltas: list of float (ttf_accuracy - direct_accuracy)
        self._accuracy_deltas: list[float] = []

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_routing(self, strategy: str, backend: str) -> None:
        """
        Record a routing decision.

        Parameters
        ----------
        strategy:
            The routing strategy chosen, e.g. ``"ttf"`` or ``"direct"``.
        backend:
            The backend selected for this request, e.g. ``"groq"`` or
            ``"vllm"``.
        """
        with self._lock:
            self._routing_decisions[strategy] += 1
            self._routing_by_backend[backend] += 1

    def record_latency(self, ms: float, backend: str) -> None:
        """
        Record an end-to-end generation latency observation.

        Parameters
        ----------
        ms:
            Latency in milliseconds.
        backend:
            The backend that served this request.
        """
        with self._lock:
            self._generation_latency[backend].append(ms)
            self._all_latencies.append(ms)

    def record_schema_validation_failure(self) -> None:
        """Increment the schema-validation failure counter by one."""
        with self._lock:
            self._schema_validation_failures += 1

    def record_fallback(self) -> None:
        """Increment the fallback-activation counter by one."""
        with self._lock:
            self._fallback_count += 1

    def record_accuracy_delta(self, delta: float) -> None:
        """
        Record an accuracy delta observation.

        Parameters
        ----------
        delta:
            Signed accuracy difference ``ttf_accuracy - direct_accuracy``.
            Positive values indicate that TTF improved accuracy.
        """
        with self._lock:
            self._accuracy_deltas.append(delta)

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """
        Return a snapshot of all current metrics as a plain dictionary.

        The dictionary is safe to serialise to JSON and suitable for
        forwarding to a log aggregator or metrics dashboard.

        Returns
        -------
        dict
            A nested dictionary with the following top-level keys:

            ``routing``
                Per-strategy request counts and per-backend request counts.
            ``latency``
                Per-backend lists of latency observations, plus global
                statistics (mean, median, p95, p99) when at least one
                latency has been recorded.
            ``schema_validation_failures``
                Total count of schema-validation failures.
            ``fallback_count``
                Total number of fallback activations.
            ``accuracy_deltas``
                List of recorded accuracy delta observations, plus summary
                statistics when available.
        """
        with self._lock:
            # Latency statistics
            latency_stats: dict[str, Any] = {
                "by_backend": {
                    backend: list(values) for backend, values in self._generation_latency.items()
                },
            }
            if self._all_latencies:
                sorted_all = sorted(self._all_latencies)
                n = len(sorted_all)
                latency_stats["mean_ms"] = statistics.mean(sorted_all)
                latency_stats["median_ms"] = statistics.median(sorted_all)
                latency_stats["p95_ms"] = sorted_all[max(0, int(n * 0.95) - 1)]
                latency_stats["p99_ms"] = sorted_all[max(0, int(n * 0.99) - 1)]
                latency_stats["count"] = n
            else:
                latency_stats["count"] = 0

            # Accuracy delta statistics
            accuracy_stats: dict[str, Any] = {
                "observations": list(self._accuracy_deltas),
            }
            if self._accuracy_deltas:
                accuracy_stats["mean_delta"] = statistics.mean(self._accuracy_deltas)
                accuracy_stats["count"] = len(self._accuracy_deltas)
                accuracy_stats["positive_count"] = sum(1 for d in self._accuracy_deltas if d > 0)
                accuracy_stats["negative_count"] = sum(1 for d in self._accuracy_deltas if d < 0)
            else:
                accuracy_stats["count"] = 0

            return {
                "routing": {
                    "by_strategy": dict(self._routing_decisions),
                    "by_backend": dict(self._routing_by_backend),
                    "total": sum(self._routing_decisions.values()),
                },
                "latency": latency_stats,
                "schema_validation_failures": self._schema_validation_failures,
                "fallback_count": self._fallback_count,
                "accuracy_deltas": accuracy_stats,
            }

    def reset(self) -> None:
        """
        Clear all recorded metrics.

        This is useful between test runs or when implementing a metrics
        scrape-and-reset pattern (counters reset to zero after each scrape).
        """
        with self._lock:
            self._reset_state()


# ---------------------------------------------------------------------------
# Prometheus integration
# ---------------------------------------------------------------------------

_LATENCY_BUCKETS = (50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 5000.0)
_ACCURACY_DELTA_BUCKETS = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)


class PrometheusMetrics:
    """Prometheus-backed metrics for FormatShield, with fallback to :class:`MetricsCollector`.

    When ``prometheus_client`` is installed, real ``Counter``, ``Histogram``,
    and ``Gauge`` objects are created and all ``record_*`` methods push
    observations to Prometheus.  When the package is absent the class
    transparently delegates every call to the underlying
    :class:`MetricsCollector` so callers never need to branch.

    Prometheus metric names follow the ``formatshield_*`` namespace:

    - ``formatshield_routing_decisions_total`` — Counter, labels: strategy, backend
    - ``formatshield_generation_latency_ms`` — Histogram, labels: backend
    - ``formatshield_schema_validation_failures_total`` — Counter
    - ``formatshield_fallback_activations_total`` — Counter
    - ``formatshield_accuracy_delta`` — Histogram

    Args:
        collector: Optional existing :class:`MetricsCollector` to delegate to
            when ``prometheus_client`` is unavailable. A new collector is
            created when ``None``.

    Examples:
        >>> prom = PrometheusMetrics()
        >>> prom.record_routing(strategy="ttf", backend="groq")
        >>> prom.get_summary()["routing"]["by_strategy"]
        {'ttf': 1}
    """

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self._collector = collector if collector is not None else MetricsCollector()
        self._prometheus_available: bool = False

        try:
            import prometheus_client  # pyright: ignore[reportMissingImports]

            self._prometheus_available = True

            self._routing_counter = prometheus_client.Counter(
                "formatshield_routing_decisions_total",
                "Total number of routing decisions made by FormatShield",
                ["strategy", "backend"],
            )
            self._latency_histogram = prometheus_client.Histogram(
                "formatshield_generation_latency_ms",
                "End-to-end generation latency in milliseconds",
                ["backend"],
                buckets=_LATENCY_BUCKETS,
            )
            self._schema_failure_counter = prometheus_client.Counter(
                "formatshield_schema_validation_failures_total",
                "Total number of JSON schema validation failures",
            )
            self._fallback_counter = prometheus_client.Counter(
                "formatshield_fallback_activations_total",
                "Total number of TTF fallback activations",
            )
            self._accuracy_delta_histogram = prometheus_client.Histogram(
                "formatshield_accuracy_delta",
                "Signed accuracy delta: ttf_accuracy minus direct_accuracy",
                buckets=_ACCURACY_DELTA_BUCKETS,
            )
        except ImportError:
            self._prometheus_available = False

    # ------------------------------------------------------------------
    # Recording methods
    # ------------------------------------------------------------------

    def record_routing(self, strategy: str, backend: str) -> None:
        """Record a routing decision.

        Increments ``formatshield_routing_decisions_total`` when
        ``prometheus_client`` is available; otherwise delegates to the
        underlying :class:`MetricsCollector`.

        Args:
            strategy: The routing strategy chosen, e.g. ``"ttf"`` or
                ``"direct"``.
            backend: The backend selected for this request, e.g. ``"groq"``
                or ``"vllm"``.
        """
        self._collector.record_routing(strategy=strategy, backend=backend)
        if self._prometheus_available:
            self._routing_counter.labels(strategy=strategy, backend=backend).inc()

    def record_latency(self, ms: float, backend: str) -> None:
        """Record an end-to-end generation latency observation.

        Observes into ``formatshield_generation_latency_ms`` when
        ``prometheus_client`` is available; otherwise delegates to the
        underlying :class:`MetricsCollector`.

        Args:
            ms: Latency in milliseconds.
            backend: The backend that served this request.
        """
        self._collector.record_latency(ms=ms, backend=backend)
        if self._prometheus_available:
            self._latency_histogram.labels(backend=backend).observe(ms)

    def record_schema_validation_failure(self) -> None:
        """Increment the schema-validation failure counter.

        Increments ``formatshield_schema_validation_failures_total`` when
        ``prometheus_client`` is available; otherwise delegates to the
        underlying :class:`MetricsCollector`.
        """
        self._collector.record_schema_validation_failure()
        if self._prometheus_available:
            self._schema_failure_counter.inc()

    def record_fallback(self) -> None:
        """Increment the fallback-activation counter.

        Increments ``formatshield_fallback_activations_total`` when
        ``prometheus_client`` is available; otherwise delegates to the
        underlying :class:`MetricsCollector`.
        """
        self._collector.record_fallback()
        if self._prometheus_available:
            self._fallback_counter.inc()

    def record_accuracy_delta(self, delta: float) -> None:
        """Record an accuracy delta observation.

        Observes into ``formatshield_accuracy_delta`` when
        ``prometheus_client`` is available; otherwise delegates to the
        underlying :class:`MetricsCollector`.

        Args:
            delta: Signed accuracy difference ``ttf_accuracy - direct_accuracy``.
                Positive values indicate TTF improved accuracy.
        """
        self._collector.record_accuracy_delta(delta=delta)
        if self._prometheus_available:
            self._accuracy_delta_histogram.observe(delta)

    # ------------------------------------------------------------------
    # Query / utility methods
    # ------------------------------------------------------------------

    def get_summary(self) -> dict[str, Any]:
        """Return a snapshot of all current metrics as a plain dictionary.

        Always delegates to the underlying :class:`MetricsCollector`, regardless
        of ``prometheus_client`` availability, so callers always receive a
        consistent JSON-serialisable structure.

        Returns:
            Nested dictionary with keys ``routing``, ``latency``,
            ``schema_validation_failures``, ``fallback_count``, and
            ``accuracy_deltas``.
        """
        return self._collector.get_summary()

    def reset(self) -> None:
        """Clear all recorded metrics in the underlying :class:`MetricsCollector`.

        Note: Prometheus counters and histograms are process-lifetime objects
        and cannot be reset at runtime.  This method resets only the in-memory
        collector used for ``get_summary()``.
        """
        self._collector.reset()

    @property
    def collector(self) -> MetricsCollector:
        """The underlying :class:`MetricsCollector` instance."""
        return self._collector


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def serve_metrics(port: int = 9090) -> None:
    """Start a Prometheus HTTP metrics server on the given port.

    Starts ``prometheus_client``'s built-in HTTP server so that a Prometheus
    scraper can collect FormatShield metrics at ``http://localhost:<port>/metrics``.

    This function is a no-op when ``prometheus_client`` is not installed; a
    warning is printed to stdout so that operators know the server was not
    started.

    Args:
        port: TCP port to listen on. Defaults to ``9090``.

    Raises:
        OSError: If the port is already in use and the server cannot bind.

    Example:
        >>> # In production: expose metrics for Prometheus to scrape
        >>> serve_metrics(port=9090)  # doctest: +SKIP
    """
    try:
        import prometheus_client  # pyright: ignore[reportMissingImports]

        prometheus_client.start_http_server(port)
    except ImportError:
        print(
            f"[FormatShield] prometheus_client is not installed; "
            f"metrics server not started on port {port}. "
            "Install it with: pip install 'formatshield[prometheus]'"
        )


def generate_metrics_text() -> str:
    """Return current metrics in Prometheus text exposition format.

    When ``prometheus_client`` is available this function calls
    ``prometheus_client.generate_latest()`` and returns the UTF-8 decoded
    text.  When the package is absent it falls back to serialising the global
    :class:`MetricsCollector` summary as a JSON string so callers always
    receive a non-empty string.

    Returns:
        A string containing either the Prometheus text exposition format or a
        JSON-encoded summary of the in-memory collector.

    Example:
        >>> text = generate_metrics_text()
        >>> isinstance(text, str)
        True
    """
    try:
        import prometheus_client  # pyright: ignore[reportMissingImports]

        raw: bytes = prometheus_client.generate_latest()
        return raw.decode("utf-8")
    except ImportError:
        collector = MetricsCollector()
        return json.dumps(collector.get_summary(), indent=2)
