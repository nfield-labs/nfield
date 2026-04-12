"""In-memory metrics collector for FormatShield observability."""

from __future__ import annotations

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
# Prometheus integration stub
# ---------------------------------------------------------------------------


class PrometheusMetrics:
    """
    Thin wrapper around :class:`MetricsCollector` that mirrors its interface
    and is designed to be swapped in for Prometheus integration in the future.

    When ``prometheus_client`` is available in the environment, subclass this
    class and override the ``record_*`` methods to push observations to real
    Prometheus counters and histograms.  Until then, this class delegates
    everything to the underlying :class:`MetricsCollector` so that callers
    do not need to change their code.

    Parameters
    ----------
    collector:
        Optional existing :class:`MetricsCollector` to delegate to.  When
        ``None``, a new collector is created.

    Examples
    --------
    >>> prom = PrometheusMetrics()
    >>> prom.record_routing(strategy="ttf", backend="groq")
    >>> prom.get_summary()["routing"]["by_strategy"]
    {'ttf': 1}
    """

    def __init__(self, collector: MetricsCollector | None = None) -> None:
        self._collector = collector if collector is not None else MetricsCollector()

    # ------------------------------------------------------------------
    # Delegating methods
    # ------------------------------------------------------------------

    def record_routing(self, strategy: str, backend: str) -> None:
        """Delegate to :meth:`MetricsCollector.record_routing`."""
        # Future: push to prometheus_client Counter
        self._collector.record_routing(strategy=strategy, backend=backend)

    def record_latency(self, ms: float, backend: str) -> None:
        """Delegate to :meth:`MetricsCollector.record_latency`."""
        # Future: push to prometheus_client Histogram
        self._collector.record_latency(ms=ms, backend=backend)

    def record_schema_validation_failure(self) -> None:
        """Delegate to :meth:`MetricsCollector.record_schema_validation_failure`."""
        # Future: increment prometheus_client Counter
        self._collector.record_schema_validation_failure()

    def record_fallback(self) -> None:
        """Delegate to :meth:`MetricsCollector.record_fallback`."""
        # Future: increment prometheus_client Counter
        self._collector.record_fallback()

    def record_accuracy_delta(self, delta: float) -> None:
        """Delegate to :meth:`MetricsCollector.record_accuracy_delta`."""
        # Future: push to prometheus_client Histogram
        self._collector.record_accuracy_delta(delta=delta)

    def get_summary(self) -> dict[str, Any]:
        """Delegate to :meth:`MetricsCollector.get_summary`."""
        return self._collector.get_summary()

    def reset(self) -> None:
        """Delegate to :meth:`MetricsCollector.reset`."""
        self._collector.reset()

    @property
    def collector(self) -> MetricsCollector:
        """The underlying :class:`MetricsCollector` instance."""
        return self._collector
