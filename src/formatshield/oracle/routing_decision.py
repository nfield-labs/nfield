"""
FormatShield routing decision dataclass.

:class:`RoutingDecision` is the output produced by
:class:`~formatshield.oracle.ThresholdOracle` for every inference request.
It encodes *which* generation strategy to use, quantified expectations about
accuracy improvement and latency overhead, and a human-readable explanation
suitable for debug logging and observability dashboards.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class RoutingDecision:
    """Describes the routing choice made by the oracle for a single request.

    Parameters
    ----------
    strategy:
        Which generation strategy to use:

        * ``"ttf"``    – Think-Then-Format: generate reasoning first, then
          produce structured output conditioned on that reasoning.
        * ``"direct"`` – Direct generation: produce structured output
          immediately without an explicit reasoning phase.
        * ``"hybrid"`` – Attempt TTF; fall back to direct if TTF fails or
          times out (reserved for future use).

    expected_accuracy_delta:
        Estimated change in field-level accuracy compared to direct generation.
        Positive values mean TTF is expected to *improve* accuracy.
        Comes from the oracle model or from per-backend heuristic constants.

    expected_overhead_pct:
        Estimated percentage increase in end-to-end latency introduced by the
        chosen strategy relative to direct generation.  A value of 0.0 means
        no overhead (i.e. ``strategy == "direct"``).

    confidence:
        Oracle confidence in this routing decision (0.0–1.0).  Derived from
        ``predict_proba`` when a trained sklearn model is available, or from a
        conservative heuristic constant otherwise.

    explanation:
        Human-readable string describing why this routing decision was made.
        Shown in debug / verbose logging and observability dashboards.

    failure_modes:
        List of failure-mode labels detected by
        :class:`~formatshield.ttf.FailureModeDetector` that influenced the
        routing decision.  Empty when no failure modes were detected.

    Properties
    ----------
    use_ttf:
        ``True`` when ``strategy == "ttf"``.
    use_direct:
        ``True`` when ``strategy == "direct"``.

    Example::

        decision = RoutingDecision(
            strategy="ttf",
            expected_accuracy_delta=0.18,
            expected_overhead_pct=25.0,
            confidence=0.82,
            explanation="High complexity score (0.74) above vllm threshold (0.60).",
        )
        if decision.use_ttf:
            result = await backend.generate_ttf(prompt, schema)
        else:
            result = await backend.generate_direct(prompt, schema)
    """

    strategy: Literal["ttf", "direct", "hybrid"]
    expected_accuracy_delta: float
    expected_overhead_pct: float
    confidence: float
    explanation: str
    failure_modes: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Guard against callers that pass None explicitly (e.g. from legacy
        # code that predates the field default_factory).
        if self.failure_modes is None:  # type: ignore[comparison-overlap]
            self.failure_modes = []

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def use_ttf(self) -> bool:
        """``True`` when the oracle recommends the TTF strategy."""
        return self.strategy == "ttf"

    @property
    def use_direct(self) -> bool:
        """``True`` when the oracle recommends direct generation."""
        return self.strategy == "direct"

    # ------------------------------------------------------------------
    # String representation
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        return (
            f"RoutingDecision(strategy={self.strategy!r}, "
            f"confidence={self.confidence:.2f}, "
            f"accuracy_delta={self.expected_accuracy_delta:+.3f}, "
            f"overhead={self.expected_overhead_pct:.1f}%)"
        )
