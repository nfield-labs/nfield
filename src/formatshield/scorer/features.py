"""
FormatShield scorer feature dataclasses.

Defines the core data structures used throughout the scoring, routing,
streaming, and benchmarking pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ComplexityFeatures:
    """
    Feature vector describing the complexity of an inference request.

    Used by ThresholdOracle to decide whether Think-Then-Format (TTF)
    is likely to improve structured-output accuracy over direct generation.
    """

    token_entropy: float
    """Normalised Shannon entropy of the prompt's token-ID distribution (0.0–1.0).
    Higher values indicate more diverse / less repetitive prompts."""

    schema_depth: int
    """Maximum nesting depth of the target JSON schema (0 = flat / no schema)."""

    required_reasoning_ops: int
    """Estimated number of multi-step reasoning operations inferred from
    chain-of-thought keyword occurrences in the prompt."""

    instruction_tune_score: float
    """Heuristic score indicating how heavily instruction-tuned / RLHF-trained
    the target model is (0.0 = base model, 1.0 = heavy RLHF)."""

    prompt_length_bucket: int
    """Coarse bucket for prompt length measured in tokens:
        0 = short  (< 50 tokens)
        1 = medium (50–200 tokens)
        2 = long   (200–1 000 tokens)
        3 = very long (> 1 000 tokens)
    """

    schema_constraint_count: int
    """Total number of constrained fields in the schema (required array items +
    enum-typed fields + pattern-constrained fields)."""

    def to_feature_vector(self) -> list[float]:
        """Convert to a flat float feature vector for ThresholdOracle logistic regression."""
        return [
            self.token_entropy,
            float(self.schema_depth),
            float(self.required_reasoning_ops),
            self.instruction_tune_score,
            float(self.prompt_length_bucket),
            float(self.schema_constraint_count),
        ]


@dataclass
class StreamEvent:
    """
    A single event emitted during streaming inference.

    Events are yielded sequentially and cover the thinking phase,
    incremental token output, and final completion with full JSON.
    """

    type: Literal["thinking", "output", "complete"]
    """Event type:
        ``thinking``  – content produced in the reasoning (<think>) phase
        ``output``    – incremental output token during structured generation
        ``complete``  – final event; carries the parsed JSON response
    """

    content: str | None = None
    """Full thinking content (set on ``thinking`` events)."""

    token: str | None = None
    """Incremental token string (set on ``output`` events)."""

    json: dict | None = None  # type: ignore[type-arg]
    """Parsed JSON response (set on ``complete`` events)."""

    backend: str = ""
    """Identifier of the inference backend that produced this event
    (e.g. ``"vllm"``, ``"ollama"``, ``"groq"``)."""

    latency_ms: float = 0.0
    """Wall-clock latency in milliseconds from request start to this event."""


@dataclass
class BenchmarkResult:
    """
    Aggregated result for a single benchmark task / backend / model combination.

    Captures accuracy and latency metrics for both direct generation and the
    Think-Then-Format (TTF) strategy, allowing head-to-head comparison.
    """

    task: str
    """Benchmark task identifier (e.g. ``"nested_address_extraction"``)."""

    backend: str
    """Inference backend identifier (e.g. ``"vllm"``, ``"groq"``)."""

    model: str
    """Model identifier as passed to the backend (e.g. ``"gpt-4o"``)."""

    direct_accuracy: float
    """Exact-match / field-level accuracy for direct (non-TTF) generation (0.0–1.0)."""

    ttf_accuracy: float
    """Exact-match / field-level accuracy for TTF generation (0.0–1.0)."""

    accuracy_delta: float
    """``ttf_accuracy - direct_accuracy``; positive values mean TTF helps."""

    direct_latency_ms: float
    """Median end-to-end latency in milliseconds for direct generation."""

    ttf_latency_ms: float
    """Median end-to-end latency in milliseconds for TTF generation."""

    overhead_pct: float
    """Percentage latency increase introduced by TTF relative to direct:
    ``(ttf_latency_ms - direct_latency_ms) / direct_latency_ms * 100``."""

    complexity_score: float
    """Single-float complexity score (0.0–1.0) computed by ComplexityScorer
    for this task's prompt and schema."""

    failure_modes_detected: list[str] = field(default_factory=list)
    """List of failure-mode labels detected by FailureModeDetector for this task."""

    def to_dict(self) -> dict:  # type: ignore[type-arg]
        """Serialise to a flat dictionary suitable for CSV export or JSON logging."""
        return {
            "task": self.task,
            "backend": self.backend,
            "model": self.model,
            "direct_accuracy": self.direct_accuracy,
            "ttf_accuracy": self.ttf_accuracy,
            "accuracy_delta": self.accuracy_delta,
            "direct_latency_ms": self.direct_latency_ms,
            "ttf_latency_ms": self.ttf_latency_ms,
            "overhead_pct": self.overhead_pct,
            "complexity_score": self.complexity_score,
            "failure_modes_detected": ",".join(self.failure_modes_detected),
        }
