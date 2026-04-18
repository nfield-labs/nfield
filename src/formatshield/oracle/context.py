"""
FormatShield Oracle-X routing context and telemetry records.

:class:`RoutingContext` carries the backend/model/task/schema identity that
makes Oracle-X backend-agnostic.  :class:`TelemetryRecord` is the canonical
per-request record written for offline analysis and online adaptation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any


@dataclass
class RoutingContext:
    """Universal routing context — backend/model/task/schema identity.

    All fields are plain strings so routing logic stays provider-neutral.
    Groq is the current backend; adding OpenAI, vLLM, etc. requires only a
    new ``backend_id`` value, no algorithm changes.

    Parameters
    ----------
    backend_id:
        Lowercase backend name, e.g. ``"groq"``, ``"vllm"``, ``"ollama"``.
    model_id:
        Bare model name without provider prefix, e.g. ``"llama-3.1-8b-instant"``.
    task_id:
        Benchmark task name or ``"unknown"`` for production traffic.
    schema_family:
        Coarse schema category used as a routing feature:
        ``"math"`` | ``"ner"`` | ``"extraction"`` | ``"code"`` |
        ``"classification"`` | ``"unknown"``.
    prompt_id:
        First 12 hex chars of ``sha256(prompt.encode())``.  Used for
        stratified splits and drift detection.
    """

    backend_id: str
    model_id: str
    task_id: str
    schema_family: str
    prompt_id: str
    phi_score: float = 0.0  # Φ(prompt, schema) ∈ [0,1] — >0.5 suggests TTF
    phi_lambda2: float = 0.0  # Fiedler value of schema dependency graph (normalized)
    phi_tau: float = 0.0  # schema constraint tightness (entropy proxy)
    phi_delta_k: float = 0.0  # NCD prompt-schema alignment gap

    @classmethod
    def from_prompt(
        cls,
        prompt: str,
        backend_id: str,
        model_id: str,
        task_id: str = "unknown",
        schema_family: str = "unknown",
    ) -> RoutingContext:
        """Convenience constructor that computes ``prompt_id`` automatically."""
        pid = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return cls(
            backend_id=backend_id,
            model_id=model_id,
            task_id=task_id,
            schema_family=schema_family,
            prompt_id=pid,
        )

    def to_dict(self) -> dict[str, str | float]:
        return {
            "backend_id": self.backend_id,
            "model_id": self.model_id,
            "task_id": self.task_id,
            "schema_family": self.schema_family,
            "prompt_id": self.prompt_id,
            "phi_score": self.phi_score,
            "phi_lambda2": self.phi_lambda2,
            "phi_tau": self.phi_tau,
            "phi_delta_k": self.phi_delta_k,
        }


@dataclass
class TelemetryRecord:
    """Canonical per-request telemetry for offline analysis and online adaptation.

    Written by :class:`~formatshield.core.FormatShield` after each
    :meth:`~formatshield.core.FormatShield.generate` call.  ``realized_outcome``
    is ``None`` in production (no ground truth available) and set post-hoc
    during benchmark runs.

    Parameters
    ----------
    features:
        Raw feature vector from ``ComplexityFeatures.to_feature_vector()``.
    routing_context:
        Backend/model/task identity at decision time.
    chosen_action:
        The strategy that was executed: ``"ttf"`` | ``"direct"`` |
        ``"hybrid"`` | ``"safe-abstain"``.
    expected_utility:
        ``U(chosen_action)`` computed by the oracle at decision time.
    realized_outcome:
        ``accuracy_delta`` observed after the request completed.
        ``None`` when ground truth is unavailable (production traffic).
    latency_ms:
        Total wall-clock latency in milliseconds.
    token_cost:
        Total tokens consumed, or ``0.0`` if unknown.
    schema_validity:
        Whether the response passed Pydantic / JSON schema validation.
    failure_modes:
        Failure mode labels from :class:`~formatshield.ttf.FailureModeDetector`.
    label_verified:
        **Fix 5** — ``True`` only when *realized_outcome* has been confirmed
        by an evaluation pipeline or human reviewer.  Online adaptive
        threshold updates in :meth:`~formatshield.oracle.OracleX.update_online`
        are a no-op when this is ``False`` (default), guarding against stale
        or unverified production labels.
    """

    features: list[float]
    routing_context: RoutingContext
    chosen_action: str
    expected_utility: float
    realized_outcome: float | None
    latency_ms: float
    token_cost: float
    schema_validity: bool
    failure_modes: list[str] = field(default_factory=list)
    label_verified: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "features": self.features,
            "routing_context": self.routing_context.to_dict(),
            "chosen_action": self.chosen_action,
            "expected_utility": self.expected_utility,
            "realized_outcome": self.realized_outcome,
            "latency_ms": self.latency_ms,
            "token_cost": self.token_cost,
            "schema_validity": self.schema_validity,
            "failure_modes": self.failure_modes,
            "label_verified": self.label_verified,
        }
