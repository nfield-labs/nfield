"""
FormatShield threshold oracle.

:class:`ThresholdOracle` decides whether each inference request should use
the Think-Then-Format (TTF) strategy or direct structured generation.

Decision flow
-------------
1. If the target model is a *native thinker* (has built-in CoT / extended
   thinking), always use direct generation – TTF would double-think.
2. If a ``latency_budget_ms`` is supplied and the estimated TTF overhead
   would exceed it, force direct generation.
3. Heuristic weighted-score threshold lookup keyed on the inference backend.
   When a :class:`~formatshield.oracle.context.RoutingContext` with a non-zero
   ``phi_score`` is provided (via :class:`OracleX`), the information-geometric
   Φ score replaces the heuristic.

Deprecated in v0.3
------------------
``from_benchmark_data()``, ``save()``, ``load()`` raise ``NotImplementedError``.
No training data or artifact files are required.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import ComplexityFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Models that have native chain-of-thought / extended thinking built in.
#: For these models, TTF is counter-productive – we always use direct mode.
NATIVE_THINKERS: frozenset[str] = frozenset(
    {
        "o1",
        "o3",
        "o1-mini",
        "o3-mini",
        "deepseek-r1",
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
    }
)

#: Per-backend complexity score thresholds used by the heuristic fallback.
#: Values are in [0, 1].  OracleX uses the Φ score when context is provided.
BACKEND_THRESHOLDS: dict[str, float] = {
    "vllm": 0.60,  # KV cache reuse → lower threshold
    "ollama": 0.65,
    "groq": 0.65,
    "openrouter": 0.67,
    "outlines": 0.62,
    "guidance": 0.63,
    "default": 0.65,
}

#: Estimated percentage latency overhead introduced by TTF per backend.
BACKEND_TTF_OVERHEAD: dict[str, float] = {
    "vllm": 10.0,
    "ollama": 25.0,
    "groq": 30.0,
    "openrouter": 35.0,
    "outlines": 20.0,
    "guidance": 22.0,
    "default": 30.0,
}

#: Expected accuracy improvement when TTF is used (positive means TTF helps).
_TTF_ACCURACY_DELTA: float = 0.17  # midpoint of 0.15–0.20 range
_DIRECT_ACCURACY_DELTA: float = 0.0

#: Heuristic confidence when no sklearn model is available.
_HEURISTIC_CONFIDENCE: float = 0.70

#: Assumed base latency (ms) used to convert TTF overhead percentage to milliseconds
#: when comparing against a caller-supplied latency_budget_ms.
_DEFAULT_BASE_LATENCY_MS: float = 500.0

#: Default path for the persisted oracle model (relative to package root).
_DEFAULT_MODEL_PATH: Path = Path(__file__).parent / "oracle_data" / "threshold_oracle_v1.pkl"

# ---------------------------------------------------------------------------
# Feature weights used for the heuristic weighted-score fallback
# ---------------------------------------------------------------------------
#
# v0.2 recalibration (live-test evidence):
#   - required_reasoning_ops cap: 20 → 5  (3 CoT keywords now = 0.6 signal,
#     vs 0.15 before; this is the strongest TTF predictor per CRANE §4.2)
#   - schema_depth weight: 0.25 → 0.10  (flat schemas dominate; depth rarely
#     exceeds 1 for real Pydantic models)
#   - required_reasoning_ops weight: 0.20 → 0.40  (CoT keywords are the
#     primary TTF signal; validated by SEAR and CRANE research)
#   - token_entropy weight: 0.20 → 0.15  (minor adjustment to keep sum=1.0)
# Result: TTF prompts (3+ CoT keywords, medium prompt) score ~0.53–0.60,
#         borderline prompts (0–1 keywords) score ~0.30–0.40  (below 0.50)

_FEATURE_WEIGHTS: list[float] = [0.15, 0.10, 0.40, 0.15, 0.10, 0.10]
_FEATURE_CAPS: list[float] = [1.0, 10.0, 5.0, 1.0, 3.0, 30.0]


class ThresholdOracle:
    """Route each inference request to either TTF or direct generation.

    Parameters
    ----------
    model_path:
        Ignored — kept for API compatibility only.  ThresholdOracle requires
        no pkl artifact.  Pass ``None`` or omit.

    Example::

        oracle = ThresholdOracle()
        features = scorer.score(prompt, schema=schema, model_id=model_id)
        decision = oracle.predict(features, backend="vllm", model_id=model_id)
        if decision.use_ttf:
            ...
    """

    def __init__(self, model_path: Path | str | None = None) -> None:
        self._model_path: Path = Path(model_path) if model_path is not None else _DEFAULT_MODEL_PATH
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None = None,
        cost_aware: bool = False,
    ) -> RoutingDecision:
        """Return a :class:`RoutingDecision` for the given request.

        Parameters
        ----------
        features:
            Feature vector computed by :class:`~formatshield.scorer.ComplexityScorer`.
        backend:
            Inference backend identifier (e.g. ``"vllm"``, ``"groq"``).
        model_id:
            Model identifier string (e.g. ``"gpt-4o"``).
        latency_budget_ms:
            Optional hard latency cap in milliseconds.  When set, TTF is
            suppressed if the estimated overhead would exceed the budget.
        cost_aware:
            When ``True`` the oracle applies a small upward bias to the
            direct-routing threshold (not yet fully implemented – reserved for
            token-cost optimisation).

        Returns
        -------
        RoutingDecision
            The routing decision.  Falls back to
            ``RoutingDecision(strategy="direct", confidence=0.3)`` on any error.
        """
        try:
            return self._predict_impl(
                features=features,
                backend=backend,
                model_id=model_id,
                latency_budget_ms=latency_budget_ms,
                cost_aware=cost_aware,
            )
        except Exception:
            logger.warning(
                "ThresholdOracle.predict: unexpected error – defaulting to direct",
                exc_info=True,
            )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.30,
                explanation="Oracle error - defaulting to direct",
            )

    @classmethod
    def from_benchmark_data(cls, *args: object, **kwargs: object) -> ThresholdOracle:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "from_benchmark_data() removed in v0.3. ThresholdOracle uses heuristic routing.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("from_benchmark_data() removed. No training data required.")

    def save(self, *args: object, **kwargs: object) -> None:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "save() removed in v0.3. ThresholdOracle has no artifact to persist.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("save() removed.")

    def load(self, *args: object, **kwargs: object) -> None:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "load() removed in v0.3. ThresholdOracle has no artifact to load.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("load() removed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """No-op — ThresholdOracle requires no pkl artifact."""
        logger.debug("ThresholdOracle: heuristic routing active, no artifact needed.")

    def _predict_impl(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None,
        cost_aware: bool,
    ) -> RoutingDecision:
        """Core prediction logic (called inside try/except in :meth:`predict`)."""

        backend_key = backend.lower() if backend else "default"
        threshold = BACKEND_THRESHOLDS.get(backend_key, BACKEND_THRESHOLDS["default"])
        estimated_overhead = BACKEND_TTF_OVERHEAD.get(backend_key, BACKEND_TTF_OVERHEAD["default"])

        # ------------------------------------------------------------------
        # Rule 1: Native thinker models – always direct
        # ------------------------------------------------------------------
        if _is_native_thinker(model_id):
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.95,
                explanation="Native thinker model detected – TTF would double-think.",
            )

        # ------------------------------------------------------------------
        # Rule 2: Latency budget exceeded – force direct
        # ------------------------------------------------------------------
        if latency_budget_ms is not None and latency_budget_ms > 0:
            # estimated_overhead is a percentage; convert to ms using a base estimate
            estimated_overhead_ms = (estimated_overhead / 100.0) * _DEFAULT_BASE_LATENCY_MS
            if estimated_overhead_ms > latency_budget_ms:
                return RoutingDecision(
                    strategy="direct",
                    expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                    expected_overhead_pct=0.0,
                    confidence=0.85,
                    explanation=(
                        f"Estimated TTF overhead ({estimated_overhead_ms:.0f} ms) "
                        f"exceeds latency budget ({latency_budget_ms:.0f} ms)."
                    ),
                )

        # ------------------------------------------------------------------
        # Rule 3: Heuristic weighted-score routing
        # ------------------------------------------------------------------
        return self._predict_heuristic(
            features=features,
            backend_key=backend_key,
            threshold=threshold,
            estimated_overhead=estimated_overhead,
            cost_aware=cost_aware,
        )

    def _predict_heuristic(
        self,
        features: ComplexityFeatures,
        backend_key: str,
        threshold: float,
        estimated_overhead: float,
        cost_aware: bool,
    ) -> RoutingDecision:
        """Heuristic fallback: weighted feature score vs. backend threshold."""
        vec = features.to_feature_vector()

        # Normalise each feature by its cap, then apply weights
        weighted_score = 0.0
        for raw_val, cap, weight in zip(vec, _FEATURE_CAPS, _FEATURE_WEIGHTS, strict=True):
            normalised = min(1.0, max(0.0, raw_val / cap if cap > 0 else 0.0))
            weighted_score += normalised * weight
        weighted_score = min(1.0, max(0.0, weighted_score))

        # cost_aware: apply a small upward bias to the threshold
        effective_threshold = threshold + (0.03 if cost_aware else 0.0)

        if weighted_score > effective_threshold:
            explanation = (
                f"Heuristic score {weighted_score:.3f} > threshold {effective_threshold:.2f} "
                f"for backend '{backend_key}' → TTF."
            )
            return RoutingDecision(
                strategy="ttf",
                expected_accuracy_delta=_TTF_ACCURACY_DELTA,
                expected_overhead_pct=estimated_overhead,
                confidence=_HEURISTIC_CONFIDENCE,
                explanation=explanation,
            )
        else:
            explanation = (
                f"Heuristic score {weighted_score:.3f} ≤ threshold {effective_threshold:.2f} "
                f"for backend '{backend_key}' → direct."
            )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=_HEURISTIC_CONFIDENCE,
                explanation=explanation,
            )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------


def _is_native_thinker(model_id: str) -> bool:
    """Return ``True`` if *model_id* (case-insensitive prefix match) is a
    known native-thinker model or its direct distillation."""
    lower = model_id.lower().strip()
    # Exact membership
    if lower in NATIVE_THINKERS:
        return True
    # Prefix match (e.g. "deepseek-r1-distill-..." variants)
    for native in NATIVE_THINKERS:
        if lower.startswith(native):
            return True
    return False
