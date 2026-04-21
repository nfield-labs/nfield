"""
FormatShield Universal Oracle-X — information-geometric routing oracle.

Oracle-X routes each inference request to direct or TTF generation using the
closed-form Φ(prompt, schema) routing score — no trained model, no pkl artifacts.

Φ is computed from three information-theoretic components:

* **λ̃₂**: spectral algebraic connectivity of the JSON schema dependency graph
  (Fiedler value, normalized by max degree + 1)
* **τ**: schema constraint tightness — entropy proxy approximating GAD
  Expected Future Grammaticality
* **ΔK**: NCD prompt-schema alignment gap (Kolmogorov complexity proxy via zlib)

Backward compatibility
----------------------
``OracleX.predict()`` accepts the same positional arguments as
``ThresholdOracle.predict()`` plus an optional ``context`` kwarg.

Deprecated in v0.3
------------------
``from_benchmark_data()``, ``save()``, ``load()`` raise ``NotImplementedError``.
Oracle-X requires no training data and no artifact files.
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

from formatshield.oracle.context import RoutingContext, TelemetryRecord
from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import ComplexityFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants (reuse from ThresholdOracle to stay backend-agnostic)
# ---------------------------------------------------------------------------

from formatshield.oracle.backend_registry import get_registry  # noqa: E402
from formatshield.oracle.threshold_oracle import (  # noqa: E402
    _DEFAULT_BASE_LATENCY_MS,
    _DIRECT_ACCURACY_DELTA,
    _FEATURE_CAPS,
    _FEATURE_WEIGHTS,
    _HEURISTIC_CONFIDENCE,
    _TTF_ACCURACY_DELTA,
    BACKEND_THRESHOLDS,
    BACKEND_TTF_OVERHEAD,
    _is_native_thinker,
)

#: Feature flag — kept for API compatibility; online adaptation is removed.
ENABLE_ONLINE_ADAPTATION: bool = False

#: λ̃₂ below this value indicates a flat schema (no dependency graph depth).
_FLAT_SCHEMA_LAMBDA2: float = 0.20

#: τ below this value, combined with a flat schema, indicates a simple extraction
#: task — TTF adds overhead without reasoning benefit.
#: At τ=0.40 the schema has almost no cross-field constraints; at τ≥0.40 there is
#: enough constraint coupling to make TTF-guided reasoning worth the latency cost.
_EXTRACTION_TAU_THRESHOLD: float = 0.40

#: Minimum required_reasoning_ops count for the extraction override to fire.
#: If the prompt contains even 1 CoT keyword (calculate, evaluate, assess, plan, …)
#: the task is semantically complex — the extraction shortcut must NOT apply.
#: Value 1 means "0 ops required" (strictly less than 1 == zero).
_SEMANTIC_REASONING_OPS_THRESHOLD: int = 1

#: Token entropy above this value indicates a lexically dense / diverse prompt —
#: used as a secondary gate when no CoT keywords are present but the vocabulary
#: richness signals a complex domain task (e.g. dense technical specifications).
#: Calibrated at 0.88: above this, virtually all tokens in the prompt are unique,
#: which strongly correlates with multi-concept technical prompts.
_ENTROPY_COMPLEXITY_THRESHOLD: float = 0.88


# ---------------------------------------------------------------------------
# OracleX
# ---------------------------------------------------------------------------


class OracleX:
    """Universal information-geometric routing oracle.

    Uses the closed-form Φ(prompt, schema) routing score when a
    :class:`~formatshield.oracle.context.RoutingContext` with ``phi_score > 0``
    is supplied; falls back to a heuristic weighted-score otherwise.

    Parameters
    ----------
    model_path:
        Ignored — kept for API compatibility only.  Oracle-X no longer loads
        any artifact from disk.
    epsilon_levels:
        Ignored — kept for API compatibility only.
    enable_adaptation:
        Ignored — kept for API compatibility only.
    """

    def __init__(
        self,
        model_path: Path | str | None = None,
        epsilon_levels: tuple[float, ...] = (0.00, 0.01, 0.02, 0.05),
        enable_adaptation: bool = ENABLE_ONLINE_ADAPTATION,
    ) -> None:
        # model_path retained for API compatibility; no pkl is loaded
        self._model_path: Path = (
            Path(model_path)
            if model_path is not None
            else Path(__file__).parent / "oracle_data" / "oracle_x_v1.pkl"
        )
        self._epsilon_levels = epsilon_levels
        self._enable_adaptation = enable_adaptation
        self._try_load_model()

    # ------------------------------------------------------------------
    # Public API — same signature as ThresholdOracle.predict() + context
    # ------------------------------------------------------------------

    def predict(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None = None,
        cost_aware: bool = False,
        context: RoutingContext | None = None,
    ) -> RoutingDecision:
        """Route a request to direct or TTF generation.

        When *context* carries a ``phi_score > 0`` the information-geometric
        Φ score drives the decision; otherwise the heuristic weighted-score is
        used as a legacy fallback.

        Falls back to a direct-generation decision on any error.
        """
        try:
            return self._predict_impl(
                features=features,
                backend=backend,
                model_id=model_id,
                latency_budget_ms=latency_budget_ms,
                cost_aware=cost_aware,
                context=context,
            )
        except Exception:
            logger.warning(
                "OracleX.predict: unexpected error — defaulting to direct",
                exc_info=True,
            )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.30,
                explanation="OracleX error — defaulting to direct",
            )

    def update_online(self, record: TelemetryRecord) -> None:
        """Deprecated — raises ``NotImplementedError`` in v0.3.

        Online adaptation is removed; Oracle-X uses the training-free Φ score.
        """
        warnings.warn(
            "update_online() removed in v0.3. Oracle-X uses information-geometric routing.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("update_online() removed. No online adaptation required.")

    @classmethod
    def from_benchmark_data(cls, *args: object, **kwargs: object) -> OracleX:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "from_benchmark_data() removed in v0.3. Oracle-X uses information-geometric routing.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("from_benchmark_data() removed. No training data required.")

    def save(self, *args: object, **kwargs: object) -> None:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "save() removed in v0.3. Oracle-X has no artifact to persist.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("save() removed.")

    def load(self, *args: object, **kwargs: object) -> None:
        """Deprecated — raises ``NotImplementedError`` in v0.3."""
        warnings.warn(
            "load() removed in v0.3. Oracle-X has no artifact to load.",
            DeprecationWarning,
            stacklevel=2,
        )
        raise NotImplementedError("load() removed.")

    # ------------------------------------------------------------------
    # Internal prediction
    # ------------------------------------------------------------------

    def _try_load_model(self) -> None:
        """No-op — Oracle-X requires no pkl artifact."""
        logger.debug("OracleX: information-geometric routing active, no artifact needed.")

    def _predict_impl(
        self,
        features: ComplexityFeatures,
        backend: str,
        model_id: str,
        latency_budget_ms: float | None,
        cost_aware: bool,
        context: RoutingContext | None,
    ) -> RoutingDecision:
        backend_key = backend.lower() if backend else "default"
        threshold = BACKEND_THRESHOLDS.get(backend_key, BACKEND_THRESHOLDS["default"])

        # Resolve overhead_pct and native_thinker from the backend registry;
        # fall back to hardcoded constants if the entry is absent.
        cap = get_registry().get(backend_key, model_id.split("/")[-1] if model_id else "*")
        overhead_pct = (
            cap.ttf_overhead_pct
            if cap.ttf_overhead_pct != 40.0 or backend_key in get_registry().known_backends()
            else BACKEND_TTF_OVERHEAD.get(backend_key, BACKEND_TTF_OVERHEAD["default"])
        )

        # Rule 1: Native thinker — check registry first, then hardcoded list
        if cap.native_thinker or _is_native_thinker(model_id):
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.95,
                explanation=f"Native thinker model '{model_id}' — direct only.",
            )

        # Rule 2: Latency budget — force direct if TTF overhead exceeds budget
        if latency_budget_ms is not None:
            estimated_overhead_ms = (overhead_pct / 100.0) * _DEFAULT_BASE_LATENCY_MS
            if estimated_overhead_ms > latency_budget_ms:
                return RoutingDecision(
                    strategy="direct",
                    expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                    expected_overhead_pct=0.0,
                    confidence=0.85,
                    explanation=(
                        f"TTF overhead ~{estimated_overhead_ms:.0f}ms exceeds "
                        f"budget {latency_budget_ms:.0f}ms."
                    ),
                )

        # Rule 3: Φ-based or heuristic routing
        return self._predict_heuristic(features, backend_key, threshold, overhead_pct, context)

    def _predict_heuristic(
        self,
        features: ComplexityFeatures,
        backend_key: str,
        threshold: float,
        overhead_pct: float,
        context: RoutingContext | None = None,
    ) -> RoutingDecision:
        """Route using Φ score (when available) or heuristic weighted-score fallback."""
        # Rule 2.5: Flat schema + low τ + no semantic complexity signals → direct.
        # Three schema signals must all be low (flat, few constraints, no ops/entropy)
        # before the extraction shortcut fires.  Any one elevated signal means the
        # prompt carries multi-step reasoning — TTF benefit outweighs the overhead.
        #
        # Two independent semantic signals guard against misclassification:
        #   1. required_reasoning_ops — explicit CoT verbs (calculate, assess, plan, …)
        #   2. token_entropy — lexical density, catches dense technical specs that
        #      have no keyword hits but have near-unique token distributions.
        _is_semantically_complex = (
            features.required_reasoning_ops >= _SEMANTIC_REASONING_OPS_THRESHOLD
            or features.token_entropy > _ENTROPY_COMPLEXITY_THRESHOLD
        )
        if (
            context is not None
            and context.phi_lambda2 < _FLAT_SCHEMA_LAMBDA2
            and context.phi_tau < _EXTRACTION_TAU_THRESHOLD
            and not _is_semantically_complex
        ):
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=0.80,
                explanation=(
                    f"Flat schema (λ̃₂={context.phi_lambda2:.3f}<{_FLAT_SCHEMA_LAMBDA2})"
                    f" + low τ={context.phi_tau:.3f}<{_EXTRACTION_TAU_THRESHOLD}"
                    " — extraction task, TTF not justified."
                ),
            )

        # Use information-geometric Φ score when context provides it
        if context is not None and context.phi_score > 0:
            phi = context.phi_score
            components = (
                f"λ̃₂={context.phi_lambda2:.3f} τ={context.phi_tau:.3f} ΔK={context.phi_delta_k:.3f}"
            )
            
            # Confidence reflects TWO signals:
            # 1. Distance from threshold (far = more confident in decision)
            # 2. Distance from 0.5 (how unambiguous is the zone itself)
            # 
            # Goal: "clearly in simple zone" (Φ=0.2) gets high confidence
            #       "clearly in complex zone" (Φ=0.8) gets high confidence
            #       "near threshold" (Φ≈0.65) gets medium confidence
            
            distance_from_threshold = abs(phi - threshold)
            # How far is phi from the "neutral" 0.5 point? (unambiguity signal)
            distance_from_neutral = abs(phi - 0.5)
            
            # Combine signals: threshold distance is primary, neutral distance is secondary
            # At threshold ± 0.02: confidence = 0.50
            # At threshold ± 0.20: confidence = 0.88
            # Very close to neutral (0.45-0.55): max confidence capped at 0.70
            
            if distance_from_threshold < 0.02:
                base_confidence = 0.50
            elif distance_from_threshold > 0.20:
                base_confidence = 0.88
            else:
                normalized = (distance_from_threshold - 0.02) / (0.20 - 0.02)
                base_confidence = 0.50 + (0.88 - 0.50) * normalized
            
            # Adjust: if very close to neutral, cap confidence (ambiguous zone)
            if distance_from_neutral < 0.05:
                base_confidence = min(base_confidence, 0.70)
            elif distance_from_neutral > 0.20:
                # Far from neutral: boost confidence slightly
                base_confidence = min(base_confidence + 0.05, 1.0)
            
            if phi > threshold:
                return RoutingDecision(
                    strategy="ttf",
                    expected_accuracy_delta=_TTF_ACCURACY_DELTA,
                    expected_overhead_pct=overhead_pct,
                    confidence=base_confidence,
                    explanation=(
                        f"OracleX Φ={phi:.3f} > threshold={threshold:.3f} → TTF ({components})."
                    ),
                )
            return RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
                expected_overhead_pct=0.0,
                confidence=base_confidence,
                explanation=(
                    f"OracleX Φ={phi:.3f} ≤ threshold={threshold:.3f} → direct ({components})."
                ),
            )

        # Heuristic fallback (legacy path when Φ is unavailable)
        fv = features.to_feature_vector()
        normalised = [
            min(fv[i] / _FEATURE_CAPS[i], 1.0) if _FEATURE_CAPS[i] > 0 else 0.0
            for i in range(len(fv))
        ]
        score = sum(n * w for n, w in zip(normalised, _FEATURE_WEIGHTS, strict=True))

        if score > threshold:
            return RoutingDecision(
                strategy="ttf",
                expected_accuracy_delta=_TTF_ACCURACY_DELTA,
                expected_overhead_pct=overhead_pct,
                confidence=_HEURISTIC_CONFIDENCE,
                explanation=(
                    f"OracleX heuristic: score={score:.3f} > threshold={threshold:.3f} "
                    f"for backend '{backend_key}'."
                ),
            )
        return RoutingDecision(
            strategy="direct",
            expected_accuracy_delta=_DIRECT_ACCURACY_DELTA,
            expected_overhead_pct=0.0,
            confidence=_HEURISTIC_CONFIDENCE,
            explanation=(
                f"OracleX heuristic: score={score:.3f} <= threshold={threshold:.3f} "
                f"for backend '{backend_key}'."
            ),
        )
