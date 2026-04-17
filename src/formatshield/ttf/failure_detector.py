"""
FormatShield FailureModeDetector.

Intercepts ThresholdOracle routing decisions and overrides TTF when it detects
cases where the two-pass strategy is likely to reduce — not improve — accuracy.

Each detected failure mode is logged for paper Table 2 ("When TTF Hurts")
which is the most academically honest contribution: systematically documenting
the boundaries of our own technique.

Failure modes implemented (6 total, per v0.0.1 spec):

1. ``simple_extraction``   — schema_depth ≤ 1 AND prompt_length_bucket ≤ 1
2. ``schema_too_constrained`` — required fields > 15 or enum count > 50
3. ``native_thinker``      — model already reasons internally (o1, DeepSeek R1)
4. ``short_prompt``        — token count too small for reasoning to matter
5. ``template_fill``       — prompt is mostly fixed structure (fill-in-the-blank)
6. ``ambiguous_schema``    — anyOf / oneOf at root level adds schema confusion

Reference: CRANE arXiv 2502.09061 § 4.2, "Format Tax" arXiv 2604.03616 § 3.
"""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any

from formatshield.oracle.routing_decision import RoutingDecision
from formatshield.scorer.features import ComplexityFeatures

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Native thinker models — they already reason internally; applying TTF on top
# doubles reasoning cost and degrades structured output quality.
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Canonical failure-mode label registry
# ---------------------------------------------------------------------------

FAILURE_MODES: list[str] = [
    "simple_extraction",  # schema too shallow + short prompt → TTF hurts
    "schema_too_constrained",  # too many required fields → constrained always better
    "native_thinker",  # model already has internal reasoning
    "short_prompt",  # not enough context for TTF to help
    "template_fill",  # highly templated prompt → no reasoning needed
    "ambiguous_schema",  # anyOf/oneOf at root → TTF can't help schema selection
]

# ---------------------------------------------------------------------------
# Native thinker models — they already reason internally; applying TTF on top
# doubles reasoning cost and degrades structured output quality.
# ---------------------------------------------------------------------------

# Failure modes that unconditionally force direct routing (hard overrides).
# Defined at module level so should_override_to_direct() avoids a local-var N806.
_OVERRIDE_MODES: frozenset[str] = frozenset(
    {
        "simple_extraction",
        "short_prompt",
        "native_thinker",
    }
)

NATIVE_THINKERS: frozenset[str] = frozenset(
    {
        "o1",
        "o1-mini",
        "o1-preview",
        "o3",
        "o3-mini",
        "o4-mini",
        "deepseek-r1",
        "deepseek-r1-distill-llama-70b",
        "deepseek-r1-distill-qwen-32b",
        "deepseek-r1-zero",
    }
)

# Fraction of prompt "slots" that must be non-variable for template detection.
_TEMPLATE_FIXED_FRACTION_THRESHOLD: float = 0.70

# Maximum count of required fields or enum values before we consider a schema
# too constrained for TTF to help.
_MAX_REQUIRED_FIELDS: int = 15
_MAX_ENUM_VALUES: int = 50


class FailureModeDetector:
    """Detect cases where TTF routing would reduce accuracy.

    :meth:`detect` runs all six checks and returns a list of detected mode
    labels.  :meth:`should_override_to_direct` decides whether the combination
    of detected modes is severe enough to force a ``"direct"`` routing override.

    The detector is fully stateless per call — it accumulates no internal
    state and is safe to share across coroutines.

    Example::

        detector = FailureModeDetector()
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        features = ComplexityFeatures(
            token_entropy=0.3,
            schema_depth=1,
            required_reasoning_ops=0,
            instruction_tune_score=0.5,
            prompt_length_bucket=0,
            schema_constraint_count=1,
        )
        modes = detector.detect(features, "groq/llama-3.3-70b-versatile", schema)
        # → ["simple_extraction", "short_prompt"]
        if detector.should_override_to_direct(modes):
            decision = RoutingDecision(strategy="direct", ...)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(
        self,
        features: ComplexityFeatures,
        model_id: str,
        schema: dict[str, Any] | None = None,
    ) -> list[str]:
        """Run all failure-mode checks and return detected mode labels.

        Parameters
        ----------
        features:
            Complexity features from :class:`~formatshield.scorer.ComplexityScorer`.
        model_id:
            Full model identifier (e.g. ``"groq/llama-3.3-70b-versatile"``).
        schema:
            Optional JSON Schema dict for schema-specific checks.

        Returns
        -------
        list[str]
            Labels of all detected failure modes.  Empty list means no
            failure modes were detected and TTF routing stands.
        """
        try:
            return self._detect_impl(features, model_id, schema or {})
        except Exception:
            logger.warning(
                "FailureModeDetector.detect: unexpected error — returning empty list",
                exc_info=True,
            )
            return []

    def should_override_to_direct(self, failure_modes: list[str]) -> bool:
        """Decide whether the detected failure modes mandate a direct-generation override.

        Returns ``True`` when any *hard-override* mode appears in *failure_modes*.

        Hard-override modes
        ~~~~~~~~~~~~~~~~~~~
        * ``simple_extraction`` – schema_depth ≤ 1 AND prompt_length_bucket ≤ 1
        * ``short_prompt``      – prompt_length_bucket == 0 (< 50 tokens)
        * ``native_thinker``    – model already reasons internally

        Advisory modes (``schema_too_constrained``, ``template_fill``,
        ``ambiguous_schema``) are attached to the
        :class:`~formatshield.oracle.RoutingDecision` for observability but do
        not themselves trigger a hard route change.

        Parameters
        ----------
        failure_modes:
            List returned by :meth:`detect`.

        Returns
        -------
        bool
            ``True`` if the routing decision should be forced to ``"direct"``.
        """
        return bool(set(failure_modes) & _OVERRIDE_MODES)

    def check(
        self,
        decision: RoutingDecision,
        features: ComplexityFeatures,
        model_id: str,
        schema: dict[str, Any] | None = None,
    ) -> tuple[RoutingDecision, list[str]]:
        """Convenience wrapper: detect failure modes and return the (possibly
        overridden) routing decision alongside the detected mode labels.

        Parameters
        ----------
        decision:
            The oracle's original routing decision.
        features:
            Complexity features from :class:`~formatshield.scorer.ComplexityScorer`.
        model_id:
            Full model identifier.
        schema:
            Optional JSON Schema dict.

        Returns
        -------
        tuple[RoutingDecision, list[str]]
            ``(updated_decision, failure_mode_labels)`` where
            ``updated_decision.strategy == "direct"`` if any hard override was
            triggered.
        """
        modes = self.detect(features, model_id, schema)

        if decision.strategy == "ttf" and self.should_override_to_direct(modes):
            explanation = f"TTF overridden to direct — failure modes detected: {', '.join(modes)}."
            overridden = RoutingDecision(
                strategy="direct",
                expected_accuracy_delta=0.0,
                expected_overhead_pct=0.0,
                confidence=decision.confidence,
                explanation=explanation,
                failure_modes=modes,
            )
            logger.info(explanation)
            return overridden, modes

        # Attach detected modes to the decision even if we don't override.
        # Use replace() to avoid mutating the caller's object.
        if modes:
            decision = replace(decision, failure_modes=modes)
        return decision, modes

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    def _detect_impl(
        self,
        features: ComplexityFeatures,
        model_id: str,
        schema: dict[str, Any],
    ) -> list[str]:
        detected: list[str] = []

        if self._is_native_thinker(model_id):
            detected.append("native_thinker")

        if self._is_short_prompt(features):
            detected.append("short_prompt")

        if self._is_simple_extraction(features):
            detected.append("simple_extraction")

        if self._is_schema_too_constrained(features, schema):
            detected.append("schema_too_constrained")

        if self._is_template_fill(features):
            detected.append("template_fill")

        if self._is_ambiguous_schema(schema):
            detected.append("ambiguous_schema")

        if detected:
            logger.debug(
                "FailureModeDetector: detected %d mode(s) for model=%s: %s",
                len(detected),
                model_id,
                detected,
            )

        return detected

    # ------------------------------------------------------------------
    # Individual failure-mode checks
    # ------------------------------------------------------------------

    @staticmethod
    def _is_native_thinker(model_id: str) -> bool:
        """Return ``True`` if the model already reasons internally.

        Checks both the raw model_id and the portion after any ``"provider/"``
        prefix to handle ``"groq/deepseek-r1"`` style identifiers.
        """
        lower = model_id.lower()
        # Strip optional backend prefix
        if "/" in lower:
            lower_stripped = lower.split("/", maxsplit=1)[1]
        else:
            lower_stripped = lower

        return any(thinker in lower or thinker in lower_stripped for thinker in NATIVE_THINKERS)

    @staticmethod
    def _is_short_prompt(features: ComplexityFeatures) -> bool:
        """Return ``True`` for very short prompts where TTF adds no value.

        Short prompts (< 50 tokens, bucket 0) lack sufficient context for a
        meaningful reasoning pass.  The extra round-trip cost never pays off.
        """
        return features.prompt_length_bucket == 0

    @staticmethod
    def _is_simple_extraction(features: ComplexityFeatures) -> bool:
        """Return ``True`` for simple, flat extraction tasks.

        A task is simple if:
        * The target schema has a nesting depth of ≤ 1 (flat object), AND
        * The prompt is short-to-medium (bucket ≤ 1, i.e. < 200 tokens).
        These tasks are well-served by direct constrained generation.
        """
        return features.schema_depth <= 1 and features.prompt_length_bucket <= 1

    @staticmethod
    def _is_schema_too_constrained(
        features: ComplexityFeatures,
        schema: dict[str, Any],
    ) -> bool:
        """Return ``True`` when the schema is so constrained that TTF cannot help.

        Checks:
        * required field count > 15 (feature: schema_constraint_count)
        * Total enum values across the schema > 50 (exhaustive enumeration)

        A heavily-constrained schema forces the model into a narrow output space
        regardless of the quality of its reasoning — TTF overhead is wasted.
        """
        if features.schema_constraint_count > _MAX_REQUIRED_FIELDS:
            return True

        # Count total enum values in schema
        enum_total = _count_enum_values(schema)
        return enum_total > _MAX_ENUM_VALUES

    @staticmethod
    def _is_template_fill(features: ComplexityFeatures) -> bool:
        """Return ``True`` for structured template-fill tasks.

        Template-fill prompts have very low token entropy (the prompt is mostly
        fixed boilerplate) and low reasoning ops (no CoT keywords).  Direct
        generation handles these best because there is nothing to "think through".
        """
        # Low entropy + very few reasoning keywords = template fill
        return features.token_entropy < 0.35 and features.required_reasoning_ops <= 1

    @staticmethod
    def _is_ambiguous_schema(schema: dict[str, Any]) -> bool:
        """Return ``True`` when the schema root uses ``anyOf`` / ``oneOf``.

        These combinators at the root level mean the model has to choose between
        multiple incompatible output shapes.  TTF may still help, but the engine
        should inject a schema hint into Pass 1 to guide schema selection.
        (This is a *soft* failure mode — does not trigger a hard direct override.)
        """
        return "anyOf" in schema or "oneOf" in schema


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _count_enum_values(schema: dict[str, Any]) -> int:
    """Recursively count the total number of enum values across *schema*."""
    if not isinstance(schema, dict):
        return 0

    total = 0

    # Count enum values at this level
    enum = schema.get("enum")
    if isinstance(enum, list):
        total += len(enum)

    # Recurse into sub-schemas
    for key in ("properties", "$defs", "definitions"):
        mapping = schema.get(key)
        if isinstance(mapping, dict):
            for sub in mapping.values():
                total += _count_enum_values(sub)

    for key in ("items", "additionalProperties", "if", "then", "else", "not"):
        sub = schema.get(key)
        if isinstance(sub, dict):
            total += _count_enum_values(sub)

    for key in ("anyOf", "oneOf", "allOf"):
        branches = schema.get(key)
        if isinstance(branches, list):
            for branch in branches:
                total += _count_enum_values(branch)

    return total
