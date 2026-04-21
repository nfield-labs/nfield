"""
Φ-Guided Thinking Controller

Shapes LLM thinking strategy using Φ components:
- λ̃₂ (schema_graph_complexity) → decomposition strategy
- τ (constraint_tightness) → constraint focus
- ΔK (alignment_gap) → vocabulary bridge

The core insight: Use information-theoretic signals to guide cognition,
not just for routing but for controlling HOW the model thinks.
"""

from typing import Optional
from formatshield.oracle.routing_score import RoutingScore
from formatshield.reasoning.reasoning_task import ThinkingShaping


class PhiController:
    """Control LLM thinking using Φ components."""

    def __init__(self, routing_score: RoutingScore):
        """
        Initialize controller with routing context.

        Args:
            routing_score: RoutingScore containing λ̃₂, τ, ΔK, phi
        """
        self.routing_score = routing_score
        self.lambda2 = routing_score.lambda2
        self.tau = routing_score.tau
        self.delta_k = routing_score.delta_k
        self.phi = routing_score.phi

    def shape(self) -> ThinkingShaping:
        """
        Main entry point: generate thinking shaping strategy.

        Combines three Φ components into a cohesive reasoning strategy.

        Returns:
            ThinkingShaping with decomposition, focus, and vocabulary bridge
        """
        decomposition = self._strategy_from_lambda2()
        constraint_focus = self._focus_from_tau()
        vocab_bridge = self._bridge_from_delta_k() if self.delta_k > 0.5 else None
        budget = self._estimate_thinking_budget()

        return ThinkingShaping(
            decomposition_strategy=decomposition,
            constraint_focus=constraint_focus,
            vocabulary_bridge=vocab_bridge,
            thinking_budget=budget,
        )

    def _strategy_from_lambda2(self) -> str:
        """
        Map λ̃₂ (schema graph complexity) to decomposition strategy.

        λ̃₂ ∈ [0, 1]:
        - 0.0–0.2: FLAT_EXTRACTION (no dependencies, direct mapping)
        - 0.2–0.4: HIERARCHICAL_EXTRACTION (parent-child structure)
        - 0.4–0.6: DEPENDENCY_AWARE (fields depend on each other)
        - 0.6–1.0: FULL_STRUCTURAL_REASONING (deeply interconnected)

        Returns:
            Strategy string describing how to decompose the problem
        """
        if self.lambda2 < 0.2:
            return (
                "FLAT_EXTRACTION: The output has independent fields with no dependencies. "
                "Extract and map each field directly. No multi-step reasoning needed."
            )

        elif self.lambda2 < 0.4:
            return (
                "HIERARCHICAL_EXTRACTION: The output has a hierarchical structure "
                "(parent → child). First understand parent concepts, then map to leaf fields. "
                "Reasoning is mostly structural, not logical."
            )

        elif self.lambda2 < 0.6:
            return (
                "DEPENDENCY_AWARE_REASONING: Fields have mutual dependencies. "
                "Reason about how fields relate to each other. "
                "Resolve conflicts or incompatibilities as you encounter them."
            )

        else:
            return (
                "FULL_STRUCTURAL_REASONING: The output is deeply interconnected. "
                "Decompose by connected components first. "
                "Reason holistically about how all parts fit together. "
                "Check for global consistency, not just local field validity."
            )

    def _focus_from_tau(self) -> str:
        """
        Map τ (constraint_tightness) to constraint focus level.

        τ ∈ [0, 1]:
        - 0.0–0.4: EXPLORATORY (schema is flexible, reasoning can roam)
        - 0.4–0.7: SOFT_CONSTRAINTS (rules matter but allow interpretation)
        - 0.7–1.0: STRICT_ENUMERATION (schema has rigid enums/ranges)

        Returns:
            Strategy string describing constraint enforcement level
        """
        if self.tau < 0.4:
            return (
                "EXPLORATORY: The schema is flexible and allows nuanced responses. "
                "You can explain uncertainty, provide alternatives, or note edge cases. "
                "Prefer explanatory clarity over strict adherence."
            )

        elif self.tau < 0.7:
            return (
                "SOFT_CONSTRAINTS: The schema has rules but allows some flexibility. "
                "Prefer exact matches when possible. "
                "If you deviate from a rule, explain why explicitly."
            )

        else:
            return (
                "STRICT_ENUMERATION: The schema has rigid enums and ranges. "
                "Every choice must come from the allowed set. "
                "No approximations or interpretations. "
                "If the correct value is not in the enum, flag this as impossible."
            )

    def _bridge_from_delta_k(self) -> Optional[str]:
        """
        Map ΔK (alignment_gap) to vocabulary bridge instruction.

        ΔK ∈ [0, 1]:
        - 0.0–0.5: Good alignment (no special action needed)
        - 0.5–1.0: Large gap (vocabulary mapping needed)

        Returns:
            Bridge instruction for vocabulary mapping, or None if alignment is good
        """
        if self.delta_k <= 0.5:
            return None

        return (
            "VOCABULARY BRIDGE: The prompt uses different terminology than the schema. "
            "Map prompt concepts to schema field names explicitly. "
            "For example, if the prompt says 'decision' and schema has 'recommendation', "
            "clarify the mapping: 'I interpret the decision (from the prompt) as recommendation (from the schema)'."
        )

    def _estimate_thinking_budget(self) -> int:
        """
        Estimate thinking token budget based on Φ.

        Φ ∈ [0, 1]:
        - Φ < 0.5: 256 tokens (simple extraction)
        - 0.5 ≤ Φ < 0.65: 512 tokens (lightweight reasoning)
        - 0.65 ≤ Φ < 0.80: 1024 tokens (standard reasoning)
        - 0.80 ≤ Φ < 0.95: 1536 tokens (deep reasoning)
        - Φ ≥ 0.95: 2048+ tokens (maximum reasoning, self-consistency)

        Returns:
            Estimated thinking budget in tokens
        """
        if self.phi < 0.5:
            return 256

        elif self.phi < 0.65:
            return 512

        elif self.phi < 0.80:
            return 1024

        elif self.phi < 0.95:
            return 1536

        else:
            return 2048


def shape_thinking_with_phi(routing_score: RoutingScore) -> ThinkingShaping:
    """
    Public API: shape thinking using Φ components.

    Args:
        routing_score: RoutingScore (contains λ̃₂, τ, ΔK, phi)

    Returns:
        ThinkingShaping with decomposition, focus, vocabulary bridge, budget
    """
    controller = PhiController(routing_score)
    return controller.shape()
