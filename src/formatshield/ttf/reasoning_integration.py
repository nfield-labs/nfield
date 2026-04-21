"""
Reasoning Module Integration for TTFEngine

Integrates the Schema-Conditioned Reasoning Engine into TTFEngine's Pass 1 prompt generation.
This module provides helpers to:
1. Compile schema to task-specific reasoning instructions
2. Extract constraints from schema
3. Shape thinking strategy using Φ components
4. Merge reasoning context into the Pass 1 prompt
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formatshield.oracle.routing_score import RoutingScore
    from formatshield.reasoning import ReasoningTask, ConstraintRule, ThinkingShaping, ReasoningTaskConfig

logger = logging.getLogger(__name__)


def build_reasoning_context(
    schema: dict[str, Any],
    prompt: str,
    routing_score: RoutingScore,
    config: ReasoningTaskConfig | None = None,
) -> dict[str, Any]:
    """Build reasoning context (task, constraints, thinking strategy) from schema and routing.

    This function safely integrates the reasoning module into TTFEngine by:
    1. Compiling the schema into task-specific instructions
    2. Extracting constraint rules
    3. Shaping thinking strategy based on Φ components
    4. Gracefully handling any errors

    Parameters
    ----------
    schema
        JSON Schema dict defining the output structure
    prompt
        Original user prompt
    routing_score
        RoutingScore containing Φ components (λ̃₂, τ, ΔK)
    config
        Optional ReasoningTaskConfig to enable/disable features

    Returns
    -------
    dict[str, Any]
        Dict with keys: 'task', 'constraints', 'thinking_shaping', 'error' (if any)
        All keys are guaranteed to be present; 'error' is None if successful.
    """
    from formatshield.reasoning import (
        compile_schema_to_task,
        extract_constraints,
        shape_thinking_with_phi,
        ReasoningTaskConfig,
    )

    context: dict[str, Any] = {
        "task": None,
        "constraints": [],
        "thinking_shaping": None,
        "error": None,
    }

    # Use defaults if config not provided
    if config is None:
        config = ReasoningTaskConfig()

    # Early exit if all features disabled
    if not config.is_any_enabled():
        logger.debug("ReasoningIntegration: all features disabled in config")
        return context

    try:
        # 1. Compile schema to task
        if config.enable_schema_aware_reasoning:
            try:
                context["task"] = compile_schema_to_task(schema, routing_score)
                logger.debug(
                    "ReasoningIntegration: compiled schema to %s task",
                    context["task"].task_type,
                )
            except Exception as e:
                logger.warning("ReasoningIntegration: schema compilation failed: %s", e)
                context["error"] = str(e)

        # 2. Extract constraints
        if config.enable_constraint_injection:
            try:
                context["constraints"] = extract_constraints(schema, prompt, routing_score)
                logger.debug(
                    "ReasoningIntegration: extracted %d constraint rules",
                    len(context["constraints"]),
                )
            except Exception as e:
                logger.warning("ReasoningIntegration: constraint extraction failed: %s", e)
                if context["error"] is None:
                    context["error"] = str(e)

        # 3. Shape thinking
        if config.enable_phi_shaping:
            try:
                context["thinking_shaping"] = shape_thinking_with_phi(routing_score)
                logger.debug(
                    "ReasoningIntegration: shaped thinking (budget=%d tokens)",
                    context["thinking_shaping"].thinking_budget,
                )
            except Exception as e:
                logger.warning("ReasoningIntegration: thinking shaping failed: %s", e)
                if context["error"] is None:
                    context["error"] = str(e)

    except Exception as e:
        logger.error("ReasoningIntegration: unexpected error building context: %s", e)
        context["error"] = str(e)

    return context


def inject_reasoning_into_prompt(
    base_prompt: str,
    reasoning_context: dict[str, Any],
    schema: dict[str, Any] | None = None,
) -> str:
    """Inject reasoning context into the base Pass 1 prompt.

    Merges task instructions, constraint rules, and thinking strategy directives
    into the prompt to guide the model's reasoning.

    Parameters
    ----------
    base_prompt
        The base Pass 1 prompt (already built by TTFEngine)
    reasoning_context
        Dict returned by build_reasoning_context()
    schema
        Optional schema dict (for context)

    Returns
    -------
    str
        Enhanced prompt with reasoning context injected
    """
    if reasoning_context.get("error"):
        logger.debug(
            "ReasoningIntegration: skipping injection due to prior error: %s",
            reasoning_context["error"],
        )
        return base_prompt

    # If no context to inject, return base
    task = reasoning_context.get("task")
    constraints = reasoning_context.get("constraints", [])
    thinking_shaping = reasoning_context.get("thinking_shaping")

    if task is None and not constraints and thinking_shaping is None:
        return base_prompt

    # Build injection sections
    sections = []

    # 1. Task-specific instructions
    if task is not None:
        sections.append("\n## REASONING TASK")
        sections.append(f"Task Type: {task.task_type.upper()}")
        sections.append("\n### Instructions")
        sections.append(task.instructions)
        if task.schema_summary:
            sections.append("\n### Schema Summary")
            sections.append(task.schema_summary)

    # 2. Constraint rules (high-priority only)
    if constraints:
        hard_constraints = [c for c in constraints if c.priority == "hard"]
        if hard_constraints:
            sections.append("\n## CONSTRAINTS")
            sections.append("These constraints are MANDATORY and must be satisfied:")
            for rule in hard_constraints[:5]:  # Limit to top 5 to avoid prompt bloat
                sections.append(f"- {rule.description}")

    # 3. Thinking strategy
    if thinking_shaping is not None:
        sections.append("\n## THINKING STRATEGY")
        sections.append(f"Decomposition: {thinking_shaping.decomposition_strategy[:100]}...")
        sections.append(f"Constraint Focus: {thinking_shaping.constraint_focus[:100]}...")
        if thinking_shaping.vocabulary_bridge:
            sections.append(
                f"Vocabulary Mapping: {thinking_shaping.vocabulary_bridge[:100]}..."
            )

    # Merge into base prompt
    injection = "\n".join(sections)
    enhanced_prompt = f"{base_prompt}\n{injection}"

    logger.debug("ReasoningIntegration: injected reasoning context (%d chars)", len(injection))

    return enhanced_prompt
