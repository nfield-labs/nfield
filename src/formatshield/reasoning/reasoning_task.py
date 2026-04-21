"""
Data contracts for Schema-Conditioned Reasoning Engine.

Defines the core dataclasses that flow through the reasoning pipeline:
- ReasoningTask: compiled schema → task instructions
- ConstraintRule: extracted rules for validation
- ThinkingShaping: Φ-controlled reasoning strategy
- ReasoningTaskConfig: feature flags and tuning parameters
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional


@dataclass
class ConstraintRule:
    """
    A single constraint extracted from schema or inferred from prompt.

    Examples:
    - Enum rule: status must be in ["pending", "approved", "rejected"]
    - Range rule: age must be between 0 and 150
    - Conditional rule: if special_category=true, then dpia_required=true
    - Dependency rule: if transfer_to_third_country=true, then transfer_mechanism must exist
    """

    rule_type: Literal["enum", "range", "conditional", "dependency", "consistency", "vocabulary"]
    description: str
    schema_path: str
    constraint_value: Any
    injection_point: Literal["pass1_system", "pass1_user", "pass2_system", "validation"]
    validator: Optional[Callable[[Any], bool]] = None
    priority: Literal["hard", "soft"] = "soft"

    def __post_init__(self) -> None:
        """Validate constraint rule structure."""
        if not self.schema_path:
            raise ValueError("schema_path must not be empty")
        if not self.description:
            raise ValueError("description must not be empty")


@dataclass
class ThinkingShaping:
    """
    Strategy for shaping LLM thinking based on Φ components.

    Computed from:
    - λ̃₂ (schema_graph_complexity) → decomposition_strategy
    - τ (constraint_tightness) → constraint_focus
    - ΔK (alignment_gap) → vocabulary_bridge
    - Φ overall → thinking_budget
    """

    decomposition_strategy: str
    constraint_focus: str
    vocabulary_bridge: Optional[str] = None
    thinking_budget: int = 256

    def __post_init__(self) -> None:
        """Validate thinking shaping."""
        if self.thinking_budget < 0:
            raise ValueError("thinking_budget must be >= 0")
        if not self.decomposition_strategy:
            raise ValueError("decomposition_strategy must not be empty")
        if not self.constraint_focus:
            raise ValueError("constraint_focus must not be empty")


@dataclass
class ReasoningTask:
    """
    Compiled schema → reasoning task program.

    This represents the complete reasoning task extracted from a JSON schema,
    ready to be injected into the LLM's system prompt during Pass 1.
    """

    task_type: Literal["extraction", "classification", "reasoning"]
    instructions: str
    constraints: List[ConstraintRule] = field(default_factory=list)
    field_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    schema_summary: str = ""
    vocabulary_bridge: Optional[str] = None
    thinking_strategy: Optional[str] = None
    estimated_tokens: int = 256

    def __post_init__(self) -> None:
        """Validate reasoning task."""
        if not self.instructions:
            raise ValueError("instructions must not be empty")
        if not self.task_type:
            raise ValueError("task_type must be one of: extraction, classification, reasoning")
        if self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be >= 0")

    def merge_with_shaping(self, shaping: ThinkingShaping) -> str:
        """
        Merge task instructions with thinking shaping.

        Returns combined prompt that includes both task steps and thinking strategy.
        """
        combined = self.instructions + "\n\n"

        if shaping.decomposition_strategy:
            combined += f"REASONING STRATEGY: {shaping.decomposition_strategy}\n"

        if shaping.constraint_focus:
            combined += f"CONSTRAINT FOCUS: {shaping.constraint_focus}\n"

        if shaping.vocabulary_bridge:
            combined += f"\nVOCABULARY BRIDGE:\n{shaping.vocabulary_bridge}\n"

        return combined


@dataclass
class ReasoningTaskConfig:
    """
    Configuration for Schema-Conditioned Reasoning Engine.

    All features default to False for backwards compatibility.
    Opt-in via explicit config.
    """

    enable_schema_aware_reasoning: bool = False
    enable_constraint_injection: bool = False
    enable_phi_shaping: bool = False

    # Tuning parameters
    vocabulary_bridge_threshold: float = 0.5
    max_task_instructions_tokens: int = 500
    cache_compiled_tasks: bool = True

    # Debugging
    debug_log_reasoning: bool = False

    def __post_init__(self) -> None:
        """Validate configuration."""
        if not 0 <= self.vocabulary_bridge_threshold <= 1:
            raise ValueError("vocabulary_bridge_threshold must be in [0, 1]")
        if self.max_task_instructions_tokens <= 0:
            raise ValueError("max_task_instructions_tokens must be > 0")

    def is_any_enabled(self) -> bool:
        """Check if any reasoning feature is enabled."""
        return (
            self.enable_schema_aware_reasoning
            or self.enable_constraint_injection
            or self.enable_phi_shaping
        )
