"""Pipeline shared state threaded through all stages S0-S6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from formatshield.assembly._blackboard import Blackboard
    from formatshield.retrieval._bmx import BMXIndex
    from formatshield.schema._types import CapacityLeaf, Field, FieldGroup, Segment

__all__ = ["PipelineState"]


@dataclass
class PipelineState:
    """Shared mutable state threaded through pipeline stages S0-S6.

    Each stage is responsible for populating its own section.
    A field belongs here only if the NEXT stage reads it.

    Example:
        >>> state = PipelineState()
        >>> state.K
        0
    """

    # Stage 0: resource calibration
    chars_per_token: float = 0.0
    C_eff: int = 0
    M_O: int = 0
    C_usable: float = 0.0

    # Optional caller-supplied prompt context: a system prompt and a user/task
    # prompt. Constant across all leaves, prepended to the built-in SFEP prompts,
    # and counted in leaf overhead so they shrink the per-leaf document budget.
    system_prompt: str = ""
    user_prompt: str = ""

    # When True, Stage 4 injects resolved upstream dependency values into a
    # dependent leaf's prompt (set by the engine from ExtractionConfig).
    inject_dependencies: bool = False

    # When True, the prompt lets the model use its own knowledge for fields the
    # document does not state (from ExtractionConfig).
    knowledge_fallback: bool = False

    # Max leaf extraction calls in flight at once (from ExtractionConfig); bounds
    # Stage 4 concurrency so wide schemas do not trip provider rate limits.
    max_concurrent_calls: int = 4

    # Stage 1: schema analysis
    fields: list[Field] = field(default_factory=list)
    field_by_path: dict[str, Field] = field(default_factory=dict)
    dep_dag: dict[str, set[str]] = field(default_factory=dict)

    # Stage 2A: structural grouping
    groups: list[FieldGroup] = field(default_factory=list)
    group_map: dict[str, FieldGroup] = field(default_factory=dict)

    # Stage 2.5: document pre-pass
    segments: list[Segment] = field(default_factory=list)
    bm25_index: BMXIndex | None = None

    # Stage 2C: capacity packing
    leaves: list[CapacityLeaf] = field(default_factory=list)
    execution_order: list[list[CapacityLeaf]] = field(default_factory=list)
    K_min: int = 1

    # Stage 4-5: extraction and validation
    blackboard: Blackboard | None = None
    K: int = 0
    retry_rounds: int = 0
