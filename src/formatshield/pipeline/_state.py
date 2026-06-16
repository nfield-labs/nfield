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

    # Optional caller steering, constant across all leaves, prepended to the
    # built-in SFEP prompt and counted in leaf overhead so it shrinks the
    # per-leaf document budget.
    instructions: str = ""

    # When True, Stage 4 injects resolved upstream dependency values into a
    # dependent leaf's prompt (set by the engine from ExtractionConfig).
    inject_dependencies: bool = False

    # When True, the prompt lets the model use its own knowledge for fields the
    # document does not state (from ExtractionConfig).
    knowledge_fallback: bool = False

    # When True, store values exactly as extracted; when False (default), normalize
    # formatted values to the schema type before writing (from ExtractionConfig).
    strict_validation: bool = False

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
    lexical_index: BMXIndex | None = None

    # Record structure: field -> record index, record -> block token cost. Drives
    # record-aware (Group Bin Packing) packing in Stage 2C. Empty = no record axis.
    record_ordinal: dict[str, int] = field(default_factory=dict)
    record_block_tokens: dict[int, int] = field(default_factory=dict)
    # Record blocks as segments (record index -> its segments; shared header segments).
    # Let Stage 5 rebuild a failed field's record block for a small, record-local retry
    # excerpt — independent of the leaf, so it works for recovery leaves too.
    record_block_segments: dict[int, list[Segment]] = field(default_factory=dict)
    record_header_segments: list[Segment] = field(default_factory=list)

    # Stage 2C: capacity packing
    leaves: list[CapacityLeaf] = field(default_factory=list)
    execution_order: list[list[CapacityLeaf]] = field(default_factory=list)
    K_min: int = 1

    # Stage 4-5: extraction and validation
    blackboard: Blackboard | None = None
    K: int = 0
    retry_rounds: int = 0
