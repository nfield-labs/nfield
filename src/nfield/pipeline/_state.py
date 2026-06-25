"""Pipeline shared state threaded through all stages S0-S6."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nfield.assembly._blackboard import Blackboard
    from nfield.retrieval._bmx import BMXIndex
    from nfield.schema._types import CapacityLeaf, Field, FieldGroup, Segment

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

    # When True, Stage 5 scores each filled value against the excerpt the model saw and
    # marks an unsupported value FAILED (anti-hallucination; from ExtractionConfig).
    ground_values: bool = False
    # Accept threshold for the grounding score (from ExtractionConfig).
    grounding_min_score: float = 0.5
    # Closed-book mode: extract from model knowledge, no document (from ExtractionConfig).
    closed_book: bool = False
    # Opt-in two-sample self-consistency abstention for closed-book (from ExtractionConfig).
    self_consistency: bool = False
    # Closed-book paths the model abstained on (NULL or no sample agreement); recovery skips them.
    abstained: set[str] = field(default_factory=set)

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

    # Grounding score in [0, 1] per filled groundable field, written by Stage 5 when
    # ``ground_values`` is set; read by Stage 6 to report the hallucination rate. A path
    # is present only if it was grounding-checked (filled and of a groundable type).
    grounding_scores: dict[str, float] = field(default_factory=dict)

    # Count of SFEP output lines whose path was not in the schema (the model emitted a
    # field outside the requested set) — a format-drift signal accumulated across all
    # extraction calls and reported in Metadata.
    unknown_lines: int = 0

    # API call counts grouped by call site, so K can be attributed to extraction,
    # validation retry, or recovery. ``in_recovery`` marks calls issued by the
    # recovery pass, which reuses the extraction and validation stages.
    in_recovery: bool = False
    calls_by_origin: dict[str, int] = field(default_factory=dict)

    # Per-field guidance for a re-extraction call: ``field path -> reason`` describing
    # why the previous attempt failed (invalid value, conflict, absence). Populated by
    # the recovery pass and rendered into the extraction prompt; empty on the first pass.
    field_reasons: dict[str, str] = field(default_factory=dict)

    def record_calls(self, origin: str, n: int = 1) -> None:
        """Count *n* API calls under *origin*, updating K and the breakdown.

        While the recovery pass runs (``in_recovery``), the origin is prefixed
        ``recovery_`` so the recovery wave's reuse of Stages 4-5 is attributed to
        recovery rather than the primary pass.

        Args:
            origin: Call-site label (e.g. ``"extract"``, ``"s5_retry"``).
            n: Number of calls to record. Default 1.
        """
        key = f"recovery_{origin}" if self.in_recovery else origin
        self.K += n
        self.calls_by_origin[key] = self.calls_by_origin.get(key, 0) + n
