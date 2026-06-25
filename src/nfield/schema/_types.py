from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "SEGMENT_TYPE_STRUCTURED",
    "SEGMENT_TYPE_TABULAR",
    "SEGMENT_TYPE_UNSTRUCTURED",
    "_VALID_SEGMENT_TYPES",
    "CapacityLeaf",
    "Field",
    "FieldGroup",
    "Segment",
]

# ---------------------------------------------------------------------------
# Segment type constants
# ---------------------------------------------------------------------------
SEGMENT_TYPE_STRUCTURED: str = "structured"
SEGMENT_TYPE_TABULAR: str = "tabular"
SEGMENT_TYPE_UNSTRUCTURED: str = "unstructured"
_VALID_SEGMENT_TYPES: frozenset[str] = frozenset(
    {SEGMENT_TYPE_STRUCTURED, SEGMENT_TYPE_TABULAR, SEGMENT_TYPE_UNSTRUCTURED}
)


# ---------------------------------------------------------------------------
# Immutable value objects
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Segment:
    """A contiguous span of text from the source document.

    Segments are the unit of retrieval in Stage 2. Each segment has a type
    that determines how it is processed (structured/tabular/unstructured).

    Attributes:
        text: Raw text content of the segment.
        start: Character offset of the first character in the original document.
        end: Character offset one past the last character (exclusive).
        segment_type: One of "structured", "tabular", or "unstructured".
        segment_id: Stable unique ID used for deduplication across passes.

    Example:
        >>> seg = Segment(text="John Smith", start=0, end=10, segment_type="unstructured")
        >>> seg.text
        'John Smith'
        >>> seg.end - seg.start
        10
    """

    text: str
    start: int
    end: int
    segment_type: str
    segment_id: int = 0

    def __post_init__(self) -> None:
        """Validate ``segment_type`` against the known set at the boundary.

        Raises:
            ValueError: If ``segment_type`` is not one of
                ``_VALID_SEGMENT_TYPES``.
        """
        if self.segment_type not in _VALID_SEGMENT_TYPES:
            raise ValueError(
                f"Invalid segment_type {self.segment_type!r}; "
                f"must be one of {sorted(_VALID_SEGMENT_TYPES)}"
            )


@dataclass(frozen=True, slots=True)
class Field:
    """A single extractable field derived from a JSON Schema node.

    Fields are created by ``flatten_schema`` and enriched by the SOTP
    (tau, var_tau), difficulty scorer (difficulty), and dependency extractor
    (dep_in, dep_out) before being consumed by Stage 2 capacity planning.

    Attributes:
        path: Dot-notation path, e.g. "address.city".
        type: JSON Schema type string — "string", "integer", "number",
            "boolean", "array", "object", "null", or "enum".
        constraints: JSON Schema constraint keywords present on this node
            (maxLength, minimum, enum, pattern, etc.).
        parent_path: Dot-notation path of the parent object. Empty string
            for top-level fields.
        schema_node: Original schema fragment for this field.
        tau: Expected output token count (computed by SOTP). 0.0 until set.
        var_tau: Variance of the token count estimate. 0.0 until set.
        difficulty: D(f) score in [0.0, 1.0]. 0.0 until set.
        dep_in: Paths that this field depends on (in-edges in the DAG).
        dep_out: Paths that depend on this field (out-edges in the DAG).
        required: True if this field is listed in the parent object's
            ``required`` array.

    Example:
        >>> f = Field(path="name", type="string", constraints={"maxLength": 100},
        ...           parent_path="", schema_node={"type": "string", "maxLength": 100})
        >>> f.path
        'name'
        >>> f.required
        False
    """

    path: str
    type: str
    constraints: dict[str, Any]
    parent_path: str
    schema_node: dict[str, Any]
    tau: float = 0.0
    var_tau: float = 0.0
    difficulty: float = 0.0
    dep_in: frozenset[str] = field(default_factory=frozenset)
    dep_out: frozenset[str] = field(default_factory=frozenset)
    required: bool = False

    def with_tau(self, *, tau: float, var_tau: float) -> Field:
        """Return a new Field with updated tau and var_tau values.

        Args:
            tau: Expected output token count (>= 1).
            var_tau: Variance of the token count estimate (>= 0).

        Returns:
            New Field instance with tau and var_tau set; all other fields
            unchanged.

        Example:
            >>> f = Field(path="x", type="integer", constraints={},
            ...           parent_path="", schema_node={})
            >>> f2 = f.with_tau(tau=2.0, var_tau=0.5)
            >>> f2.tau
            2.0
        """
        return Field(
            path=self.path,
            type=self.type,
            constraints=self.constraints,
            parent_path=self.parent_path,
            schema_node=self.schema_node,
            tau=tau,
            var_tau=var_tau,
            difficulty=self.difficulty,
            dep_in=self.dep_in,
            dep_out=self.dep_out,
            required=self.required,
        )

    def with_difficulty(self, difficulty: float) -> Field:
        """Return a new Field with updated difficulty score.

        Args:
            difficulty: D(f) score in [0.0, 1.0].

        Returns:
            New Field instance with difficulty set; all other fields unchanged.

        Example:
            >>> f = Field(path="x", type="boolean", constraints={},
            ...           parent_path="", schema_node={})
            >>> f2 = f.with_difficulty(0.025)
            >>> f2.difficulty
            0.025
        """
        return Field(
            path=self.path,
            type=self.type,
            constraints=self.constraints,
            parent_path=self.parent_path,
            schema_node=self.schema_node,
            tau=self.tau,
            var_tau=self.var_tau,
            difficulty=difficulty,
            dep_in=self.dep_in,
            dep_out=self.dep_out,
            required=self.required,
        )

    def with_deps(self, *, dep_in: frozenset[str], dep_out: frozenset[str]) -> Field:
        """Return a new Field with updated dependency sets.

        Args:
            dep_in: Paths that this field depends on.
            dep_out: Paths that depend on this field.

        Returns:
            New Field instance with dep_in and dep_out set; all other fields
            unchanged.

        Example:
            >>> f = Field(path="city", type="string", constraints={},
            ...           parent_path="address", schema_node={})
            >>> f2 = f.with_deps(dep_in=frozenset({"has_address"}), dep_out=frozenset())
            >>> "has_address" in f2.dep_in
            True
        """
        return Field(
            path=self.path,
            type=self.type,
            constraints=self.constraints,
            parent_path=self.parent_path,
            schema_node=self.schema_node,
            tau=self.tau,
            var_tau=self.var_tau,
            difficulty=self.difficulty,
            dep_in=dep_in,
            dep_out=dep_out,
            required=self.required,
        )


# ---------------------------------------------------------------------------
# Mutable aggregation objects
# ---------------------------------------------------------------------------


@dataclass
class FieldGroup:
    """A group of fields sharing a common parent path.

    FieldGroups are the unit of retrieval routing in Stage 2A-2B. The
    ``D_cost`` attribute is pre-computed token cost of matched segments,
    used for capacity planning in Stage 2C.

    Attributes:
        parent_path: Common dot-notation path prefix for all fields in the group.
        fields: Ordered list of Field objects in this group.
        matched_segments: Document segments matched to this group in Stage 2.5.
        segment_scores: BMX relevance scores parallel to matched_segments.
        D_cost: Estimated token cost of the matched_segments concatenated.
        field_best_segment: ``field_path -> segment_id`` of the matched segment
            that best supports each field, used by Stage 3 field-level coverage.
            Empty on the small-doc fast path.

    Example:
        >>> g = FieldGroup(parent_path="address")
        >>> g.parent_path
        'address'
    """

    parent_path: str
    fields: list[Field] = field(default_factory=list)
    matched_segments: list[Segment] = field(default_factory=list)
    segment_scores: list[float] = field(default_factory=list)
    D_cost: int = 0
    field_best_segment: dict[str, int] = field(default_factory=dict)


@dataclass
class CapacityLeaf:
    """A single LLM call unit after Stage 2C capacity planning.

    Each CapacityLeaf will be sent as exactly one structured-extraction
    call to the LLM. The ``document_excerpt`` is the retrieval result
    for this leaf's fields, trimmed to fit within the context window.

    Attributes:
        fields: All Field objects to extract in this call.
        groups: The FieldGroups contributing fields to this leaf.
        document_excerpt: Concatenated, trimmed document text for context.
        overhead: Token overhead from system prompt, schema, and JSON scaffolding.
        safe_output: Available output tokens after overhead is subtracted.
        leaf_id: Stable unique ID used for result assembly.

    Example:
        >>> leaf = CapacityLeaf(leaf_id=1)
        >>> leaf.leaf_id
        1
        >>> leaf.document_excerpt
        ''
    """

    fields: list[Field] = field(default_factory=list)
    groups: list[FieldGroup] = field(default_factory=list)
    document_excerpt: str = ""
    overhead: int = 0
    safe_output: int = 0
    leaf_id: int = 0
