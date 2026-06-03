"""PAPT (Prompt-Adaptive Template Selection) for extraction prompts.

PAPT selects the verbosity tier for extraction prompts based on the available
token budget. Three tiers trade off schema detail against token cost:

* ``CONCISE``  — field names only; used when the budget is very tight.
* ``STANDARD`` — field names + one-line descriptions; the normal tier.
* ``VERBOSE``  — full schema descriptions + constraint examples; used when
  the budget allows extra context for difficult fields.

Five cluster-type variants further tune the prompt style:

* ``SIMPLE``    — all-boolean / all-integer fields (short values).
* ``STANDARD``  — mixed-type fields (default).
* ``COMPLEX``   — deeply nested objects or high-difficulty fields.
* ``LIST``      — groups dominated by array fields.
* ``REFERENCE`` — fields referencing other schema objects (``$ref`` patterns).

Post-MVP: TEP (Think-Extract-Parse) two-phase variants for fields where
``D(f) >= 0.5`` are stubbed here but not yet implemented.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from formatshield.schema._types import Field

__all__ = [
    "ClusterType",
    "TemplateType",
    "classify_cluster",
    "describe_field",
    "select_template",
]

# ---------------------------------------------------------------------------
# Budget thresholds (tokens available for schema description block)
# ---------------------------------------------------------------------------

_BUDGET_CONCISE_MAX: int = 300  # Below this → CONCISE (field names only)
_BUDGET_VERBOSE_MIN: int = 800  # Above this → VERBOSE (full descriptions)

# Difficulty threshold for "complex" cluster classification
_COMPLEXITY_DIFFICULTY_THRESHOLD: float = 0.5
_COMPLEXITY_FIELD_FRACTION: float = 0.4  # 40% of fields must exceed threshold

# Fraction of array fields needed to classify a cluster as LIST
_LIST_FIELD_FRACTION: float = 0.5


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TemplateType(Enum):
    """Verbosity tier for extraction prompt generation.

    Attributes:
        CONCISE: Field names only — minimum token cost.
        STANDARD: Field names plus one-line schema descriptions.
        VERBOSE: Full schema descriptions including constraints and examples.

    Example:
        >>> TemplateType.STANDARD.value
        'standard'
    """

    CONCISE = "concise"
    STANDARD = "standard"
    VERBOSE = "verbose"


class ClusterType(Enum):
    """Structural classification of a field group for template selection.

    Attributes:
        SIMPLE: All-boolean or all-integer fields.
        STANDARD: Mixed-type fields (default).
        COMPLEX: High-difficulty fields or deeply nested objects.
        LIST: Groups dominated by array fields.
        REFERENCE: Fields with schema object references.

    Example:
        >>> ClusterType.LIST.value
        'list'
    """

    SIMPLE = "simple"
    STANDARD = "standard"
    COMPLEX = "complex"
    LIST = "list"
    REFERENCE = "reference"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def select_template(
    fields: list[Field],
    budget_tokens: int,
) -> TemplateType:
    """Select the prompt verbosity tier for a field group.

    Chooses based on the available *budget_tokens* for the schema description
    block. Tighter budgets produce shorter prompts; larger budgets allow richer
    context that improves accuracy on difficult fields.

    Args:
        fields: Fields to be extracted in this prompt call. Used for
            cluster classification (affects future TEP routing; no-op in MVP).
        budget_tokens: Available tokens for the schema description block.
            Should be ``C_usable - overhead - safe_output`` for the leaf.

    Returns:
        The appropriate :class:`TemplateType` for this budget.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("name", "string", {}, "", {})
        >>> select_template([f], budget_tokens=100)
        <TemplateType.CONCISE: 'concise'>
        >>> select_template([f], budget_tokens=500)
        <TemplateType.STANDARD: 'standard'>
        >>> select_template([f], budget_tokens=1000)
        <TemplateType.VERBOSE: 'verbose'>
    """
    if budget_tokens < _BUDGET_CONCISE_MAX:
        return TemplateType.CONCISE
    if budget_tokens >= _BUDGET_VERBOSE_MIN:
        return TemplateType.VERBOSE
    return TemplateType.STANDARD


def classify_cluster(fields: list[Field]) -> ClusterType:
    """Classify a field group into a structural cluster type.

    Used by the prompt builder to select cluster-specific phrasing.
    In MVP this affects only comment text; post-MVP it will select
    distinct TEP prompt variants.

    Args:
        fields: Fields in the group to classify.

    Returns:
        The :class:`ClusterType` best describing this group's structure.

    Example:
        >>> from formatshield.schema._types import Field
        >>> bools = [Field(f"f{i}", "boolean", {}, "", {}) for i in range(3)]
        >>> classify_cluster(bools)
        <ClusterType.SIMPLE: 'simple'>
    """
    if not fields:
        return ClusterType.STANDARD

    # Reference: any field with a schema_node containing "$ref"
    if any("$ref" in f.schema_node for f in fields):
        return ClusterType.REFERENCE

    # List: majority are array fields
    array_count = sum(1 for f in fields if f.type == "array")
    if array_count / len(fields) >= _LIST_FIELD_FRACTION:
        return ClusterType.LIST

    # Complex: many high-difficulty fields
    complex_count = sum(1 for f in fields if f.difficulty >= _COMPLEXITY_DIFFICULTY_THRESHOLD)
    if complex_count / len(fields) >= _COMPLEXITY_FIELD_FRACTION:
        return ClusterType.COMPLEX

    # Simple: all primitive / low-variance types
    simple_types = frozenset({"boolean", "integer", "null"})
    if all(f.type in simple_types for f in fields):
        return ClusterType.SIMPLE

    return ClusterType.STANDARD


def describe_field(
    field: Field,
    template_type: TemplateType,
) -> str:
    """Produce a human-readable description line for a single field.

    The detail level follows *template_type*:
    - ``CONCISE``:  ``"field.path (type)"``
    - ``STANDARD``: ``"field.path (type): description"``
    - ``VERBOSE``:  ``"field.path (type): description — constraints"``

    Args:
        field: The field to describe.
        template_type: Controls how much schema detail to include.

    Returns:
        A single-line string describing the field.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("age", "integer", {"minimum": 0}, "", {"description": "Patient age"})
        >>> describe_field(f, TemplateType.CONCISE)
        'age (integer)'
        >>> describe_field(f, TemplateType.STANDARD)
        'age (integer): Patient age'
    """
    path = field.path
    ftype = field.type

    if template_type == TemplateType.CONCISE:
        return f"{path} ({ftype})"

    description = _extract_description(field.schema_node)

    if template_type == TemplateType.STANDARD:
        if description:
            return f"{path} ({ftype}): {description}"
        return f"{path} ({ftype})"

    # VERBOSE: include constraints
    parts = [f"{path} ({ftype})"]
    if description:
        parts.append(f": {description}")
    constraint_text = _format_constraints(field.constraints)
    if constraint_text:
        parts.append(f" — {constraint_text}")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _extract_description(schema_node: dict[str, Any]) -> str:
    """Extract a human-readable description from a schema node.

    Args:
        schema_node: Raw JSON Schema fragment for the field.

    Returns:
        Description string, or empty string if not present.
    """
    desc = schema_node.get("description", "")
    if isinstance(desc, str):
        return desc.strip()
    return ""


def _format_constraints(constraints: dict[str, Any]) -> str:
    """Format schema constraints into a compact human-readable string.

    Args:
        constraints: Constraint dict from the field (minLength, enum, etc.).

    Returns:
        Formatted constraint summary, or empty string if no notable constraints.
    """
    parts: list[str] = []

    if "enum" in constraints:
        options = constraints["enum"]
        if len(options) <= 5:
            parts.append(f"one of: {', '.join(str(o) for o in options)}")
        else:
            parts.append(f"{len(options)} allowed values")

    if "minimum" in constraints or "maximum" in constraints:
        lo = constraints.get("minimum")
        hi = constraints.get("maximum")
        if lo is not None and hi is not None:
            parts.append(f"range [{lo}, {hi}]")
        elif lo is not None:
            parts.append(f">= {lo}")
        else:
            parts.append(f"<= {hi}")

    if "minLength" in constraints or "maxLength" in constraints:
        lo = constraints.get("minLength")
        hi = constraints.get("maxLength")
        if lo is not None and hi is not None:
            parts.append(f"length [{lo}, {hi}]")
        elif hi is not None:
            parts.append(f"max {hi} chars")
        elif lo is not None:
            parts.append(f"min {lo} chars")

    if "pattern" in constraints:
        parts.append(f"pattern: {constraints['pattern']}")

    if "format" in constraints:
        parts.append(f"format: {constraints['format']}")

    return "; ".join(parts)
