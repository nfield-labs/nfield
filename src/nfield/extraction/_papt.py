"""PAPT (Prompt-Adaptive Template Selection) for extraction prompts.

PAPT selects the verbosity tier for extraction prompts based on the available
token budget. Three tiers trade off schema detail against token cost:

* ``CONCISE``  - field names only; used when the budget is very tight.
* ``STANDARD`` - field names + one-line descriptions; the normal tier.
* ``VERBOSE``  - full schema descriptions + constraint examples; used when
  the budget allows extra context for difficult fields.

Five cluster-type variants further tune the prompt style:

* ``SIMPLE``    - all-boolean / all-integer fields (short values).
* ``STANDARD``  - mixed-type fields (default).
* ``COMPLEX``   - deeply nested objects or high-difficulty fields.
* ``LIST``      - groups dominated by array fields.
* ``REFERENCE`` - fields referencing other schema objects (``$ref`` patterns).

Post-MVP: TEP (Think-Extract-Parse) two-phase variants for fields where
``D(f) >= 0.5`` are stubbed here but not yet implemented.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nfield.schema._types import Field

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
        CONCISE: Field names only - minimum token cost.
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
        >>> from nfield.schema._types import Field
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
        >>> from nfield.schema._types import Field
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
    """Produce a complete, human-readable description line for a single field.

    Renders EVERYTHING the schema says about the field so the model has the full
    contract - name, type, description, title, every constraint, the element
    schema for arrays, and any examples. None of this is dropped to save tokens:
    a field's own meaning and the shape of a valid value are exactly what raises
    extraction accuracy (schema-in-prompt grounding; constraints + examples cut
    type/format errors). The *template_type* tier governs only the surrounding
    prompt scaffolding, never a field's own spec.

    Format (clauses appear only when the schema provides them)::

        field.path (type): description [title] - constraints | items: <elem> | e.g. <examples>

    Args:
        field: The field to describe.
        template_type: Retained for prompt-scaffolding selection; does not strip
            any part of a field's spec.

    Returns:
        A single-line string fully describing the field.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("age", "integer", {"minimum": 0}, "", {"description": "Patient age"})
        >>> describe_field(f, TemplateType.CONCISE)
        'age (integer): Patient age - >= 0'
        >>> g = Field("tags", "array", {}, "", {"items": {"type": "string"}, "examples": [["a", "b"]]})
        >>> describe_field(g, TemplateType.STANDARD)
        'tags (array) | items: string | e.g. ["a", "b"]'
    """
    parts = [f"{field.path} ({field.type})"]

    description = _extract_description(field.schema_node)
    title = field.schema_node.get("title", "")
    if (
        isinstance(title, str)
        and title.strip()
        and title.strip().lower() not in description.lower()
    ):
        # A title adds signal only when it is not already echoed by the description.
        description = f"{description} [{title.strip()}]" if description else title.strip()
    if description:
        parts.append(f": {description}")

    constraint_text = _format_constraints(field.constraints)
    if constraint_text:
        parts.append(f" - {constraint_text}")

    item_text = _format_array_items(field)
    if item_text:
        parts.append(f" | items: {item_text}")

    example_text = _format_examples(field.schema_node)
    if example_text:
        parts.append(f" | e.g. {example_text}")

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
    # Track which keys we render with bespoke phrasing; everything else falls
    # through the generic catch-all so NO constraint is ever silently dropped.
    handled: set[str] = set()

    if "const" in constraints:
        parts.append(f"must equal {constraints['const']}")
        handled.add("const")

    if "enum" in constraints:
        options = constraints["enum"]
        if len(options) <= 5:
            parts.append(f"one of: {', '.join(str(o) for o in options)}")
        else:
            parts.append(f"{len(options)} allowed values")
        handled.add("enum")

    if any(
        k in constraints for k in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum")
    ):
        lo = constraints.get("minimum")
        hi = constraints.get("maximum")
        xlo = constraints.get("exclusiveMinimum")
        xhi = constraints.get("exclusiveMaximum")
        if lo is not None and hi is not None:
            parts.append(f"range [{lo}, {hi}]")
        elif lo is not None:
            parts.append(f">= {lo}")
        elif hi is not None:
            parts.append(f"<= {hi}")
        if xlo is not None:
            parts.append(f"> {xlo}")
        if xhi is not None:
            parts.append(f"< {xhi}")
        handled.update({"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"})

    if "multipleOf" in constraints:
        parts.append(f"multiple of {constraints['multipleOf']}")
        handled.add("multipleOf")

    if "minLength" in constraints or "maxLength" in constraints:
        lo = constraints.get("minLength")
        hi = constraints.get("maxLength")
        if lo is not None and hi is not None:
            parts.append(f"length [{lo}, {hi}]")
        elif hi is not None:
            parts.append(f"max {hi} chars")
        elif lo is not None:
            parts.append(f"min {lo} chars")
        handled.update({"minLength", "maxLength"})

    if "pattern" in constraints:
        parts.append(f"pattern: {constraints['pattern']}")
        handled.add("pattern")

    if "format" in constraints:
        parts.append(f"format: {constraints['format']}")
        handled.add("format")

    # Catch-all: any constraint keyword we don't phrase specially is still sent
    # verbatim (e.g. minItems, maxItems, uniqueItems) so the model sees the full
    # contract - a field's properties are never dropped.
    parts.extend(f"{key}: {constraints[key]}" for key in sorted(constraints) if key not in handled)

    return "; ".join(parts)


# Schema keywords describing an array's element, surfaced so the model knows the
# shape of each item (type + its own constraints + meaning), not just "array".
_ITEM_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "enum",
        "const",
        "multipleOf",
    }
)
# Cap on examples shown per field - enough to anchor the format without bloating
# the prompt (a couple of concrete shapes is what helps; more is noise).
_MAX_EXAMPLES_SHOWN: int = 3


def _format_array_items(field: Field) -> str:
    """Describe an array field's element schema (type, constraints, meaning).

    Args:
        field: The field to inspect; only ``array`` fields produce output.

    Returns:
        A compact element description (e.g. ``"string, format: date (ISO day)"``),
        or empty string when the field is not an array or has no item schema.
    """
    if field.type != "array":
        return ""
    items = field.schema_node.get("items")
    if not isinstance(items, dict):
        return ""
    elem_type = items.get("type", "string")
    elem_type = elem_type[0] if isinstance(elem_type, list) and elem_type else elem_type
    text = str(elem_type)
    elem_constraints = {k: v for k, v in items.items() if k in _ITEM_CONSTRAINT_KEYS}
    elem_constraint_text = _format_constraints(elem_constraints)
    if elem_constraint_text:
        text += f", {elem_constraint_text}"
    elem_desc = _extract_description(items)
    if elem_desc:
        text += f" ({elem_desc})"
    return text


def _format_examples(schema_node: dict[str, Any]) -> str:
    """Render up to a few schema examples for a field, if any are provided.

    Supports both JSON Schema ``examples`` (a list) and the OpenAPI-style
    singular ``example``. Examples anchor the expected value shape and are one
    of the cheapest, most effective accuracy levers for structured extraction.

    Args:
        schema_node: Raw JSON Schema fragment for the field.

    Returns:
        Comma-separated example values (JSON-rendered), or empty string.
    """
    raw = schema_node.get("examples")
    if raw is None and "example" in schema_node:
        raw = [schema_node["example"]]
    if not isinstance(raw, list) or not raw:
        return ""
    return ", ".join(_render_example(ex) for ex in raw[:_MAX_EXAMPLES_SHOWN])


def _render_example(example: object) -> str:
    """Render one schema example as JSON, falling back to ``str`` if not JSON-able."""
    try:
        return json.dumps(example, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(example)
