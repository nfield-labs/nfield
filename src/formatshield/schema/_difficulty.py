from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Field

__all__ = ["compute_difficulty"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# D_type values from DeepJSONEval accuracy inversion (architecture engine §1.3)
_D_TYPE: dict[str, float] = {
    "boolean": 0.05,
    "null": 0.05,
    "enum": 0.10,
    "integer": 0.15,
    "number": 0.20,
    "string": 0.40,  # constrained string (has maxLength/pattern/format)
    "array": 0.60,
    "object": 0.80,  # nested_object — should be rare after flattening
}
_D_TYPE_STRING_UNCONSTRAINED: float = 0.70

_D_WEIGHT_TYPE: float = 0.5
_D_WEIGHT_CONSTRAINT: float = 0.3
_D_WEIGHT_DEP: float = 0.2

_MAX_DEP_DEGREE: int = 10  # normalize dep degree against this ceiling

# Constraint difficulty weights
_CONSTRAINT_WEIGHTS: dict[str, float] = {
    "pattern": 0.5,
    "format": 0.3,
    "enum": 0.1,
    "minimum": 0.15,
    "maximum": 0.15,
    "exclusiveMinimum": 0.15,
    "exclusiveMaximum": 0.15,
    "minLength": 0.1,
    "maxLength": 0.1,
    "minItems": 0.1,
    "maxItems": 0.1,
    "uniqueItems": 0.2,
    "multipleOf": 0.2,
}
_UNKNOWN_CONSTRAINT_WEIGHT: float = 0.05


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_difficulty(
    field: Field,
    dep_dag: dict[str, set[str]],
    *,
    reverse_dep_dag: dict[str, set[str]] | None = None,
) -> float:
    """Compute extraction difficulty D(f) for a field.

    D(f) = 0.5 * D_type + 0.3 * D_constraint + 0.2 * D_dep

    Higher values indicate fields that need more document context,
    more specific prompting, and are more likely to fail on first attempt.

    Args:
        field: The Field to score.
        dep_dag: Dependency graph mapping field_path -> set of paths
            it depends on. Used to compute D_dep from in/out-degree.
        reverse_dep_dag: Optional pre-computed reverse dependency map
            (field_path -> set of fields that depend on it). If not provided,
            computed from dep_dag. Pass this when scoring multiple fields
            to avoid O(N²) recomputation.

    Returns:
        Difficulty score in [0.0, 1.0]. Higher = harder.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field(path="active", type="boolean", constraints={},
        ...           parent_path="", schema_node={})
        >>> d = compute_difficulty(f, dep_dag={})
        >>> 0.0 <= d <= 1.0
        True
        >>> d == 0.05 * 0.5  # D_type only, no constraints, no deps
        True
    """
    if reverse_dep_dag is None:
        reverse_dep_dag = _compute_reverse_dep_dag(dep_dag)

    d_type = _compute_d_type(field)
    d_constraint = _compute_d_constraint(field)
    d_dep = _compute_d_dep(field, dep_dag, reverse_dep_dag)

    score = _D_WEIGHT_TYPE * d_type + _D_WEIGHT_CONSTRAINT * d_constraint + _D_WEIGHT_DEP * d_dep
    # Clamp to [0.0, 1.0] for safety
    return max(0.0, min(score, 1.0))


# ---------------------------------------------------------------------------
# Component functions
# ---------------------------------------------------------------------------


def _compute_d_type(field: Field) -> float:
    """Compute type difficulty component for a field.

    Args:
        field: The Field to score.

    Returns:
        D_type in [0.0, 1.0].
    """
    if field.type == "string":
        constraints = field.constraints
        is_constrained = any(k in constraints for k in ("maxLength", "pattern", "format", "enum"))
        return _D_TYPE["string"] if is_constrained else _D_TYPE_STRING_UNCONSTRAINED
    return _D_TYPE.get(field.type, _D_TYPE_STRING_UNCONSTRAINED)


def _compute_d_constraint(field: Field) -> float:
    """Compute constraint difficulty component for a field.

    Scales from 0.0 (no constraints) to 1.0 (many complex constraints).

    Args:
        field: The Field to score.

    Returns:
        D_constraint in [0.0, 1.0].
    """
    if not field.constraints:
        return 0.0

    total = sum(_CONSTRAINT_WEIGHTS.get(k, _UNKNOWN_CONSTRAINT_WEIGHT) for k in field.constraints)
    return min(total, 1.0)


def _compute_reverse_dep_dag(dep_dag: dict[str, set[str]]) -> dict[str, set[str]]:
    """Compute reverse dependency index: field_path -> set of fields that depend on it.

    This is a pre-computation helper to avoid O(N) per-field scans in _compute_d_dep.

    Args:
        dep_dag: Forward dependency graph mapping field_path -> set of paths it depends on.

    Returns:
        Reverse dependency map mapping field_path -> set of paths that depend on it.

    Example:
        >>> dep_dag = {"b": {"a"}, "c": {"a", "b"}}
        >>> reverse = _compute_reverse_dep_dag(dep_dag)
        >>> reverse["a"]
        {'b', 'c'}
        >>> reverse["b"]
        {'c'}
    """
    reverse: dict[str, set[str]] = {}
    for field_path, deps in dep_dag.items():
        for dep in deps:
            if dep not in reverse:
                reverse[dep] = set()
            reverse[dep].add(field_path)
    return reverse


def _compute_d_dep(
    field: Field,
    dep_dag: dict[str, set[str]],
    reverse_dep_dag: dict[str, set[str]],
) -> float:
    """Compute dependency difficulty component for a field.

    Normalizes in-degree + out-degree against _MAX_DEP_DEGREE ceiling.
    Uses pre-computed reverse_dep_dag for O(1) out-degree lookup.

    Args:
        field: The Field to score.
        dep_dag: Dependency graph mapping field_path -> set of dependency paths.
        reverse_dep_dag: Reverse index (field_path -> set of dependents).

    Returns:
        D_dep in [0.0, 1.0].
    """
    in_degree = len(dep_dag.get(field.path, set()))
    out_degree = len(reverse_dep_dag.get(field.path, set()))
    total_degree = in_degree + out_degree
    return min(total_degree / _MAX_DEP_DEGREE, 1.0)
