from __future__ import annotations

from typing import Any

__all__ = ["extract_dependencies"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_dependencies(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Extract field dependency graph from a JSON Schema.

    Parses four JSON Schema 2020-12 dependency keywords:
    1. dependentRequired: {"fieldA": ["fieldB"]} → A requires B present
    2. dependentSchemas: {"fieldA": {...}} → A's value activates a sub-schema
    3. if/then/else: conditional schemas create soft dependencies
    4. allOf cross-refs: allOf members that reference other field paths

    Args:
        schema: A complete JSON Schema dict.

    Returns:
        A DAG as dict[field_path, set[field_path_it_depends_on]].
        Each value is the set of paths that the key field depends on.
        Empty dict if no dependencies found.

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "has_address": {"type": "boolean"},
        ...         "city": {"type": "string"},
        ...     },
        ...     "dependentRequired": {"city": ["has_address"]},
        ... }
        >>> deps = extract_dependencies(schema)
        >>> deps["city"] == {"has_address"}
        True
    """
    if not isinstance(schema, dict):
        return {}

    deps: dict[str, set[str]] = {}

    for source_deps in (
        _extract_dependent_required(schema),
        _extract_dependent_schemas(schema),
        _extract_if_then_else(schema),
        _extract_allof_deps(schema),
    ):
        for field_path, field_deps in source_deps.items():
            deps.setdefault(field_path, set()).update(field_deps)

    return deps


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _extract_dependent_required(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Extract deps from dependentRequired keyword.

    Args:
        schema: A JSON Schema dict.

    Returns:
        Dict mapping field_path to set of paths it depends on.
    """
    deps: dict[str, set[str]] = {}
    dependent_required = schema.get("dependentRequired", {})
    if not isinstance(dependent_required, dict):
        return deps
    for field_name, required_fields in dependent_required.items():
        if isinstance(required_fields, list):
            deps.setdefault(str(field_name), set()).update(
                str(r) for r in required_fields
            )
    return deps


def _extract_dependent_schemas(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Extract deps from dependentSchemas keyword (field activates sub-schema).

    Args:
        schema: A JSON Schema dict.

    Returns:
        Dict mapping field_path to set of paths it depends on.
    """
    deps: dict[str, set[str]] = {}
    dependent_schemas = schema.get("dependentSchemas", {})
    if not isinstance(dependent_schemas, dict):
        return deps
    for field_name, sub in dependent_schemas.items():
        # The sub-schema is activated when field_name is present.
        # Fields required in the sub-schema depend on field_name.
        if not isinstance(sub, dict):
            continue
        for required_field in sub.get("required", []):
            deps.setdefault(str(required_field), set()).add(str(field_name))
    return deps


def _extract_if_then_else(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Extract deps from if/then/else conditional schemas.

    Args:
        schema: A JSON Schema dict.

    Returns:
        Dict mapping field_path to set of paths it depends on.
    """
    deps: dict[str, set[str]] = {}
    if_schema = schema.get("if")
    if not isinstance(if_schema, dict):
        return deps

    # Fields required in "then" or "else" softly depend on fields in "if"
    if_fields = set(if_schema.get("properties", {}).keys())
    if_required = set(if_schema.get("required", []))
    condition_fields = if_fields | if_required

    for branch_key in ("then", "else"):
        branch = schema.get(branch_key)
        if isinstance(branch, dict):
            for req_field in branch.get("required", []):
                deps.setdefault(str(req_field), set()).update(condition_fields)

    return deps


def _extract_allof_deps(schema: dict[str, Any]) -> dict[str, set[str]]:
    """Extract cross-references from allOf sub-schemas.

    Args:
        schema: A JSON Schema dict.

    Returns:
        Dict mapping field_path to set of paths it depends on.
    """
    deps: dict[str, set[str]] = {}
    allof = schema.get("allOf", [])
    if not isinstance(allof, list):
        return deps
    for sub in allof:
        if not isinstance(sub, dict):
            continue
        sub_deps = extract_dependencies(sub)
        for field_path, field_deps in sub_deps.items():
            deps.setdefault(field_path, set()).update(field_deps)
    return deps
