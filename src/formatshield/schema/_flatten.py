from __future__ import annotations

from typing import Any

from formatshield.exceptions import SchemaError

from ._types import Field

__all__ = ["flatten_schema"]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_LEAF_TYPES: frozenset[str] = frozenset({"string", "integer", "number", "boolean", "null"})
_ARRAY_SUFFIX: str = "[]"
_WILDCARD_SUFFIX: str = ".*"
MAX_SCHEMA_DEPTH: int = 32
# Ceiling on total node expansions T during the iterative DFS flatten.
#
# A "node expansion" = one stack pop = one schema fragment processed (the root,
# every nested object/array, every $ref/anyOf/oneOf/allOf re-push, AND every leaf).
# It is NOT the emitted-field count: a field requires a pop but most pops emit no
# field, so |fields| ≤ T. We bound T (not |fields|) because a $ref fan-out inflates
# the non-leaf pops exponentially even when zero leaves are emitted.
#
# Let D = MAX_SCHEMA_DEPTH and b = max $ref reuse per node (a $def referenced b
# times per level). The per-branch cycle guard lets a node re-expand on each
# distinct branch, so
#     T = Σ_{i=0..D} b^i = (b^(D+1) - 1)/(b - 1) = Θ(b^D)      (b ≥ 2),
# i.e. T is EXPONENTIAL in depth. MAX_SCHEMA_DEPTH bounds D, not T (e.g. b=2,
# D=32 ⇒ T ≈ 2^32 ≈ 4.3e9 expansions ⇒ OOM/hang).
#
# This constant bounds T directly: the loop raises once pops > C, so work is O(C)
# for any (b, D); and since |fields| ≤ pops, memory is O(T) ≤ O(C) too — one bound
# covers both. Choose C by the constraint
#     N_legit ≤ C ≪ b^D,
# where N_legit = node count of the largest schema to admit (a flat schema of F
# fields has N ≈ F, so C must exceed the max supported field count). Tunable.
MAX_TOTAL_NODES: int = 5_000_000

_CONSTRAINT_KEYS: frozenset[str] = frozenset(
    {
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "minLength",
        "maxLength",
        "pattern",
        "format",
        "minItems",
        "maxItems",
        "uniqueItems",
        "enum",
        "const",
        "multipleOf",
        "minProperties",
        "maxProperties",
    }
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def flatten_schema(schema: dict[str, Any]) -> list[Field]:
    """Flatten a nested JSON Schema to a list of dot-notation Field objects.

    Uses an iterative stack-based DFS to avoid Python recursion limits on
    deeply nested schemas.

    Handles:
    - $ref with cycle detection (tracks visited ref strings)
    - anyOf/oneOf — takes first non-null option
    - prefixItems — creates indexed paths path[0], path[1], ...
    - allOf — merges all sub-schemas' properties
    - patternProperties — creates wildcard path with .* suffix
    - items (homogeneous arrays) — creates path[] path
    - additionalProperties (dict schema) — creates wildcard path with .* suffix

    Args:
        schema: A valid JSON Schema dict. Top-level must be a dict.

    Returns:
        List of Field objects with unique dot-notation paths, ordered by
        DFS discovery order.

    Raises:
        SchemaError: If schema is not a dict.
        SchemaError: If schema depth exceeds MAX_SCHEMA_DEPTH.
        SchemaError: If total node expansions exceed MAX_TOTAL_NODES (guards
            against $ref fan-out / pathological exponential blow-up).

    Example:
        >>> schema = {
        ...     "type": "object",
        ...     "properties": {
        ...         "name": {"type": "string"},
        ...         "age": {"type": "integer"},
        ...     }
        ... }
        >>> fields = flatten_schema(schema)
        >>> [f.path for f in fields]
        ['name', 'age']
    """
    if not isinstance(schema, dict):
        raise SchemaError(
            "Schema must be a dict",
            hint="Pass a parsed JSON Schema dict, not a string or list.",
        )

    fields: list[Field] = []
    seen_paths: set[str] = set()

    # Stack items: (node, path, parent_path, depth, required_set, ref_chain).
    # ref_chain holds the $refs already expanded on the CURRENT branch — this
    # detects cycles (a ref pointing back into its own ancestry) without
    # suppressing diamonds (the same $def reused by two sibling fields, e.g. a
    # Pydantic model with two Address fields). A global "seen refs" set would
    # wrongly drop the second use; a per-branch set keeps both.
    # path="" means top-level; we push object children with their names.
    initial_required: frozenset[str] = frozenset(schema.get("required", []))
    stack: list[tuple[dict[str, Any], str, str, int, frozenset[str], frozenset[str]]] = [
        (schema, "", "", 0, initial_required, frozenset())
    ]

    nodes_processed = 0

    while stack:
        nodes_processed += 1
        if nodes_processed > MAX_TOTAL_NODES:
            raise SchemaError(
                f"Schema expansion exceeded MAX_TOTAL_NODES={MAX_TOTAL_NODES}",
                hint=(
                    "A $ref fan-out or pathological schema is expanding far beyond "
                    "any real schema's size. Check for a $def referenced repeatedly."
                ),
            )

        node, path, parent_path, depth, required_set, ref_chain = stack.pop()

        if depth > MAX_SCHEMA_DEPTH:
            raise SchemaError(
                f"Schema depth exceeds MAX_SCHEMA_DEPTH={MAX_SCHEMA_DEPTH}",
                field=path,
                hint="Reduce nesting or increase MAX_SCHEMA_DEPTH.",
            )

        # ── $ref resolution ──────────────────────────────────────────────
        if "$ref" in node:
            ref_str = node["$ref"]
            # Cycle guard: only skip if this ref is already on the current
            # branch. A ref reused elsewhere in the tree is expanded again.
            if ref_str in ref_chain:
                continue
            try:
                resolved = _resolve_ref(schema, ref_str)
            except SchemaError:
                continue
            # Merge sibling keys (e.g. description, required) with resolved node
            merged = {**resolved, **{k: v for k, v in node.items() if k != "$ref"}}
            stack.append((merged, path, parent_path, depth, required_set, ref_chain | {ref_str}))
            continue

        # ── anyOf / oneOf — pick first non-null option ───────────────────
        for combo_key in ("anyOf", "oneOf"):
            if combo_key in node:
                options: list[dict[str, Any]] = node[combo_key]
                chosen = _first_non_null_option(options)
                if chosen is not None:
                    # Merge sibling keys into chosen option
                    merged_chosen = {
                        **chosen,
                        **{k: v for k, v in node.items() if k not in (combo_key, "$ref")},
                    }
                    stack.append(
                        (merged_chosen, path, parent_path, depth, required_set, ref_chain)
                    )
                break
        else:
            # Only process this node normally if no anyOf/oneOf handled it
            _process_node(
                node=node,
                path=path,
                parent_path=parent_path,
                depth=depth,
                required_set=required_set,
                ref_chain=ref_chain,
                stack=stack,
                fields=fields,
                seen_paths=seen_paths,
            )

    return fields


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _process_node(
    *,
    node: dict[str, Any],
    path: str,
    parent_path: str,
    depth: int,
    required_set: frozenset[str],
    ref_chain: frozenset[str],
    stack: list[tuple[dict[str, Any], str, str, int, frozenset[str], frozenset[str]]],
    fields: list[Field],
    seen_paths: set[str],
) -> None:
    """Process one schema node and push children onto the stack or emit a Field."""
    # ── allOf — merge all sub-schema properties ───────────────────────────
    if "allOf" in node:
        merged_props: dict[str, Any] = {}
        merged_req: list[str] = list(node.get("required", []))
        for sub in node["allOf"]:
            if not isinstance(sub, dict):
                continue
            merged_props.update(sub.get("properties", {}))
            merged_req.extend(sub.get("required", []))
        # Also include any properties directly on the node
        merged_props.update(node.get("properties", {}))
        if merged_props:
            synthetic = {**node, "properties": merged_props, "required": merged_req}
            del synthetic["allOf"]
            stack.append((synthetic, path, parent_path, depth, frozenset(merged_req), ref_chain))
            return

    node_type = _determine_type(node)

    # ── object — push children ────────────────────────────────────────────
    if node_type == "object" or "properties" in node:
        child_required: frozenset[str] = frozenset(node.get("required", []))
        properties: dict[str, Any] = node.get("properties", {})
        # Push in reverse so DFS pops in definition order
        for prop_name, prop_node in reversed(list(properties.items())):
            child_path = f"{path}.{prop_name}" if path else prop_name
            stack.append((prop_node, child_path, path, depth + 1, child_required, ref_chain))

        # patternProperties → wildcard field
        for pat_node in node.get("patternProperties", {}).values():
            wildcard_path = f"{path}{_WILDCARD_SUFFIX}" if path else _WILDCARD_SUFFIX
            if wildcard_path not in seen_paths:
                seen_paths.add(wildcard_path)
                fields.append(
                    _make_field(
                        node=pat_node,
                        path=wildcard_path,
                        parent_path=path,
                        required_set=required_set,
                    )
                )

        # additionalProperties (dict schema, not bool)
        addl = node.get("additionalProperties")
        if isinstance(addl, dict):
            wildcard_path = f"{path}{_WILDCARD_SUFFIX}" if path else _WILDCARD_SUFFIX
            if wildcard_path not in seen_paths:
                seen_paths.add(wildcard_path)
                fields.append(
                    _make_field(
                        node=addl,
                        path=wildcard_path,
                        parent_path=path,
                        required_set=required_set,
                    )
                )

        # If no properties at all and no children pushed, emit object leaf
        if (
            not properties
            and "patternProperties" not in node
            and not isinstance(addl, dict)
            and path
            and path not in seen_paths
        ):
            seen_paths.add(path)
            field_name = path.rsplit(".", 1)[-1]
            is_required = field_name in required_set
            constraints = _extract_constraints(node)
            fields.append(
                Field(
                    path=path,
                    type="object",
                    constraints=constraints,
                    parent_path=parent_path,
                    schema_node=node,
                    required=is_required,
                )
            )
        return

    # ── array ─────────────────────────────────────────────────────────────
    if node_type == "array":
        prefix_items = node.get("prefixItems")
        if isinstance(prefix_items, list):
            for idx, item_node in enumerate(prefix_items):
                indexed_path = f"{path}[{idx}]"
                stack.append((item_node, indexed_path, path, depth + 1, frozenset(), ref_chain))
            return

        items_node = node.get("items")
        if isinstance(items_node, dict):
            array_path = f"{path}{_ARRAY_SUFFIX}"
            stack.append((items_node, array_path, path, depth + 1, frozenset(), ref_chain))
            return

        # Array with no items/prefixItems — emit array leaf
        if path and path not in seen_paths:
            seen_paths.add(path)
            field_name = path.rsplit(".", 1)[-1].rstrip("[]")
            is_required = field_name in required_set
            constraints = _extract_constraints(node)
            fields.append(
                Field(
                    path=path,
                    type="array",
                    constraints=constraints,
                    parent_path=parent_path,
                    schema_node=node,
                    required=is_required,
                )
            )
        return

    # ── leaf (string, integer, number, boolean, null, enum) ───────────────
    if path and path not in seen_paths:
        seen_paths.add(path)
        fields.append(
            _make_field(
                node=node,
                path=path,
                parent_path=parent_path,
                required_set=required_set,
            )
        )


def _make_field(
    *,
    node: dict[str, Any],
    path: str,
    parent_path: str,
    required_set: frozenset[str],
) -> Field:
    """Construct a Field from a leaf schema node."""
    node_type = _determine_type(node)
    constraints = _extract_constraints(node)
    field_name = path.rsplit(".", 1)[-1].rstrip("[]").rstrip(".*")
    is_required = field_name in required_set
    return Field(
        path=path,
        type=node_type,
        constraints=constraints,
        parent_path=parent_path,
        schema_node=node,
        required=is_required,
    )


def _resolve_ref(root_schema: dict[str, Any], ref: str) -> dict[str, Any]:
    """Resolve a $ref string to a schema node.

    Handles: #/$defs/Name, #/definitions/Name, #/components/schemas/Name

    Args:
        root_schema: The top-level schema dict.
        ref: The $ref string to resolve.

    Returns:
        The resolved schema node dict.

    Raises:
        SchemaError: If ref cannot be resolved.
    """
    if not ref.startswith("#/"):
        raise SchemaError(
            f"Cannot resolve non-local $ref: {ref!r}",
            hint="Only local refs (#/...) are supported.",
        )
    parts = ref.lstrip("#/").split("/")
    node: Any = root_schema
    for part in parts:
        # JSON Pointer unescaping
        part = part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, dict) or part not in node:
            raise SchemaError(
                f"Cannot resolve $ref {ref!r}: key {part!r} not found",
                hint="Check that $defs or definitions contains this name.",
            )
        node = node[part]
    if not isinstance(node, dict):
        raise SchemaError(
            f"$ref {ref!r} resolved to a non-dict value",
            hint="$ref targets must be schema objects.",
        )
    return node


def _extract_constraints(node: dict[str, Any]) -> dict[str, Any]:
    """Extract JSON Schema constraint keywords from a schema node.

    Args:
        node: A JSON Schema node dict.

    Returns:
        Dict containing only constraint keywords present in node.

    Example:
        >>> _extract_constraints({"type": "string", "maxLength": 100, "format": "email"})
        {'maxLength': 100, 'format': 'email'}
    """
    return {k: v for k, v in node.items() if k in _CONSTRAINT_KEYS}


def _infer_type(node: dict[str, Any]) -> str:
    """Infer the JSON Schema type from a schema node's structure.

    Args:
        node: A schema node dict without an explicit "type" key.

    Returns:
        The inferred type string.

    Example:
        >>> _infer_type({"enum": ["a", "b"]})
        'enum'
    """
    if "enum" in node:
        return "enum"
    if "const" in node:
        const_val = node["const"]
        if isinstance(const_val, bool):
            return "boolean"
        if isinstance(const_val, int):
            return "integer"
        if isinstance(const_val, float):
            return "number"
        return "string"
    if "properties" in node or node.get("type") == "object":
        return "object"
    if "items" in node or "prefixItems" in node:
        return "array"
    return "string"  # fallback


def _determine_type(node: dict[str, Any]) -> str:
    """Determine the effective type of a schema node.

    Handles list types by picking the first non-null type.

    Args:
        node: A schema node dict.

    Returns:
        The effective type string.
    """
    raw_type = node.get("type")
    if isinstance(raw_type, list):
        # e.g. ["string", "null"] → "string"
        non_null = [t for t in raw_type if t != "null"]
        if non_null:
            return str(non_null[0])
        return "null"
    if isinstance(raw_type, str):
        return raw_type
    return _infer_type(node)


def _first_non_null_option(
    options: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Return the first non-null option from anyOf/oneOf list.

    Args:
        options: List of schema option dicts.

    Returns:
        First option that is not purely {"type": "null"}, or None if empty.
    """
    for opt in options:
        if not isinstance(opt, dict):
            continue
        if opt.get("type") == "null" and len(opt) == 1:
            continue
        return opt
    return None
