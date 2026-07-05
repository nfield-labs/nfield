from __future__ import annotations

from typing import Any

from nfield.exceptions import SchemaError

from ._types import Field

__all__ = ["OPEN_MAP_MARKER", "flatten_schema"]

# Marks an open-map list-leaf; assembly folds its [{key, value}] list back to a dict.
OPEN_MAP_MARKER: str = "x-open-map"
# Marks an open map that shares its object with fixed properties (additionalProperties
# beside named keys); assembly folds it and merges its keys into the parent object.
OPEN_MAP_MERGE_MARKER: str = "x-open-map-merge"
# An array|object anyOf emits both branches, the array under this shadow suffix. These
# tag a branch with its base path and kind so assembly keeps the one the document filled.
UNION_BASE_MARKER: str = "x-union-base"
UNION_KIND_MARKER: str = "x-union-kind"
UNION_ARRAY_SUFFIX: str = "__uarr"

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
# for any (b, D); and since |fields| ≤ pops, memory is O(T) ≤ O(C) too - one bound
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
        UNION_BASE_MARKER,
        UNION_KIND_MARKER,
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
    - anyOf/oneOf - takes first non-null option
    - prefixItems - creates indexed paths path[0], path[1], ...
    - allOf - merges all sub-schemas' properties
    - patternProperties - creates wildcard path with .* suffix
    - array of scalars - creates path[] leaf
    - array of objects - creates one array "list-leaf" whose ``constraints["items"]``
      holds the item schema, so a variable-length object array is extracted as a
      whole list instead of collapsing to a single templated element
    - additionalProperties (dict schema) - creates wildcard path with .* suffix

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
    # ref_chain holds the $refs already expanded on the CURRENT branch - this
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

        # ── anyOf / oneOf - pick first non-null option ───────────────────
        for combo_key in ("anyOf", "oneOf"):
            if combo_key in node:
                options: list[dict[str, Any]] = node[combo_key]
                siblings = {k: v for k, v in node.items() if k not in (combo_key, "$ref")}
                plan = _union_plan(options)
                if plan is not None:
                    array_branch, object_branch = plan
                    if array_branch is not None:
                        # Array-vs-object union: emit both, the array under a shadow path;
                        # assembly keeps whichever branch the document populated.
                        stack.append(
                            (
                                _stamp_union({**object_branch, **siblings}, path, "object"),
                                path,
                                parent_path,
                                depth,
                                required_set,
                                ref_chain,
                            )
                        )
                        stack.append(
                            (
                                _stamp_union({**array_branch, **siblings}, path, "array"),
                                f"{path}{UNION_ARRAY_SUFFIX}",
                                parent_path,
                                depth,
                                frozenset(),
                                ref_chain,
                            )
                        )
                    else:
                        # Several object branches: extract the union of their fields so
                        # whichever shape the document uses is covered.
                        stack.append(
                            (
                                {**object_branch, **siblings},
                                path,
                                parent_path,
                                depth,
                                required_set,
                                ref_chain,
                            )
                        )
                    break
                chosen = _first_non_null_option(options)
                if chosen is not None:
                    merged_chosen = {**chosen, **siblings}
                    stack.append(
                        (merged_chosen, path, parent_path, depth, required_set, ref_chain)
                    )
                elif path:
                    # Every branch is null: emit a null leaf so the field still appears
                    # in the output rather than vanishing.
                    stack.append(
                        (
                            {**siblings, "type": "null"},
                            path,
                            parent_path,
                            depth,
                            required_set,
                            ref_chain,
                        )
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
                root_schema=schema,
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
    root_schema: dict[str, Any],
) -> None:
    """Process one schema node and push children onto the stack or emit a Field."""
    # ── allOf - merge all sub-schema properties ───────────────────────────
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

    # ── object - push children ────────────────────────────────────────────
    if node_type == "object" or "properties" in node:
        child_required: frozenset[str] = frozenset(node.get("required", []))
        properties: dict[str, Any] = node.get("properties", {})
        # Push in reverse so DFS pops in definition order
        for prop_name, prop_node in reversed(list(properties.items())):
            child_path = f"{path}.{prop_name}" if path else prop_name
            stack.append((prop_node, child_path, path, depth + 1, child_required, ref_chain))

        # A pure dynamic-key object extracts as one {key, value} list (open map).
        value_schema = _open_map_value_schema(node)
        if value_schema is not None and not properties and path and path not in seen_paths:
            seen_paths.add(path)
            fields.append(
                _open_map_leaf(
                    path, parent_path, value_schema, required_set, root_schema, source=node
                )
            )
            return

        # patternProperties → wildcard field (mixed object: keep the wildcard)
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

        # additionalProperties (dict, not bool): an open map sharing the object; its
        # dynamic keys extract as {key, value} rows and merge into the parent at assembly.
        addl = node.get("additionalProperties")
        if isinstance(addl, dict):
            wildcard_path = f"{path}{_WILDCARD_SUFFIX}" if path else _WILDCARD_SUFFIX
            if wildcard_path not in seen_paths:
                seen_paths.add(wildcard_path)
                leaf = _open_map_leaf(wildcard_path, path, addl, required_set, root_schema)
                leaf.constraints[OPEN_MAP_MERGE_MARKER] = True
                fields.append(leaf)

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
            if "anyOf" in items_node or "oneOf" in items_node:
                # Per-element union: resolve to the richest element shape so object
                # elements are extracted as objects, not flattened to scalars.
                combo = items_node.get("anyOf") or items_node.get("oneOf")
                resolved = _first_non_null_option(combo) if isinstance(combo, list) else None
                if resolved is not None:
                    items_node = {
                        **resolved,
                        **{k: v for k, v in items_node.items() if k not in ("anyOf", "oneOf")},
                    }
            # Object, scalar, and nested-array items become one list-leaf holding the item
            # schema; recursing into path[] would add an extra array level at assembly.
            if (
                _items_is_object(items_node, root_schema)
                or _items_is_scalar(items_node, root_schema)
                or _items_is_array(items_node, root_schema)
            ):
                if path and path not in seen_paths:
                    seen_paths.add(path)
                    field_name = path.rsplit(".", 1)[-1].rstrip("[]")
                    resolved_items = _resolve_items_node(items_node, root_schema)
                    constraints = {**_extract_constraints(node), "items": resolved_items}
                    fields.append(
                        Field(
                            path=path,
                            type="array",
                            constraints=constraints,
                            parent_path=parent_path,
                            schema_node=node,
                            required=field_name in required_set,
                        )
                    )
                return
            array_path = f"{path}{_ARRAY_SUFFIX}"
            stack.append((items_node, array_path, path, depth + 1, frozenset(), ref_chain))
            return

        # Array with no items/prefixItems - emit array leaf
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


def _open_map_value_schema(node: dict[str, Any]) -> dict[str, Any] | None:
    """Return the value schema of an open map (patternProperties/additionalProperties)."""
    patterns = node.get("patternProperties")
    if isinstance(patterns, dict) and patterns:
        first = next(iter(patterns.values()))
        return first if isinstance(first, dict) else {}
    addl = node.get("additionalProperties")
    if isinstance(addl, dict):
        return addl
    return None


def _open_map_leaf(
    path: str,
    parent_path: str,
    value_schema: dict[str, Any],
    required_set: frozenset[str],
    root_schema: dict[str, Any],
    source: dict[str, Any] | None = None,
) -> Field:
    """Build one array list-leaf of {key, value} for an open map."""
    item = {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "The map key."},
            "value": _resolve_items_node(value_schema, root_schema),
        },
    }
    constraints: dict[str, Any] = {"items": item, OPEN_MAP_MARKER: True}
    for marker in (UNION_BASE_MARKER, UNION_KIND_MARKER):
        if source is not None and marker in source:
            constraints[marker] = source[marker]
    field_name = path.rsplit(".", 1)[-1]
    return Field(
        path=path,
        type="array",
        constraints=constraints,
        parent_path=parent_path,
        schema_node={"type": "array", "items": item},
        required=field_name in required_set,
    )


def _items_is_object(items_node: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Return ``True`` if an array's ``items`` schema is (or $refs) an object.

    Resolves a single ``$ref`` so that ``items: {"$ref": "#/$defs/metric_entry"}``
    is recognised as an object array. A resolution failure is treated as non-object.
    """
    node = items_node
    if "$ref" in node:
        try:
            node = _resolve_ref(root_schema, node["$ref"])
        except SchemaError:
            return False
    return node.get("type") == "object" or "properties" in node


def _items_is_scalar(items_node: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Return ``True`` if an array's ``items`` schema is a scalar (not object or array).

    A scalar item is a string/integer/number/boolean/enum leaf - anything that is not
    an object (handled as an object list-leaf) and not itself an array (handled as a
    nested-array list-leaf). Resolves a single ``$ref``.
    """
    node = items_node
    if "$ref" in node:
        try:
            node = _resolve_ref(root_schema, node["$ref"])
        except SchemaError:
            return False
    if node.get("type") == "object" or "properties" in node:
        return False
    return not (node.get("type") == "array" or "items" in node or "prefixItems" in node)


def _items_is_array(items_node: dict[str, Any], root_schema: dict[str, Any]) -> bool:
    """Return ``True`` if an array's ``items`` schema is itself an array or tuple.

    An array-of-array becomes one list-leaf whose item schema is the inner array, so
    the whole nested list is extracted at this path. Resolves a single ``$ref``.
    """
    node = items_node
    if "$ref" in node:
        try:
            node = _resolve_ref(root_schema, node["$ref"])
        except SchemaError:
            return False
    return node.get("type") == "array" or "items" in node or "prefixItems" in node


def _resolve_items_node(items_node: dict[str, Any], root_schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve a single ``$ref`` in an array's items so the item shape is concrete.

    Returns the resolved node merged with any sibling keys (e.g. ``description``).
    A non-ref or unresolvable node is returned unchanged, so the caller always has
    a usable item schema for the prompt and the JSON-array cast.
    """
    node = items_node
    if "$ref" in node:
        try:
            resolved = _resolve_ref(root_schema, node["$ref"])
        except SchemaError:
            return node
        node = {**resolved, **{k: v for k, v in node.items() if k != "$ref"}}
    # Resolve one level of property $refs so each item field carries its real type.
    props = node.get("properties")
    if isinstance(props, dict):
        node = {
            **node,
            "properties": {n: _resolve_prop(sub, root_schema) for n, sub in props.items()},
        }
    return node


def _resolve_prop(sub: Any, root_schema: dict[str, Any]) -> Any:
    """Resolve a single-level ``$ref`` in an item property, merging sibling keys."""
    if not (isinstance(sub, dict) and "$ref" in sub):
        return sub
    try:
        resolved = _resolve_ref(root_schema, sub["$ref"])
    except SchemaError:
        return sub
    return {**resolved, **{k: v for k, v in sub.items() if k != "$ref"}}


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
    """Choose the anyOf/oneOf branch to flatten, preferring the richest shape.

    Null-only options are skipped. Among the rest the richest branch wins - object
    over array over scalar - because an object (or open map) can represent both the
    grouped and the flat form of a polymorphic field, while a scalar or array branch
    cannot represent the grouped form. Equal ranks keep declaration order.

    Args:
        options: List of schema option dicts.

    Returns:
        The chosen option, or ``None`` if every option is null-only or non-dict.
    """
    candidates = [
        opt
        for opt in options
        if isinstance(opt, dict) and not (opt.get("type") == "null" and len(opt) == 1)
    ]
    if not candidates:
        return None
    # max() is stable, so the first branch of the top rank wins (declaration order).
    return max(candidates, key=_option_rank)


def _union_plan(
    options: list[dict[str, Any]],
) -> tuple[dict[str, Any] | None, dict[str, Any]] | None:
    """Plan how to flatten a document-dependent anyOf, or ``None`` for the default.

    Returns ``(array_branch, object_branch)`` when the branch shape depends on the
    document and a single richest-branch choice would drop fields:

    * array branch present -> structural union; the caller emits both, the array under
      a shadow path, and assembly keeps the one the document filled.
    * no array but several object branches -> ``array_branch`` is ``None`` and
      ``object_branch`` is their merged superset, so every possible field is extracted.

    A scalar union (string vs int) or a single object branch is not document-dependent
    and returns ``None`` so the caller keeps the existing single-branch choice.
    """
    candidates = [
        opt
        for opt in options
        if isinstance(opt, dict) and not (opt.get("type") == "null" and len(opt) == 1)
    ]
    array_branch = next((opt for opt in candidates if _option_rank(opt) == 1), None)
    object_branches = [opt for opt in candidates if _option_rank(opt) == 2]
    if not object_branches:
        return None
    merged_object = (
        _merge_object_branches(object_branches) if len(object_branches) > 1 else object_branches[0]
    )
    if array_branch is not None:
        return array_branch, merged_object
    if len(object_branches) > 1:
        return None, merged_object
    return None


def _merge_object_branches(objects: list[dict[str, Any]]) -> dict[str, Any]:
    """Union the properties of several object branches into one object schema.

    The first branch wins a property-name collision (declaration order). An open map
    (``additionalProperties``) on any branch is carried through. Nothing is marked
    required - a union guarantees no single field.
    """
    properties: dict[str, Any] = {}
    additional: dict[str, Any] | None = None
    for obj in objects:
        for name, sub in (obj.get("properties") or {}).items():
            properties.setdefault(name, sub)
        if additional is None and isinstance(obj.get("additionalProperties"), dict):
            additional = obj["additionalProperties"]
    merged: dict[str, Any] = {"type": "object", "properties": properties}
    if additional is not None:
        merged["additionalProperties"] = additional
    return merged


def _stamp_union(node: dict[str, Any], base: str, kind: str) -> dict[str, Any]:
    """Tag a union branch node with its base path and kind for assembly-time resolution."""
    return {**node, UNION_BASE_MARKER: base, UNION_KIND_MARKER: kind}


def _option_rank(option: dict[str, Any]) -> int:
    """Structural richness of an anyOf branch: object (2) > array (1) > scalar (0)."""
    if (
        option.get("type") == "object"
        or "properties" in option
        or "additionalProperties" in option
    ):
        return 2
    if option.get("type") == "array" or "items" in option or "prefixItems" in option:
        return 1
    return 0
