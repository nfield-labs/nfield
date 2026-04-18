"""
Schema constraint tightness τ — information-theoretic entropy proxy.

Walks the JSON Schema type tree and computes per-leaf entropy h(v), then
returns the normalized constraint tightness:

    τ = 1 - mean(h(v)) / H₀   ∈ [0, 1]

where H₀ = log₂(128_000) ≈ 16.97 bits (typical LLM vocabulary size).

τ = 0 → fully unconstrained schema (all fields are free strings)
τ = 1 → maximally constrained (all fields are single-value enums / booleans)

Per-leaf entropy h(v)
---------------------
* ``boolean``                     → log₂(2) = 1.0 bit
* ``enum`` with k choices         → log₂(k) bits
* ``integer`` in [a, b]           → log₂(b − a + 1) bits
* ``string`` with ``format``      → 0.5 · H₀ bits  (format halves the space)
* ``null``                        → 0.0 bits
* everything else (unconstrained) → H₀ bits

Deps: stdlib only (``math``).
"""

from __future__ import annotations

import math
from typing import Any

#: Reference entropy for an unconstrained token (log₂ of typical LLM vocab)
H0: float = math.log2(128_000)


def _leaf_entropy(field_schema: dict[str, Any]) -> float:
    """Return per-leaf Shannon entropy h(v) in bits."""
    if not isinstance(field_schema, dict):
        return H0

    ftype = field_schema.get("type")
    enum = field_schema.get("enum")
    const = field_schema.get("const")

    # const → single value, zero entropy
    if const is not None:
        return 0.0

    # enum → log₂(|choices|)
    if enum is not None:
        k = max(len(enum), 1)
        return math.log2(k)

    if ftype == "boolean":
        return 1.0

    if ftype == "null":
        return 0.0

    if ftype == "integer" or ftype == "number":
        minimum = field_schema.get("minimum", field_schema.get("exclusiveMinimum"))
        maximum = field_schema.get("maximum", field_schema.get("exclusiveMaximum"))
        if minimum is not None and maximum is not None:
            try:
                span = float(maximum) - float(minimum) + 1
                if span > 1:
                    return math.log2(span)
                return 0.0
            except (TypeError, ValueError):
                pass
        return H0

    if ftype == "string":
        if field_schema.get("format") or field_schema.get("pattern"):
            return 0.5 * H0
        max_length = field_schema.get("maxLength")
        if max_length is not None:
            try:
                return math.log2(max(int(max_length), 1))
            except (TypeError, ValueError):
                pass
        return H0

    if ftype == "array":
        # Entropy of the array itself: consider item count constraints
        max_items = field_schema.get("maxItems")
        min_items = field_schema.get("minItems", 0)
        if max_items is not None:
            try:
                span = int(max_items) - int(min_items) + 1
                return math.log2(max(span, 1))
            except (TypeError, ValueError):
                pass
        return H0

    # object, any, or unknown → unconstrained
    return H0


def _walk_schema(schema: dict[str, Any]) -> list[float]:
    """Collect per-leaf entropy values from all leaf fields."""
    entropies: list[float] = []

    props = schema.get("properties", {})
    required = set(schema.get("required", []))

    for name, sub in props.items():
        if not isinstance(sub, dict):
            entropies.append(H0)
            continue

        sub_type = sub.get("type")

        if sub_type == "object" or sub.get("properties"):
            # Recurse into nested objects
            entropies.extend(_walk_schema(sub))
        elif sub_type == "array":
            items = sub.get("items", {})
            if isinstance(items, dict) and items.get("properties"):
                # Array of objects — recurse into item schema
                entropies.extend(_walk_schema(items))
            else:
                entropies.append(_leaf_entropy(sub))
        else:
            # Optional fields (not required) are slightly less constrained
            h = _leaf_entropy(sub)
            if name not in required:
                h = min(H0, h + 1.0)  # +1 bit for presence/absence
            entropies.append(h)

    # Handle allOf/anyOf/oneOf — use minimum entropy (tightest constraint)
    for kw in ("allOf", "anyOf", "oneOf"):
        clauses = schema.get(kw, [])
        if clauses:
            clause_entropies = [
                _walk_schema(c) for c in clauses if isinstance(c, dict)
            ]
            if clause_entropies:
                # allOf: all must hold → use min entropy (most constrained)
                # anyOf/oneOf: at least one holds → use max entropy (least constrained)
                if kw == "allOf":
                    for per_clause in clause_entropies:
                        entropies.extend(per_clause)
                else:
                    # Take the average over clauses for anyOf/oneOf
                    flattened = [e for clause in clause_entropies for e in clause]
                    entropies.extend(flattened)

    return entropies


def constraint_tightness(schema: dict[str, Any]) -> float:
    """Return schema constraint tightness τ ∈ [0, 1].

    τ = 0: all fields unconstrained (free text everywhere).
    τ = 1: all fields maximally constrained (booleans, single-value enums).

    Parameters
    ----------
    schema:
        JSON Schema dict (e.g. from ``pydantic_model.model_json_schema()``).
    """
    if not isinstance(schema, dict):
        return 0.0

    entropies = _walk_schema(schema)

    if not entropies:
        # No leaf fields found — treat as unconstrained
        return 0.0

    mean_h = sum(entropies) / len(entropies)
    tau = 1.0 - mean_h / H0
    return max(0.0, min(1.0, tau))
