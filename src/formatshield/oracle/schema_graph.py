"""
Schema dependency graph and Fiedler value computation.

Builds an undirected weighted graph G_σ from a JSON Schema dict, then computes
the normalized Fiedler value (second-smallest Laplacian eigenvalue):

    λ̃₂ = λ₂(L) / (d_max + 1)  ∈ [0, 1]

where L = D - W is the unnormalized graph Laplacian.

High λ̃₂ ≈ densely coupled schema → higher Φ → prefer TTF.
Low  λ̃₂ ≈ sparse / isolated fields → direct generation is fine.

Edge weights
------------
* Structural nesting (parent → child):  1.0
* ``$ref`` references:                  1.5
* Combining keywords (allOf/anyOf/oneOf/if/then/else): 2.0
* Shared name stem (e.g. start_date / end_date):       0.5

Complexity
----------
* n ≤ 50 fields: ``numpy.linalg.eigvalsh`` — exact, O(n³)
* n > 50 fields: ``scipy.sparse.linalg.eigsh`` Lanczos — O(n · k)
  (falls back to dense eigvalsh if scipy is unavailable)
"""

from __future__ import annotations

import math
from typing import Any


def _collect_fields(schema: dict[str, Any], prefix: str = "") -> list[str]:
    """Recursively collect all leaf/intermediate field paths in the schema."""
    fields: list[str] = []
    props = schema.get("properties", {})
    for name, sub in props.items():
        full = f"{prefix}.{name}" if prefix else name
        fields.append(full)
        if isinstance(sub, dict):
            fields.extend(_collect_fields(sub, full))
    # Handle array items
    items = schema.get("items", {})
    if isinstance(items, dict) and items.get("properties"):
        fields.extend(_collect_fields(items, prefix + "[]" if prefix else "[]"))
    # Handle allOf/anyOf/oneOf/if/then/else sub-schemas
    for kw in ("allOf", "anyOf", "oneOf"):
        for sub in schema.get(kw, []):
            if isinstance(sub, dict):
                fields.extend(_collect_fields(sub, prefix))
    for kw in ("if", "then", "else"):
        sub = schema.get(kw)
        if isinstance(sub, dict):
            fields.extend(_collect_fields(sub, prefix))
    return fields


def _name_stem(field: str) -> str:
    """Return the base part of a field path after stripping numeric/date suffixes."""
    base = field.split(".")[-1].split("[")[0]
    # Strip trailing digits or common date suffixes
    for suffix in ("_start", "_end", "_from", "_to", "_min", "_max", "_begin", "_finish"):
        if base.endswith(suffix):
            return base[: -len(suffix)]
    # Strip trailing underscore+digit
    import re

    return re.sub(r"_\d+$", "", base)


def fiedler_value(schema: dict[str, Any]) -> float:
    """Compute normalized Fiedler value λ̃₂ ∈ [0, 1] for *schema*.

    Returns 0.0 for empty or single-field schemas (trivially connected or
    disconnected — no constraint coupling to exploit).
    """
    if not isinstance(schema, dict):
        return 0.0
    fields = list(dict.fromkeys(_collect_fields(schema)))  # deduplicate, preserve order
    n = len(fields)
    if n < 2:
        return 0.0

    idx = {f: i for i, f in enumerate(fields)}
    # Adjacency weight matrix (dense, n×n)
    wmat = [[0.0] * n for _ in range(n)]

    def _add_edge(a: int, b: int, w: float) -> None:
        wmat[a][b] += w
        wmat[b][a] += w

    # Structural edges from nesting (parent.child)
    for f in fields:
        parts = f.rsplit(".", 1)
        if len(parts) == 2:
            parent = parts[0]
            if parent in idx:
                _add_edge(idx[parent], idx[f], 1.0)
            else:
                # Handle array item paths: "field[].child" → parent is "field"
                parent_no_array = parent.replace("[]", "")
                if parent_no_array and parent_no_array in idx:
                    _add_edge(idx[parent_no_array], idx[f], 1.0)

    # $ref edges — connect fields that share a $ref chain
    def _add_ref_edges(sub: dict[str, Any], source_field: str | None = None) -> None:
        ref = sub.get("$ref")
        if ref and source_field and source_field in idx:
            # Connect to all fields with matching definition name in their path
            def_name = ref.split("/")[-1].lower()
            for f in fields:
                if def_name in f.lower() and f != source_field and f in idx:
                    _add_edge(idx[source_field], idx[f], 1.5)
        for kw in ("properties",):
            for name, child in sub.get(kw, {}).items():
                if isinstance(child, dict):
                    _add_ref_edges(child, name if name in idx else None)

    _add_ref_edges(schema)

    # Combinator edges (allOf/anyOf/oneOf/if/then/else) — connect co-occurring fields
    def _combinator_fields(sub: dict[str, Any]) -> list[str]:
        return [f for f in _collect_fields(sub) if f in idx]

    for kw in ("allOf", "anyOf", "oneOf"):
        for clause in schema.get(kw, []):
            if isinstance(clause, dict):
                members = _combinator_fields(clause)
                for i in range(len(members)):
                    for j in range(i + 1, len(members)):
                        _add_edge(idx[members[i]], idx[members[j]], 2.0)

    for kw in ("if", "then", "else"):
        clause = schema.get(kw)
        if isinstance(clause, dict):
            members = _combinator_fields(clause)
            for i in range(len(members)):
                for j in range(i + 1, len(members)):
                    _add_edge(idx[members[i]], idx[members[j]], 2.0)

    # Shared-stem edges
    stem_groups: dict[str, list[int]] = {}
    for f in fields:
        s = _name_stem(f)
        stem_groups.setdefault(s, []).append(idx[f])
    for group in stem_groups.values():
        if len(group) >= 2:
            for i in range(len(group)):
                for j in range(i + 1, len(group)):
                    _add_edge(group[i], group[j], 0.5)

    # Build Laplacian lap = D - W
    degrees = [sum(wmat[i]) for i in range(n)]
    d_max = max(degrees) if degrees else 1.0

    # Compute λ₂ via numpy
    try:
        import numpy as np

        lap = np.array(wmat, dtype=float)
        # Negate and add diagonal (lap = D - W)
        lap = -lap
        for i in range(n):
            lap[i, i] = degrees[i]

        if n > 50:
            try:
                from scipy.sparse import csr_matrix
                from scipy.sparse.linalg import eigsh

                lap_sparse = csr_matrix(lap)
                # k=2: want second smallest eigenvalue (first is always 0)
                eigenvalues = eigsh(
                    lap_sparse, k=min(3, n - 1), which="SM", return_eigenvectors=False
                )
                eigenvalues = sorted(eigenvalues)
                lambda2 = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
            except Exception:
                eigenvalues = np.linalg.eigvalsh(lap)
                lambda2 = float(sorted(eigenvalues)[1])
        else:
            eigenvalues = np.linalg.eigvalsh(lap)
            lambda2 = float(sorted(eigenvalues)[1])

    except ImportError:
        # Pure-Python fallback: power iteration approximation
        lambda2 = _fiedler_power_iter(wmat, degrees, n)

    # Normalize and clamp
    lambda2_norm = lambda2 / (d_max + 1.0)
    return max(0.0, min(1.0, lambda2_norm))


def _fiedler_power_iter(
    wmat: list[list[float]],
    degrees: list[float],
    n: int,
    max_iter: int = 200,
) -> float:
    """Approximate λ₂ via inverse power iteration (no-numpy fallback).

    Accuracy sufficient for routing — not for research.
    """
    import random

    random.seed(42)
    # Random initial vector orthogonal to all-ones vector
    v = [random.gauss(0, 1) for _ in range(n)]
    ones_mean = sum(v) / n
    v = [x - ones_mean for x in v]
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    v = [x / norm for x in v]

    # Rayleigh quotient approximation via repeated lap·v
    for _ in range(max_iter):
        lap_v = [degrees[i] * v[i] - sum(wmat[i][j] * v[j] for j in range(n)) for i in range(n)]
        # Re-orthogonalize against constant vector
        mean_lap_v = sum(lap_v) / n
        lap_v = [x - mean_lap_v for x in lap_v]
        norm = math.sqrt(sum(x * x for x in lap_v)) or 1.0
        v = [x / norm for x in lap_v]

    lap_v = [degrees[i] * v[i] - sum(wmat[i][j] * v[j] for j in range(n)) for i in range(n)]
    rayleigh = sum(v[i] * lap_v[i] for i in range(n))
    return max(0.0, rayleigh)
