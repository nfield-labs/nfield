"""AC-2001 arc consistency propagation (Research status).

This module implements AC-2001 arc consistency propagation with O(ed²) complexity,
where e is the number of constraint edges and d is the domain size.

AC-2001 (Bessiere & Régin, 2001) improves on AC-3 by maintaining a list of
valid supports for each (variable, value) pair, reducing redundant checks.

Status: Research — will be benchmarked against simpler constraint propagation
before implementing. Only needed if schemas with dense constraint dependencies
show accuracy degradation without it.

Reference: arXiv:2405.15434 (constraint-aware LLM extraction)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ._types import Field

__all__ = ["propagate_arc_consistency"]


def propagate_arc_consistency(
    fields: list[Field],
    dep_dag: dict[str, set[str]],
    *,
    max_iterations: int = 100,
) -> dict[str, set[str]]:
    """Propagate AC-2001 arc consistency through the field dependency graph.

    Reduces the effective domain of constrained fields by propagating
    arc consistency constraints from the dependency DAG. Fields that are
    arc-inconsistent are flagged for additional prompting context.

    Args:
        fields: List of Field objects to analyze.
        dep_dag: Dependency DAG from extract_dependencies().
        max_iterations: Safety limit on propagation rounds (default: 100).

    Returns:
        Updated dep_dag with arc-inconsistent edges removed.

    Raises:
        NotImplementedError: Always — this algorithm is Research status.
            Benchmark constrained extraction accuracy first.

    Note:
        Complexity: O(ed²) where e = edges in dep_dag, d = max domain size.
        Only beneficial when dep_dag has > 20 constraint edges. For typical
        JSON Schemas (5-15 deps), simple topological ordering is sufficient.
    """
    raise NotImplementedError(
        "AC-2001 arc consistency is Research status. "
        "Benchmark constrained extraction first to determine if needed. "
        "See: formatshield/schema/_ac2001.py for implementation notes."
    )
