"""Stage 1: Schema Analysis.

Zero API calls. Flattens the JSON Schema, computes per-field token costs
(tau/var_tau), extracts the dependency DAG, and scores each field's
extraction difficulty D(f). Initialises the Blackboard.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from nfield.assembly._blackboard import Blackboard
from nfield.exceptions import SchemaError
from nfield.schema._deps import extract_dependencies
from nfield.schema._difficulty import compute_difficulty
from nfield.schema._flatten import flatten_schema
from nfield.schema._tau import compute_tau

if TYPE_CHECKING:
    from nfield.pipeline._state import PipelineState

__all__ = ["run_stage_1"]


def run_stage_1(state: PipelineState, schema: dict[str, Any]) -> PipelineState:
    """Analyse schema: flatten, score tokens, extract deps, score difficulty.

    Populates:
    - ``state.fields`` - ordered list of ``Field`` objects with dot-notation paths
    - ``state.field_by_path`` - O(1) lookup by path
    - ``state.dep_dag`` - dependency adjacency dict
    - ``state.blackboard`` - Blackboard initialised with all field paths

    Args:
        state: Pipeline state from Stage 0 (must have ``chars_per_token`` set).
        schema: Raw JSON Schema dict.

    Returns:
        Updated ``PipelineState``.

    Raises:
        SchemaError: If the schema produces zero extractable fields.

    """
    # --- 1. Flatten schema to dot-notation fields ---
    fields = flatten_schema(schema)
    if not fields:
        raise SchemaError(
            "Schema produced zero extractable fields after flattening.",
            hint="Check that the schema has at least one leaf property.",
        )

    # --- 2. Compute tau (token cost) per field ---
    enriched_with_tau = []
    for f in fields:
        tau, var_tau = compute_tau(f, state.chars_per_token)
        enriched_with_tau.append(f.with_tau(tau=tau, var_tau=var_tau))
    enriched = enriched_with_tau

    # --- 3. Extract dependency DAG ---
    dep_dag = extract_dependencies(schema)

    # --- 4. Score difficulty per field ---
    enriched = [f.with_difficulty(compute_difficulty(f, dep_dag)) for f in enriched]

    # --- 5. Populate dep_in / dep_out on each field ---
    # dep_dag maps path -> set[paths it depends on] (in-edges)
    # dep_out: which fields depend on this one (out-edges)
    dep_out_map: dict[str, set[str]] = {}
    for path, deps in dep_dag.items():
        for dep_path in deps:
            dep_out_map.setdefault(dep_path, set()).add(path)

    enriched = [
        f.with_deps(
            dep_in=frozenset(dep_dag.get(f.path, set())),
            dep_out=frozenset(dep_out_map.get(f.path, set())),
        )
        for f in enriched
    ]

    # --- 6. Build path index ---
    field_by_path = {f.path: f for f in enriched}
    if len(field_by_path) != len(enriched):
        raise SchemaError(
            "Schema produced duplicate field paths after flattening.",
            hint="Check for ambiguous $ref or allOf merges that produce path collisions.",
        )

    # --- 7. Initialise Blackboard ---
    blackboard = Blackboard([f.path for f in enriched])

    state.fields = enriched
    state.field_by_path = field_by_path
    state.dep_dag = dep_dag
    state.blackboard = blackboard
    return state
