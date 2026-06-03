"""Stage 2C: Capacity Packing.

Decide how many LLM calls the extraction needs and which fields travel together
in each call. Pure computation, zero API calls.

Pipeline of this stage:

* ``compute_K_min`` — a lower bound on the number of calls, from total output
  volume and the count of fields too large to share a call.
* ``fits`` — a dual feasibility test (input context budget AND output budget)
  for a candidate set of fields.
* ``_greedy_ffd`` — First-Fit Decreasing bin packing (Johnson, "Near-optimal
  bin packing algorithms", 1973), O(N log N), with per-field splitting when a
  single group cannot fit one call.
* ``tarjan_scc`` + ``compute_execution_order`` — Tarjan's strongly-connected-
  components algorithm (Tarjan, "Depth-first search and linear graph
  algorithms", SIAM J. Comput. 1972) condenses dependency cycles, then Kahn's
  topological sort (Kahn, "Topological sorting of large networks", CACM 1962)
  groups leaves into parallel execution rounds. Both are O(V+E).
"""

from __future__ import annotations

import math
from collections import deque
from typing import TYPE_CHECKING

from formatshield.schema._types import CapacityLeaf

if TYPE_CHECKING:
    from collections.abc import Iterator

    from formatshield.config import ExtractionConfig
    from formatshield.pipeline._state import PipelineState
    from formatshield.schema._types import Field, FieldGroup

__all__ = [
    "compute_K_min",
    "compute_execution_order",
    "fits",
    "run_stage_2c",
    "tarjan_scc",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# LLM output lengths are heavy-tailed (log-t), not normal, so a Gaussian safety
# margin under-reserves. Inflating the z-score by 1.5x compensates for the fat
# tail (ProD, arXiv:2604.07931; TIE, arXiv:2604.00499).
_Z_HEAVY_TAIL_FACTOR: float = 1.5
_MIN_SAFE_OUTPUT_TOKENS: int = 50  # floor so every call can emit a few fields
# Format Token Overhead: tokens spent on the system prompt, format instructions,
# and per-field schema description before any value is produced. Fixed estimates
# used for planning; the assembled prompt is the source of truth at call time.
_SYSTEM_PROMPT_OVERHEAD_TOKENS: int = 100
_PER_FIELD_SCHEMA_TOKENS: int = 15
# SFEP emits one "path = value" line per field; tau(f) predicts only the value
# length, so reserve a fixed allowance for the path, " = " separator, and newline
# to avoid truncating the final lines of a large response.
_SFEP_LINE_OVERHEAD_TOKENS: int = 8
# English-average characters per token; used only as a fallback before the
# model-specific ratio measured at calibration is threaded into this estimate.
_FALLBACK_CHARS_PER_TOKEN: float = 4.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_K_min(fields: list[Field], safe_output: float) -> int:  # noqa: N802
    """Lower bound on the number of LLM calls any packing can achieve.

    The bound is the larger of two independent constraints:

    * **Volume bound** — total predicted output cannot exceed the per-call output
      budget, so at least ``ceil(Στ(f) / safe_output)`` calls are required.
    * **Large-field bound** — any field whose own output would consume more than
      half a call's budget effectively monopolises a call, so the count of such
      fields is itself a lower bound.

    Reported alongside the actual call count K as the optimality gap (K / K_min).

    Args:
        fields: All schema fields, each carrying its predicted output size tau.
        safe_output: Per-call output budget (the model output ceiling).

    Returns:
        Minimum number of calls, at least 1.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("x", "integer", {}, "", {}, tau=10.0)
        >>> compute_K_min([f], safe_output=100.0)
        1
    """
    if safe_output <= 0:
        return len(fields)
    sum_tau = sum(f.tau for f in fields)
    # Fields whose individual tau exceeds half the output budget need their own leaf
    large_field_count = sum(1 for f in fields if f.tau > 0.5 * safe_output)
    return max(1, math.ceil(sum_tau / safe_output), large_field_count)


def fits(
    leaf_fields: list[Field],
    D_cost: int,  # noqa: N803
    overhead: int,
    C_usable: float,  # noqa: N803
    output_ceiling: float,
) -> bool:
    """Check whether a set of fields fits in a single leaf under dual constraints.

    Capacity is computed from the model's real limits, not a fixed field count:

    * Context: ``overhead + D_cost + output_needed <= C_usable`` (C_usable is a
      fraction of the model's real context window).
    * Output:  ``output_needed <= output_ceiling`` (the model's real max-output
      minus the heavy-tail margin).

    Args:
        leaf_fields: Fields to pack into this leaf.
        D_cost: Token cost of document segments for these fields.
        overhead: Fixed token overhead (system prompt + schema description).
        C_usable: Usable slice of the context window. Held well below 100% of
            the window because extraction accuracy degrades as the context fills
            (BABILong arXiv:2406.10149; RULER arXiv:2404.06654; MECW
            arXiv:2509.21361; Sequential-NIAH arXiv:2504.04713).
        output_ceiling: Largest output a single call may safely emit, the model
            output limit minus the heavy-tail margin.

    Returns:
        ``True`` if both constraints are satisfied.

    Example:
        >>> from formatshield.schema._types import Field
        >>> f = Field("x", "integer", {}, "", {}, tau=5.0)
        >>> fits([f], D_cost=100, overhead=50, C_usable=500.0, output_ceiling=200.0)
        True
    """
    output_needed = math.ceil(sum(f.tau for f in leaf_fields))
    if output_needed > output_ceiling:
        return False
    return overhead + D_cost + output_needed <= C_usable


def tarjan_scc(graph: dict[str, set[str]]) -> list[list[str]]:
    """Find strongly connected components (Tarjan, 1972). O(V+E).

    A single depth-first pass assigns each node a discovery index and a low-link
    value; a component closes when a node's low-link equals its own index.
    Cyclically dependent nodes share a component, letting the caller schedule
    mutually dependent fields together instead of deadlocking a topological sort.

    Implemented with an explicit work stack rather than recursion, so it cannot
    exceed the interpreter recursion limit on a very long dependency chain
    (mirrors the iterative schema flattener for the same reason).

    Args:
        graph: Adjacency dict where ``graph[A] = {B}`` means A depends on B.

    Returns:
        List of SCCs. Each SCC is a list of node IDs; members appear in
        stack-pop order, which is not sorted. Singleton SCCs (no cycle)
        contain a single node.

    Example:
        >>> tarjan_scc({"a": {"b"}, "b": {"a"}, "c": set()})
        [['b', 'a'], ['c']]
    """
    counter = 0
    index: dict[str, int] = {}
    lowlink: dict[str, int] = {}
    on_stack: dict[str, bool] = {}
    scc_stack: list[str] = []
    sccs: list[list[str]] = []

    # All nodes, including those referenced only as dependency targets.
    all_nodes: set[str] = set(graph.keys())
    for deps in graph.values():
        all_nodes.update(deps)

    for root in sorted(all_nodes):  # sorted for deterministic component order
        if root in index:
            continue

        # Each work-stack entry is (node, iterator over its sorted neighbours).
        index[root] = lowlink[root] = counter
        counter += 1
        scc_stack.append(root)
        on_stack[root] = True
        work: list[tuple[str, Iterator[str]]] = [(root, iter(sorted(graph.get(root, set()))))]

        while work:
            node, neighbours = work[-1]
            descended = False
            for nbr in neighbours:
                if nbr not in index:
                    index[nbr] = lowlink[nbr] = counter
                    counter += 1
                    scc_stack.append(nbr)
                    on_stack[nbr] = True
                    work.append((nbr, iter(sorted(graph.get(nbr, set())))))
                    descended = True
                    break
                if on_stack.get(nbr, False):
                    lowlink[node] = min(lowlink[node], index[nbr])
            if descended:
                continue

            # All neighbours of `node` are explored; propagate low-link upward.
            work.pop()
            if work:
                parent = work[-1][0]
                lowlink[parent] = min(lowlink[parent], lowlink[node])

            if lowlink[node] == index[node]:
                component: list[str] = []
                while True:
                    w = scc_stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == node:
                        break
                sccs.append(component)

    return sccs


def compute_execution_order(
    leaves: list[CapacityLeaf],
    dep_dag: dict[str, set[str]],
) -> list[list[CapacityLeaf]]:
    """Group leaves into parallel execution rounds respecting dependencies.

    Lifts the field-level dependency graph to the leaf level, condenses any
    leaf cycle into one super-node with Tarjan's SCC algorithm (so mutually
    dependent fields run together rather than deadlocking the sort), then runs
    Kahn's topological sort (Kahn, 1962) over the condensed DAG. Each
    topological level becomes one round: leaves within a round are independent
    and may run concurrently, while rounds run in sequence.

    Args:
        leaves: All CapacityLeafs produced by Stage 2C packing.
        dep_dag: Field-level dependency graph from Stage 1.

    Returns:
        List of rounds; each round is a list of CapacityLeafs that can run
        in parallel. An empty list of leaves returns ``[[]]``.

    Example:
        >>> leaf = CapacityLeaf(leaf_id=0)
        >>> compute_execution_order([leaf], {})
        [[...]]
    """
    if not leaves:
        return [[]]

    n = len(leaves)
    path_to_leaf: dict[str, int] = {
        f.path: i for i, leaf in enumerate(leaves) for f in leaf.fields
    }

    # Leaf-level dependency graph (string keys for tarjan_scc): leaf i depends
    # on leaf j if any field in i depends on a field in j.
    leaf_graph: dict[str, set[str]] = {str(i): set() for i in range(n)}
    for i, leaf in enumerate(leaves):
        for f in leaf.fields:
            for dep_path in dep_dag.get(f.path, set()):
                j = path_to_leaf.get(dep_path)
                if j is not None and j != i:
                    leaf_graph[str(i)].add(str(j))

    # Condense cycles: each SCC becomes one super-node executed together.
    sccs = tarjan_scc(leaf_graph)
    scc_of: dict[str, int] = {node: sid for sid, comp in enumerate(sccs) for node in comp}
    num_scc = len(sccs)

    # Condensed DAG between SCCs (acyclic by construction).
    scc_deps: list[set[int]] = [set() for _ in range(num_scc)]
    for node, deps in leaf_graph.items():
        a = scc_of[node]
        for dep in deps:
            b = scc_of[dep]
            if a != b:
                scc_deps[a].add(b)

    # Kahn toposort over SCCs, grouping each topological level into one round.
    in_degree = [len(scc_deps[s]) for s in range(num_scc)]
    reverse: list[set[int]] = [set() for _ in range(num_scc)]
    for a in range(num_scc):
        for b in scc_deps[a]:
            reverse[b].add(a)

    queue: deque[int] = deque(s for s in range(num_scc) if in_degree[s] == 0)
    rounds: list[list[CapacityLeaf]] = []
    while queue:
        level_size = len(queue)
        current_round: list[CapacityLeaf] = []
        for _ in range(level_size):
            sid = queue.popleft()
            current_round.extend(leaves[int(node)] for node in sccs[sid])
            for dependent in reverse[sid]:
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)
        rounds.append(current_round)

    return rounds


def run_stage_2c(state: PipelineState, config: ExtractionConfig) -> PipelineState:
    """Pack FieldGroups into CapacityLeafs and compute execution order.

    Steps:
    1. Derive the output ceiling (model limit minus heavy-tail margin).
    2. Compute K_min, the lower bound on call count.
    3. Pack groups into leaves with First-Fit Decreasing (splitting per field
       when a group is too large for one call).
    4. Order the leaves into parallel rounds (Tarjan SCC + Kahn).

    Populates:
    - ``state.leaves`` — list of CapacityLeafs (one per API call)
    - ``state.execution_order`` — parallel rounds
    - ``state.K_min`` — theoretical minimum API calls

    Args:
        state: Pipeline state from Stage 2.5 (must have groups with D_cost).
        config: Extraction configuration (z_target, etc.).

    Returns:
        Updated ``PipelineState``.
    """
    z_eff = config.z_target * _Z_HEAVY_TAIL_FACTOR

    # The output ceiling is the most any single call may safely emit: the model
    # output limit minus the heavy-tail margin over all fields. fits() and K_min
    # check against this shared ceiling, while each leaf separately reserves only
    # what its own fields need (see _leaf_safe_output) so the document excerpt
    # keeps the bulk of the input budget.
    sum_var_all = sum(f.var_tau for f in state.fields)
    output_ceiling = max(
        float(_MIN_SAFE_OUTPUT_TOKENS),
        state.M_O - z_eff * math.sqrt(sum_var_all),
    )

    state.K_min = compute_K_min(state.fields, output_ceiling)

    leaves = _greedy_ffd(state.groups, state, z_eff=z_eff, output_ceiling=output_ceiling)
    state.leaves = leaves
    state.execution_order = compute_execution_order(leaves, state.dep_dag)
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_leaf_overhead(fields: list[Field]) -> int:
    """Estimate FTO (Format Token Overhead) for a leaf's fields.

    Args:
        fields: Fields in the leaf.

    Returns:
        Token overhead estimate.
    """
    return _SYSTEM_PROMPT_OVERHEAD_TOKENS + _PER_FIELD_SCHEMA_TOKENS * len(fields)


def _leaf_safe_output(fields: list[Field], z_eff: float) -> int:
    """Output tokens to reserve for one leaf: predicted output + safety margin.

    This is the per-leaf reservation, distinct from the model-wide output
    ceiling. It sets ``max_tokens`` for the call, so it is deliberately small —
    only what this leaf's fields need — leaving the remainder of the input
    budget for the document excerpt.

    Args:
        fields: Fields packed into the leaf.
        z_eff: Heavy-tail-inflated z-score (z_target * 1.5).

    Returns:
        Reserved output token count (>= _MIN_SAFE_OUTPUT_TOKENS).
    """
    output_needed = sum(f.tau for f in fields) + len(fields) * _SFEP_LINE_OVERHEAD_TOKENS
    margin = z_eff * math.sqrt(sum(f.var_tau for f in fields))
    return max(_MIN_SAFE_OUTPUT_TOKENS, math.ceil(output_needed + margin))


def _greedy_ffd(
    groups: list[FieldGroup],
    state: PipelineState,
    *,
    z_eff: float,
    output_ceiling: float,
) -> list[CapacityLeaf]:
    """First-Fit Decreasing bin packing of groups (with per-field split) into leaves.

    Sorts groups by total predicted output descending (heaviest first), the
    standard First-Fit Decreasing heuristic that keeps the packing within ~11/9
    of optimal (Johnson, 1973). Each whole group is placed in the first leaf it
    fits. A group too large for any single call is refined lazily — its fields
    are packed into fresh leaves one at a time — so a wide flat group (e.g. 200
    sibling fields) splits across several calls instead of overflowing one.

    Args:
        groups: FieldGroups from Stage 2A.
        state: PipelineState with C_usable and chars_per_token.
        z_eff: Heavy-tail-inflated z-score, for per-leaf output reservation.
        output_ceiling: Max tokens any single call may emit (M_O - margin).

    Returns:
        List of CapacityLeaf objects with fields, groups, overhead, and the
        per-leaf safe_output reservation set.
    """
    # Sort groups heaviest first (descending tau sum)
    sorted_groups = sorted(groups, key=lambda g: sum(f.tau for f in g.fields), reverse=True)

    leaves: list[CapacityLeaf] = []

    def _new_leaf(fields: list[Field], group: FieldGroup) -> None:
        leaves.append(
            CapacityLeaf(
                fields=list(fields),
                groups=[group],
                overhead=_compute_leaf_overhead(fields),
                safe_output=_leaf_safe_output(fields, z_eff),
                leaf_id=len(leaves),
            )
        )

    for group in sorted_groups:
        # 1. Try to place the whole group into an existing leaf.
        placed = False
        for leaf in leaves:
            candidate_fields = leaf.fields + group.fields
            candidate_dcost = _aggregate_dcost([*leaf.groups, group], state.chars_per_token)
            overhead = _compute_leaf_overhead(candidate_fields)
            if fits(candidate_fields, candidate_dcost, overhead, state.C_usable, output_ceiling):
                leaf.fields.extend(group.fields)
                leaf.groups.append(group)
                leaf.overhead = overhead
                leaf.safe_output = _leaf_safe_output(leaf.fields, z_eff)
                placed = True
                break
        if placed:
            continue

        # 2. Whole group did not fit any existing leaf — pack its fields into
        #    fresh leaves greedily. A group that fits a fresh leaf becomes one
        #    leaf; an oversized group splits across several. D_cost stays the
        #    whole group's cost (conservative: keeps fits() safe after split).
        current: list[Field] = []
        for f in group.fields:
            candidate = [*current, f]
            overhead = _compute_leaf_overhead(candidate)
            if current and not fits(
                candidate, group.D_cost, overhead, state.C_usable, output_ceiling
            ):
                _new_leaf(current, group)
                current = [f]
            else:
                current = candidate
        if current:
            _new_leaf(current, group)

    return (
        leaves
        if leaves
        else [
            CapacityLeaf(
                fields=list(state.fields),
                groups=list(state.groups),
                overhead=_compute_leaf_overhead(state.fields),
                safe_output=_leaf_safe_output(state.fields, z_eff),
                leaf_id=0,
            )
        ]
    )


def _aggregate_dcost(groups: list[FieldGroup], chars_per_token: float) -> int:
    """Sum D_cost across groups, deduplicating shared segments.

    Segments appearing in multiple groups are only counted once, so a chunk
    serving several groups is not double-charged against the leaf budget.

    Args:
        groups: FieldGroups whose D_cost to aggregate.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0), so the
            token estimate matches the one Stage 2.5 and Stage 3 use. Falls back
            to the English average if non-positive.

    Returns:
        Total unique segment token cost.
    """
    seen_ids: set[int] = set()
    total = 0
    for g in groups:
        for seg in g.matched_segments:
            if seg.segment_id not in seen_ids:
                seen_ids.add(seg.segment_id)
                total += len(seg.text)
    # If no matched_segments (small-doc fast path), use D_cost directly.
    if not seen_ids:
        return sum(g.D_cost for g in groups)
    ratio = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    return max(1, math.ceil(total / ratio))
