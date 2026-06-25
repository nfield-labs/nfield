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

from nfield.extraction._papt import TemplateType, describe_field
from nfield.extraction._prompt import builtin_system_message
from nfield.pipeline._coverage import coverage_tokens
from nfield.retrieval._chunker import _DEFAULT_TARGET_TOKENS
from nfield.schema._types import CapacityLeaf

if TYPE_CHECKING:
    from collections.abc import Iterator

    from nfield.config import ExtractionConfig
    from nfield.pipeline._state import PipelineState
    from nfield.schema._types import Field, FieldGroup

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
# Floor so a call's output budget never collapses to zero.
_MIN_SAFE_OUTPUT_TOKENS: int = 50
# Sort key for shared/global fields (no record): packs them after all records.
_UNGROUPED: int = 1 << 30
# Min document room per leaf = one chunk (chunker target). Stage 3 trims larger pools.
_MIN_EXCERPT_TOKENS: int = _DEFAULT_TARGET_TOKENS
# Format-only wrapper around a field's dynamic schema-description text (the
# "(type): " punctuation), added on top of the path/description token estimate.
_SCHEMA_DESC_FORMAT_TOKENS: int = 4
# SFEP emits one "path = value" line per field; tau(f) predicts only the value
# length. The path is echoed in every line (computed dynamically), and this is a
# small fixed allowance for the " = " separator and newline on top of it.
_SFEP_LINE_OVERHEAD_TOKENS: int = 8
# English-average characters per token; used only as a fallback before the
# model-specific ratio measured at calibration is threaded into this estimate.
_FALLBACK_CHARS_PER_TOKEN: float = 4.0
# Tokens for the "[Resolved dependency values ...]" header when a leaf injects
# upstream dependency values (Dependency Field Injection).
_INJECTION_HEADER_TOKENS: int = 12
# Difficulty penalty in the per-leaf reliability budget. A field's reliability
# load is 1 + λ·D(f): a trivial field (D=0) costs 1 unit, a maximally-hard field
# (D=1) costs 1+λ. Extraction reliability falls with field COUNT — and harder
# fields consume more of the model's attention — so the budget is spent in
# difficulty-weighted units, not raw counts (IFScale arXiv:2507.11538; instance-
# count collapse arXiv:2603.22608).
_DIFFICULTY_WEIGHT: float = 1.0


def _reliability_load(fields: list[Field]) -> float:
    """Difficulty-weighted reliability load of a field set: ``Σ (1 + λ·D(f))``.

    Replaces a raw field count as the leaf-size limit, so a leaf can hold many
    easy fields or fewer hard ones — the model's reliable capacity depends on both
    how many fields and how hard they are, not the count alone.

    Args:
        fields: The fields to weigh.

    Returns:
        The summed reliability load (>= ``len(fields)``).
    """
    return sum(1.0 + _DIFFICULTY_WEIGHT * f.difficulty for f in fields)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def _field_schema_tokens(field: Field, chars_per_token: float) -> int:
    """Tokens to *describe* one field in the prompt — costed from the REAL line.

    Renders the exact ``describe_field`` line the prompt will contain (path,
    type, description, title, every constraint, array-item schema, examples) and
    measures it, so the packing budget matches what is actually sent. A char
    estimate over a subset of keys underestimated wide/rich schemas and let a
    leaf overflow the real prompt.

    Args:
        field: The field to describe.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        Estimated schema-description token count for this field.
    """
    cpt = chars_per_token or _FALLBACK_CHARS_PER_TOKEN
    line = describe_field(field, TemplateType.STANDARD)
    return math.ceil(len(line) / cpt) + _SCHEMA_DESC_FORMAT_TOKENS


def _field_output_tokens(field: Field, chars_per_token: float) -> int:
    """Tokens for one SFEP output line ``path = value`` (path echoed + value).

    tau(f) predicts only the *value* length; the field's path is echoed in every
    output line, so it must be counted too. Dynamic per field, so long nested
    paths are charged their real output cost.

    Args:
        field: The field being extracted.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        Estimated output token count for this field's SFEP line.
    """
    cpt = chars_per_token or _FALLBACK_CHARS_PER_TOKEN
    return math.ceil(len(field.path) / cpt) + math.ceil(field.tau) + _SFEP_LINE_OVERHEAD_TOKENS


def compute_K_min(  # noqa: N802
    fields: list[Field],
    safe_output: float,
    chars_per_token: float,
) -> int:
    """Lower bound on the number of LLM calls any packing can achieve.

    The bound is the larger of two independent constraints:

    * **Volume bound** — total predicted output (value *and* echoed path per
      field) cannot exceed the per-call output budget, so at least
      ``ceil(Σ output(f) / safe_output)`` calls are required.
    * **Large-field bound** — any field whose own output would consume more than
      half a call's budget effectively monopolises a call, so the count of such
      fields is itself a lower bound.

    Reported alongside the actual call count K as the optimality gap (K / K_min).

    Args:
        fields: All schema fields, each carrying its predicted output size tau.
        safe_output: Per-call output budget (the model output ceiling).
        chars_per_token: Calibrated characters-per-token ratio (Stage 0), used to
            cost each field's echoed path in the output line.

    Returns:
        Minimum number of calls, at least 1.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("x", "integer", {}, "", {}, tau=10.0)
        >>> compute_K_min([f], safe_output=100.0, chars_per_token=4.0)
        1
    """
    if safe_output <= 0:
        return len(fields)
    sum_out = sum(_field_output_tokens(f, chars_per_token) for f in fields)
    # Fields whose own output exceeds half the budget effectively need their own leaf
    large_field_count = sum(
        1 for f in fields if _field_output_tokens(f, chars_per_token) > 0.5 * safe_output
    )
    return max(1, math.ceil(sum_out / safe_output), large_field_count)


def fits(
    leaf_fields: list[Field],
    D_cost: int,  # noqa: N803
    overhead: int,
    C_usable: float,  # noqa: N803
    output_ceiling: float,
    chars_per_token: float,
) -> bool:
    """Check whether a set of fields fits in a single leaf.

    The binding per-call limit is OUTPUT, not the document. The document excerpt
    is shared across all of a leaf's fields and is trimmed to the leftover budget
    by Stage 3, so it never needs more than ``C_usable - overhead - output`` and
    cannot, on its own, make a leaf infeasible. Two checks therefore apply:

    * Output:  ``output_needed <= output_ceiling`` (model max-output minus margin).
    * Context: after overhead and output, at least a minimal document excerpt
      must still fit — ``overhead + output_needed + min(D_cost, MIN_EXCERPT)
      <= C_usable`` (a small group needing little document uses its real D_cost;
      a large retrieval pool is capped, since Stage 3 will trim it to fit).

    Args:
        leaf_fields: Fields to pack into this leaf.
        D_cost: Token cost of document segments matched for these fields (the
            retrieval pool); only its small-group portion binds, as Stage 3 trims.
        overhead: Fixed token overhead (system prompt + schema description).
        C_usable: Usable slice of the context window. Held well below 100% of
            the window because extraction accuracy degrades as the context fills
            (BABILong arXiv:2406.10149; RULER arXiv:2404.06654; MECW
            arXiv:2509.21361; Sequential-NIAH arXiv:2504.04713).
        output_ceiling: Largest output a single call may safely emit, the model
            output limit minus the heavy-tail margin.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0), used to
            cost each field's echoed path in the output line.

    Returns:
        ``True`` if both constraints are satisfied.

    Example:
        >>> from nfield.schema._types import Field
        >>> f = Field("x", "integer", {}, "", {}, tau=5.0)
        >>> fits([f], D_cost=100, overhead=50, C_usable=500.0, output_ceiling=200.0,
        ...      chars_per_token=4.0)
        True
    """
    output_needed = sum(_field_output_tokens(f, chars_per_token) for f in leaf_fields)
    if output_needed > output_ceiling:
        return False
    # The document is shared and trimmed to the leftover budget, so only its
    # small-group portion binds; a large retrieval pool is capped at MIN_EXCERPT.
    # Output is NOT charged against C_usable: it generates into the window's
    # headroom (see ``output_ceiling``), so only the prompt (overhead + excerpt)
    # is held under the reliability ceiling — input and output are decoupled.
    doc_needed = min(D_cost, _MIN_EXCERPT_TOKENS)
    return overhead + doc_needed <= C_usable


def _coverage_fits(
    groups: list[FieldGroup],
    leaf_fields: list[Field],
    overhead: int,
    c_usable: float,
    output_ceiling: float,
    chars_per_token: float,
) -> bool:
    """Evidence-aware feasibility (Set-Union Bin Packing): does the leaf's coverage
    set — each group's best segment plus each typed field's own best segment — fit
    alongside overhead and output?

    Complements :func:`fits` (which bounds output + a minimal excerpt): here the
    document term is the real coverage floor (the same set Stage 3 keeps, see
    :mod:`_coverage`), deduplicated. When it fails, the leaf must split rather than
    have Stage 3 trim away a field's only evidence — keeping each leaf small and
    focused, which also avoids the accuracy loss of over-long context
    (Lost-in-the-Middle, arXiv:2307.03172).

    Args:
        groups: Candidate groups for the leaf.
        leaf_fields: All fields those groups contribute (for output cost).
        overhead: Fixed token overhead (system/user prompt + schema description).
        c_usable: Usable slice of the context window.
        output_ceiling: Largest output a single call may safely emit.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        ``True`` if overhead + output + best-segment union fits ``c_usable``.
    """
    output_needed = sum(_field_output_tokens(f, chars_per_token) for f in leaf_fields)
    if output_needed > output_ceiling:
        return False
    # Only the prompt (overhead + the evidence coverage floor) is charged against
    # C_usable; output generates into the window headroom (decoupled budgets).
    floor = coverage_tokens(groups, {f.path for f in leaf_fields}, chars_per_token)
    return overhead + floor <= c_usable


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
        >>> len(compute_execution_order([leaf], {}))
        1
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

    # The output ceiling is the most any single call may safely emit. Output is
    # DECOUPLED from the input budget: the prompt (overhead + excerpt) is capped at
    # C_usable for reliability (lost-in-the-middle), but the model generates its
    # answer into the window's *headroom* (C_eff - C_usable), not out of the input
    # budget. So the ceiling is the model output limit (minus the heavy-tail
    # margin) bounded by that headroom — guaranteeing prompt + output <= C_eff
    # without ever stealing excerpt space (the truncation-vs-starvation trade-off
    # the coupled budget forced on verbose, instruction-driven values).
    sum_var_all = sum(f.var_tau for f in state.fields)
    headroom = max(float(_MIN_SAFE_OUTPUT_TOKENS), state.C_eff - state.C_usable)
    output_ceiling = max(
        float(_MIN_SAFE_OUTPUT_TOKENS),
        min(state.M_O - z_eff * math.sqrt(sum_var_all), headroom),
    )

    cpt = state.chars_per_token or _FALLBACK_CHARS_PER_TOKEN
    # K_min also respects the reliability budget: no packing can use fewer than
    # ceil(total reliability load / budget) calls, however generous the token
    # budget is. The load is difficulty-weighted, so a hard schema needs more calls.
    cap = max(1, config.max_fields_per_call)
    token_k_min = compute_K_min(state.fields, output_ceiling, cpt)
    reliability_k_min = math.ceil(_reliability_load(state.fields) / cap) if state.fields else 1
    state.K_min = max(token_k_min, reliability_k_min)

    # Fixed prompt cost (real SFEP system message + caller prompts), measured not guessed.
    builtin_sys = builtin_system_message(knowledge_fallback=state.knowledge_fallback)
    fixed_prompt_chars = len(builtin_sys) + len(state.instructions)
    prompt_overhead = math.ceil(fixed_prompt_chars / cpt)

    # Record-aware packing when the document has a record structure (Group Bin
    # Packing): one record's fields share a leaf, so each leaf maps to <=2 adjacent
    # blocks. Otherwise the lexical First-Fit Decreasing packer.
    if state.record_ordinal:
        leaves = _greedy_record_ffd(
            state.groups,
            state,
            output_ceiling=output_ceiling,
            z_eff=z_eff,
            prompt_overhead=prompt_overhead,
            inject_dependencies=config.inject_dependencies,
            max_fields_per_call=cap,
        )
    else:
        leaves = _greedy_ffd(
            state.groups,
            state,
            output_ceiling=output_ceiling,
            z_eff=z_eff,
            prompt_overhead=prompt_overhead,
            inject_dependencies=config.inject_dependencies,
            max_fields_per_call=cap,
        )
    state.leaves = leaves
    state.execution_order = compute_execution_order(leaves, state.dep_dag)
    return state


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _compute_leaf_overhead(
    fields: list[Field],
    chars_per_token: float,
    prompt_overhead: int = 0,
) -> int:
    """Estimate FTO (Format Token Overhead) for a leaf's fields.

    Both parts are measured, not guessed: the per-field schema-description cost is
    the real rendered ``describe_field`` line (see ``_field_schema_tokens``), and
    the fixed prompt cost (built-in SFEP system message + caller prompts) is passed
    in via ``prompt_overhead`` already measured from the real strings.

    Args:
        fields: Fields in the leaf.
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).
        prompt_overhead: Tokens for the fixed prompt text constant across leaves
            (built-in SFEP system message + caller system/user prompts), measured
            in ``run_stage_2c``. Total overhead = this + per-field schema lines.

    Returns:
        Token overhead estimate.
    """
    schema_tokens = sum(_field_schema_tokens(f, chars_per_token) for f in fields)
    return schema_tokens + prompt_overhead


def _injection_cost(
    fields: list[Field],
    dep_dag: dict[str, set[str]],
    field_by_path: dict[str, Field],
    chars_per_token: float,
) -> int:
    """Estimate tokens to inject this leaf's *cross-leaf* dependency values.

    A dependency field that lands in the SAME leaf costs nothing (the value is
    already present). A dependency in another leaf must be injected as a
    ``path = value`` line, so its tokens are charged to this leaf's overhead —
    this is what can push a leaf over budget and force a split.

    Depends only on the leaf's own field set and the dep graph (not the global
    partition), so it is well-defined during First-Fit packing.

    Args:
        fields: Candidate fields for the leaf.
        dep_dag: Field dependency graph (path -> set of paths it depends on).
        field_by_path: Lookup for predicted token cost (tau) of a dep field.
        chars_per_token: Calibrated ratio for path-string token estimation.

    Returns:
        Estimated injected-block token count (0 if no cross-leaf dependencies).
    """
    cpt = chars_per_token or _FALLBACK_CHARS_PER_TOKEN
    leaf_paths = {f.path for f in fields}
    cross_deps: set[str] = set()
    for f in fields:
        for dep_path in dep_dag.get(f.path, set()):
            if dep_path not in leaf_paths and dep_path in field_by_path:
                cross_deps.add(dep_path)
    if not cross_deps:
        return 0
    total = _INJECTION_HEADER_TOKENS
    for dep_path in cross_deps:
        total += _field_output_tokens(field_by_path[dep_path], cpt)
    return total


def _leaf_safe_output(
    fields: list[Field],
    output_ceiling: float,
    z_eff: float,
    chars_per_token: float,
) -> int:
    """Output reservation for ONE leaf: its predicted output, not the model max.

    Each call reserves ``max_tokens`` against the provider's tokens-per-minute
    budget, so reserving the full ceiling on every call starves the rate limit
    (a leaf emitting ~50 short fields would book the whole model output anyway).
    Reserve what THIS leaf is predicted to emit — ``Σ output(f)`` plus the same
    heavy-tail margin ``z·√Σ var(f)`` used for the global ceiling — capped at the
    ceiling (never exceed it, so ``fits()`` stays valid) and floored so a tiny
    leaf keeps usable room.

    Args:
        fields: The leaf's fields, each carrying tau and var_tau.
        output_ceiling: Upper bound any single call may emit (M_O - margin).
        z_eff: Heavy-tail-inflated z-score (same as the global ceiling).
        chars_per_token: Calibrated characters-per-token ratio (Stage 0).

    Returns:
        Per-leaf ``max_tokens`` in ``[_MIN_SAFE_OUTPUT_TOKENS, output_ceiling]``.
    """
    need = sum(_field_output_tokens(f, chars_per_token) for f in fields)
    variance = sum(f.var_tau for f in fields)
    sized = math.ceil(need + z_eff * math.sqrt(variance))
    return max(_MIN_SAFE_OUTPUT_TOKENS, min(int(output_ceiling), sized))


def _greedy_ffd(
    groups: list[FieldGroup],
    state: PipelineState,
    *,
    output_ceiling: float,
    z_eff: float,
    prompt_overhead: int = 0,
    inject_dependencies: bool = False,
    max_fields_per_call: int = 50,
) -> list[CapacityLeaf]:
    """First-Fit Decreasing bin packing of groups (with per-field split) into leaves.

    Sorts groups by total predicted output descending (heaviest first), the
    standard First-Fit Decreasing heuristic that keeps the packing within ~11/9
    of optimal (Johnson, 1973). Each whole group is placed in the first leaf it
    fits. A group too large for any single call is refined lazily — its fields
    are packed into fresh leaves one at a time — so a wide flat group (e.g. 200
    sibling fields) splits across several calls instead of overflowing one.

    Complexity: O(G·L) placement scans (G groups, L leaves) — first-fit inherently
    rescans existing leaves, and each scan re-costs the candidate's coverage set.
    Comfortable to thousands of leaves (N up to ~10^4 fields). For far larger N
    (10^5 to 10^6) this quadratic leaf scan is the pipeline's scaling bottleneck and
    will need an indexed/sharded first-fit; every other stage is linear (future work).

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

    cpt = state.chars_per_token

    def _overhead(fields: list[Field]) -> int:
        total = _compute_leaf_overhead(fields, cpt, prompt_overhead)
        if inject_dependencies:
            total += _injection_cost(fields, state.dep_dag, state.field_by_path, cpt)
        return total

    def _new_leaf(fields: list[Field], group: FieldGroup) -> None:
        leaves.append(
            CapacityLeaf(
                fields=list(fields),
                groups=[group],
                overhead=_overhead(fields),
                safe_output=_leaf_safe_output(fields, output_ceiling, z_eff, cpt),
                leaf_id=len(leaves),
            )
        )

    for group in sorted_groups:
        # 1. Try to place the whole group into an existing leaf. The leaf must pass
        #    BOTH the output/minimal-excerpt test (fits) AND the evidence-coverage
        #    test (_coverage_fits): the deduped union of every group's best segment
        #    must still fit. When coverage fails the group starts a new leaf — the
        #    Set-Union Bin Packing split, locality-aware because groups sharing
        #    segments have a small union and pack together.
        placed = False
        for leaf in leaves:
            candidate_fields = leaf.fields + group.fields
            # Reliability cap: keep the difficulty-weighted load within budget, even
            # when the token budget would allow more — many easy fields or fewer
            # hard ones, never a raw count that ignores difficulty.
            if _reliability_load(candidate_fields) > max_fields_per_call:
                continue
            candidate_groups = [*leaf.groups, group]
            candidate_dcost = _aggregate_dcost(candidate_groups, state.chars_per_token)
            overhead = _overhead(candidate_fields)
            if fits(
                candidate_fields, candidate_dcost, overhead, state.C_usable, output_ceiling, cpt
            ) and _coverage_fits(
                candidate_groups, candidate_fields, overhead, state.C_usable, output_ceiling, cpt
            ):
                leaf.fields.extend(group.fields)
                leaf.groups.append(group)
                leaf.overhead = overhead
                leaf.safe_output = _leaf_safe_output(leaf.fields, output_ceiling, z_eff, cpt)
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
            overhead = _overhead(candidate)
            # Split when the reliability budget is hit, the token budget is exceeded,
            # or the field-level coverage set would no longer fit (so a typed field's
            # evidence is never trimmed away — same set Stage 3 keeps).
            over_cap = _reliability_load(candidate) > max_fields_per_call
            if current and (
                over_cap
                or not fits(candidate, group.D_cost, overhead, state.C_usable, output_ceiling, cpt)
                or not _coverage_fits(
                    [group], candidate, overhead, state.C_usable, output_ceiling, cpt
                )
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
                overhead=_overhead(state.fields),
                safe_output=_leaf_safe_output(state.fields, output_ceiling, z_eff, cpt),
                leaf_id=0,
            )
        ]
    )


def _greedy_record_ffd(
    groups: list[FieldGroup],
    state: PipelineState,
    *,
    output_ceiling: float,
    z_eff: float,
    prompt_overhead: int = 0,
    inject_dependencies: bool = False,
    max_fields_per_call: int = 50,
) -> list[CapacityLeaf]:
    """Record-aware Next-Fit packing: one record's fields land in adjacent leaves.

    Group Bin Packing (Gilmore-Gomory set-partitioning): the record is the group
    whose items must share a bin. Fields are taken in record (document) order and
    Next-Fit packed (Johnson 1974, <=2x OPT) — only the open leaf is tried, so each
    leaf holds a contiguous run of records and never scatters. A leaf closes when
    either the reliability load exceeds the cap (so K stays at its floor) or the
    leaf's records' blocks no longer fit ``C_usable`` (so the prompt cannot
    overflow). Result: each leaf maps to <=2 adjacent blocks.

    Args:
        groups: FieldGroups from Stage 2A.
        state: PipelineState with ``record_ordinal`` and ``record_block_tokens``.
        output_ceiling: Max tokens any single call may emit (M_O - margin).
        z_eff: Heavy-tail-inflated z-score, for per-leaf output reservation.
        prompt_overhead: Fixed system/user prompt token cost.
        inject_dependencies: Whether resolved dependency values are injected.
        max_fields_per_call: Reliability cap on difficulty-weighted load per leaf.

    Returns:
        List of CapacityLeaf objects in document order.
    """
    cpt = state.chars_per_token
    record_of = state.record_ordinal
    block_tokens = state.record_block_tokens

    def group_record(group: FieldGroup) -> int:
        for f in group.fields:
            if f.path in record_of:
                return record_of[f.path]
        return -1  # global/shared fields (e.g. a document-level header section)

    def overhead(fields: list[Field]) -> int:
        total = _compute_leaf_overhead(fields, cpt, prompt_overhead)
        if inject_dependencies:
            total += _injection_cost(fields, state.dep_dag, state.field_by_path, cpt)
        return total

    def block_cost(records: set[int]) -> int:
        return sum(block_tokens.get(r, 0) for r in records if r >= 0)

    # Document order: by record index, shared fields (-1) last so they pack together.
    ordered = sorted(groups, key=lambda g: group_record(g) if group_record(g) >= 0 else _UNGROUPED)

    leaves: list[CapacityLeaf] = []
    cur_fields: list[Field] = []
    cur_groups: list[FieldGroup] = []
    cur_records: set[int] = set()

    def close() -> None:
        if cur_fields:
            leaves.append(
                CapacityLeaf(
                    fields=list(cur_fields),
                    groups=list(cur_groups),
                    overhead=overhead(cur_fields),
                    safe_output=_leaf_safe_output(cur_fields, output_ceiling, z_eff, cpt),
                    leaf_id=len(leaves),
                )
            )

    for group in ordered:
        rec = group_record(group)
        for f in group.fields:
            candidate = [*cur_fields, f]
            candidate_records = cur_records | ({rec} if rec >= 0 else set())
            over_cap = _reliability_load(candidate) > max_fields_per_call
            # Only a NEW record may overflow the budget: a single record bigger than
            # the budget keeps packing by the field cap (Stage 3 trims its block),
            # rather than degenerating to one field per leaf.
            adds_record = bool(candidate_records - cur_records)
            over_budget = adds_record and (
                overhead(candidate) + block_cost(candidate_records) > state.C_usable
            )
            if cur_fields and (over_cap or over_budget):
                close()
                cur_fields, cur_groups, cur_records = [f], [group], ({rec} if rec >= 0 else set())
            else:
                cur_fields = candidate
                if not cur_groups or cur_groups[-1] is not group:
                    cur_groups.append(group)
                cur_records = candidate_records
    close()

    return leaves or [
        CapacityLeaf(
            fields=list(state.fields),
            groups=list(state.groups),
            overhead=overhead(state.fields),
            safe_output=_leaf_safe_output(state.fields, output_ceiling, z_eff, cpt),
            leaf_id=0,
        )
    ]


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
    # Small-doc fast path: groups carry no matched_segments and every group's
    # D_cost is the same shared full document, which appears in the leaf prompt
    # exactly once. Count it once (max), not once per group (sum) — summing would
    # phantom-inflate the leaf's document cost and force needless extra leaves.
    if not seen_ids:
        return max((g.D_cost for g in groups), default=0)
    ratio = chars_per_token if chars_per_token > 0 else _FALLBACK_CHARS_PER_TOKEN
    return max(1, math.ceil(total / ratio))
