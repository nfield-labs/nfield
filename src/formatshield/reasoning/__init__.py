"""
Schema-Conditioned Reasoning Engine

Transform FormatShield from "structured output formatter" to "cognitive compiler for LLMs."

Eight core modules:
1. schema_compiler:        Convert JSON schema → reasoning task program
2. constraint_engine:      Extract rules from schema + prompt
3. phi_controller:         Shape thinking using Φ components (λ̃₂, τ, ΔK)
4. execution_plan:         Generate binding step-by-step execution protocols
5. aggregation_compiler:   Derive aggregation rules from schema structure + post-gen verification
6. step_gate:              Forward-steering enforcement at step boundaries
7. constraint_graph:       Bidirectional semantic constraint propagation graph
8. retry_budget:           Schema complexity-aware retry budget allocation + failure triage

Usage:
    from formatshield.reasoning import (
        compile_schema_to_task, extract_constraints,
        shape_thinking_with_phi, build_execution_plan,
        compile_aggregation_rules, verify_aggregation_rules,
        check_execution_steps,
        build_constraint_graph,
        allocate_retry_budget, classify_failure, build_surgical_reask,
    )
    from formatshield.oracle.routing_score import compute_routing_score

    schema = {...}
    prompt = "..."
    routing_score = compute_routing_score(prompt, schema)

    # Core pipeline
    task = compile_schema_to_task(schema, routing_score, prompt=prompt)
    rules = extract_constraints(schema, prompt, routing_score)
    shaping = shape_thinking_with_phi(routing_score)
    plan = build_execution_plan(schema, routing_score)
    pass1_system = task.merge_with_shaping(shaping)

    # Derive + verify aggregation rules (boolean/numeric/enum consistency)
    agg_rules = compile_aggregation_rules(schema)
    # After Pass 2:
    agg_result = verify_aggregation_rules(output, agg_rules)
    if not agg_result.passed:
        reask = build_aggregation_reask(output, agg_result.failed_rules)

    # Check step completion after Pass 1 (forward-steering if incomplete)
    gate_result = check_execution_steps(plan, partial_output, trace_text)
    if not gate_result.all_complete:
        context = gate_result.combined_injection + context

    # Bidirectional constraint propagation
    cpg = build_constraint_graph(schema)
    batch_result = cpg.propagate_batch(output)
    if batch_result.inconsistencies:
        ...  # handle inconsistencies found by propagation

    # Complexity-aware retry budget
    budget = allocate_retry_budget(schema, lambda2=0.7, tau=0.8)
    failure = classify_failure(error_msg, output, agg_rules)
    reask = build_surgical_reask(output, [failure])
"""

from formatshield.reasoning.aggregation_compiler import (
    AggregationCompiler,
    AggregationPattern,
    AggregationRule,
    AggregationVerificationResult,
    build_aggregation_reask,
    compile_aggregation_rules,
    verify_aggregation_rules,
)
from formatshield.reasoning.constraint_engine import (
    ConstraintExtractor,
    extract_constraints,
)
from formatshield.reasoning.constraint_graph import (
    ConstraintPropagationGraph,
    DomainReduction,
    EdgeType,
    GraphEdge,
    PropagationResult,
    build_constraint_graph,
)
from formatshield.reasoning.execution_plan import (
    ExecutionPlan,
    ExecutionPlanBuilder,
    ExecutionStep,
    build_execution_plan,
    render_execution_plan,
)
from formatshield.reasoning.phi_controller import (
    PhiController,
    shape_thinking_with_phi,
)
from formatshield.reasoning.reasoning_task import (
    ConstraintRule,
    ReasoningTask,
    ReasoningTaskConfig,
    ThinkingShaping,
)
from formatshield.reasoning.retry_budget import (
    BudgetAllocation,
    FailureClassification,
    FailureTriager,
    FailureType,
    RetryBudgetAllocator,
    SurgicalReasker,
    allocate_retry_budget,
    build_surgical_reask,
    classify_failure,
)
from formatshield.reasoning.schema_compiler import (
    SchemaCompiler,
    compile_schema_to_task,
)
from formatshield.reasoning.step_gate import (
    GateResult,
    StepCheckResult,
    TemporalStepGate,
    check_execution_steps,
    parse_partial_output,
)

__all__ = [
    "AggregationCompiler",
    "AggregationPattern",
    "AggregationRule",
    "AggregationVerificationResult",
    "BudgetAllocation",
    "ConstraintExtractor",
    "ConstraintPropagationGraph",
    "ConstraintRule",
    "DomainReduction",
    "EdgeType",
    "ExecutionPlan",
    "ExecutionPlanBuilder",
    "ExecutionStep",
    "FailureClassification",
    "FailureTriager",
    "FailureType",
    "GateResult",
    "GraphEdge",
    "PhiController",
    "PropagationResult",
    "ReasoningTask",
    "ReasoningTaskConfig",
    "RetryBudgetAllocator",
    "SchemaCompiler",
    "StepCheckResult",
    "SurgicalReasker",
    "TemporalStepGate",
    "ThinkingShaping",
    "allocate_retry_budget",
    "build_aggregation_reask",
    "build_constraint_graph",
    "build_execution_plan",
    "build_surgical_reask",
    "check_execution_steps",
    "classify_failure",
    "compile_aggregation_rules",
    "compile_schema_to_task",
    "extract_constraints",
    "parse_partial_output",
    "render_execution_plan",
    "shape_thinking_with_phi",
    "verify_aggregation_rules",
]
