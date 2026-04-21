"""
Schema-Conditioned Reasoning Engine

Transform FormatShield from "structured output formatter" to "cognitive compiler for LLMs."

Three core modules:
1. schema_compiler: Convert JSON schema → reasoning task program
2. constraint_engine: Extract rules from schema + prompt
3. phi_controller: Shape thinking using Φ components (λ̃₂, τ, ΔK)

Usage:
    from formatshield.reasoning import compile_schema_to_task, extract_constraints, shape_thinking_with_phi
    from formatshield.oracle.routing_score import compute_routing_score

    schema = {...}
    prompt = "..."
    routing_score = compute_routing_score(prompt, schema)

    # 1. Compile schema to task
    task = compile_schema_to_task(schema, routing_score)

    # 2. Extract constraints
    rules = extract_constraints(schema, prompt, routing_score)

    # 3. Shape thinking
    shaping = shape_thinking_with_phi(routing_score)

    # 4. Merge into Pass 1 prompt
    pass1_system = task.merge_with_shaping(shaping)
"""

from formatshield.reasoning.reasoning_task import (
    ConstraintRule,
    ReasoningTask,
    ReasoningTaskConfig,
    ThinkingShaping,
)
from formatshield.reasoning.schema_compiler import (
    SchemaCompiler,
    compile_schema_to_task,
)
from formatshield.reasoning.constraint_engine import (
    ConstraintExtractor,
    extract_constraints,
)
from formatshield.reasoning.phi_controller import (
    PhiController,
    shape_thinking_with_phi,
)

__all__ = [
    # Data contracts
    "ReasoningTask",
    "ConstraintRule",
    "ThinkingShaping",
    "ReasoningTaskConfig",
    # Public APIs
    "compile_schema_to_task",
    "extract_constraints",
    "shape_thinking_with_phi",
    # Classes (advanced usage)
    "SchemaCompiler",
    "ConstraintExtractor",
    "PhiController",
]
