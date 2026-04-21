"""
Schema-Aware Reasoning Compiler

Converts JSON schema into structured reasoning task instructions.

Core insight: Φ components signal what KIND of reasoning is needed.
- λ̃₂ < 0.2 (flat) → pure extraction, no reasoning
- 0.2 ≤ λ̃₂ < 0.6 (moderate) → mapping with light reasoning
- λ̃₂ ≥ 0.6 (complex) → deep structural reasoning

This module generates task instructions per task type.
"""

from typing import Any

from formatshield.oracle.routing_score import RoutingScore
from formatshield.reasoning.reasoning_task import ReasoningTask

# Keywords that strongly indicate a reasoning task regardless of λ̃₂
_REASONING_KEYWORDS = frozenset(
    {
        "analyze",
        "analyse",
        "evaluate",
        "logical",
        "validity",
        "valid",
        "argue",
        "argument",
        "reason",
        "fallacy",
        "fallacies",
        "assess",
        "judge",
        "judgement",
        "judgment",
        "critique",
        "infer",
        "inference",
        "deduce",
        "deduction",
        "premise",
        "premises",
        "conclusion",
        "evidence",
        "sound",
        "unsound",
        "identify flaw",
        "identify error",
        "step by step",
        "step-by-step",
    }
)

_REASONING_KEYWORD_THRESHOLD = 2  # Number of keywords needed to override λ̃₂


class SchemaCompiler:
    """Compile JSON schema into structured reasoning task."""

    def __init__(self, routing_score: RoutingScore):
        """
        Initialize compiler with routing context.

        Args:
            routing_score: RoutingScore containing λ̃₂, τ, ΔK, phi
        """
        self.routing_score = routing_score
        self.lambda2 = routing_score.lambda2
        self.tau = routing_score.tau
        self.delta_k = routing_score.delta_k
        self.phi = routing_score.phi

    def compile(self, schema: dict, prompt: str = "") -> ReasoningTask:
        """
        Main entry point: convert schema → reasoning task.

        Args:
            schema: JSON Schema dict
            prompt: Optional user prompt for semantic task-type detection

        Returns:
            ReasoningTask with instructions, constraints, dependencies
        """
        task_type = self._detect_task_type(prompt)
        instructions = self._generate_instructions(schema, task_type)
        dependencies = self._extract_dependencies(schema)
        summary = self._summarize_schema(schema)
        vocab_bridge = self._detect_vocabulary_gap(schema)

        return ReasoningTask(
            task_type=task_type,
            instructions=instructions,
            field_dependencies=dependencies,
            schema_summary=summary,
            vocabulary_bridge=vocab_bridge,
            estimated_tokens=self._estimate_tokens(task_type, instructions),
        )

    def _detect_task_type(self, prompt: str = "") -> str:
        """
        Detect task type using λ̃₂ AND semantic prompt analysis.

        Semantic override: if the prompt contains ≥2 reasoning keywords
        (evaluate, analyze, logical, fallacy, etc.) the task type is
        upgraded to "reasoning" regardless of λ̃₂.

        λ̃₂ fallback (no strong semantic signal):
        - 0.0–0.2: flat, no dependencies → EXTRACTION
        - 0.2–0.6: moderate connectivity → CLASSIFICATION
        - 0.6–1.0: highly connected → REASONING
        """
        # Semantic override: count reasoning keywords in prompt
        if prompt:
            prompt_lower = prompt.lower()
            keyword_hits = sum(1 for kw in _REASONING_KEYWORDS if kw in prompt_lower)
            if keyword_hits >= _REASONING_KEYWORD_THRESHOLD:
                return "reasoning"

        # Fall back to λ̃₂ thresholds
        if self.lambda2 < 0.2:
            return "extraction"
        elif self.lambda2 < 0.6:
            return "classification"
        else:
            return "reasoning"

    def _generate_instructions(self, schema: dict, task_type: str) -> str:
        """
        Generate step-by-step reasoning task instructions.

        Task type determines tone:
        - extraction: direct mapping, no reasoning needed
        - classification: lightweight reasoning with rules
        - reasoning: deep structural reasoning
        """
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        if task_type == "extraction":
            return self._instructions_extraction(properties, required)
        elif task_type == "classification":
            return self._instructions_classification(properties, required)
        else:
            return self._instructions_reasoning(properties, required)

    def _instructions_extraction(self, properties: dict, required: list[str]) -> str:
        """Generate extraction task: direct field mapping with binding rules."""
        instructions = (
            "TASK: Extract and map the following fields EXACTLY as they appear. "
            "Do NOT infer or add information beyond what is stated.\n\n"
        )

        instructions += "REQUIRED FIELDS (MANDATORY — MUST be populated):\n"
        for field_name in required:
            field_spec = properties.get(field_name, {})
            field_type = field_spec.get("type", "string")
            description = field_spec.get("description", field_name)
            instructions += f"- {field_name} ({field_type}): {description}\n"

        optional_fields = set(properties.keys()) - set(required)
        if optional_fields:
            instructions += "\nOPTIONAL FIELDS (populate if present in source):\n"
            for field_name in sorted(optional_fields):
                field_spec = properties[field_name]
                field_type = field_spec.get("type", "string")
                instructions += f"- {field_name} ({field_type})\n"

        instructions += "\nRULES — FOLLOW EXACTLY:\n"
        instructions += "1. Extract exactly what is present in the input. Nothing more.\n"
        instructions += "2. Do NOT infer, generate, or add information not explicitly stated.\n"
        instructions += "3. If a required field is missing from the source, return null.\n"
        instructions += "4. NEVER fabricate field values.\n"

        return instructions

    def _instructions_classification(self, properties: dict, required: list[str]) -> str:
        """Generate classification task: map to categories. FOLLOW EVERY STEP."""
        instructions = (
            "TASK: Classify and map inputs to the appropriate schema fields. "
            "FOLLOW EVERY STEP — do NOT skip.\n\n"
        )

        instructions += "STEP 1: Understand what you are classifying\n"
        instructions += (
            "Review ALL fields below. Understand their meaning and relationships "
            "BEFORE assigning any values.\n\n"
        )

        instructions += "STEP 2: Classify the input\n"
        instructions += (
            "For each field, determine the EXACT match or classification. "
            "You MUST select a value for every required field.\n\n"
        )

        # Find enum fields and highlight them
        enum_fields = [(name, spec) for name, spec in properties.items() if "enum" in spec]
        if enum_fields:
            instructions += "ENUMERATED FIELDS (choose EXACTLY ONE value):\n"
            for field_name, spec in enum_fields:
                options = spec.get("enum", [])
                opts_str = ", ".join(str(o) for o in options)
                instructions += f"- {field_name}: MUST be one of → {opts_str}\n"
            instructions += "\n"

        instructions += "STEP 3: Validate your choices\n"
        instructions += "Ensure each required field is assigned a valid value.\n"
        instructions += (
            "If any enum field has no clear match, pick the closest option and explain why "
            "— do NOT invent values outside the allowed list.\n"
        )

        return instructions

    def _instructions_reasoning(self, properties: dict, required: list[str]) -> str:
        """Generate reasoning task: deep structural reasoning. FOLLOW EVERY STEP IN ORDER."""
        instructions = (
            "TASK: Reason through the problem and produce a structured response. "
            "FOLLOW EVERY STEP IN ORDER — do NOT skip or reorder.\n\n"
        )

        instructions += "STEP 1: Understand the structure\n"
        instructions += (
            "The output has multiple interconnected fields. You MUST understand "
            "their relationships BEFORE making any decisions.\n\n"
        )

        instructions += "STEP 2: Reason step by step\n"
        instructions += "For each decision or field, you MUST:\n"
        instructions += "1. State EXACTLY what you are deciding\n"
        instructions += "2. List the alternatives and constraints available\n"
        instructions += "3. Explain your choice with supporting evidence\n"
        instructions += "4. Note ALL dependencies on other fields\n\n"

        instructions += "STEP 3: Check for consistency BEFORE finalizing\n"
        instructions += "You MUST verify:\n"
        instructions += "- No contradictions between fields\n"
        instructions += "- All required fields are addressed\n"
        instructions += "- All interdependencies are satisfied\n"
        instructions += "- Boolean verdict fields reflect sub-component results\n\n"

        if required:
            instructions += "REQUIRED FIELDS (MANDATORY — all must be populated):\n"
            for field_name in required:
                instructions += f"- {field_name}\n"

        return instructions

    def _extract_dependencies(self, schema: dict) -> dict[str, list[str]]:
        """
        Extract field dependencies from schema.

        Detects:
        - Direct dependencies (if-then-else schemas)
        - Transitive dependencies (nested objects)
        - Conditional requirements
        """
        dependencies: dict[str, list[str]] = {}
        properties = schema.get("properties", {})

        # Initialize: each field depends on nothing by default
        for field_name in properties:
            dependencies[field_name] = []

        # Detect conditional dependencies (dependentSchemas)
        if "dependentSchemas" in schema:
            dep_schemas = schema["dependentSchemas"]
            if "if" in dep_schemas and "then" in dep_schemas:
                # Try to extract field names from if/then
                # This is a simplification; real schemas may be more complex
                if_spec = dep_schemas["if"]
                then_spec = dep_schemas["then"]

                if_fields = self._extract_field_names(if_spec)
                then_fields = self._extract_field_names(then_spec)

                # If any if_field, then all then_fields are needed
                for then_field in then_fields:
                    if then_field in dependencies:
                        dependencies[then_field] = list(set(dependencies[then_field] + if_fields))

        # Detect nested object dependencies
        for field_name, field_spec in properties.items():
            if field_spec.get("type") == "object" and "properties" in field_spec:
                nested_props = field_spec["properties"]
                dependencies[field_name] = list(nested_props.keys())

        return dependencies

    def _extract_field_names(self, spec: Any) -> list[str]:
        """Recursively extract field names from schema spec."""
        fields: list[str] = []

        if isinstance(spec, dict):
            if "properties" in spec:
                fields.extend(spec["properties"].keys())
            if "required" in spec and isinstance(spec["required"], list):
                fields.extend(spec["required"])

            # Recurse into nested specs
            for value in spec.values():
                if isinstance(value, dict):
                    fields.extend(self._extract_field_names(value))
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            fields.extend(self._extract_field_names(item))

        return list(set(fields))

    def _summarize_schema(self, schema: dict) -> str:
        """Generate human-readable schema summary."""
        properties = schema.get("properties", {})
        required = schema.get("required", [])

        summary = f"Output has {len(properties)} fields, {len(required)} required.\n\n"

        # Count field types
        types: dict[str, int] = {}
        for field_spec in properties.values():
            field_type = field_spec.get("type", "object")
            types[field_type] = types.get(field_type, 0) + 1

        if types:
            type_str = ", ".join(f"{count} {t}" for t, count in sorted(types.items()))
            summary += f"Field types: {type_str}\n"

        # List enums
        enums = [name for name, spec in properties.items() if "enum" in spec]
        if enums:
            summary += f"Enumerated fields: {', '.join(enums)}\n"

        return summary

    def _detect_vocabulary_gap(self, schema: dict) -> str | None:
        """
        Detect vocabulary mismatch when ΔK is high.

        ΔK > 0.5 indicates prompt and schema use different terminology.
        """
        if self.delta_k <= 0.5:
            return None

        properties = schema.get("properties", {})
        field_names = list(properties.keys())

        return (
            f"The schema uses field names: {', '.join(field_names[:5])}...\n"
            f"Your prompt may use different terms. Map them explicitly if needed."
        )

    def _estimate_tokens(self, task_type: str, instructions: str) -> int:
        """Rough estimate of tokens in instructions."""
        # Average ~4 chars per token
        base_tokens = len(instructions) // 4

        # Add overhead by task type
        overhead = {"extraction": 0, "classification": 100, "reasoning": 200}.get(task_type, 0)

        return max(256, base_tokens + overhead)


def compile_schema_to_task(
    schema: dict,
    routing_score: RoutingScore,
    prompt: str = "",
) -> ReasoningTask:
    """
    Public API: compile schema to reasoning task.

    Args:
        schema: JSON Schema dict
        routing_score: RoutingScore (contains λ̃₂, τ, ΔK)
        prompt: Optional user prompt for semantic task-type detection.
                When provided, reasoning keywords in the prompt can upgrade
                the task type to "reasoning" even for flat schemas.

    Returns:
        ReasoningTask with instructions and constraints
    """
    compiler = SchemaCompiler(routing_score)
    return compiler.compile(schema, prompt=prompt)
