"""
Schema-Aware Reasoning Compiler

Converts JSON schema into structured reasoning task instructions.

Core insight: Φ components signal what KIND of reasoning is needed.
- λ̃₂ < 0.2 (flat) → pure extraction, no reasoning
- 0.2 ≤ λ̃₂ < 0.6 (moderate) → mapping with light reasoning
- λ̃₂ ≥ 0.6 (complex) → deep structural reasoning

This module generates task instructions per task type.
"""

from typing import Any, Dict, List, Optional, Set
from formatshield.oracle.routing_score import RoutingScore
from formatshield.reasoning.reasoning_task import ReasoningTask


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

    def compile(self, schema: dict) -> ReasoningTask:
        """
        Main entry point: convert schema → reasoning task.

        Args:
            schema: JSON Schema dict

        Returns:
            ReasoningTask with instructions, constraints, dependencies
        """
        task_type = self._detect_task_type()
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

    def _detect_task_type(self) -> str:
        """
        Detect task type from λ̃₂ (schema graph complexity).

        λ̃₂ ∈ [0, 1]:
        - 0.0–0.2: flat, no dependencies → EXTRACTION
        - 0.2–0.6: moderate connectivity → CLASSIFICATION
        - 0.6–1.0: highly connected → REASONING
        """
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

    def _instructions_extraction(self, properties: dict, required: List[str]) -> str:
        """Generate extraction task: direct field mapping."""
        instructions = "TASK: Extract and map the following fields exactly as they appear.\n\n"

        instructions += "REQUIRED FIELDS:\n"
        for field_name in required:
            field_spec = properties.get(field_name, {})
            field_type = field_spec.get("type", "string")
            description = field_spec.get("description", field_name)
            instructions += f"- {field_name} ({field_type}): {description}\n"

        optional_fields = set(properties.keys()) - set(required)
        if optional_fields:
            instructions += "\nOPTIONAL FIELDS:\n"
            for field_name in sorted(optional_fields):
                field_spec = properties[field_name]
                field_type = field_spec.get("type", "string")
                instructions += f"- {field_name} ({field_type})\n"

        instructions += "\nRULES:\n"
        instructions += "1. Extract exactly what is present in the input.\n"
        instructions += "2. Do not infer or add information not stated.\n"
        instructions += "3. If a required field is missing, return null or error.\n"

        return instructions

    def _instructions_classification(self, properties: dict, required: List[str]) -> str:
        """Generate classification task: map to categories with light reasoning."""
        instructions = "TASK: Classify and map inputs to the appropriate schema fields.\n\n"

        instructions += "STEP 1: Understand what you're classifying\n"
        instructions += "Review all the fields below and understand their meaning and relationships.\n\n"

        instructions += "STEP 2: Classify the input\n"
        instructions += "For each field, determine the best match or classification.\n\n"

        # Find enum fields and highlight them
        enum_fields = [
            (name, spec) for name, spec in properties.items()
            if "enum" in spec
        ]
        if enum_fields:
            instructions += "ENUMERATED FIELDS (choose exactly one value):\n"
            for field_name, spec in enum_fields:
                options = spec.get("enum", [])
                instructions += f"- {field_name}: {', '.join(str(o) for o in options)}\n"
            instructions += "\n"

        instructions += "STEP 3: Validate your choices\n"
        instructions += "Ensure each required field is assigned a valid value.\n"
        instructions += "If uncertain, explain your reasoning.\n"

        return instructions

    def _instructions_reasoning(self, properties: dict, required: List[str]) -> str:
        """Generate reasoning task: deep structural reasoning."""
        instructions = "TASK: Reason through the problem and produce a structured response.\n\n"

        instructions += "STEP 1: Understand the structure\n"
        instructions += "The output has multiple interconnected fields. Understanding their relationships is key.\n\n"

        instructions += "STEP 2: Reason step by step\n"
        instructions += "For each decision or field:\n"
        instructions += "1. State what you're deciding\n"
        instructions += "2. List alternatives or constraints\n"
        instructions += "3. Explain your choice\n"
        instructions += "4. Note any dependencies on other fields\n\n"

        instructions += "STEP 3: Check for consistency\n"
        instructions += "Before finalizing:\n"
        instructions += "- Ensure no contradictions between fields\n"
        instructions += "- Verify all required fields are addressed\n"
        instructions += "- Confirm interdependencies are satisfied\n\n"

        instructions += "REQUIRED FIELDS:\n"
        for field_name in required:
            instructions += f"- {field_name}\n"

        return instructions

    def _extract_dependencies(self, schema: dict) -> Dict[str, List[str]]:
        """
        Extract field dependencies from schema.

        Detects:
        - Direct dependencies (if-then-else schemas)
        - Transitive dependencies (nested objects)
        - Conditional requirements
        """
        dependencies: Dict[str, List[str]] = {}
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

    def _extract_field_names(self, spec: Any) -> List[str]:
        """Recursively extract field names from schema spec."""
        fields: List[str] = []

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
        types: Dict[str, int] = {}
        for field_spec in properties.values():
            field_type = field_spec.get("type", "object")
            types[field_type] = types.get(field_type, 0) + 1

        if types:
            summary += "Field types: " + ", ".join(f"{count} {t}" for t, count in sorted(types.items())) + "\n"

        # List enums
        enums = [
            name for name, spec in properties.items()
            if "enum" in spec
        ]
        if enums:
            summary += f"Enumerated fields: {', '.join(enums)}\n"

        return summary

    def _detect_vocabulary_gap(self, schema: dict) -> Optional[str]:
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


def compile_schema_to_task(schema: dict, routing_score: RoutingScore) -> ReasoningTask:
    """
    Public API: compile schema to reasoning task.

    Args:
        schema: JSON Schema dict
        routing_score: RoutingScore (contains λ̃₂, τ, ΔK)

    Returns:
        ReasoningTask with instructions and constraints
    """
    compiler = SchemaCompiler(routing_score)
    return compiler.compile(schema)
