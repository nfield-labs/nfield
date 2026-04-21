"""
Constraint Propagation Engine

Extracts logical rules from JSON schema and links them to prompt.

Three types of constraints:
1. Schema-native: enums, ranges, patterns, type constraints
2. Inferred: if-then dependencies, conditional requirements
3. Vocabulary: mapping between prompt language and schema terms
"""

import re
from typing import Any, Callable, List, Optional, Set
from formatshield.oracle.routing_score import RoutingScore
from formatshield.reasoning.reasoning_task import ConstraintRule


class ConstraintExtractor:
    """Extract validation rules from schema and prompt."""

    def __init__(self, routing_score: RoutingScore):
        """
        Initialize extractor with routing context.

        Args:
            routing_score: RoutingScore containing λ̃₂, τ, ΔK, phi
        """
        self.routing_score = routing_score
        self.lambda2 = routing_score.lambda2
        self.tau = routing_score.tau
        self.delta_k = routing_score.delta_k
        self.phi = routing_score.phi

    def extract(self, schema: dict, prompt: str) -> List[ConstraintRule]:
        """
        Main entry point: extract all constraints from schema + prompt.

        Args:
            schema: JSON Schema dict
            prompt: User prompt string

        Returns:
            List of ConstraintRule in priority order (hard rules first)
        """
        all_rules: List[ConstraintRule] = []

        # Extract schema-native constraints
        all_rules.extend(self._extract_enum_rules(schema))
        all_rules.extend(self._extract_range_rules(schema))
        all_rules.extend(self._extract_type_rules(schema))
        all_rules.extend(self._extract_pattern_rules(schema))

        # Extract inferred constraints
        all_rules.extend(self._extract_conditional_rules(schema))
        all_rules.extend(self._extract_dependency_rules(schema))

        # Extract vocabulary bridges if high ΔK
        vocabulary_threshold = 0.5  # Default threshold
        if self.delta_k > vocabulary_threshold:
            all_rules.extend(self._extract_vocabulary_rules(schema, prompt))

        # Sort: hard rules first, then by specificity
        all_rules.sort(key=lambda r: (r.priority != "hard", -len(r.schema_path)))

        return all_rules

    def _extract_enum_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract enumeration constraints (choice from fixed set)."""
        rules: List[ConstraintRule] = []

        def walk_enum(node: Any, path: str = "") -> None:
            """Recursively find enum fields."""
            if isinstance(node, dict):
                if "enum" in node:
                    enum_values = node["enum"]
                    rules.append(
                        ConstraintRule(
                            rule_type="enum",
                            description=f"Must be one of: {', '.join(str(v) for v in enum_values)}",
                            schema_path=path or "root",
                            constraint_value=enum_values,
                            injection_point="pass1_system",
                            validator=lambda x, values=enum_values: x in values,
                            priority="hard",
                        )
                    )

                # Recurse
                if "properties" in node:
                    for name, spec in node["properties"].items():
                        walk_enum(spec, f"{path}.properties.{name}" if path else f"properties.{name}")

                if "items" in node:
                    walk_enum(node["items"], f"{path}.items")

        walk_enum(schema)
        return rules

    def _extract_range_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract numeric range constraints (min/max, minItems/maxItems)."""
        rules: List[ConstraintRule] = []

        def walk_range(node: Any, path: str = "") -> None:
            """Recursively find range constraints."""
            if isinstance(node, dict):
                # Numeric ranges
                if "minimum" in node or "maximum" in node:
                    min_val = node.get("minimum")
                    max_val = node.get("maximum")
                    description = f"Must be in range [{min_val}, {max_val}]"

                    def validate_range(x: Any, min_v=min_val, max_v=max_val) -> bool:
                        try:
                            num = float(x) if isinstance(x, (int, float)) else float(str(x))
                            if min_v is not None and num < min_v:
                                return False
                            if max_v is not None and num > max_v:
                                return False
                            return True
                        except (TypeError, ValueError):
                            return False

                    rules.append(
                        ConstraintRule(
                            rule_type="range",
                            description=description,
                            schema_path=path or "root",
                            constraint_value=(min_val, max_val),
                            injection_point="validation",
                            validator=validate_range,
                            priority="soft",
                        )
                    )

                # Array cardinality
                if "minItems" in node or "maxItems" in node:
                    min_items = node.get("minItems")
                    max_items = node.get("maxItems")
                    description = f"Array must have {min_items}-{max_items} items"

                    def validate_items(x: Any, min_i=min_items, max_i=max_items) -> bool:
                        if not isinstance(x, list):
                            return False
                        if min_i is not None and len(x) < min_i:
                            return False
                        if max_i is not None and len(x) > max_i:
                            return False
                        return True

                    rules.append(
                        ConstraintRule(
                            rule_type="range",
                            description=description,
                            schema_path=path or "root",
                            constraint_value=(min_items, max_items),
                            injection_point="validation",
                            validator=validate_items,
                            priority="soft",
                        )
                    )

                # Recurse
                if "properties" in node:
                    for name, spec in node["properties"].items():
                        walk_range(spec, f"{path}.properties.{name}" if path else f"properties.{name}")

        walk_range(schema)
        return rules

    def _extract_type_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract type constraints (string, number, boolean, etc.)."""
        rules: List[ConstraintRule] = []

        def walk_types(node: Any, path: str = "") -> None:
            """Recursively find type constraints."""
            if isinstance(node, dict) and "type" in node:
                field_type = node["type"]
                description = f"Must be of type: {field_type}"

                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict,
                }
                expected_types = type_map.get(field_type, object)

                def validate_type(x: Any, types=expected_types) -> bool:
                    return isinstance(x, types)

                rules.append(
                    ConstraintRule(
                        rule_type="consistency",
                        description=description,
                        schema_path=path or "root",
                        constraint_value=field_type,
                        injection_point="validation",
                        validator=validate_type,
                        priority="hard",
                    )
                )

            # Recurse
            if isinstance(node, dict):
                if "properties" in node:
                    for name, spec in node["properties"].items():
                        walk_types(spec, f"{path}.properties.{name}" if path else f"properties.{name}")

        walk_types(schema)
        return rules

    def _extract_pattern_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract regex pattern constraints."""
        rules: List[ConstraintRule] = []

        def walk_patterns(node: Any, path: str = "") -> None:
            """Recursively find pattern constraints."""
            if isinstance(node, dict) and "pattern" in node:
                pattern = node["pattern"]
                description = f"Must match pattern: {pattern}"

                def validate_pattern(x: Any, pat=pattern) -> bool:
                    try:
                        return bool(re.match(pat, str(x)))
                    except Exception:
                        return False

                rules.append(
                    ConstraintRule(
                        rule_type="consistency",
                        description=description,
                        schema_path=path or "root",
                        constraint_value=pattern,
                        injection_point="validation",
                        validator=validate_pattern,
                        priority="soft",
                    )
                )

            # Recurse
            if isinstance(node, dict):
                if "properties" in node:
                    for name, spec in node["properties"].items():
                        walk_patterns(spec, f"{path}.properties.{name}" if path else f"properties.{name}")

        walk_patterns(schema)
        return rules

    def _extract_conditional_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract if-then-else conditional rules."""
        rules: List[ConstraintRule] = []

        if "dependentSchemas" not in schema:
            return rules

        dep = schema.get("dependentSchemas", {})

        # Simple if-then: if some field exists, then some other field must exist
        if "if" in dep and "then" in dep:
            if_clause = dep["if"]
            then_clause = dep["then"]

            description = f"Conditional: if ({if_clause}) then ({then_clause})"

            rules.append(
                ConstraintRule(
                    rule_type="conditional",
                    description=description,
                    schema_path="conditional",
                    constraint_value={"if": if_clause, "then": then_clause},
                    injection_point="pass1_system",
                    priority="soft",
                )
            )

        return rules

    def _extract_dependency_rules(self, schema: dict) -> List[ConstraintRule]:
        """Extract field interdependency rules."""
        rules: List[ConstraintRule] = []

        # additionalProperties: false → no unknown fields allowed
        if schema.get("additionalProperties") is False:
            props = list(schema.get("properties", {}).keys())
            description = f"No additional properties. Only allowed: {', '.join(props)}"

            rules.append(
                ConstraintRule(
                    rule_type="dependency",
                    description=description,
                    schema_path="additionalProperties",
                    constraint_value=False,
                    injection_point="validation",
                    priority="hard",
                )
            )

        # Required fields
        required = schema.get("required", [])
        if required:
            description = f"Required fields: {', '.join(required)}"

            def validate_required(x: Any, req=required) -> bool:
                if not isinstance(x, dict):
                    return False
                return all(field in x for field in req)

            rules.append(
                ConstraintRule(
                    rule_type="dependency",
                    description=description,
                    schema_path="required",
                    constraint_value=required,
                    injection_point="validation",
                    validator=validate_required,
                    priority="hard",
                )
            )

        return rules

    def _extract_vocabulary_rules(self, schema: dict, prompt: str) -> List[ConstraintRule]:
        """Extract vocabulary mapping rules when ΔK > threshold."""
        rules: List[ConstraintRule] = []

        props = schema.get("properties", {})
        prompt_lower = prompt.lower()

        # Simple heuristic: if schema field name doesn't appear in prompt, flag it
        unmapped_fields = [
            name for name in props.keys()
            if name.lower() not in prompt_lower and name.replace("_", " ").lower() not in prompt_lower
        ]

        if unmapped_fields:
            description = (
                f"Vocabulary bridge needed. Schema fields not mentioned in prompt: {unmapped_fields}. "
                f"Map them explicitly using schema understanding."
            )

            rules.append(
                ConstraintRule(
                    rule_type="vocabulary",
                    description=description,
                    schema_path="vocabulary_bridge",
                    constraint_value=unmapped_fields,
                    injection_point="pass1_system",
                    priority="soft",
                )
            )

        return rules


def extract_constraints(schema: dict, prompt: str, routing_score: RoutingScore) -> List[ConstraintRule]:
    """
    Public API: extract constraints from schema and prompt.

    Args:
        schema: JSON Schema dict
        prompt: User prompt string
        routing_score: RoutingScore (contains λ̃₂, τ, ΔK)

    Returns:
        List of ConstraintRule in priority order
    """
    extractor = ConstraintExtractor(routing_score)
    return extractor.extract(schema, prompt)
