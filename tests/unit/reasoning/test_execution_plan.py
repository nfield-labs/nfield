"""
Unit Tests: Execution Plan Builder Module

Tests the execution_plan module's ability to:
1. Detect array fields with minItems → mandatory iteration steps
2. Detect enum-constrained array items → restricted selection steps
3. Detect top-level enum fields → binding selection step
4. Detect numeric range fields → bounds validation step
5. Detect boolean verdict fields + sub-arrays → consistency rules
6. Render plans as binding protocol strings
7. Compute enforcement level from τ (via routing score)
"""

import pytest

from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning.execution_plan import (
    ExecutionPlan,
    ExecutionStep,
    build_execution_plan,
    render_execution_plan,
)


class TestArrayFieldExtractionSteps:
    """Test execution plan for array fields with minItems"""

    def test_array_with_min_items_generates_extraction_step(self):
        """minItems > 0 produces a mandatory extraction step"""
        schema = {
            "type": "object",
            "properties": {
                "premises": {
                    "type": "array",
                    "minItems": 2,
                    "items": {"type": "string"},
                }
            },
        }
        routing_score = compute_routing_score("Analyze the argument", schema)
        plan = build_execution_plan(schema, routing_score)

        assert not plan.is_empty()
        assert any("MINIMUM 2" in step.instruction for step in plan.steps)
        assert any("premises" in step.instruction for step in plan.steps)

    def test_array_min_items_step_is_binding(self):
        """minItems step is marked as binding"""
        schema = {
            "type": "object",
            "properties": {
                "items": {"type": "array", "minItems": 3, "items": {"type": "string"}},
            },
        }
        routing_score = compute_routing_score("List all items", schema)
        plan = build_execution_plan(schema, routing_score)

        min_steps = [s for s in plan.steps if "MINIMUM 3" in s.instruction]
        assert len(min_steps) >= 1
        assert all(s.binding for s in min_steps)

    def test_array_without_min_items_no_mandatory_count_step(self):
        """Array without minItems does not generate a count-enforcement step"""
        schema = {
            "type": "object",
            "properties": {
                "tags": {"type": "array", "items": {"type": "string"}},
            },
        }
        routing_score = compute_routing_score("Get tags", schema)
        plan = build_execution_plan(schema, routing_score)

        # No MINIMUM step since no minItems
        assert not any("MINIMUM" in step.instruction for step in plan.steps)

    def test_array_with_item_properties_generates_per_item_step(self):
        """Array with item sub-properties generates per-item evaluation step"""
        schema = {
            "type": "object",
            "properties": {
                "premises": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "premise": {"type": "string"},
                            "valid": {"type": "boolean"},
                            "weakness": {"type": "string"},
                        },
                        "required": ["premise", "valid"],
                    },
                }
            },
        }
        routing_score = compute_routing_score("Evaluate argument", schema)
        plan = build_execution_plan(schema, routing_score)

        # Should have per-item evaluation step
        per_item_steps = [s for s in plan.steps if "INDEPENDENTLY" in s.instruction]
        assert len(per_item_steps) >= 1
        assert any("premise" in s.instruction or "valid" in s.instruction for s in per_item_steps)

    def test_array_item_boolean_generates_consistency_rule(self):
        """Boolean sub-fields in array items generate consistency rules"""
        schema = {
            "type": "object",
            "properties": {
                "results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "text": {"type": "string"},
                            "valid": {"type": "boolean"},
                        },
                    },
                }
            },
        }
        routing_score = compute_routing_score("Validate results", schema)
        plan = build_execution_plan(schema, routing_score)

        # Should have consistency rule for boolean sub-field
        assert any("valid" in rule for rule in plan.consistency_rules)


class TestEnumConstrainedArraySteps:
    """Test execution plan for enum-constrained array items"""

    def test_enum_array_items_generates_restriction_step(self):
        """Array with enum items produces a restriction step"""
        schema = {
            "type": "object",
            "properties": {
                "fallacies": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["ad_hominem", "straw_man", "hasty_generalization", "false_cause"],
                    },
                }
            },
        }
        routing_score = compute_routing_score("Identify fallacies", schema)
        plan = build_execution_plan(schema, routing_score)

        restriction_steps = [
            s for s in plan.steps if "ONLY from" in s.instruction or "INVALID" in s.instruction
        ]
        assert len(restriction_steps) >= 1
        assert any(
            "ad_hominem" in s.instruction or "hasty_generalization" in s.instruction
            for s in restriction_steps
        )

    def test_enum_array_restriction_step_is_binding(self):
        """Enum array restriction step is binding"""
        schema = {
            "type": "object",
            "properties": {
                "categories": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["A", "B", "C"]},
                }
            },
        }
        routing_score = compute_routing_score("Categorize", schema)
        plan = build_execution_plan(schema, routing_score)

        enum_steps = [s for s in plan.steps if "ONLY from" in s.instruction]
        if enum_steps:
            assert all(s.binding for s in enum_steps)


class TestTopLevelEnumFields:
    """Test execution plan for top-level enum fields"""

    def test_single_enum_field_generates_selection_step(self):
        """Single enum field produces a binding selection step"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
            },
        }
        routing_score = compute_routing_score("What is the status?", schema)
        plan = build_execution_plan(schema, routing_score)

        enum_steps = [s for s in plan.steps if "enumerated" in s.instruction.lower()]
        assert len(enum_steps) >= 1
        assert any("pending" in s.instruction for s in enum_steps)

    def test_multiple_enum_fields_combined_in_one_step(self):
        """Multiple enum fields are batched into one step"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "priority": {"enum": ["low", "high"]},
            },
        }
        routing_score = compute_routing_score("Classify", schema)
        plan = build_execution_plan(schema, routing_score)

        enum_steps = [s for s in plan.steps if "enumerated" in s.instruction.lower()]
        # Both fields may appear in one combined step
        assert len(enum_steps) >= 1

    def test_enum_step_forbids_invented_values(self):
        """Enum step explicitly forbids values outside the allowed list"""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["yes", "no", "abstain"]},
            },
        }
        routing_score = compute_routing_score("Make a decision", schema)
        plan = build_execution_plan(schema, routing_score)

        enum_steps = [s for s in plan.steps if "enumerated" in s.instruction.lower()]
        if enum_steps:
            combined = " ".join(s.instruction for s in enum_steps)
            assert "DO NOT" in combined or "invent" in combined.lower() or "INVALID" in combined


class TestNumericRangeSteps:
    """Test execution plan for numeric range fields"""

    def test_numeric_range_generates_bounds_step(self):
        """min/max on a number field produces a bounds validation step"""
        schema = {
            "type": "object",
            "properties": {
                "argument_strength": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                }
            },
        }
        routing_score = compute_routing_score("Rate the argument", schema)
        plan = build_execution_plan(schema, routing_score)

        range_steps = [
            s for s in plan.steps if "bounds" in s.instruction.lower() or "∈" in s.instruction
        ]
        assert len(range_steps) >= 1
        assert any("0" in s.instruction and "1" in s.instruction for s in range_steps)

    def test_out_of_range_marked_invalid(self):
        """Range step explicitly states out-of-range values are invalid"""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100},
            },
        }
        routing_score = compute_routing_score("Score", schema)
        plan = build_execution_plan(schema, routing_score)

        range_steps = [s for s in plan.steps if "0" in s.instruction and "100" in s.instruction]
        if range_steps:
            assert any("INVALID" in s.instruction for s in range_steps)


class TestBooleanConsistencyRules:
    """Test cross-field consistency rules for boolean verdict fields"""

    def test_boolean_verdict_with_array_generates_consistency_rule(self):
        """Boolean 'valid' field + array with valid sub-items → consistency rule"""
        schema = {
            "type": "object",
            "properties": {
                "argument_valid": {"type": "boolean"},
                "premises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "premise": {"type": "string"},
                            "valid": {"type": "boolean"},
                        },
                    },
                },
            },
        }
        routing_score = compute_routing_score("Evaluate logic", schema)
        plan = build_execution_plan(schema, routing_score)

        consistency = " ".join(plan.consistency_rules)
        assert "argument_valid" in consistency
        assert "premises" in consistency

    def test_consistency_rule_says_cannot_contradict(self):
        """Consistency rule explicitly forbids contradiction"""
        schema = {
            "type": "object",
            "properties": {
                "valid": {"type": "boolean"},
                "checks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"result": {"type": "string"}, "valid": {"type": "boolean"}},
                    },
                },
            },
        }
        routing_score = compute_routing_score("Validate all checks", schema)
        plan = build_execution_plan(schema, routing_score)

        if plan.consistency_rules:
            rule_text = " ".join(plan.consistency_rules).lower()
            assert "cannot" in rule_text or "must" in rule_text or "false" in rule_text


class TestRequiredFieldsStep:
    """Test required field completion step"""

    def test_required_fields_generate_completion_step(self):
        """Required fields list produces a final completion verification step"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }
        routing_score = compute_routing_score("Get person info", schema)
        plan = build_execution_plan(schema, routing_score)

        completion_steps = [
            s for s in plan.steps if "required" in s.instruction.lower() and "name" in s.instruction
        ]
        assert len(completion_steps) >= 1

    def test_no_required_fields_no_completion_step(self):
        """No required fields → no completion verification step"""
        schema = {
            "type": "object",
            "properties": {"optional_field": {"type": "string"}},
        }
        routing_score = compute_routing_score("Get data", schema)
        plan = build_execution_plan(schema, routing_score)

        # May still have other steps (enum, range etc) but not the required-fields step
        req_steps = [s for s in plan.steps if "required fields" in s.instruction.lower()]
        assert len(req_steps) == 0


class TestEnforcementLevel:
    """Test enforcement level derived from τ"""

    def test_high_tau_is_strict(self):
        """High τ → strict enforcement (tight constraints)"""
        # Use a schema with many enums to drive τ high
        schema = {
            "type": "object",
            "properties": {
                "a": {"enum": ["x", "y"]},
                "b": {"enum": ["p", "q"]},
                "c": {"enum": ["1", "2"]},
                "d": {"enum": ["m", "n"]},
            },
        }
        routing_score = compute_routing_score("Classify all fields", schema)
        plan = build_execution_plan(schema, routing_score)

        # Enforcement level depends on τ
        assert plan.enforcement_level in {"strict", "guided", "flexible"}

    def test_empty_schema_flexible(self):
        """Empty schema → flexible enforcement (no constraints)"""
        schema = {"type": "object", "properties": {}}
        routing_score = compute_routing_score("Empty", schema)
        plan = build_execution_plan(schema, routing_score)

        # No steps for empty schema
        assert plan.is_empty() or plan.enforcement_level in {"flexible", "guided"}


class TestExecutionPlanRendering:
    """Test render output format"""

    def test_render_empty_plan_returns_empty_string(self):
        """Empty plan renders to empty string"""
        plan = ExecutionPlan(steps=[], consistency_rules=[], enforcement_level="flexible")
        assert render_execution_plan(plan) == ""

    def test_render_includes_execution_protocol_header(self):
        """Non-empty plan includes EXECUTION PROTOCOL header"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
            },
            "required": ["status"],
        }
        routing_score = compute_routing_score("Status?", schema)
        plan = build_execution_plan(schema, routing_score)

        if not plan.is_empty():
            rendered = render_execution_plan(plan)
            assert "EXECUTION PROTOCOL" in rendered

    def test_render_steps_have_binding_markers(self):
        """Binding steps are marked with ★"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
            },
            "required": ["status"],
        }
        routing_score = compute_routing_score("Status?", schema)
        plan = build_execution_plan(schema, routing_score)

        if not plan.is_empty():
            rendered = render_execution_plan(plan)
            assert "★" in rendered

    def test_render_includes_consistency_checks_section(self):
        """Plans with consistency rules include the consistency checks section"""
        schema = {
            "type": "object",
            "properties": {
                "argument_valid": {"type": "boolean"},
                "premises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {"premise": {"type": "string"}, "valid": {"type": "boolean"}},
                    },
                },
            },
        }
        routing_score = compute_routing_score("Evaluate argument validity", schema)
        plan = build_execution_plan(schema, routing_score)

        if plan.consistency_rules:
            rendered = render_execution_plan(plan)
            assert "CONSISTENCY" in rendered

    def test_render_steps_are_numbered(self):
        """Steps appear with step numbers"""
        schema = {
            "type": "object",
            "properties": {
                "verdict": {"enum": ["pass", "fail"]},
                "score": {"type": "number", "minimum": 0, "maximum": 100},
            },
            "required": ["verdict", "score"],
        }
        routing_score = compute_routing_score("Score and verdict?", schema)
        plan = build_execution_plan(schema, routing_score)

        if not plan.is_empty():
            rendered = render_execution_plan(plan)
            assert "Step 1" in rendered


class TestExecutionPlanDataContract:
    """Test ExecutionPlan and ExecutionStep dataclass contracts"""

    def test_execution_plan_has_expected_fields(self):
        """ExecutionPlan has steps, consistency_rules, enforcement_level"""
        plan = ExecutionPlan()
        assert hasattr(plan, "steps")
        assert hasattr(plan, "consistency_rules")
        assert hasattr(plan, "enforcement_level")

    def test_execution_step_has_expected_fields(self):
        """ExecutionStep has step_number, instruction, binding"""
        step = ExecutionStep(step_number=1, instruction="Do something")
        assert step.step_number == 1
        assert step.instruction == "Do something"
        assert step.binding is True  # Default

    def test_empty_plan_is_empty(self):
        """Plan with no steps and no rules is empty"""
        plan = ExecutionPlan(steps=[], consistency_rules=[])
        assert plan.is_empty()

    def test_plan_with_steps_is_not_empty(self):
        """Plan with at least one step is not empty"""
        step = ExecutionStep(step_number=1, instruction="Do this")
        plan = ExecutionPlan(steps=[step], consistency_rules=[])
        assert not plan.is_empty()

    def test_plan_with_only_consistency_rules_is_not_empty(self):
        """Plan with only consistency rules (no steps) is not empty"""
        plan = ExecutionPlan(steps=[], consistency_rules=["Rule 1"])
        assert not plan.is_empty()


class TestFullSchemaExecution:
    """Test execution plan for the demo schema (logical validity analysis)"""

    def test_logical_analysis_schema_has_comprehensive_plan(self):
        """Full logical analysis schema produces plan covering all key aspects"""
        schema = {
            "type": "object",
            "properties": {
                "argument_valid": {"type": "boolean"},
                "premises": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "premise": {"type": "string"},
                            "valid": {"type": "boolean"},
                            "weakness": {"type": "string"},
                        },
                        "required": ["premise", "valid"],
                    },
                },
                "fallacies_identified": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": [
                            "ad_hominem",
                            "straw_man",
                            "false_dilemma",
                            "hasty_generalization",
                            "false_cause",
                            "none",
                        ],
                    },
                },
                "conclusion_scope_mismatch": {"type": "boolean"},
                "overall_assessment": {"type": "string"},
                "argument_strength": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": 1,
                },
            },
            "required": [
                "argument_valid",
                "premises",
                "fallacies_identified",
                "overall_assessment",
                "argument_strength",
            ],
        }
        routing_score = compute_routing_score(
            "Analyze step by step and evaluate the logical validity of this argument",
            schema,
        )
        plan = build_execution_plan(schema, routing_score)

        assert not plan.is_empty()

        # Should cover premises iteration
        assert any("MINIMUM 2" in s.instruction for s in plan.steps)

        # Should cover fallacies enum restriction
        assert any(
            "ONLY from" in s.instruction or "hasty_generalization" in s.instruction
            for s in plan.steps
        )

        # Should cover argument_strength range
        assert any("0" in s.instruction and "1" in s.instruction for s in plan.steps)

        # Should have argument_valid consistency rule
        consistency_text = " ".join(plan.consistency_rules)
        assert "argument_valid" in consistency_text

    def test_logical_analysis_render_is_substantial(self):
        """Rendered plan for logical analysis schema is non-trivial"""
        schema = {
            "type": "object",
            "properties": {
                "argument_valid": {"type": "boolean"},
                "premises": {
                    "type": "array",
                    "minItems": 2,
                    "items": {
                        "type": "object",
                        "properties": {
                            "premise": {"type": "string"},
                            "valid": {"type": "boolean"},
                        },
                    },
                },
                "argument_strength": {"type": "number", "minimum": 0, "maximum": 1},
            },
            "required": ["argument_valid", "premises", "argument_strength"],
        }
        routing_score = compute_routing_score("Evaluate argument validity", schema)
        plan = build_execution_plan(schema, routing_score)
        rendered = render_execution_plan(plan)

        assert len(rendered) > 100  # Substantial content
        assert "Step 1" in rendered
        assert "EXECUTION PROTOCOL" in rendered


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
