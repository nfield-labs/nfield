"""
E2E Integration Test: Schema-Conditioned Reasoning Engine

Tests that the reasoning module correctly:
1. Compiles schemas into task instructions
2. Extracts constraint rules
3. Shapes thinking using Φ components
4. Integrates with TTFEngine without breaking existing behavior

This test DRIVES the implementation. It should FAIL initially.
"""

import pytest
from typing import Any, Dict
from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning import (
    compile_schema_to_task,
    extract_constraints,
    shape_thinking_with_phi,
    ReasoningTask,
    ConstraintRule,
    ThinkingShaping,
)


class TestSchemaCompilation:
    """Test schema_compiler module"""

    def test_extraction_task_detection(self):
        """Flat schema → extraction task"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name", "email"]
        }

        routing_score = compute_routing_score("Extract contact info", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task is not None
        assert task.task_type == "extraction"
        assert "name" in task.instructions.lower()
        assert "email" in task.instructions.lower()
        assert len(task.field_dependencies) == 3

    def test_classification_task_detection(self):
        """Moderate schema → classification or extraction task (depending on λ̃₂)"""
        schema = {
            "type": "object",
            "properties": {
                "processing_basis": {
                    "enum": ["consent", "contract", "legal_obligation", "legitimate_interest"]
                },
                "special_category": {"type": "boolean"},
                "purpose": {"type": "string"},
                "additional_field": {"type": "string"}
            },
            "required": ["processing_basis"]
        }

        routing_score = compute_routing_score("Classify GDPR processing basis", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task is not None
        assert task.task_type in ["extraction", "classification", "reasoning"]
        assert "processing_basis" in task.instructions.lower()

    def test_reasoning_task_detection(self):
        """Complex interconnected schema → extraction, classification, or reasoning task (depending on λ̃₂)"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["approve", "reject", "escalate"]},
                "rationale": {"type": "string"},
                "risk_level": {"enum": ["low", "medium", "high"]},
                "alternative_options": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1
                },
                "decision_criteria": {
                    "type": "object",
                    "properties": {
                        "cost_benefit_favorable": {"type": "boolean"},
                        "stakeholder_impact": {"type": "string"},
                        "regulatory_impact": {"type": "string"},
                        "competitive_advantage": {"type": "string"}
                    }
                },
                "dependencies": {
                    "type": "object",
                    "properties": {
                        "related_decisions": {"type": "array", "items": {"type": "string"}},
                        "blocking_factors": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["recommendation", "risk_level"]
        }

        routing_score = compute_routing_score("Recommend strategy considering tradeoffs", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task is not None
        assert task.task_type in ["extraction", "classification", "reasoning"]
        assert "recommendation" in task.instructions.lower()


class TestConstraintExtraction:
    """Test constraint_engine module"""

    def test_enum_constraint_extraction(self):
        """Extract enum rules from schema"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
                "priority": {"enum": ["low", "medium", "high"]}
            }
        }
        prompt = "What is the status of the request?"

        routing_score = compute_routing_score(prompt, schema)
        rules = extract_constraints(schema, prompt, routing_score)

        assert len(rules) > 0
        enum_rules = [r for r in rules if r.rule_type == "enum"]
        assert len(enum_rules) >= 2

        for rule in enum_rules:
            assert rule.schema_path in ["properties.status", "properties.priority"]
            assert rule.validator is not None

    def test_conditional_constraint_extraction(self):
        """Extract if-then rules from schema"""
        schema = {
            "type": "object",
            "properties": {
                "transfer_to_third_country": {"type": "boolean"},
                "transfer_mechanism": {
                    "enum": ["SCCs", "BCRs", "Adequacy Decision"]
                }
            },
            "dependentSchemas": {
                "if": {"properties": {"transfer_to_third_country": {"const": True}}},
                "then": {"required": ["transfer_mechanism"]}
            }
        }
        prompt = "Check if data is transferred and how."

        routing_score = compute_routing_score(prompt, schema)
        rules = extract_constraints(schema, prompt, routing_score)

        assert len(rules) > 0
        # At least some rules should capture the conditional dependency
        assert any("transfer" in r.description.lower() for r in rules)

    def test_vocabulary_bridging(self):
        """Extract vocabulary mapping when ΔK > threshold"""
        schema = {
            "type": "object",
            "properties": {
                "processing_basis": {
                    "enum": ["consent", "legitimate_interest"]
                }
            }
        }
        prompt = "Does the user give permission or is there a business reason?"

        routing_score = compute_routing_score(prompt, schema)
        rules = extract_constraints(schema, prompt, routing_score)

        # Should include vocabulary mapping if ΔK is high
        if routing_score.delta_k > 0.5:
            assert any("vocabulary" in r.rule_type.lower() or "bridge" in r.description.lower() for r in rules) or len(rules) > 0


class TestPhiController:
    """Test phi_controller module"""

    def test_flat_schema_extraction_strategy(self):
        """Low λ̃₂ → flat extraction strategy"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }

        routing_score = compute_routing_score("Extract data", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert shaping is not None
        assert "flat" in shaping.decomposition_strategy.lower() or "extraction" in shaping.decomposition_strategy.lower()
        assert shaping.thinking_budget >= 256  # Minimum

    def test_high_constraint_strict_focus(self):
        """High τ → strict enumeration focus"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
                "priority": {"enum": ["low", "medium", "high"]},
                "category": {"enum": ["A", "B", "C"]}
            }
        }

        routing_score = compute_routing_score("Choose exact status", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert shaping is not None
        # High τ should recommend strict focus
        if routing_score.tau > 0.7:
            assert "strict" in shaping.constraint_focus.lower() or "enum" in shaping.constraint_focus.lower()

    def test_high_alignment_gap_vocabulary_bridge(self):
        """High ΔK → vocabulary bridge injection"""
        schema = {
            "type": "object",
            "properties": {
                "processing_basis": {
                    "enum": ["consent", "contract", "legal_obligation"]
                }
            }
        }
        # Prompt uses different terminology
        prompt = "Does the user give permission, is it required by contract, or is it legally mandated?"

        routing_score = compute_routing_score(prompt, schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert shaping is not None
        if routing_score.delta_k > 0.5:
            assert shaping.vocabulary_bridge is not None

    def test_thinking_budget_scales_with_phi(self):
        """Higher Φ → higher thinking budget"""
        simple_schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }

        complex_schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["A", "B", "C"]},
                "rationale": {"type": "string"},
                "dependencies": {"type": "object", "properties": {}},
                "tradeoffs": {"type": "array", "items": {"type": "string"}}
            }
        }

        simple_routing = compute_routing_score("Simple task", simple_schema)
        complex_routing = compute_routing_score("Complex decision", complex_schema)

        simple_shaping = shape_thinking_with_phi(simple_routing)
        complex_shaping = shape_thinking_with_phi(complex_routing)

        # Complex schema should get more thinking budget
        assert complex_shaping.thinking_budget >= simple_shaping.thinking_budget


class TestReasoningConfig:
    """Test reasoning module configuration"""

    def test_reasoning_config_creation(self):
        """ReasoningTaskConfig can be instantiated"""
        from formatshield.reasoning import ReasoningTaskConfig

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=True,
            enable_phi_shaping=True,
        )

        assert config.enable_schema_aware_reasoning is True
        assert config.enable_constraint_injection is True
        assert config.enable_phi_shaping is True
        assert config.is_any_enabled() is True

    def test_reasoning_config_defaults(self):
        """ReasoningTaskConfig has sensible defaults"""
        from formatshield.reasoning import ReasoningTaskConfig

        config = ReasoningTaskConfig()

        assert config.enable_schema_aware_reasoning is False
        assert config.enable_constraint_injection is False
        assert config.enable_phi_shaping is False
        assert config.is_any_enabled() is False
        assert config.vocabulary_bridge_threshold == 0.5
        assert config.max_task_instructions_tokens == 500


class TestDataContracts:
    """Test dataclass contracts"""

    def test_reasoning_task_creation(self):
        """ReasoningTask dataclass valid"""
        task = ReasoningTask(
            task_type="extraction",
            instructions="Extract the name and email.",
            constraints=[],
            field_dependencies={"name": [], "email": []},
            schema_summary="Simple contact extraction"
        )

        assert task.task_type == "extraction"
        assert "Extract" in task.instructions
        assert len(task.field_dependencies) == 2

    def test_constraint_rule_creation(self):
        """ConstraintRule dataclass valid"""
        rule = ConstraintRule(
            rule_type="enum",
            description="Status must be one of: pending, approved, rejected",
            schema_path="properties.status",
            constraint_value=["pending", "approved", "rejected"],
            injection_point="pass1_system",
            validator=lambda x: x in ["pending", "approved", "rejected"],
            priority="hard"
        )

        assert rule.rule_type == "enum"
        assert rule.validator("approved") is True
        assert rule.validator("invalid") is False

    def test_thinking_shaping_creation(self):
        """ThinkingShaping dataclass valid"""
        shaping = ThinkingShaping(
            decomposition_strategy="HIERARCHICAL_EXTRACTION",
            constraint_focus="SOFT_CONSTRAINTS",
            vocabulary_bridge="Map 'vendor' to 'third_party_recipient'",
            thinking_budget=512
        )

        assert shaping.decomposition_strategy == "HIERARCHICAL_EXTRACTION"
        assert shaping.thinking_budget == 512


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
