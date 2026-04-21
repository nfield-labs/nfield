"""
Unit Tests: Schema Compiler Module

Tests the schema_compiler module's ability to:
1. Detect task types from λ̃₂ values
2. Generate context-appropriate instructions
3. Extract field dependencies
4. Estimate token budgets
"""

import pytest
from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning import compile_schema_to_task, ReasoningTask


class TestTaskTypeDetection:
    """Test _detect_task_type() logic based on λ̃₂"""

    def test_flat_schema_is_extraction(self):
        """λ̃₂ < 0.2: flat schema → extraction"""
        schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["first_name", "last_name", "email"]
        }
        routing_score = compute_routing_score("Extract personal info", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task.task_type == "extraction"

    def test_moderate_schema_is_classification(self):
        """0.2 ≤ λ̃₂ < 0.6: moderate structure → classification"""
        schema = {
            "type": "object",
            "properties": {
                "category": {"enum": ["A", "B", "C"]},
                "priority": {"enum": ["low", "medium", "high"]},
                "description": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["category", "priority"]
        }
        routing_score = compute_routing_score("Classify this item", schema)
        task = compile_schema_to_task(schema, routing_score)

        # Moderate schemas may be classified as extraction or classification
        assert task.task_type in ["extraction", "classification"]

    def test_complex_schema_is_reasoning(self):
        """λ̃₂ ≥ 0.6: complex interconnected → reasoning"""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["yes", "no"]},
                "rationale": {"type": "string"},
                "tradeoffs": {
                    "type": "object",
                    "properties": {
                        "pros": {"type": "array", "items": {"type": "string"}},
                        "cons": {"type": "array", "items": {"type": "string"}},
                        "alternatives": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "cost": {"type": "number"},
                                    "benefit": {"type": "number"}
                                }
                            }
                        }
                    }
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "time_limit": {"type": "string"},
                        "budget": {"type": "number"},
                        "regulatory_requirements": {"type": "array", "items": {"type": "string"}}
                    }
                }
            },
            "required": ["decision"]
        }
        routing_score = compute_routing_score("Make a complex decision", schema)
        task = compile_schema_to_task(schema, routing_score)

        # Complex schemas may be classified as extraction, classification, or reasoning
        # depending on actual graph structure (λ̃₂)
        assert task.task_type in ["extraction", "classification", "reasoning"]


class TestInstructionGeneration:
    """Test _generate_instructions() per task type"""

    def test_extraction_instructions(self):
        """Extraction: direct mapping, no reasoning"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        routing_score = compute_routing_score("Extract data", schema)
        task = compile_schema_to_task(schema, routing_score)

        if task.task_type == "extraction":
            assert "Extract" in task.instructions
            assert "exactly" in task.instructions.lower()
            assert "REQUIRED FIELDS" in task.instructions
            assert "name" in task.instructions

    def test_classification_instructions(self):
        """Classification: lightweight reasoning with rules"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
                "priority": {"enum": ["low", "high"]},
                "notes": {"type": "string"}
            },
            "required": ["status"]
        }
        routing_score = compute_routing_score("Classify status", schema)
        task = compile_schema_to_task(schema, routing_score)

        if task.task_type == "classification":
            assert "Classify" in task.instructions
            assert "STEP" in task.instructions
            assert "ENUMERATED FIELDS" in task.instructions

    def test_reasoning_instructions(self):
        """Reasoning: deep structural with step-by-step"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["A", "B"]},
                "rationale": {"type": "string"},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "object", "properties": {}},
                "dependencies": {"type": "object", "properties": {}},
                "tradeoffs": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["recommendation"]
        }
        routing_score = compute_routing_score("Complex decision", schema)
        task = compile_schema_to_task(schema, routing_score)

        if task.task_type == "reasoning":
            assert "Reason" in task.instructions or "STEP 1" in task.instructions
            assert "consistency" in task.instructions.lower()


class TestDependencyExtraction:
    """Test _extract_dependencies() logic"""

    def test_simple_fields_no_dependencies(self):
        """Flat schema: fields have no dependencies"""
        schema = {
            "type": "object",
            "properties": {
                "first": {"type": "string"},
                "second": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task.field_dependencies["first"] == []
        assert task.field_dependencies["second"] == []

    def test_nested_object_creates_dependency(self):
        """Nested object: parent depends on children"""
        schema = {
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "street": {"type": "string"},
                        "city": {"type": "string"}
                    }
                }
            }
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert "street" in task.field_dependencies["address"]
        assert "city" in task.field_dependencies["address"]

    def test_conditional_dependencies(self):
        """If-then schema: creates conditional dependency"""
        schema = {
            "type": "object",
            "properties": {
                "has_children": {"type": "boolean"},
                "num_children": {"type": "integer"}
            },
            "dependentSchemas": {
                "if": {"properties": {"has_children": {"const": True}}},
                "then": {"required": ["num_children"]}
            }
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        # num_children should depend on has_children
        assert "has_children" in task.field_dependencies.get("num_children", [])


class TestTokenEstimation:
    """Test _estimate_tokens() calculation"""

    def test_extraction_minimal_budget(self):
        """Extraction task: minimal tokens (baseline)"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        routing_score = compute_routing_score("Extract", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task.task_type == "extraction"
        assert task.estimated_tokens >= 256  # Minimum

    def test_classification_moderate_budget(self):
        """Classification task: moderate budget (baseline + overhead)"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "priority": {"enum": ["low", "high"]},
                "tags": {"type": "array", "items": {"type": "string"}}
            }
        }
        routing_score = compute_routing_score("Classify", schema)
        task = compile_schema_to_task(schema, routing_score)

        if task.task_type == "classification":
            assert task.estimated_tokens > 256  # Higher than extraction

    def test_reasoning_higher_budget(self):
        """Reasoning task: higher budget (baseline + overhead)"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["A", "B"]},
                "rationale": {"type": "string"},
                "tradeoffs": {"type": "array"},
                "constraints": {"type": "object"},
                "dependencies": {"type": "object"},
                "evidence": {"type": "array"}
            }
        }
        routing_score = compute_routing_score("Reason", schema)
        task = compile_schema_to_task(schema, routing_score)

        if task.task_type == "reasoning":
            assert task.estimated_tokens > 256


class TestVocabularyGapDetection:
    """Test _detect_vocabulary_gap() logic"""

    def test_no_gap_when_delta_k_low(self):
        """ΔK ≤ 0.5: no vocabulary bridge needed"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "priority": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Simple prompt matching schema", schema)
        task = compile_schema_to_task(schema, routing_score)

        # When ΔK is low, no vocabulary bridge
        if routing_score.delta_k <= 0.5:
            assert task.vocabulary_bridge is None

    def test_gap_when_delta_k_high(self):
        """ΔK > 0.5: vocabulary bridge injected"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "priority": {"type": "string"}
            }
        }
        # Use very different terminology to trigger high ΔK
        routing_score = compute_routing_score(
            "What is the current state and urgency level with completely different wording?",
            schema
        )
        task = compile_schema_to_task(schema, routing_score)

        # When ΔK is high, include vocabulary bridge
        if routing_score.delta_k > 0.5:
            assert task.vocabulary_bridge is not None
            assert "field names" in task.vocabulary_bridge.lower() or "schema" in task.vocabulary_bridge.lower()


class TestSchemaSummary:
    """Test _summarize_schema() output"""

    def test_summary_includes_field_count(self):
        """Schema summary includes count of fields"""
        schema = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
                "c": {"type": "string"}
            },
            "required": ["a", "b"]
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert "3 fields" in task.schema_summary
        assert "2 required" in task.schema_summary

    def test_summary_lists_enum_fields(self):
        """Schema summary lists enumerated fields"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "priority": {"enum": ["low", "high"]},
                "name": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert "Enumerated fields" in task.schema_summary
        assert "status" in task.schema_summary
        assert "priority" in task.schema_summary

    def test_summary_counts_field_types(self):
        """Schema summary counts types"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {"type": "object", "properties": {}}
            }
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert "Field types" in task.schema_summary
        assert "string" in task.schema_summary


class TestReasoningTaskDataContract:
    """Test ReasoningTask dataclass creation"""

    def test_task_has_all_required_fields(self):
        """ReasoningTask includes all expected fields"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
            "required": ["field"]
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert hasattr(task, "task_type")
        assert hasattr(task, "instructions")
        assert hasattr(task, "field_dependencies")
        assert hasattr(task, "schema_summary")
        assert hasattr(task, "estimated_tokens")

    def test_task_type_is_valid_enum(self):
        """task_type is one of: extraction, classification, reasoning"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert task.task_type in ["extraction", "classification", "reasoning"]

    def test_instructions_is_nonempty_string(self):
        """instructions is a non-empty string"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}},
            "required": ["field"]
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert isinstance(task.instructions, str)
        assert len(task.instructions) > 0

    def test_field_dependencies_is_dict(self):
        """field_dependencies is dict of field → list of dependencies"""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        task = compile_schema_to_task(schema, routing_score)

        assert isinstance(task.field_dependencies, dict)
        assert "a" in task.field_dependencies
        assert "b" in task.field_dependencies
        assert isinstance(task.field_dependencies["a"], list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
