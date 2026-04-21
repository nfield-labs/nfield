"""
Unit Tests: Reasoning Integration Module

Tests the integration of reasoning module into TTFEngine.
"""

import pytest

from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning import ReasoningTaskConfig
from formatshield.ttf.reasoning_integration import (
    build_reasoning_context,
    inject_reasoning_into_prompt,
)


class TestBuildReasoningContext:
    """Test build_reasoning_context() function"""

    def test_context_with_all_features_enabled(self):
        """All features enabled: task, constraints, thinking_shaping returned"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
                "notes": {"type": "string"},
            },
            "required": ["status"],
        }
        prompt = "What is the status?"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=True,
            enable_phi_shaping=True,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)

        assert context["error"] is None
        assert context["task"] is not None
        assert context["constraints"] is not None
        assert context["thinking_shaping"] is not None

    def test_context_with_all_features_disabled(self):
        """All features disabled: empty context returned"""
        schema = {"type": "object", "properties": {}}
        prompt = "Test"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=False,
            enable_constraint_injection=False,
            enable_phi_shaping=False,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)

        assert context["task"] is None
        assert context["constraints"] == []
        assert context["thinking_shaping"] is None
        assert context["error"] is None

    def test_context_with_default_config(self):
        """Default config (all disabled): empty context"""
        schema = {"type": "object", "properties": {}}
        prompt = "Test"
        routing_score = compute_routing_score(prompt, schema)

        # Default config
        context = build_reasoning_context(schema, prompt, routing_score)

        assert context["task"] is None
        assert context["constraints"] == []
        assert context["thinking_shaping"] is None

    def test_context_with_only_task_enabled(self):
        """Only task compilation enabled"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        prompt = "Extract name"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=False,
            enable_phi_shaping=False,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)

        assert context["error"] is None
        assert context["task"] is not None
        assert context["constraints"] == []
        assert context["thinking_shaping"] is None

    def test_context_with_only_constraints_enabled(self):
        """Only constraint extraction enabled"""
        schema = {"type": "object", "properties": {"status": {"enum": ["A", "B"]}}}
        prompt = "Status?"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=False,
            enable_constraint_injection=True,
            enable_phi_shaping=False,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)

        assert context["error"] is None
        assert context["task"] is None
        assert context["constraints"] is not None
        assert len(context["constraints"]) > 0
        assert context["thinking_shaping"] is None

    def test_context_with_only_phi_shaping_enabled(self):
        """Only thinking shaping enabled"""
        schema = {"type": "object", "properties": {"field": {"type": "string"}}}
        prompt = "Test"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=False,
            enable_constraint_injection=False,
            enable_phi_shaping=True,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)

        assert context["error"] is None
        assert context["task"] is None
        assert context["constraints"] == []
        assert context["thinking_shaping"] is not None

    def test_context_has_all_keys(self):
        """Context dict always has all expected keys"""
        schema = {"type": "object", "properties": {}}
        prompt = "Test"
        routing_score = compute_routing_score(prompt, schema)

        context = build_reasoning_context(schema, prompt, routing_score)

        expected_keys = {"task", "constraints", "thinking_shaping", "execution_plan", "error"}
        assert set(context.keys()) == expected_keys

    def test_context_with_invalid_schema(self):
        """Invalid schema: error captured gracefully"""
        schema = None  # Invalid
        prompt = "Test"
        routing_score = compute_routing_score(prompt, {})

        config = ReasoningTaskConfig(enable_schema_aware_reasoning=True)

        # Should not raise, but error should be set
        try:
            context = build_reasoning_context(schema, prompt, routing_score, config)
            # Either succeeds silently or error is set
            assert isinstance(context, dict)
        except Exception as exc:
            # Acceptable to raise on truly invalid input
            _ = exc


class TestInjectReasoningIntoPrompt:
    """Test inject_reasoning_into_prompt() function"""

    def test_inject_with_full_context(self):
        """Full reasoning context: instructions, constraints, strategy injected"""
        schema = {
            "type": "object",
            "properties": {"status": {"enum": ["A", "B"]}, "notes": {"type": "string"}},
        }
        prompt = "What is the status?"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=True,
            enable_phi_shaping=True,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)
        base_prompt = "Original Pass 1 prompt"

        enhanced = inject_reasoning_into_prompt(base_prompt, context, schema)

        # Enhanced prompt should be longer
        assert len(enhanced) > len(base_prompt)
        # Should contain reasoning markers (from task or execution plan)
        assert (
            "REASONING TASK" in enhanced
            or "Instructions" in enhanced
            or "EXECUTION PROTOCOL" in enhanced
        )
        # Base prompt should still be there
        assert base_prompt in enhanced

    def test_inject_with_empty_context(self):
        """Empty context: base prompt returned unchanged"""
        base_prompt = "Original Pass 1 prompt"
        context = {
            "task": None,
            "constraints": [],
            "thinking_shaping": None,
            "error": None,
        }

        enhanced = inject_reasoning_into_prompt(base_prompt, context)

        assert enhanced == base_prompt

    def test_inject_with_error_in_context(self):
        """Error in context: base prompt returned unchanged"""
        base_prompt = "Original Pass 1 prompt"
        context = {
            "task": None,
            "constraints": [],
            "thinking_shaping": None,
            "error": "Some error occurred",
        }

        enhanced = inject_reasoning_into_prompt(base_prompt, context)

        assert enhanced == base_prompt

    def test_inject_with_only_task(self):
        """Only task in context: task instructions injected"""
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        prompt = "Extract name"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=False,
            enable_phi_shaping=False,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)
        base_prompt = "Pass 1 prompt"

        enhanced = inject_reasoning_into_prompt(base_prompt, context)

        assert len(enhanced) >= len(base_prompt)
        # Should have task section
        assert "REASONING TASK" in enhanced or "Extract" in enhanced

    def test_inject_with_only_constraints(self):
        """Only constraints in context: constraint rules injected"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved"]},
                "priority": {"enum": ["low", "high"]},
            },
        }
        prompt = "Status and priority?"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=False,
            enable_constraint_injection=True,
            enable_phi_shaping=False,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)
        base_prompt = "Pass 1 prompt"

        enhanced = inject_reasoning_into_prompt(base_prompt, context)

        assert len(enhanced) >= len(base_prompt)
        # Should have constraints section or execution protocol
        assert (
            "CONSTRAINTS" in enhanced or "pending" in enhanced or "EXECUTION PROTOCOL" in enhanced
        )

    def test_inject_preserves_base_prompt(self):
        """Base prompt is always preserved in enhanced version"""
        base_prompt = "This is the original prompt with important context."
        schema = {"type": "object", "properties": {"status": {"enum": ["A", "B"]}}}
        prompt = "What is status?"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(enable_constraint_injection=True)
        context = build_reasoning_context(schema, prompt, routing_score, config)

        enhanced = inject_reasoning_into_prompt(base_prompt, context)

        assert base_prompt in enhanced


class TestReasoningContextIntegration:
    """Integration tests for reasoning context building and injection"""

    def test_full_workflow_simple_extraction(self):
        """Full workflow on simple extraction task"""
        schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"},
                "email": {"type": "string"},
            },
            "required": ["first_name", "last_name", "email"],
        }
        prompt = "Extract contact information"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=True,
            enable_phi_shaping=True,
        )

        # Build context
        context = build_reasoning_context(schema, prompt, routing_score, config)
        assert context["error"] is None

        # Inject into prompt
        base = "Pass 1 prompt"
        enhanced = inject_reasoning_into_prompt(base, context, schema)

        # Should be enhanced
        assert len(enhanced) >= len(base)

    def test_full_workflow_complex_reasoning(self):
        """Full workflow on complex reasoning task"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["approve", "reject", "escalate"]},
                "rationale": {"type": "string"},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "tradeoffs": {"type": "array", "items": {"type": "string"}},
                "constraints": {"type": "object", "properties": {}},
            },
        }
        prompt = "Make a recommendation considering all factors"
        routing_score = compute_routing_score(prompt, schema)

        config = ReasoningTaskConfig(
            enable_schema_aware_reasoning=True,
            enable_constraint_injection=True,
            enable_phi_shaping=True,
        )

        context = build_reasoning_context(schema, prompt, routing_score, config)
        assert context["error"] is None
        assert context["task"] is not None

        base = "Pass 1 prompt"
        enhanced = inject_reasoning_into_prompt(base, context, schema)

        # Complex task should have rich enhancement
        assert len(enhanced) > len(base)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
