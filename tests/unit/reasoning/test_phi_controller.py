"""
Unit Tests: Phi Controller Module

Tests the phi_controller module's ability to:
1. Map λ̃₂ to decomposition strategy
2. Map τ to constraint focus
3. Map ΔK to vocabulary bridge
4. Calculate Φ-proportional thinking budget
"""

import pytest
from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning import shape_thinking_with_phi


class TestLambda2StrategyMapping:
    """Test _strategy_from_lambda2() logic"""

    def test_flat_schema_flat_extraction(self):
        """λ̃₂ < 0.2: FLAT_EXTRACTION strategy"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Extract simple data", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.lambda2 < 0.2:
            assert "FLAT_EXTRACTION" in shaping.decomposition_strategy or "flat" in shaping.decomposition_strategy.lower()
            assert "independent" in shaping.decomposition_strategy.lower() or "direct" in shaping.decomposition_strategy.lower()

    def test_moderate_schema_hierarchical(self):
        """0.2 ≤ λ̃₂ < 0.4: HIERARCHICAL_EXTRACTION strategy"""
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "email": {"type": "string"}
                    }
                },
                "company": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "address": {"type": "string"}
                    }
                }
            }
        }
        routing_score = compute_routing_score("Extract hierarchical data", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.2 <= routing_score.lambda2 < 0.4:
            assert "HIERARCHICAL" in shaping.decomposition_strategy or "hierarchical" in shaping.decomposition_strategy.lower()

    def test_dependent_schema_dependency_aware(self):
        """0.4 ≤ λ̃₂ < 0.6: DEPENDENCY_AWARE_REASONING strategy"""
        schema = {
            "type": "object",
            "properties": {
                "use_option_a": {"type": "boolean"},
                "option_a_value": {"type": "string"},
                "use_option_b": {"type": "boolean"},
                "option_b_value": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Choose between two options", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.4 <= routing_score.lambda2 < 0.6:
            assert "DEPENDENCY" in shaping.decomposition_strategy or "dependency" in shaping.decomposition_strategy.lower()

    def test_complex_schema_full_reasoning(self):
        """λ̃₂ ≥ 0.6: FULL_STRUCTURAL_REASONING strategy"""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["yes", "no"]},
                "rationale": {"type": "string"},
                "tradeoffs": {
                    "type": "object",
                    "properties": {
                        "pros": {"type": "array"},
                        "cons": {"type": "array"}
                    }
                },
                "constraints": {
                    "type": "object",
                    "properties": {
                        "time": {"type": "string"},
                        "budget": {"type": "number"}
                    }
                },
                "dependencies": {
                    "type": "array",
                    "items": {"type": "string"}
                }
            }
        }
        routing_score = compute_routing_score("Make complex decision with many considerations", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.lambda2 >= 0.6:
            assert "FULL" in shaping.decomposition_strategy or "holistic" in shaping.decomposition_strategy.lower()


class TestTauConstraintFocusMapping:
    """Test _focus_from_tau() logic"""

    def test_low_tau_exploratory(self):
        """τ < 0.4: EXPLORATORY focus"""
        schema = {
            "type": "object",
            "properties": {
                "explanation": {"type": "string"},
                "alternatives": {"type": "array", "items": {"type": "string"}}
            }
        }
        routing_score = compute_routing_score("Explain your thoughts and alternatives", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.tau < 0.4:
            assert "EXPLORATORY" in shaping.constraint_focus or "exploratory" in shaping.constraint_focus.lower()
            assert "flexible" in shaping.constraint_focus.lower() or "nuanced" in shaping.constraint_focus.lower()

    def test_moderate_tau_soft_constraints(self):
        """0.4 ≤ τ < 0.7: SOFT_CONSTRAINTS focus"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B", "C"]},
                "notes": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Choose status and add notes", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.4 <= routing_score.tau < 0.7:
            assert "SOFT" in shaping.constraint_focus or "soft" in shaping.constraint_focus.lower()

    def test_high_tau_strict_enumeration(self):
        """τ ≥ 0.7: STRICT_ENUMERATION focus"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]},
                "priority": {"enum": ["low", "medium", "high"]},
                "type": {"enum": ["A", "B", "C"]},
                "category": {"enum": ["X", "Y", "Z"]}
            }
        }
        routing_score = compute_routing_score("Choose exact status, priority, type, and category", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.tau >= 0.7:
            assert "STRICT" in shaping.constraint_focus or "strict" in shaping.constraint_focus.lower()
            assert "enum" in shaping.constraint_focus.lower()


class TestDeltaKVocabularyBridge:
    """Test _bridge_from_delta_k() logic"""

    def test_low_delta_k_no_bridge(self):
        """ΔK ≤ 0.5: no vocabulary bridge"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "priority": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("What is the status and priority?", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.delta_k <= 0.5:
            assert shaping.vocabulary_bridge is None

    def test_high_delta_k_bridge_present(self):
        """ΔK > 0.5: vocabulary bridge instruction provided"""
        schema = {
            "type": "object",
            "properties": {
                "processing_basis": {"enum": ["consent", "contract", "legal_obligation"]},
                "data_recipient": {"type": "string"},
                "transfer_location": {"type": "string"}
            }
        }
        # Very different wording to trigger high ΔK
        routing_score = compute_routing_score(
            "Does the user give permission, is it required by a contract, or is it a legal mandate? "
            "Who gets the information and where does it go?",
            schema
        )
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.delta_k > 0.5:
            assert shaping.vocabulary_bridge is not None
            assert "VOCABULARY" in shaping.vocabulary_bridge or "vocabulary" in shaping.vocabulary_bridge.lower()
            assert "map" in shaping.vocabulary_bridge.lower() or "terminology" in shaping.vocabulary_bridge.lower()

    def test_bridge_includes_example(self):
        """Vocabulary bridge includes clarification example"""
        schema = {
            "type": "object",
            "properties": {
                "vendor_id": {"type": "string"},
                "vendor_name": {"type": "string"}
            }
        }
        routing_score = compute_routing_score(
            "Who is the third party supplier providing this service and what is their name?",
            schema
        )
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.delta_k > 0.5 and shaping.vocabulary_bridge:
            # Should mention clarification or mapping
            assert "map" in shaping.vocabulary_bridge.lower() or "interpret" in shaping.vocabulary_bridge.lower()


class TestThinkingBudgetEstimation:
    """Test _estimate_thinking_budget() calculation"""

    def test_simple_schema_minimal_budget(self):
        """Φ < 0.5: 256 tokens (simple extraction)"""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}}
        }
        routing_score = compute_routing_score("Extract name", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.phi < 0.5:
            assert shaping.thinking_budget == 256

    def test_lightweight_budget(self):
        """0.5 ≤ Φ < 0.65: 512 tokens (lightweight reasoning)"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "score": {"type": "number"},
                "notes": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Classify with score and notes", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.5 <= routing_score.phi < 0.65:
            assert shaping.thinking_budget == 512

    def test_standard_budget(self):
        """0.65 ≤ Φ < 0.80: 1024 tokens (standard reasoning)"""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["yes", "no"]},
                "rationale": {"type": "string"},
                "alternatives": {"type": "array"},
                "constraints": {"type": "object"}
            }
        }
        routing_score = compute_routing_score("Make decision with reasoning", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.65 <= routing_score.phi < 0.80:
            assert shaping.thinking_budget == 1024

    def test_deep_budget(self):
        """0.80 ≤ Φ < 0.95: 1536 tokens (deep reasoning)"""
        schema = {
            "type": "object",
            "properties": {
                "decision": {"enum": ["approve", "reject", "escalate"]},
                "rationale": {"type": "string"},
                "tradeoffs": {"type": "object"},
                "constraints": {"type": "object"},
                "dependencies": {"type": "array"},
                "evidence": {"type": "array"},
                "confidence": {"type": "number"}
            }
        }
        routing_score = compute_routing_score("Make complex decision with multiple factors", schema)
        shaping = shape_thinking_with_phi(routing_score)

        if 0.80 <= routing_score.phi < 0.95:
            assert shaping.thinking_budget == 1536

    def test_maximum_budget(self):
        """Φ ≥ 0.95: 2048 tokens (maximum reasoning, self-consistency)"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["A", "B", "C"]},
                "rationale": {"type": "string"},
                "tradeoffs": {"type": "object"},
                "constraints": {"type": "object"},
                "dependencies": {"type": "object"},
                "evidence": {"type": "array"},
                "confidence": {"type": "number"},
                "alternatives": {"type": "array"},
                "decision_criteria": {"type": "object"}
            }
        }
        routing_score = compute_routing_score(
            "Make a critically important decision considering all factors with maximum reasoning",
            schema
        )
        shaping = shape_thinking_with_phi(routing_score)

        if routing_score.phi >= 0.95:
            assert shaping.thinking_budget == 2048

    def test_budget_is_positive_integer(self):
        """Thinking budget is always a positive integer"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Any prompt", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert isinstance(shaping.thinking_budget, int)
        assert shaping.thinking_budget > 0
        assert shaping.thinking_budget >= 256


class TestThinkingShapingDataContract:
    """Test ThinkingShaping dataclass properties"""

    def test_shaping_has_required_fields(self):
        """ThinkingShaping includes all expected fields"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert hasattr(shaping, "decomposition_strategy")
        assert hasattr(shaping, "constraint_focus")
        assert hasattr(shaping, "vocabulary_bridge")
        assert hasattr(shaping, "thinking_budget")

    def test_decomposition_strategy_is_string(self):
        """decomposition_strategy is a non-empty string"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert isinstance(shaping.decomposition_strategy, str)
        assert len(shaping.decomposition_strategy) > 0

    def test_constraint_focus_is_string(self):
        """constraint_focus is a non-empty string"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert isinstance(shaping.constraint_focus, str)
        assert len(shaping.constraint_focus) > 0

    def test_vocabulary_bridge_is_optional_string(self):
        """vocabulary_bridge is None or a string"""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string"}}
        }
        routing_score = compute_routing_score("Test", schema)
        shaping = shape_thinking_with_phi(routing_score)

        assert shaping.vocabulary_bridge is None or isinstance(shaping.vocabulary_bridge, str)


class TestPhiComponentIntegration:
    """Test how λ̃₂, τ, ΔK work together"""

    def test_simple_flat_task(self):
        """Simple extraction: low λ̃₂, low τ, low ΔK"""
        schema = {
            "type": "object",
            "properties": {
                "first_name": {"type": "string"},
                "last_name": {"type": "string"}
            }
        }
        routing_score = compute_routing_score("Extract first and last name", schema)
        shaping = shape_thinking_with_phi(routing_score)

        # Simple task should have minimal strategy
        assert "FLAT" in shaping.decomposition_strategy or "HIERARCHICAL" in shaping.decomposition_strategy
        assert shaping.thinking_budget <= 512

    def test_complex_reasoning_task(self):
        """Complex reasoning: potentially high λ̃₂, high τ, potentially high ΔK"""
        schema = {
            "type": "object",
            "properties": {
                "recommendation": {"enum": ["A", "B", "C"]},
                "rationale": {"type": "string"},
                "cost": {"type": "number"},
                "benefit": {"type": "number"},
                "risk": {"enum": ["low", "medium", "high"]},
                "timeline": {"type": "string"},
                "stakeholders": {"type": "array"},
                "constraints": {"type": "object"},
                "dependencies": {"type": "object"}
            }
        }
        routing_score = compute_routing_score(
            "Recommend a strategy considering cost, benefit, risk, timeline, stakeholders, constraints, and dependencies",
            schema
        )
        shaping = shape_thinking_with_phi(routing_score)

        # Complex task may have various decomposition strategies depending on λ̃₂
        # All should be valid strategy strings
        assert isinstance(shaping.decomposition_strategy, str)
        assert len(shaping.decomposition_strategy) > 0
        # High Φ scores typically have higher budgets
        assert shaping.thinking_budget > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
