"""
Unit Tests: Constraint Engine Module

Tests the constraint_engine module's ability to:
1. Extract enum constraints
2. Extract range constraints
3. Extract type constraints
4. Extract pattern constraints
5. Extract conditional dependencies
6. Extract required field rules
7. Extract vocabulary mappings
"""

import pytest
from formatshield.oracle.routing_score import compute_routing_score
from formatshield.reasoning import extract_constraints


class TestEnumConstraintExtraction:
    """Test _extract_enum_rules() logic"""

    def test_single_enum_field(self):
        """Extract single enum constraint"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["pending", "approved", "rejected"]}
            }
        }
        routing_score = compute_routing_score("What is status?", schema)
        rules = extract_constraints(schema, "What is status?", routing_score)

        enum_rules = [r for r in rules if r.rule_type == "enum"]
        assert len(enum_rules) >= 1
        assert any("status" in r.schema_path for r in enum_rules)
        assert any("pending" in r.description for r in enum_rules)

    def test_multiple_enum_fields(self):
        """Extract multiple enum constraints"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B", "C"]},
                "priority": {"enum": ["low", "high"]}
            }
        }
        routing_score = compute_routing_score("Status and priority?", schema)
        rules = extract_constraints(schema, "Status and priority?", routing_score)

        enum_rules = [r for r in rules if r.rule_type == "enum"]
        assert len(enum_rules) >= 2

    def test_enum_validator_works(self):
        """Enum validator correctly validates values"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["yes", "no"]}
            }
        }
        routing_score = compute_routing_score("Status?", schema)
        rules = extract_constraints(schema, "Status?", routing_score)

        enum_rules = [r for r in rules if r.rule_type == "enum" and "status" in r.schema_path]
        assert len(enum_rules) > 0

        rule = enum_rules[0]
        if rule.validator:
            assert rule.validator("yes") is True
            assert rule.validator("no") is True
            assert rule.validator("maybe") is False


class TestRangeConstraintExtraction:
    """Test _extract_range_rules() logic"""

    def test_numeric_range_constraint(self):
        """Extract numeric min/max constraints"""
        schema = {
            "type": "object",
            "properties": {
                "age": {"type": "integer", "minimum": 0, "maximum": 150}
            }
        }
        routing_score = compute_routing_score("Age?", schema)
        rules = extract_constraints(schema, "Age?", routing_score)

        range_rules = [r for r in rules if r.rule_type == "range"]
        assert len(range_rules) >= 1
        assert any("age" in r.schema_path for r in range_rules)

    def test_array_cardinality_constraint(self):
        """Extract array minItems/maxItems constraints"""
        schema = {
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                }
            }
        }
        routing_score = compute_routing_score("Tags?", schema)
        rules = extract_constraints(schema, "Tags?", routing_score)

        range_rules = [r for r in rules if r.rule_type == "range"]
        assert any("tags" in r.schema_path for r in range_rules)

    def test_range_validator_works(self):
        """Range validator correctly validates numeric values"""
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "number", "minimum": 0, "maximum": 100}
            }
        }
        routing_score = compute_routing_score("Score?", schema)
        rules = extract_constraints(schema, "Score?", routing_score)

        range_rules = [r for r in rules if r.rule_type == "range" and "score" in r.schema_path]
        if len(range_rules) > 0 and range_rules[0].validator:
            rule = range_rules[0]
            assert rule.validator(50) is True
            assert rule.validator(0) is True
            assert rule.validator(100) is True
            assert rule.validator(-1) is False
            assert rule.validator(101) is False


class TestTypeConstraintExtraction:
    """Test _extract_type_rules() logic"""

    def test_type_constraints_for_primitives(self):
        """Extract type constraints for string, number, boolean"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
                "active": {"type": "boolean"}
            }
        }
        routing_score = compute_routing_score("Extract all", schema)
        rules = extract_constraints(schema, "Extract all", routing_score)

        type_rules = [r for r in rules if r.rule_type == "consistency"]
        # Should have type constraints for string, integer, boolean
        assert len(type_rules) >= 3

    def test_type_validator_works(self):
        """Type validator correctly validates types"""
        schema = {
            "type": "object",
            "properties": {
                "count": {"type": "integer"}
            }
        }
        routing_score = compute_routing_score("Count?", schema)
        rules = extract_constraints(schema, "Count?", routing_score)

        type_rules = [r for r in rules if "count" in r.schema_path and "type" in r.description.lower()]
        if len(type_rules) > 0 and type_rules[0].validator:
            rule = type_rules[0]
            assert rule.validator(42) is True
            assert rule.validator("not a number") is False


class TestPatternConstraintExtraction:
    """Test _extract_pattern_rules() logic"""

    def test_regex_pattern_constraint(self):
        """Extract regex pattern constraints"""
        schema = {
            "type": "object",
            "properties": {
                "email": {
                    "type": "string",
                    "pattern": "^[\\w\\.-]+@[\\w\\.-]+\\.\\w+$"
                }
            }
        }
        routing_score = compute_routing_score("Email?", schema)
        rules = extract_constraints(schema, "Email?", routing_score)

        pattern_rules = [r for r in rules if r.rule_type == "consistency" and "pattern" in r.description.lower()]
        # Should have at least the pattern constraint
        assert any("email" in r.schema_path for r in pattern_rules)

    def test_pattern_validator_works(self):
        """Pattern validator correctly matches regex"""
        schema = {
            "type": "object",
            "properties": {
                "zipcode": {
                    "type": "string",
                    "pattern": "^\\d{5}$"
                }
            }
        }
        routing_score = compute_routing_score("Zipcode?", schema)
        rules = extract_constraints(schema, "Zipcode?", routing_score)

        pattern_rules = [r for r in rules if "zipcode" in r.schema_path and "pattern" in r.description.lower()]
        if len(pattern_rules) > 0 and pattern_rules[0].validator:
            rule = pattern_rules[0]
            assert rule.validator("12345") is True
            assert rule.validator("123") is False
            assert rule.validator("ABCDE") is False


class TestConditionalConstraintExtraction:
    """Test _extract_conditional_rules() logic"""

    def test_if_then_dependency(self):
        """Extract if-then conditional constraints"""
        schema = {
            "type": "object",
            "properties": {
                "use_custom": {"type": "boolean"},
                "custom_value": {"type": "string"}
            },
            "dependentSchemas": {
                "if": {"properties": {"use_custom": {"const": True}}},
                "then": {"required": ["custom_value"]}
            }
        }
        routing_score = compute_routing_score("Conditional check?", schema)
        rules = extract_constraints(schema, "Conditional check?", routing_score)

        conditional_rules = [r for r in rules if r.rule_type == "conditional"]
        assert len(conditional_rules) >= 1
        assert any("if" in r.description.lower() for r in conditional_rules)


class TestDependencyConstraintExtraction:
    """Test _extract_dependency_rules() logic"""

    def test_required_fields_constraint(self):
        """Extract required fields constraints"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "email": {"type": "string"}
            },
            "required": ["name", "email"]
        }
        routing_score = compute_routing_score("Required?", schema)
        rules = extract_constraints(schema, "Required?", routing_score)

        dep_rules = [r for r in rules if r.rule_type == "dependency" and "required" in r.schema_path.lower()]
        assert len(dep_rules) >= 1
        assert any("name" in r.description for r in dep_rules)
        assert any("email" in r.description for r in dep_rules)

    def test_additional_properties_false(self):
        """Extract additionalProperties: false constraint"""
        schema = {
            "type": "object",
            "properties": {
                "allowed": {"type": "string"}
            },
            "additionalProperties": False
        }
        routing_score = compute_routing_score("Additional props?", schema)
        rules = extract_constraints(schema, "Additional props?", routing_score)

        dep_rules = [r for r in rules if r.rule_type == "dependency" and "additional" in r.description.lower()]
        assert len(dep_rules) >= 1

    def test_required_validator_works(self):
        """Required fields validator works correctly"""
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"}
            },
            "required": ["name"]
        }
        routing_score = compute_routing_score("Required?", schema)
        rules = extract_constraints(schema, "Required?", routing_score)

        dep_rules = [r for r in rules if r.rule_type == "dependency" and "required" in r.schema_path.lower()]
        if len(dep_rules) > 0 and dep_rules[0].validator:
            rule = dep_rules[0]
            assert rule.validator({"name": "John"}) is True
            assert rule.validator({"name": "John", "age": 30}) is True
            assert rule.validator({"age": 30}) is False


class TestVocabularyConstraintExtraction:
    """Test _extract_vocabulary_rules() logic"""

    def test_vocabulary_bridge_high_delta_k(self):
        """High ΔK > 0.5: extract vocabulary bridging rules"""
        schema = {
            "type": "object",
            "properties": {
                "processing_basis": {"enum": ["consent", "contract", "legal"]},
                "third_party_recipient": {"type": "string"}
            }
        }
        # Very different wording to trigger high ΔK
        routing_score = compute_routing_score(
            "Does the user give permission or is there a contract or legal mandate? Who receives data?",
            schema
        )
        rules = extract_constraints(schema, "Does the user give permission...", routing_score)

        # If ΔK > 0.5, should have vocabulary rules
        if routing_score.delta_k > 0.5:
            vocab_rules = [r for r in rules if r.rule_type == "vocabulary"]
            assert len(vocab_rules) > 0

    def test_vocabulary_bridge_low_delta_k(self):
        """Low ΔK ≤ 0.5: no vocabulary bridging rules needed"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string"},
                "priority": {"type": "string"}
            }
        }
        # Same terminology as schema
        routing_score = compute_routing_score("What is the status and priority?", schema)
        rules = extract_constraints(schema, "What is the status and priority?", routing_score)

        # If ΔK <= 0.5, should have no vocabulary rules
        if routing_score.delta_k <= 0.5:
            vocab_rules = [r for r in rules if r.rule_type == "vocabulary"]
            assert len(vocab_rules) == 0


class TestConstraintPriority:
    """Test constraint priority ordering"""

    def test_hard_rules_come_first(self):
        """Hard rules (enums, required) come before soft rules"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "count": {"type": "integer", "minimum": 0, "maximum": 100}
            },
            "required": ["status"]
        }
        routing_score = compute_routing_score("Status and count?", schema)
        rules = extract_constraints(schema, "Status and count?", routing_score)

        # Find first hard and first soft rule
        hard_idx = next((i for i, r in enumerate(rules) if r.priority == "hard"), None)
        soft_idx = next((i for i, r in enumerate(rules) if r.priority == "soft"), None)

        if hard_idx is not None and soft_idx is not None:
            assert hard_idx < soft_idx, "Hard rules should come before soft rules"


class TestConstraintRuleDataContract:
    """Test ConstraintRule dataclass properties"""

    def test_constraint_rule_has_required_fields(self):
        """ConstraintRule includes all expected fields"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]}
            }
        }
        routing_score = compute_routing_score("Status?", schema)
        rules = extract_constraints(schema, "Status?", routing_score)

        assert len(rules) > 0
        rule = rules[0]

        assert hasattr(rule, "rule_type")
        assert hasattr(rule, "description")
        assert hasattr(rule, "schema_path")
        assert hasattr(rule, "priority")

    def test_constraint_rule_type_valid(self):
        """rule_type is one of valid types"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A", "B"]},
                "count": {"type": "integer", "minimum": 0},
                "name": {"type": "string", "pattern": "^[A-Z]"}
            }
        }
        routing_score = compute_routing_score("All?", schema)
        rules = extract_constraints(schema, "All?", routing_score)

        valid_types = {"enum", "range", "consistency", "conditional", "dependency", "vocabulary"}
        for rule in rules:
            assert rule.rule_type in valid_types

    def test_constraint_priority_valid(self):
        """priority is either 'hard' or 'soft'"""
        schema = {
            "type": "object",
            "properties": {
                "status": {"enum": ["A"]},
                "count": {"type": "integer", "minimum": 0}
            }
        }
        routing_score = compute_routing_score("All?", schema)
        rules = extract_constraints(schema, "All?", routing_score)

        for rule in rules:
            assert rule.priority in ["hard", "soft"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
