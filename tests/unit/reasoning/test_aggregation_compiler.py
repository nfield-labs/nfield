"""
Unit tests for Stage 1A — Semantic Aggregation Compiler (SAC).

Tests cover:
- Pattern detection: boolean ALL, numeric SUM/MEAN, enum FAIL-if-any
- Rule generation correctness
- Post-generation verification (verify_aggregation_rules)
- Surgical reask builder (build_aggregation_reask)
- Edge cases: empty schemas, no aggregatable patterns, partial outputs
"""

from __future__ import annotations

import pytest

from formatshield.reasoning.aggregation_compiler import (
    AggregationCompiler,
    AggregationPattern,
    AggregationRule,
    AggregationVerificationResult,
    build_aggregation_reask,
    compile_aggregation_rules,
    verify_aggregation_rules,
)

# ---------------------------------------------------------------------------
# Fixtures — sample schemas
# ---------------------------------------------------------------------------

BOOLEAN_ALL_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "argument_valid": {"type": "boolean"},
        "premises": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "string"},
                    "valid": {"type": "boolean"},
                },
            },
        },
    },
}

NUMERIC_SUM_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "total_score": {"type": "number"},
        "criteria": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "score": {"type": "number"},
                },
            },
        },
    },
}

NUMERIC_MEAN_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "average_confidence": {"type": "number"},
        "predictions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "confidence": {"type": "number"},
                },
            },
        },
    },
}

ENUM_FAIL_ANY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "overall": {"type": "string", "enum": ["PASS", "FAIL"]},
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "result": {"enum": ["PASS", "FAIL"]},
                },
            },
        },
    },
}

FLAT_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
    },
}

EMPTY_SCHEMA: dict = {"type": "object", "properties": {}}


# ---------------------------------------------------------------------------
# Tests: Pattern detection
# ---------------------------------------------------------------------------


class TestBooleanAllDetection:
    def test_detects_boolean_all_pattern(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        assert len(rules) == 1
        assert rules[0].pattern == AggregationPattern.BOOLEAN_ALL

    def test_correct_parent_and_array_fields(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        rule = rules[0]
        assert rule.parent_field == "argument_valid"
        assert rule.array_field == "premises"
        assert rule.child_field == "valid"

    def test_instruction_says_derived_field(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        assert "DERIVED FIELD" in rules[0].execution_step_instruction
        assert "DO NOT" in rules[0].execution_step_instruction.upper()

    def test_instruction_contains_field_names(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        instr = rules[0].execution_step_instruction
        assert "argument_valid" in instr
        assert "premises" in instr

    def test_verification_description_non_empty(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        assert rules[0].verification_description != ""


class TestNumericSumDetection:
    def test_detects_numeric_sum_for_total(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_SUM_SCHEMA)
        sum_rules = [r for r in rules if r.pattern == AggregationPattern.NUMERIC_SUM]
        assert len(sum_rules) >= 1

    def test_sum_rule_has_correct_fields(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_SUM_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.NUMERIC_SUM)
        assert rule.parent_field == "total_score"
        assert rule.array_field == "criteria"
        assert rule.child_field == "score"

    def test_sum_instruction_references_sum_op(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_SUM_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.NUMERIC_SUM)
        assert "SUM" in rule.execution_step_instruction


class TestNumericMeanDetection:
    def test_detects_numeric_mean_for_average(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_MEAN_SCHEMA)
        mean_rules = [r for r in rules if r.pattern == AggregationPattern.NUMERIC_MEAN]
        assert len(mean_rules) >= 1

    def test_mean_rule_has_correct_fields(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_MEAN_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.NUMERIC_MEAN)
        assert rule.parent_field == "average_confidence"
        assert rule.array_field == "predictions"
        assert rule.child_field == "confidence"

    def test_mean_instruction_references_mean_op(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_MEAN_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.NUMERIC_MEAN)
        assert "MEAN" in rule.execution_step_instruction


class TestEnumFailAnyDetection:
    def test_detects_enum_fail_any(self) -> None:
        rules = compile_aggregation_rules(ENUM_FAIL_ANY_SCHEMA)
        fail_rules = [r for r in rules if r.pattern == AggregationPattern.ENUM_FAIL_ANY]
        assert len(fail_rules) >= 1

    def test_fail_values_populated(self) -> None:
        rules = compile_aggregation_rules(ENUM_FAIL_ANY_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.ENUM_FAIL_ANY)
        assert "FAIL" in rule.fail_values or "fail" in rule.fail_values

    def test_enum_instruction_mentions_fail_values(self) -> None:
        rules = compile_aggregation_rules(ENUM_FAIL_ANY_SCHEMA)
        rule = next(r for r in rules if r.pattern == AggregationPattern.ENUM_FAIL_ANY)
        assert "FAIL" in rule.execution_step_instruction.upper()


class TestNoPatternSchemas:
    def test_flat_schema_returns_no_rules(self) -> None:
        rules = compile_aggregation_rules(FLAT_SCHEMA)
        assert rules == []

    def test_empty_schema_returns_no_rules(self) -> None:
        rules = compile_aggregation_rules(EMPTY_SCHEMA)
        assert rules == []

    def test_schema_without_properties_key(self) -> None:
        rules = compile_aggregation_rules({})
        assert rules == []

    def test_array_without_matching_parent_no_rule(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "items": {"properties": {"score": {"type": "number"}}},
                },
                "name": {"type": "string"},
                # No numeric parent that matches "score"
            },
        }
        rules = compile_aggregation_rules(schema)
        assert rules == []


# ---------------------------------------------------------------------------
# Tests: AggregationRule dataclass
# ---------------------------------------------------------------------------


class TestAggregationRuleValidation:
    def test_empty_parent_field_raises(self) -> None:
        with pytest.raises(ValueError, match="parent_field"):
            AggregationRule(
                parent_field="",
                array_field="items",
                child_field="valid",
                pattern=AggregationPattern.BOOLEAN_ALL,
                execution_step_instruction="instruction",
                verification_description="desc",
            )

    def test_empty_array_field_raises(self) -> None:
        with pytest.raises(ValueError, match="array_field"):
            AggregationRule(
                parent_field="result",
                array_field="",
                child_field="valid",
                pattern=AggregationPattern.BOOLEAN_ALL,
                execution_step_instruction="instruction",
                verification_description="desc",
            )

    def test_empty_child_field_raises(self) -> None:
        with pytest.raises(ValueError, match="child_field"):
            AggregationRule(
                parent_field="result",
                array_field="items",
                child_field="",
                pattern=AggregationPattern.BOOLEAN_ALL,
                execution_step_instruction="instruction",
                verification_description="desc",
            )

    def test_empty_instruction_raises(self) -> None:
        with pytest.raises(ValueError, match="execution_step_instruction"):
            AggregationRule(
                parent_field="result",
                array_field="items",
                child_field="valid",
                pattern=AggregationPattern.BOOLEAN_ALL,
                execution_step_instruction="",
                verification_description="desc",
            )


# ---------------------------------------------------------------------------
# Tests: Post-generation verifier
# ---------------------------------------------------------------------------


class TestVerifyAggregationRules:
    def test_boolean_all_consistent_true(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            "argument_valid": True,
            "premises": [
                {"statement": "P1", "valid": True},
                {"statement": "P2", "valid": True},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True
        assert result.is_consistent() is True

    def test_boolean_all_inconsistency_detected(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            "argument_valid": True,  # WRONG: one premise is false
            "premises": [
                {"statement": "P1", "valid": True},
                {"statement": "P2", "valid": False},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is False
        assert len(result.failed_rules) == 1

    def test_boolean_all_false_parent_with_false_child_ok(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            "argument_valid": False,
            "premises": [
                {"statement": "P1", "valid": False},
                {"statement": "P2", "valid": True},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True

    def test_numeric_sum_correct(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_SUM_SCHEMA)
        output = {
            "total_score": 30.0,
            "criteria": [
                {"name": "A", "score": 10.0},
                {"name": "B", "score": 20.0},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True

    def test_numeric_sum_incorrect_detected(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_SUM_SCHEMA)
        output = {
            "total_score": 15.0,  # WRONG: should be 30.0
            "criteria": [
                {"name": "A", "score": 10.0},
                {"name": "B", "score": 20.0},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is False

    def test_numeric_mean_correct(self) -> None:
        rules = compile_aggregation_rules(NUMERIC_MEAN_SCHEMA)
        output = {
            "average_confidence": 0.75,
            "predictions": [
                {"label": "A", "confidence": 0.8},
                {"label": "B", "confidence": 0.7},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True

    def test_enum_fail_any_consistent(self) -> None:
        rules = compile_aggregation_rules(ENUM_FAIL_ANY_SCHEMA)
        output = {
            "overall": "FAIL",
            "checks": [
                {"name": "C1", "result": "PASS"},
                {"name": "C2", "result": "FAIL"},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True

    def test_enum_fail_any_incorrect_detected(self) -> None:
        rules = compile_aggregation_rules(ENUM_FAIL_ANY_SCHEMA)
        output = {
            "overall": "PASS",  # WRONG: one check failed
            "checks": [
                {"name": "C1", "result": "PASS"},
                {"name": "C2", "result": "FAIL"},
            ],
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is False

    def test_missing_parent_field_skipped(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            # argument_valid is absent
            "premises": [{"valid": True}],
        }
        result = verify_aggregation_rules(output, rules)
        # Graceful skip — cannot verify without parent
        assert result.passed is True

    def test_missing_array_field_skipped(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            "argument_valid": True,
            # premises is absent
        }
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True

    def test_empty_rules_always_passes(self) -> None:
        result = verify_aggregation_rules({"x": 1}, [])
        assert result.passed is True

    def test_empty_array_skipped_gracefully(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {"argument_valid": True, "premises": []}
        # Empty array: no child values → nothing to violate
        result = verify_aggregation_rules(output, rules)
        assert result.passed is True


# ---------------------------------------------------------------------------
# Tests: Surgical reask builder
# ---------------------------------------------------------------------------


class TestBuildAggregationReask:
    def test_empty_failed_rules_returns_empty(self) -> None:
        reask = build_aggregation_reask({}, [])
        assert reask == ""

    def test_reask_contains_field_name(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {
            "argument_valid": True,
            "premises": [{"valid": False}],
        }
        result = verify_aggregation_rules(output, rules)
        reask = build_aggregation_reask(output, result.failed_rules)
        assert "argument_valid" in reask

    def test_reask_says_update_only(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {"argument_valid": True, "premises": [{"valid": False}]}
        result = verify_aggregation_rules(output, rules)
        reask = build_aggregation_reask(output, result.failed_rules)
        assert "Update ONLY" in reask

    def test_reask_shows_current_value(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {"argument_valid": True, "premises": [{"valid": False}]}
        result = verify_aggregation_rules(output, rules)
        reask = build_aggregation_reask(output, result.failed_rules)
        # Should show current (incorrect) value for context
        assert "True" in reask or "true" in reask

    def test_reask_correction_required_header(self) -> None:
        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        output = {"argument_valid": True, "premises": [{"valid": False}]}
        result = verify_aggregation_rules(output, rules)
        reask = build_aggregation_reask(output, result.failed_rules)
        assert "CORRECTION REQUIRED" in reask


# ---------------------------------------------------------------------------
# Tests: AggregationVerificationResult
# ---------------------------------------------------------------------------


class TestAggregationVerificationResult:
    def test_is_consistent_true_when_passed(self) -> None:
        result = AggregationVerificationResult(passed=True, failed_rules=[])
        assert result.is_consistent() is True

    def test_is_consistent_false_when_failed(self) -> None:
        dummy_rule = AggregationRule(
            parent_field="x",
            array_field="items",
            child_field="v",
            pattern=AggregationPattern.BOOLEAN_ALL,
            execution_step_instruction="inst",
            verification_description="desc",
        )
        result = AggregationVerificationResult(passed=False, failed_rules=[(dummy_rule, "reason")])
        assert result.is_consistent() is False


# ---------------------------------------------------------------------------
# Tests: Multiple patterns in one schema
# ---------------------------------------------------------------------------


class TestMultiPatternSchema:
    def test_multiple_rules_detected(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "argument_valid": {"type": "boolean"},
                "total_score": {"type": "number"},
                "premises": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "valid": {"type": "boolean"},
                            "score": {"type": "number"},
                        },
                    },
                },
            },
        }
        rules = compile_aggregation_rules(schema)
        patterns = {r.pattern for r in rules}
        assert AggregationPattern.BOOLEAN_ALL in patterns
        assert AggregationPattern.NUMERIC_SUM in patterns

    def test_compile_aggregation_rules_public_api(self) -> None:
        """Public API function produces same result as class method."""
        rules_fn = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        rules_cls = AggregationCompiler().compile(BOOLEAN_ALL_SCHEMA)
        assert len(rules_fn) == len(rules_cls)
        for r1, r2 in zip(rules_fn, rules_cls, strict=True):
            assert r1.parent_field == r2.parent_field
            assert r1.pattern == r2.pattern
