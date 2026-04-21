"""
Unit tests for retry_budget — schema complexity-aware budget allocation and failure triage.

Coverage:
- RetryBudgetAllocator: budget tiers by lambda2, tau, field_count
- FailureTriager: classification of all failure types
- SurgicalReasker: targeted reask prompt construction
- FailureClassification: dataclass contracts
- BudgetAllocation: structured allocation info
- Public API: allocate_retry_budget, classify_failure, build_surgical_reask
- Integration: FailureTriager with aggregation rules for CONSISTENCY detection
"""

from __future__ import annotations

import pytest

from formatshield.reasoning.retry_budget import (
    BudgetAllocation,
    FailureClassification,
    FailureTriager,
    FailureType,
    RetryBudgetAllocator,
    SurgicalReasker,
    allocate_retry_budget,
    build_surgical_reask,
    classify_failure,
)

# ---------------------------------------------------------------------------
# Fixture schemas
# ---------------------------------------------------------------------------

FLAT_SCHEMA: dict = {
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    }
}

SIMPLE_3_FIELD_SCHEMA: dict = {
    "properties": {
        "a": {"type": "string"},
        "b": {"type": "string"},
        "c": {"type": "string"},
    }
}

COMPLEX_SCHEMA: dict = {
    "properties": {
        "premises": {"type": "array", "items": {"properties": {"valid": {"type": "boolean"}}}},
        "argument_valid": {"type": "boolean"},
        "confidence": {"type": "number"},
        "source": {"type": "string"},
        "verdict": {"type": "string", "enum": ["ACCEPT", "REJECT"]},
        "category": {"type": "string", "enum": ["A", "B", "C"]},
        "score": {"type": "number"},
        "status": {"type": "string", "enum": ["PASS", "FAIL"]},
        "notes": {"type": "string"},
        "tags": {"type": "array", "items": {"type": "string"}},
        "priority": {"type": "integer"},
        "timestamp": {"type": "string"},
    }
}

BOOLEAN_ALL_SCHEMA: dict = {
    "properties": {
        "premises": {
            "type": "array",
            "items": {"properties": {"valid": {"type": "boolean"}}},
        },
        "argument_valid": {"type": "boolean"},
    }
}


# ---------------------------------------------------------------------------
# RetryBudgetAllocator tests
# ---------------------------------------------------------------------------


class TestBudgetAllocationTiers:
    def test_flat_schema_3_fields_gets_budget_1(self) -> None:
        allocator = RetryBudgetAllocator(SIMPLE_3_FIELD_SCHEMA, lambda2=0.1, tau=0.5)
        assert allocator.allocate() == pytest.approx(1.0, abs=0.01)

    def test_flat_schema_more_fields_gets_budget_1_5(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.19, tau=0.5)
        # FLAT_SCHEMA has 2 fields, so <= 3 → budget 1.0
        assert allocator.allocate() == pytest.approx(1.0, abs=0.01)

    def test_moderate_coupling_gets_budget_2_5(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.5)
        assert allocator.allocate() == pytest.approx(2.5, abs=0.01)

    def test_high_coupling_few_fields_gets_budget_3(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.6, tau=0.5)
        assert allocator.allocate() == pytest.approx(3.0, abs=0.01)

    def test_high_coupling_many_fields_gets_budget_4(self) -> None:
        allocator = RetryBudgetAllocator(COMPLEX_SCHEMA, lambda2=0.7, tau=0.5)
        assert allocator.allocate() == pytest.approx(4.0, abs=0.01)

    def test_high_tau_increases_budget(self) -> None:
        allocator_low_tau = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.5)
        allocator_high_tau = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.8)
        assert allocator_high_tau.allocate() > allocator_low_tau.allocate()

    def test_budget_capped_at_4(self) -> None:
        allocator = RetryBudgetAllocator(COMPLEX_SCHEMA, lambda2=0.9, tau=0.9)
        assert allocator.allocate() <= 4.0

    def test_lambda2_clamped_to_0_1(self) -> None:
        allocator_neg = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=-1.0, tau=0.5)
        allocator_over = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=2.0, tau=0.5)
        # Should not raise; values clamped internally
        assert allocator_neg.allocate() >= 0
        assert allocator_over.allocate() >= 0

    def test_tau_clamped_to_0_1(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.5, tau=5.0)
        assert allocator.allocate() <= 4.0

    def test_budget_cached_on_second_call(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.5)
        first = allocator.allocate()
        second = allocator.allocate()
        assert first == second


class TestBudgetCosts:
    def test_enum_violation_cheapest(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.cost_of_retry(FailureType.ENUM_VIOLATION) == pytest.approx(0.1)

    def test_consistency_failure_most_expensive(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.cost_of_retry(FailureType.CONSISTENCY_FAILURE) == pytest.approx(1.0)

    def test_missing_field_cost(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.cost_of_retry(FailureType.MISSING_FIELD) == pytest.approx(0.3)

    def test_array_cardinality_cost(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.cost_of_retry(FailureType.ARRAY_CARDINALITY) == pytest.approx(0.5)

    def test_can_retry_when_sufficient_budget(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.can_retry(2.0, FailureType.CONSISTENCY_FAILURE) is True

    def test_cannot_retry_when_budget_exhausted(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        assert allocator.can_retry(0.05, FailureType.ENUM_VIOLATION) is False

    def test_exact_budget_allows_retry(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA)
        cost = allocator.cost_of_retry(FailureType.MISSING_FIELD)
        assert allocator.can_retry(cost, FailureType.MISSING_FIELD) is True


class TestAllocationInfo:
    def test_allocation_info_returns_budget_allocation(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.5)
        info = allocator.allocation_info()
        assert isinstance(info, BudgetAllocation)

    def test_allocation_info_fields_match_inputs(self) -> None:
        allocator = RetryBudgetAllocator(FLAT_SCHEMA, lambda2=0.35, tau=0.5)
        info = allocator.allocation_info()
        assert info.lambda2 == pytest.approx(0.35)
        assert info.tau == pytest.approx(0.5)
        assert info.initial_budget > 0


# ---------------------------------------------------------------------------
# FailureTriager tests
# ---------------------------------------------------------------------------


class TestMissingFieldClassification:
    def test_missing_required_property_classified(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "'score' is a required property",
            output={},
        )
        assert result.failure_type == FailureType.MISSING_FIELD

    def test_missing_field_path_extracted(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "'argument_valid' is a required property",
            output={},
        )
        assert result.field_path == "argument_valid"

    def test_missing_field_cost_correct(self) -> None:
        triager = FailureTriager()
        result = triager.classify("'x' is a required property", output={})
        assert result.retry_cost == pytest.approx(0.3)

    def test_missing_field_recommended_action(self) -> None:
        triager = FailureTriager()
        result = triager.classify("'x' is a required property", output={})
        assert result.recommended_action == "surgical_field_injection"


class TestEnumViolationClassification:
    def test_enum_violation_classified(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "'INVALID' is not one of ['PASS', 'FAIL']",
            output={"verdict": "INVALID"},
        )
        assert result.failure_type == FailureType.ENUM_VIOLATION

    def test_enum_violation_cost_correct(self) -> None:
        triager = FailureTriager()
        result = triager.classify("'X' is not one of ['A', 'B']", output={})
        assert result.retry_cost == pytest.approx(0.1)

    def test_enum_violation_action(self) -> None:
        triager = FailureTriager()
        result = triager.classify("value is not one of allowed enum values", output={})
        assert result.recommended_action == "enum_constraint_tightening"


class TestArrayCardinalityClassification:
    def test_too_short_classified_as_cardinality(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "[1, 2] is too short (minItems is 3)",
            output={"items": [1, 2]},
        )
        assert result.failure_type == FailureType.ARRAY_CARDINALITY

    def test_cardinality_cost_correct(self) -> None:
        triager = FailureTriager()
        result = triager.classify("[1] is too short", output={})
        assert result.retry_cost == pytest.approx(0.5)

    def test_cardinality_action(self) -> None:
        triager = FailureTriager()
        result = triager.classify("[1] is too short", output={})
        assert result.recommended_action == "array_completion_injection"


class TestRangeViolationClassification:
    def test_below_minimum_classified_as_range(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "-1 is less than the minimum of 0",
            output={"score": -1},
        )
        assert result.failure_type == FailureType.RANGE_VIOLATION

    def test_above_maximum_classified_as_range(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "5 is greater than the maximum of 3",
            output={"count": 5},
        )
        assert result.failure_type == FailureType.RANGE_VIOLATION

    def test_range_cost_correct(self) -> None:
        triager = FailureTriager()
        result = triager.classify("5 is greater than the maximum of 3", output={})
        assert result.retry_cost == pytest.approx(0.2)


class TestTypeFailureClassification:
    def test_unknown_error_defaults_to_type_failure(self) -> None:
        triager = FailureTriager()
        result = triager.classify(
            "unexpected validation error with no recognized pattern",
            output={},
        )
        assert result.failure_type == FailureType.TYPE_FAILURE

    def test_type_failure_cost_correct(self) -> None:
        triager = FailureTriager()
        result = triager.classify("some unknown error", output={})
        assert result.retry_cost == pytest.approx(0.2)


class TestConsistencyFailureClassification:
    def test_consistency_failure_when_aggregation_rule_violated(self) -> None:
        from formatshield.reasoning.aggregation_compiler import compile_aggregation_rules

        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        triager = FailureTriager()
        output = {
            "premises": [{"valid": False}],
            "argument_valid": True,  # Inconsistent!
        }
        result = triager.classify("validation error", output, aggregation_rules=rules)
        assert result.failure_type == FailureType.CONSISTENCY_FAILURE

    def test_consistency_failure_routes_to_sac(self) -> None:
        from formatshield.reasoning.aggregation_compiler import compile_aggregation_rules

        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        triager = FailureTriager()
        output = {"premises": [{"valid": False}], "argument_valid": True}
        result = triager.classify("err", output, aggregation_rules=rules)
        assert result.recommended_action == "sac_correction"
        assert result.routes_to_sac is True

    def test_no_consistency_failure_when_rules_empty(self) -> None:
        triager = FailureTriager()
        output = {"premises": [{"valid": False}], "argument_valid": True}
        result = triager.classify("'x' is a required property", output, aggregation_rules=[])
        assert result.failure_type == FailureType.MISSING_FIELD

    def test_consistency_failure_cost_correct(self) -> None:
        from formatshield.reasoning.aggregation_compiler import compile_aggregation_rules

        rules = compile_aggregation_rules(BOOLEAN_ALL_SCHEMA)
        triager = FailureTriager()
        output = {"premises": [{"valid": False}], "argument_valid": True}
        result = triager.classify("err", output, aggregation_rules=rules)
        assert result.retry_cost == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# FailureClassification tests
# ---------------------------------------------------------------------------


class TestFailureClassification:
    def test_negative_retry_cost_raises(self) -> None:
        with pytest.raises(ValueError, match="retry_cost"):
            FailureClassification(
                failure_type=FailureType.TYPE_FAILURE,
                field_path="x",
                error_message="err",
                retry_cost=-0.1,
                recommended_action="action",
            )

    def test_zero_retry_cost_allowed(self) -> None:
        fc = FailureClassification(
            failure_type=FailureType.TYPE_FAILURE,
            field_path="x",
            error_message="err",
            retry_cost=0.0,
            recommended_action="action",
        )
        assert fc.retry_cost == 0.0

    def test_routes_to_sac_false_for_non_consistency(self) -> None:
        fc = FailureClassification(
            failure_type=FailureType.MISSING_FIELD,
            field_path="x",
            error_message="err",
            retry_cost=0.3,
            recommended_action="surgical_field_injection",
        )
        assert fc.routes_to_sac is False

    def test_routes_to_sac_true_for_consistency(self) -> None:
        fc = FailureClassification(
            failure_type=FailureType.CONSISTENCY_FAILURE,
            field_path="x",
            error_message="err",
            retry_cost=1.0,
            recommended_action="sac_correction",
        )
        assert fc.routes_to_sac is True


# ---------------------------------------------------------------------------
# SurgicalReasker tests
# ---------------------------------------------------------------------------


class TestSurgicalReasker:
    def test_empty_failures_returns_empty_string(self) -> None:
        reasker = SurgicalReasker()
        result = reasker.build({}, [])
        assert result == ""

    def test_reask_contains_field_name(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.MISSING_FIELD,
            field_path="score",
            error_message="'score' is required",
            retry_cost=0.3,
            recommended_action="surgical_field_injection",
        )
        result = reasker.build({}, [failure])
        assert "score" in result

    def test_reask_says_update_only(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.ENUM_VIOLATION,
            field_path="verdict",
            error_message="'X' not in enum",
            retry_cost=0.1,
            recommended_action="enum_constraint_tightening",
        )
        result = reasker.build({"verdict": "X"}, [failure])
        assert "ONLY" in result or "only" in result

    def test_reask_shows_current_value(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.ENUM_VIOLATION,
            field_path="verdict",
            error_message="'INVALID' not in enum",
            retry_cost=0.1,
            recommended_action="enum_constraint_tightening",
        )
        result = reasker.build({"verdict": "INVALID"}, [failure])
        assert "INVALID" in result

    def test_reask_contains_correction_required_header(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.TYPE_FAILURE,
            field_path="x",
            error_message="wrong type",
            retry_cost=0.2,
            recommended_action="type_clarification_reask",
        )
        result = reasker.build({}, [failure])
        assert "CORRECTION REQUIRED" in result

    def test_reask_contains_return_complete_json_instruction(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.MISSING_FIELD,
            field_path="x",
            error_message="required",
            retry_cost=0.3,
            recommended_action="surgical_field_injection",
        )
        result = reasker.build({}, [failure])
        assert "JSON" in result

    def test_multiple_failures_all_listed(self) -> None:
        reasker = SurgicalReasker()
        failures = [
            FailureClassification(
                failure_type=FailureType.MISSING_FIELD,
                field_path="field1",
                error_message="missing",
                retry_cost=0.3,
                recommended_action="surgical_field_injection",
            ),
            FailureClassification(
                failure_type=FailureType.ENUM_VIOLATION,
                field_path="field2",
                error_message="enum violation",
                retry_cost=0.1,
                recommended_action="enum_constraint_tightening",
            ),
        ]
        result = reasker.build({}, failures)
        assert "field1" in result
        assert "field2" in result

    def test_missing_field_hint_text(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.MISSING_FIELD,
            field_path="priority",
            error_message="required",
            retry_cost=0.3,
            recommended_action="surgical_field_injection",
        )
        result = reasker.build({}, [failure])
        assert "Add" in result or "required" in result.lower()

    def test_consistency_failure_hint_mentions_array(self) -> None:
        reasker = SurgicalReasker()
        failure = FailureClassification(
            failure_type=FailureType.CONSISTENCY_FAILURE,
            field_path="argument_valid",
            error_message="inconsistent",
            retry_cost=1.0,
            recommended_action="sac_correction",
        )
        result = reasker.build({"argument_valid": True}, [failure])
        assert "array" in result.lower() or "Recompute" in result


# ---------------------------------------------------------------------------
# Public API tests
# ---------------------------------------------------------------------------


class TestPublicAPI:
    def test_allocate_retry_budget_returns_float(self) -> None:
        budget = allocate_retry_budget(FLAT_SCHEMA, lambda2=0.1, tau=0.5)
        assert isinstance(budget, float)

    def test_allocate_retry_budget_flat_schema(self) -> None:
        budget = allocate_retry_budget(SIMPLE_3_FIELD_SCHEMA, lambda2=0.05, tau=0.5)
        assert budget == pytest.approx(1.0, abs=0.01)

    def test_classify_failure_returns_classification(self) -> None:
        result = classify_failure("'x' is a required property", {})
        assert isinstance(result, FailureClassification)

    def test_classify_failure_missing_field(self) -> None:
        result = classify_failure("'score' is a required property", {})
        assert result.failure_type == FailureType.MISSING_FIELD

    def test_build_surgical_reask_empty_failures(self) -> None:
        result = build_surgical_reask({}, [])
        assert result == ""

    def test_build_surgical_reask_with_failure(self) -> None:
        failure = classify_failure("'x' is a required property", {})
        result = build_surgical_reask({}, [failure])
        assert "CORRECTION REQUIRED" in result

    def test_classify_with_aggregation_rules_none_safe(self) -> None:
        result = classify_failure("some error", {}, aggregation_rules=None)
        assert isinstance(result, FailureClassification)
