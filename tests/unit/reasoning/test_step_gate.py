"""
Unit tests for Stage 1B — Temporal Step Gating Engine (TSGE).

Tests cover:
- StepCheckResult and GateResult dataclasses
- Array step completeness checks (minItems enforcement)
- Evaluation step checks (per-item evaluation)
- Required fields step checks
- Enum step checks
- Generic binding step heuristic checks
- Forward-steering injection text generation
- gate_plan full-plan integration
- parse_partial_output helper
- Edge cases: empty plan, no output, full output
"""

from __future__ import annotations

from formatshield.reasoning.execution_plan import ExecutionPlan, ExecutionStep
from formatshield.reasoning.step_gate import (
    GateResult,
    TemporalStepGate,
    check_execution_steps,
    parse_partial_output,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_step(n: int, instruction: str, binding: bool = True) -> ExecutionStep:
    return ExecutionStep(step_number=n, instruction=instruction, binding=binding)


def _array_step(n: int, field: str, min_items: int) -> ExecutionStep:
    return _make_step(
        n,
        f"Extract ALL '{field}' — MINIMUM {min_items} required. "
        f"Do NOT proceed to the next step until you have identified "
        f"at least {min_items} distinct {field}.",
    )


def _eval_step(n: int, field: str) -> ExecutionStep:
    return _make_step(n, f"Evaluate EACH item in '{field}' INDEPENDENTLY.")


def _required_step(n: int, fields: list[str]) -> ExecutionStep:
    return _make_step(
        n,
        "Verify ALL required fields are populated before finalizing: " + ", ".join(fields) + ".",
    )


def _enum_step(n: int, assignments: dict[str, list[str]]) -> ExecutionStep:
    clauses = [
        f"'{name}' → exactly one of {{{', '.join(repr(v) for v in vals)}}}"
        for name, vals in assignments.items()
    ]
    return _make_step(
        n,
        "Assign enumerated fields — each MUST be exactly one allowed value: "
        + "; ".join(clauses)
        + ".",
    )


# ---------------------------------------------------------------------------
# Tests: Array step
# ---------------------------------------------------------------------------


class TestArrayStep:
    def test_array_step_complete_exact_count(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 2)
        result = gate.check_step(
            step,
            {"premises": [{"valid": True}, {"valid": False}]},
        )
        assert result.complete is True

    def test_array_step_complete_surplus_count(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 2)
        result = gate.check_step(
            step,
            {"premises": [{"v": 1}, {"v": 2}, {"v": 3}]},
        )
        assert result.complete is True

    def test_array_step_incomplete_one_item(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 3)
        result = gate.check_step(step, {"premises": [{"valid": True}]})
        assert result.complete is False
        assert result.injection_text != ""

    def test_array_step_incomplete_empty_array(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 2)
        result = gate.check_step(step, {"premises": []})
        assert result.complete is False

    def test_array_step_missing_field(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 2)
        result = gate.check_step(step, {})
        assert result.complete is False

    def test_array_step_injection_mentions_field(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 3)
        result = gate.check_step(step, {"premises": [{"v": 1}]})
        assert "premises" in result.injection_text

    def test_array_step_injection_mentions_count(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 3)
        result = gate.check_step(step, {"premises": [{"v": 1}]})
        assert "2" in result.injection_text  # Need 2 more

    def test_array_step_trace_heuristic_sufficient(self) -> None:
        gate = TemporalStepGate()
        step = _array_step(1, "premises", 2)
        trace = (
            "Let me identify the premises:\n1. First premise about X\n2. Second premise about Y\n"
        )
        result = gate.check_step(step, {}, trace_text=trace)
        assert result.complete is True


# ---------------------------------------------------------------------------
# Tests: Eval step
# ---------------------------------------------------------------------------


class TestEvalStep:
    def test_eval_step_complete_with_populated_items(self) -> None:
        gate = TemporalStepGate()
        step = _eval_step(2, "premises")
        result = gate.check_step(
            step,
            {"premises": [{"statement": "S1", "valid": True}, {"statement": "S2", "valid": False}]},
        )
        assert result.complete is True

    def test_eval_step_incomplete_empty_items(self) -> None:
        gate = TemporalStepGate()
        step = _eval_step(2, "premises")
        result = gate.check_step(step, {"premises": []})
        assert result.complete is False

    def test_eval_step_trace_mention_sufficient(self) -> None:
        gate = TemporalStepGate()
        step = _eval_step(2, "premises")
        trace = "Evaluating each premise in premises individually..."
        result = gate.check_step(step, {}, trace_text=trace)
        assert result.complete is True

    def test_eval_step_incomplete_missing_field(self) -> None:
        gate = TemporalStepGate()
        step = _eval_step(2, "premises")
        result = gate.check_step(step, {})
        assert result.complete is False


# ---------------------------------------------------------------------------
# Tests: Required fields step
# ---------------------------------------------------------------------------


class TestRequiredStep:
    def test_required_step_all_present(self) -> None:
        gate = TemporalStepGate()
        step = _required_step(3, ["name", "score", "verdict"])
        result = gate.check_step(step, {"name": "X", "score": 5.0, "verdict": "PASS"})
        assert result.complete is True

    def test_required_step_one_missing(self) -> None:
        gate = TemporalStepGate()
        step = _required_step(3, ["name", "score", "verdict"])
        result = gate.check_step(step, {"name": "X", "score": 5.0})
        # verdict is missing
        assert result.complete is False
        assert "verdict" in result.missing_description

    def test_required_step_all_missing(self) -> None:
        gate = TemporalStepGate()
        step = _required_step(3, ["name", "score"])
        result = gate.check_step(step, {})
        assert result.complete is False

    def test_required_step_injection_mentions_missing(self) -> None:
        gate = TemporalStepGate()
        step = _required_step(3, ["verdict"])
        result = gate.check_step(step, {})
        assert "verdict" in result.injection_text

    def test_required_step_no_colon_graceful(self) -> None:
        gate = TemporalStepGate()
        step = _make_step(3, "Verify ALL required fields are populated before finalizing")
        result = gate.check_step(step, {})
        # No fields extracted → assume complete
        assert result.complete is True


# ---------------------------------------------------------------------------
# Tests: Enum step
# ---------------------------------------------------------------------------


class TestEnumStep:
    def test_enum_step_valid_assignment(self) -> None:
        gate = TemporalStepGate()
        step = _enum_step(4, {"status": ["pending", "approved", "rejected"]})
        result = gate.check_step(step, {"status": "approved"})
        assert result.complete is True

    def test_enum_step_invalid_value(self) -> None:
        gate = TemporalStepGate()
        step = _enum_step(4, {"status": ["pending", "approved", "rejected"]})
        result = gate.check_step(step, {"status": "unknown"})
        assert result.complete is False
        assert "status" in result.missing_description

    def test_enum_step_missing_field_skipped(self) -> None:
        gate = TemporalStepGate()
        step = _enum_step(4, {"status": ["pending", "approved"]})
        # Field not yet in output → skip (cannot check what isn't generated)
        result = gate.check_step(step, {})
        assert result.complete is True

    def test_enum_step_injection_on_violation(self) -> None:
        gate = TemporalStepGate()
        step = _enum_step(4, {"status": ["pending", "approved"]})
        result = gate.check_step(step, {"status": "unknown"})
        assert result.injection_text != ""
        txt = result.injection_text.lower()
        assert "enum" in txt or "incomplete" in txt


# ---------------------------------------------------------------------------
# Tests: Generic step
# ---------------------------------------------------------------------------


class TestGenericStep:
    def test_generic_binding_step_no_trace_complete(self) -> None:
        gate = TemporalStepGate()
        step = _make_step(5, "Compute numeric fields within required bounds: score ∈ [0, 100].")
        # No trace → assume complete (avoid false positives)
        result = gate.check_step(step, {})
        assert result.complete is True

    def test_generic_binding_step_with_trace_matches(self) -> None:
        gate = TemporalStepGate()
        step = _make_step(5, "Compute score within bounds.")
        trace = "I will compute the score ensuring it is within bounds."
        result = gate.check_step(step, {}, trace_text=trace)
        assert result.complete is True

    def test_non_binding_step_always_complete(self) -> None:
        gate = TemporalStepGate()
        step = _make_step(6, "Optional: provide additional context.", binding=False)
        result = gate.check_step(step, {})
        assert result.complete is True


# ---------------------------------------------------------------------------
# Tests: TemporalStepGate.inject_continuation
# ---------------------------------------------------------------------------


class TestInjectContinuation:
    def test_inject_has_step_number(self) -> None:
        step = _array_step(3, "items", 5)
        txt = TemporalStepGate.inject_continuation(step, "need more items", "items", 2)
        assert "3" in txt

    def test_inject_has_missing_count(self) -> None:
        step = _array_step(3, "items", 5)
        txt = TemporalStepGate.inject_continuation(step, "need more items", "items", 2)
        assert "2" in txt

    def test_inject_zero_count_no_count_hint(self) -> None:
        step = _make_step(1, "Do something.")
        txt = TemporalStepGate.inject_continuation(step, "not done", "field", 0)
        # Should not mention a count when 0
        assert "Need 0 more" not in txt


# ---------------------------------------------------------------------------
# Tests: gate_plan (full plan integration)
# ---------------------------------------------------------------------------


class TestGatePlan:
    def test_all_complete_empty_plan(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(steps=[], consistency_rules=[])
        result = gate.gate_plan(plan)
        assert result.all_complete is True

    def test_all_complete_with_satisfied_steps(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(
            steps=[_array_step(1, "premises", 2)],
            consistency_rules=[],
        )
        output = {"premises": [{"v": 1}, {"v": 2}]}
        result = gate.gate_plan(plan, output)
        assert result.all_complete is True
        assert result.incomplete_count == 0

    def test_incomplete_step_captured(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(
            steps=[_array_step(1, "premises", 3)],
            consistency_rules=[],
        )
        output = {"premises": [{"v": 1}]}
        result = gate.gate_plan(plan, output)
        assert result.all_complete is False
        assert result.incomplete_count == 1

    def test_combined_injection_non_empty_on_incomplete(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(
            steps=[_array_step(1, "premises", 3)],
            consistency_rules=[],
        )
        output = {"premises": []}
        result = gate.gate_plan(plan, output)
        assert result.has_injections() is True
        assert result.combined_injection != ""

    def test_non_binding_steps_not_counted(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(
            steps=[_make_step(1, "Optional guidance.", binding=False)],
            consistency_rules=[],
        )
        result = gate.gate_plan(plan, {})
        assert result.all_complete is True
        assert result.incomplete_count == 0

    def test_multiple_steps_partial_completion(self) -> None:
        gate = TemporalStepGate()
        plan = ExecutionPlan(
            steps=[
                _array_step(1, "premises", 2),
                _required_step(2, ["verdict"]),
            ],
            consistency_rules=[],
        )
        # premises OK but verdict missing
        output = {"premises": [{"v": 1}, {"v": 2}]}
        result = gate.gate_plan(plan, output)
        assert result.all_complete is False
        assert result.incomplete_count == 1
        assert len(result.step_results) == 2


# ---------------------------------------------------------------------------
# Tests: GateResult
# ---------------------------------------------------------------------------


class TestGateResult:
    def test_has_injections_false_when_empty(self) -> None:
        result = GateResult(all_complete=True, combined_injection="")
        assert result.has_injections() is False

    def test_has_injections_true_when_set(self) -> None:
        result = GateResult(all_complete=False, combined_injection="Inject this")
        assert result.has_injections() is True


# ---------------------------------------------------------------------------
# Tests: check_execution_steps public API
# ---------------------------------------------------------------------------


class TestCheckExecutionSteps:
    def test_public_api_produces_gate_result(self) -> None:
        plan = ExecutionPlan(
            steps=[_array_step(1, "items", 2)],
            consistency_rules=[],
        )
        result = check_execution_steps(plan, {"items": [{"x": 1}, {"x": 2}]})
        assert isinstance(result, GateResult)
        assert result.all_complete is True

    def test_public_api_incomplete(self) -> None:
        plan = ExecutionPlan(
            steps=[_array_step(1, "items", 3)],
            consistency_rules=[],
        )
        result = check_execution_steps(plan, {"items": []})
        assert result.all_complete is False


# ---------------------------------------------------------------------------
# Tests: parse_partial_output
# ---------------------------------------------------------------------------


class TestParsePartialOutput:
    def test_parses_valid_json(self) -> None:
        raw = '{"name": "Alice", "score": 9}'
        result = parse_partial_output(raw)
        assert result == {"name": "Alice", "score": 9}

    def test_returns_empty_dict_on_invalid(self) -> None:
        result = parse_partial_output("not json at all")
        assert result == {}

    def test_extracts_json_object_from_text(self) -> None:
        raw = 'Here is my answer: {"verdict": "PASS"} and more text'
        result = parse_partial_output(raw)
        assert result.get("verdict") == "PASS"

    def test_returns_empty_on_empty_string(self) -> None:
        result = parse_partial_output("")
        assert result == {}

    def test_returns_empty_for_json_array(self) -> None:
        # We only support dict outputs
        result = parse_partial_output("[1, 2, 3]")
        assert result == {}
