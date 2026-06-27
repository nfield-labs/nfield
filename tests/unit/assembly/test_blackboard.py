"""Unit tests for assembly._blackboard - field state machine."""

from __future__ import annotations

import pytest

from nfield.assembly._blackboard import Blackboard, FieldState
from nfield.exceptions import AssemblyError

# ---------------------------------------------------------------------------
# Blackboard construction
# ---------------------------------------------------------------------------


class TestBlackboardConstruction:
    def test_all_paths_start_empty(self):
        bb = Blackboard(["a", "b", "c"])
        assert bb.get_missing() == ["a", "b", "c"]

    def test_empty_paths_list(self):
        bb = Blackboard([])
        assert bb.get_missing() == []
        assert bb.get_filled() == {}

    def test_duplicate_paths_raise(self):
        with pytest.raises(ValueError, match="unique"):
            Blackboard(["x", "x"])

    def test_all_paths_returns_sorted(self):
        bb = Blackboard(["c", "a", "b"])
        assert bb.all_paths() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# write - state transitions
# ---------------------------------------------------------------------------


class TestBlackboardWrite:
    def test_write_transitions_empty_to_filled(self):
        bb = Blackboard(["x"])
        bb.write("x", 42)
        assert bb.get_state("x") == FieldState.FILLED

    def test_written_value_retrievable(self):
        bb = Blackboard(["name"])
        bb.write("name", "Alice")
        assert bb.get_filled() == {"name": "Alice"}

    def test_same_value_twice_no_conflict(self):
        bb = Blackboard(["x"])
        bb.write("x", 1)
        bb.write("x", 1)
        assert bb.get_state("x") == FieldState.FILLED

    def test_different_value_causes_conflict(self):
        bb = Blackboard(["x"])
        bb.write("x", 1)
        bb.write("x", 2)
        assert bb.get_state("x") == FieldState.CONFLICT

    def test_conflict_values_stored(self):
        bb = Blackboard(["x"])
        bb.write("x", "first")
        bb.write("x", "second")
        vals = bb.get_conflict_values("x")
        assert "first" in vals
        assert "second" in vals

    def test_failed_field_recovers_on_write(self):
        bb = Blackboard(["y"])
        bb.mark_failed("y", "parse error")
        assert bb.get_state("y") == FieldState.FAILED
        bb.write("y", "recovered")
        assert bb.get_state("y") == FieldState.FILLED

    def test_needs_revalidation_blocks_write(self):
        bb = Blackboard(["z"])
        bb.mark_needs_revalidation("z")
        bb.write("z", "new_value")  # Should be silently ignored
        assert bb.get_state("z") == FieldState.NEEDS_REVALIDATION

    def test_unknown_path_raises(self):
        bb = Blackboard(["known"])
        with pytest.raises(AssemblyError):
            bb.write("unknown", "value")


# ---------------------------------------------------------------------------
# write_raw - dep-change safe write
# ---------------------------------------------------------------------------


class TestWriteRaw:
    def test_empty_to_filled(self):
        bb = Blackboard(["a"])
        bb.write_raw("a", 10)
        assert bb.get_state("a") == FieldState.FILLED

    def test_filled_field_transitions_to_needs_revalidation(self):
        bb = Blackboard(["dep"])
        bb.write("dep", "original")
        assert bb.get_state("dep") == FieldState.FILLED
        bb.write_raw("dep", "updated")
        assert bb.get_state("dep") == FieldState.NEEDS_REVALIDATION

    def test_filled_value_not_overwritten(self):
        bb = Blackboard(["dep"])
        bb.write("dep", "original")
        bb.write_raw("dep", "new")
        # write_raw on FILLED transitions to NEEDS_REVALIDATION (dep changed)
        # The field is removed from get_filled() - it needs revalidation
        assert bb.get_state("dep") == FieldState.NEEDS_REVALIDATION
        assert "dep" not in bb.get_filled()

    def test_failed_state_blocks_write_raw(self):
        bb = Blackboard(["f"])
        bb.mark_failed("f", "error")
        bb.write_raw("f", "recovery_attempt")
        assert bb.get_state("f") == FieldState.FAILED


# ---------------------------------------------------------------------------
# mark_failed and mark_needs_revalidation
# ---------------------------------------------------------------------------


class TestStateTransitions:
    def test_mark_failed(self):
        bb = Blackboard(["f"])
        bb.mark_failed("f", "constraint violation")
        assert bb.get_state("f") == FieldState.FAILED

    def test_error_message_stored(self):
        bb = Blackboard(["f"])
        bb.mark_failed("f", "parse error")
        assert bb.get_error("f") == "parse error"

    def test_no_error_for_non_failed_field(self):
        bb = Blackboard(["ok"])
        bb.write("ok", "value")
        assert bb.get_error("ok") is None

    def test_mark_needs_revalidation(self):
        bb = Blackboard(["n"])
        bb.write("n", "value")
        bb.mark_needs_revalidation("n")
        assert bb.get_state("n") == FieldState.NEEDS_REVALIDATION

    def test_mark_pending_from_empty(self):
        bb = Blackboard(["p"])
        bb.mark_pending("p")
        assert bb.get_state("p") == FieldState.PENDING

    def test_mark_pending_only_from_empty(self):
        bb = Blackboard(["p"])
        bb.write("p", "val")
        bb.mark_pending("p")  # Already FILLED - should not change
        assert bb.get_state("p") == FieldState.FILLED


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------


class TestReadOperations:
    def test_get_missing_after_partial_fill(self):
        bb = Blackboard(["a", "b", "c"])
        bb.write("a", 1)
        missing = bb.get_missing()
        assert "a" not in missing
        assert "b" in missing
        assert "c" in missing

    def test_get_conflicts(self):
        bb = Blackboard(["x", "y"])
        bb.write("x", 1)
        bb.write("x", 2)
        assert bb.get_conflicts() == ["x"]

    def test_get_needs_revalidation(self):
        bb = Blackboard(["r"])
        bb.mark_needs_revalidation("r")
        assert bb.get_needs_revalidation() == ["r"]

    def test_get_failed(self):
        bb = Blackboard(["f"])
        bb.mark_failed("f", "err")
        assert bb.get_failed() == ["f"]

    def test_get_filled_excludes_conflicted(self):
        bb = Blackboard(["a", "b"])
        bb.write("a", 1)
        bb.write("b", "first")
        bb.write("b", "second")
        filled = bb.get_filled()
        assert "a" in filled
        assert "b" not in filled  # CONFLICT - not in filled

    def test_summary_counts_all_states(self):
        bb = Blackboard(["a", "b", "c", "d", "e", "f"])
        bb.write("a", 1)
        bb.write("b", 1)
        bb.write("b", 2)  # conflict
        bb.mark_failed("c", "err")
        bb.mark_needs_revalidation("d")
        bb.mark_pending("e")
        summary = bb.summary()
        assert summary["filled"] == 1
        assert summary["conflict"] == 1
        assert summary["failed"] == 1
        assert summary["needs_revalidation"] == 1
        assert summary["pending"] == 1
        assert summary["empty"] == 1  # f


# ---------------------------------------------------------------------------
# get_state
# ---------------------------------------------------------------------------


class TestGetState:
    def test_unknown_path_raises(self):
        bb = Blackboard(["known"])
        with pytest.raises(AssemblyError):
            bb.get_state("unknown")

    def test_known_path_returns_state(self):
        bb = Blackboard(["x"])
        assert bb.get_state("x") == FieldState.EMPTY


# ---------------------------------------------------------------------------
# Honest counting: a None ("confirmed absent") value is NOT a fill
# ---------------------------------------------------------------------------
class TestNoneIsNotFilled:
    def test_none_excluded_from_get_filled(self):
        bb = Blackboard(["name", "nickname"])
        bb.write("name", "Alice")
        bb.write_raw("nickname", None)  # recovery: confirmed absent
        filled = bb.get_filled()
        assert filled == {"name": "Alice"}, "None confirmed-absent must not count as filled"

    def test_real_value_still_filled(self):
        bb = Blackboard(["x"])
        bb.write("x", 0)  # 0 / False are real values, not None
        assert bb.get_filled() == {"x": 0}

    def test_false_and_empty_string_are_real(self):
        bb = Blackboard(["flag", "note"])
        bb.write("flag", False)
        bb.write("note", "")
        assert bb.get_filled() == {"flag": False, "note": ""}


# ---------------------------------------------------------------------------
# Call-failure (transient) tracked apart from a genuine absent FAILED
# ---------------------------------------------------------------------------
class TestCallFailedTracking:
    def test_transient_failure_is_call_failed(self):
        bb = Blackboard(["a", "b"])
        bb.mark_failed("a", "provider error: timeout", transient=True)
        bb.mark_failed("b", "field not found in document")  # genuine absence
        assert bb.get_call_failed() == ["a"]
        assert bb.get_failed() == ["a", "b"]  # both still FAILED

    def test_recovered_call_failure_is_cleared(self):
        bb = Blackboard(["a"])
        bb.mark_failed("a", "provider error", transient=True)
        assert bb.get_call_failed() == ["a"]
        bb.write("a", "value")  # a retry succeeded
        assert bb.get_call_failed() == []
        assert bb.get_filled() == {"a": "value"}

    def test_absent_then_not_call_failed(self):
        bb = Blackboard(["a"])
        bb.mark_failed("a", "absent")  # default transient=False
        assert bb.get_call_failed() == []
