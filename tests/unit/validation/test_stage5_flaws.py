"""Unit tests for the Stage 5 flaw fixes C (round count), D (conflict/reval), E (call failures)."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from nfield.assembly._blackboard import Blackboard, FieldState
from nfield.config import ExtractionConfig
from nfield.schema._types import CapacityLeaf, Field
from nfield.validation._retry import orchestrate_retry, surgical_field_retry


def _field(path: str, ftype: str = "string", tau: float = 2.0) -> Field:
    return Field(path=path, type=ftype, constraints={}, parent_path="", schema_node={}, tau=tau)


def _leaf(fields: list[Field]) -> CapacityLeaf:
    return CapacityLeaf(fields=fields, document_excerpt="doc", safe_output=256, leaf_id=1)


# --------------------------------------------------------------------------- C


@pytest.mark.asyncio
async def test_rounds_counter_reports_true_count() -> None:
    """flaw C: rounds_counter reflects the actual number of rounds run."""
    fields = [_field("a")]
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value="")  # never recovers → uses all rounds
    rounds = [0]
    await orchestrate_retry(
        failed_fields=fields,
        errors={"a": "missing"},
        provider=provider,
        leaf=_leaf(fields),
        dep_dag={},
        config=ExtractionConfig(max_retry_rounds=2),
        rounds_counter=rounds,
    )
    assert rounds[0] == 2


@pytest.mark.asyncio
async def test_rounds_counter_stops_when_recovered() -> None:
    """A field recovered in round 1 means only one round ran."""
    fields = [_field("a")]
    provider = AsyncMock()
    provider.complete = AsyncMock(return_value="a = hello")
    rounds = [0]
    recovered = await orchestrate_retry(
        failed_fields=fields,
        errors={"a": "missing"},
        provider=provider,
        leaf=_leaf(fields),
        dep_dag={},
        config=ExtractionConfig(max_retry_rounds=2),
        rounds_counter=rounds,
    )
    assert recovered == {"a": "hello"}
    assert rounds[0] == 1


# --------------------------------------------------------------------------- E


@pytest.mark.asyncio
async def test_call_failure_is_recorded_not_silent() -> None:
    """flaw E: a provider exception records a per-field reason instead of {} silence."""
    fields = [_field("a"), _field("b")]
    provider = AsyncMock()
    provider.complete = AsyncMock(side_effect=RuntimeError("boom 503"))
    failures: dict[str, str] = {}
    result = await surgical_field_retry(
        fields,
        {"a": "x", "b": "y"},
        provider,
        _leaf(fields),
        call_failures=failures,
    )
    assert result == {}
    assert set(failures) == {"a", "b"}
    assert "boom 503" in failures["a"]


@pytest.mark.asyncio
async def test_call_failure_cleared_when_later_call_succeeds() -> None:
    """flaw E (H1): a later round's successful call clears a stale failure label."""
    fields = [_field("a")]
    provider = AsyncMock()
    # round 1 raises; round 2 succeeds but returns nothing for the field.
    provider.complete = AsyncMock(side_effect=[RuntimeError("boom"), ""])
    failures: dict[str, str] = {}
    await orchestrate_retry(
        failed_fields=fields,
        errors={"a": "missing"},
        provider=provider,
        leaf=_leaf(fields),
        dep_dag={},
        config=ExtractionConfig(max_retry_rounds=2),
        call_failures=failures,
    )
    # The field did not recover, but its retry call did NOT fail on the last attempt,
    # so it must not be mislabelled "retry call failed".
    assert "a" not in failures


# --------------------------------------------------------------------------- D (blackboard primitive)


def test_reopen_for_retry_from_terminal_states() -> None:
    """flaw D: CONFLICT / NEEDS_REVALIDATION / FAILED reopen to PENDING, cleared."""
    bb = Blackboard(["c", "n", "f", "ok"])
    # conflict
    bb.write("c", 1)
    bb.write("c", 2)
    assert bb.get_state("c") == FieldState.CONFLICT
    # needs revalidation
    bb.mark_needs_revalidation("n")
    # failed
    bb.mark_failed("f", "nope")
    # filled (should NOT reopen)
    bb.write("ok", "v")

    assert bb.reopen_for_retry("c") is True
    assert bb.reopen_for_retry("n") is True
    assert bb.reopen_for_retry("f") is True
    assert bb.reopen_for_retry("ok") is False

    assert bb.get_state("c") == FieldState.PENDING
    assert bb.get_state("n") == FieldState.PENDING
    assert bb.get_state("f") == FieldState.PENDING
    assert bb.get_conflict_values("c") == []  # cleared
    assert bb.get_error("f") is None  # cleared

    # A reopened field can now accept a fresh value.
    bb.write("c", 99)
    assert bb.get_state("c") == FieldState.FILLED
    assert bb.get_filled()["c"] == 99
