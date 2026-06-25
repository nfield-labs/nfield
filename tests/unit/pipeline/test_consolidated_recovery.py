"""Tests for the consolidated Stage 5 / 5.5 retry path.

Covers, deterministically and without network, the ``consolidated_recovery``
configuration in which Stage 5 only validates (no API calls) and the recovery
pass performs every re-extraction:

- Stage 5 makes no provider calls and settles state (PENDING -> FAILED, invalid
  FILLED -> FAILED) so the recovery pass has the full pool.
- the recovery pass recovers a missed field and tags its calls ``recovery_*``,
- a CONFLICT field is reopened and re-extracted when ``recover_conflicts`` is set,
- a validated field is never re-touched,
- the pass is a no-op when nothing needs recovery.
"""

from __future__ import annotations

from nfield.assembly._blackboard import Blackboard, FieldState
from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s5_validate import run_stage_5
from nfield.pipeline.s5b_recover import run_recovery_pass
from nfield.schema._types import CapacityLeaf, Field, FieldGroup, Segment

_DOC = "a is one and b is recovered_value"


def _field(path: str, ftype: str = "string", constraints: dict | None = None) -> Field:
    return Field(
        path=path, type=ftype, constraints=constraints or {}, parent_path="", schema_node={}
    ).with_tau(tau=5.0, var_tau=1.0)


class _RecoverProvider:
    """Returns a value for the missed field ``b`` (recovers it)."""

    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/recover"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        return "b = recovered_value"

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _state(fields: list[Field]) -> PipelineState:
    seg = Segment(text=_DOC, start=0, end=len(_DOC), segment_type="unstructured", segment_id=0)
    group = FieldGroup(parent_path="", fields=fields, matched_segments=[seg], segment_scores=[1.0])
    leaf = CapacityLeaf(
        fields=list(fields), groups=[group], document_excerpt=_DOC, overhead=50, safe_output=256
    )
    state = PipelineState(
        fields=list(fields),
        field_by_path={f.path: f for f in fields},
        dep_dag={},
        chars_per_token=4.0,
        C_eff=8192,
        M_O=2048,
        C_usable=4000.0,
    )
    state.groups = [group]
    state.segments = [seg]
    state.leaves = [leaf]
    state.blackboard = Blackboard([f.path for f in fields])
    return state


class TestValidationOnlyStage5:
    async def test_no_provider_calls(self):
        a, b = _field("a"), _field("b")
        state = _state([a, b])
        state.blackboard.write("a", "one")
        state.blackboard.mark_pending("b")
        provider = _RecoverProvider()
        await run_stage_5(state, provider, ExtractionConfig())
        assert provider.calls == 0

    async def test_pending_becomes_failed(self):
        a, b = _field("a"), _field("b")
        state = _state([a, b])
        state.blackboard.write("a", "one")
        state.blackboard.mark_pending("b")  # extracted but never returned
        await run_stage_5(state, _RecoverProvider(), ExtractionConfig())
        assert state.blackboard.get_state("b") == FieldState.FAILED
        assert state.blackboard.get_state("a") == FieldState.FILLED

    async def test_invalid_filled_becomes_failed(self):
        age = _field("age", "integer", {"maximum": 120})
        state = _state([age])
        state.blackboard.write("age", 200)  # type-valid but violates the constraint
        await run_stage_5(state, _RecoverProvider(), ExtractionConfig())
        assert state.blackboard.get_state("age") == FieldState.FAILED


class TestConsolidatedRecovery:
    async def test_recovers_failed_field_tagged_recovery(self):
        a, b = _field("a"), _field("b")
        state = _state([a, b])
        state.blackboard.write("a", "one")
        state.blackboard.mark_failed("b", "field not found in document")
        provider = _RecoverProvider()
        await run_recovery_pass(state, provider, ExtractionConfig())
        assert state.blackboard.get_filled().get("b") == "recovered_value"
        assert state.blackboard.get_filled().get("a") == "one"
        assert any(k.startswith("recovery_") for k in state.calls_by_origin)

    async def test_conflict_is_reopened_and_recovered(self):
        a, b = _field("a"), _field("b")
        state = _state([a, b])
        state.blackboard.write("a", "one")
        state.blackboard.write("b", "first")
        state.blackboard.write("b", "second")  # divergent second write -> CONFLICT
        assert state.blackboard.get_state("b") == FieldState.CONFLICT
        cfg = ExtractionConfig(recover_conflicts=True)
        await run_recovery_pass(state, _RecoverProvider(), cfg)
        assert state.blackboard.get_filled().get("b") == "recovered_value"

    async def test_noop_when_nothing_missing(self):
        a = _field("a")
        state = _state([a])
        state.blackboard.write("a", "one")
        provider = _RecoverProvider()
        await run_recovery_pass(state, provider, ExtractionConfig())
        assert provider.calls == 0
