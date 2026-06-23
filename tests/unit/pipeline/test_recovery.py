"""Tests for the Stage 5.5 missing-field recovery pass (MFRP).

Covers, deterministically and without network:
- no-op when nothing is missing,
- a missed field is recovered by the second pass,
- validated fields are never re-touched,
- the pass is bounded to one round (a permanently-missing field stays FAILED),
- state.leaves / execution_order are restored after the pass.
"""

from __future__ import annotations

from formatshield.assembly._blackboard import Blackboard, FieldState
from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s5b_recover import run_recovery_pass
from formatshield.schema._types import CapacityLeaf, Field, FieldGroup, Segment


def _field(path: str, *, tau: float = 5.0) -> Field:
    return Field(
        path=path, type="string", constraints={}, parent_path="", schema_node={}
    ).with_tau(tau=tau, var_tau=1.0)


class _RecoverProvider:
    """Returns a value only for the missed field ``b`` (recovers it)."""

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


class _SilentProvider:
    """Returns nothing — the missed field stays missing (bound check)."""

    context_window = 8192
    max_output_tokens = 8192
    model_name = "mock/silent"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages: list[dict[str, str]], *, max_tokens: int) -> str:
        self.calls += 1
        return ""

    async def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


def _state_with_missing() -> PipelineState:
    a, b = _field("a"), _field("b")
    seg = Segment(
        text="a is one and b is recovered_value",
        start=0,
        end=33,
        segment_type="unstructured",
        segment_id=0,
    )
    group = FieldGroup(parent_path="", fields=[a, b], matched_segments=[seg], segment_scores=[1.0])
    state = PipelineState(
        fields=[a, b],
        field_by_path={"a": a, "b": b},
        dep_dag={},
        chars_per_token=4.0,
        C_eff=8192,
        M_O=2048,
        C_usable=4000.0,
    )
    state.groups = [group]
    state.segments = [seg]
    state.blackboard = Blackboard(["a", "b"])
    state.blackboard.write("a", "one")  # FILLED — must not be re-touched
    state.blackboard.mark_failed("b", "field absent after retry")  # the missed field
    return state


class TestRecoveryPass:
    async def test_nothing_missing_is_noop(self):
        state = _state_with_missing()
        state.blackboard.write("b", "already")  # now nothing missing
        provider = _RecoverProvider()
        await run_recovery_pass(state, provider, ExtractionConfig())
        assert provider.calls == 0

    async def test_recovers_missed_field(self):
        state = _state_with_missing()
        provider = _RecoverProvider()
        cfg = ExtractionConfig(max_retry_rounds=0)
        await run_recovery_pass(state, provider, cfg)
        assert provider.calls >= 1
        assert state.blackboard.get_state("b") == FieldState.FILLED
        assert state.blackboard.get_filled()["b"] == "recovered_value"

    async def test_call_failed_field_is_re_extracted_by_default(self):
        # A field whose Stage 4 call exhausted its transient budget (429 / timeout) is,
        # by default, given one more bounded attempt in recovery: the rolling-window rate
        # limit has refilled by the time recovery runs, so the retry usually lands.
        state = _state_with_missing()
        bb = Blackboard(["a", "b"])
        bb.write("a", "one")
        bb.mark_failed("b", "provider error: 429 rate limit", transient=True)
        state.blackboard = bb
        provider = _RecoverProvider()  # recovers "b" if asked
        await run_recovery_pass(state, provider, ExtractionConfig(max_retry_rounds=0))
        assert provider.calls >= 1  # recovery DID give the call-failed field another try
        assert state.blackboard.get_state("b") == FieldState.FILLED
        assert state.blackboard.get_filled()["b"] == "recovered_value"

    async def test_call_failed_excluded_when_flag_off(self):
        # With recover_call_failed=False, the conservative behaviour is preserved: a
        # transient call failure is left unrecovered, adding no load to a throttled API.
        state = _state_with_missing()
        bb = Blackboard(["a", "b"])
        bb.write("a", "one")
        bb.mark_failed("b", "provider error: 429 rate limit", transient=True)
        state.blackboard = bb
        provider = _RecoverProvider()  # would recover "b" if asked
        cfg = ExtractionConfig(max_retry_rounds=0, recover_call_failed=False)
        await run_recovery_pass(state, provider, cfg)
        assert provider.calls == 0  # recovery did not fire into the throttled API
        assert state.blackboard.get_state("b") == FieldState.FAILED

    async def test_filled_field_untouched(self):
        state = _state_with_missing()
        cfg = ExtractionConfig(max_retry_rounds=0)
        await run_recovery_pass(state, _RecoverProvider(), cfg)
        # 'a' was FILLED before the pass and must keep its original value.
        assert state.blackboard.get_filled()["a"] == "one"

    async def test_bounded_one_pass(self):
        state = _state_with_missing()
        provider = _SilentProvider()
        cfg = ExtractionConfig(max_retry_rounds=0)
        await run_recovery_pass(state, provider, cfg)
        # Still missing after one pass → stays FAILED, no infinite loop.
        assert state.blackboard.get_state("b") == FieldState.FAILED

    async def test_restores_leaves(self):
        state = _state_with_missing()
        original_leaf = CapacityLeaf(fields=list(state.fields), groups=state.groups, leaf_id=0)
        state.leaves = [original_leaf]
        state.execution_order = [[original_leaf]]
        cfg = ExtractionConfig(max_retry_rounds=0)
        await run_recovery_pass(state, _RecoverProvider(), cfg)
        assert state.leaves == [original_leaf]
        assert state.execution_order == [[original_leaf]]
