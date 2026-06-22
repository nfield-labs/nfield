"""Stage 5.5 fallback escalation: stragglers re-extracted on a stronger model."""

from __future__ import annotations

import asyncio

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import run_stage_2c
from formatshield.pipeline.s3_excerpt import run_stage_3
from formatshield.pipeline.s4_extract import run_stage_4
from formatshield.pipeline.s5_validate import run_stage_5
from formatshield.pipeline.s5b_recover import run_recovery_pass

SCHEMA = {
    "type": "object",
    "properties": {
        "company": {"type": "string"},
        "year": {"type": "integer", "minimum": 1800, "maximum": 2100},
    },
}
DOCUMENT = "Acme Corp was founded in 1947."


class _FixedProvider:
    """Returns the same canned completion on every call (incl. recovery re-extraction)."""

    context_window = 8192
    max_output_tokens = 1024
    model_name = "mock/model"

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls = 0

    async def complete(self, messages, *, max_tokens):
        self.calls += 1
        return self.response

    async def count_tokens(self, text):
        return max(1, len(text) // 4)


def _settled_state_with_failed_year() -> PipelineState:
    """Run S1-S5 with a primary that cannot produce a valid ``year``."""
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    config = ExtractionConfig()
    state = run_stage_1(state, SCHEMA)
    state = run_stage_2a(state)
    state = run_stage_2b(state, DOCUMENT, config)
    state = run_stage_2c(state, config)
    state = run_stage_3(state)
    primary = _FixedProvider("company = Acme Corp\nyear = not_a_number\n")
    asyncio.run(run_stage_4(state, primary))
    asyncio.run(run_stage_5(state, primary, config))
    return state


def test_fallback_recovers_what_primary_cannot() -> None:
    state = _settled_state_with_failed_year()
    assert state.blackboard is not None
    assert "year" not in state.blackboard.get_filled()  # primary failed it

    primary = _FixedProvider("company = Acme Corp\nyear = not_a_number\n")
    fallback = _FixedProvider("year = 1947\n")
    config = ExtractionConfig()
    asyncio.run(run_recovery_pass(state, primary, config, fallback_provider=fallback))

    # The stronger model rescued the straggler the primary kept failing.
    assert state.blackboard.get_filled().get("year") == 1947
    assert fallback.calls >= 1


def test_no_fallback_leaves_straggler_unrecovered() -> None:
    state = _settled_state_with_failed_year()
    assert state.blackboard is not None

    primary = _FixedProvider("company = Acme Corp\nyear = not_a_number\n")
    config = ExtractionConfig()
    asyncio.run(run_recovery_pass(state, primary, config, fallback_provider=None))

    # Without escalation, the field the primary cannot produce stays unrecovered.
    assert "year" not in state.blackboard.get_filled()
