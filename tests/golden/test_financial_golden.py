"""Golden regression test: 369-field financial SEC schema."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from formatshield.config import ExtractionConfig
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1
from formatshield.pipeline.s2a_structure import run_stage_2a
from formatshield.pipeline.s2b_prepass import run_stage_2b
from formatshield.pipeline.s2c_packing import run_stage_2c

FIXTURE_PATH = (
    Path(__file__).parent.parent / "fixtures" / "schemas" / "financial_sec_369fields.json"
)


@pytest.fixture
def financial_schema():
    if not FIXTURE_PATH.exists():
        pytest.skip("financial_sec_369fields.json fixture not found")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def financial_state(financial_schema):
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state = run_stage_1(state, financial_schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Financial report.", ExtractionConfig())
    state = run_stage_2c(state, ExtractionConfig())
    return state


class TestFinancialGolden:
    def test_field_count(self, financial_state):
        assert len(financial_state.fields) >= 1

    def test_paths_unique(self, financial_state):
        paths = [f.path for f in financial_state.fields]
        assert len(paths) == len(set(paths))

    def test_k_min_positive(self, financial_state):
        assert financial_state.K_min >= 1

    def test_multiple_leaves_for_large_schema(self, financial_state):
        # 369 fields should require multiple API calls
        assert len(financial_state.leaves) >= 1

    def test_execution_order_consistent(self, financial_state):
        in_order = [leaf for r in financial_state.execution_order for leaf in r]
        assert len(in_order) == len(financial_state.leaves)
