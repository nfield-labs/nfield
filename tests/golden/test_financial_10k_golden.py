"""Golden regression test: the realistic ~1000-field 10-K financial schema.

Loads ``tests/fixtures/schemas/financial_10k_realistic.json`` — built from real
US-GAAP XBRL concepts and SEC-DEI cover-page tags across seven fiscal years —
and runs Stages S1-S2C to confirm the planner handles a real, wide enterprise
schema. No API calls.
"""

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
    Path(__file__).parent.parent / "fixtures" / "schemas" / "financial_10k_realistic.json"
)


@pytest.fixture
def schema():
    if not FIXTURE_PATH.exists():
        pytest.skip("financial_10k_realistic.json fixture not found")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def planned_state(schema):
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=8192, C_usable=4096.0)
    state = run_stage_1(state, schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Annual report excerpt.", ExtractionConfig())
    state = run_stage_2c(state, ExtractionConfig())
    return state


class TestFinancial10KGolden:
    def test_field_count_near_1000(self, planned_state):
        # Real US-GAAP concepts fanned out over 7 fiscal years.
        assert len(planned_state.fields) == 1074

    def test_real_concept_paths_present(self, planned_state):
        paths = {f.path for f in planned_state.fields}
        assert "fiscal_year_2024.income_statement.net_income_loss" in paths
        assert "fiscal_year_2024.balance_sheet.goodwill" in paths
        assert "fiscal_year_2023.cash_flow.net_cash_provided_by_operating_activities" in paths
        assert "company_profile.entity_central_index_key" in paths

    def test_paths_unique(self, planned_state):
        paths = [f.path for f in planned_state.fields]
        assert len(paths) == len(set(paths))

    def test_constraints_carried_through(self, planned_state):
        by_path = {f.path: f for f in planned_state.fields}
        # minimum 0 on an asset amount
        assert by_path["fiscal_year_2024.balance_sheet.assets"].constraints.get("minimum") == 0
        # enum on the exchange field
        assert "enum" in by_path["company_profile.security_exchange_name"].constraints

    def test_packs_into_many_leaves(self, planned_state):
        assert len(planned_state.leaves) > 1

    def test_every_field_packed_exactly_once(self, planned_state):
        leaf_paths = [f.path for leaf in planned_state.leaves for f in leaf.fields]
        assert sorted(leaf_paths) == sorted(f.path for f in planned_state.fields)

    def test_execution_order_covers_all_leaves(self, planned_state):
        in_order = [leaf for r in planned_state.execution_order for leaf in r]
        assert len(in_order) == len(planned_state.leaves)
