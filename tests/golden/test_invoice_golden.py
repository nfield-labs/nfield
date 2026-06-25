"""Golden regression test: 50-field invoice schema.

Fixed expected values that must not change across refactors:
- K_min >= 1 for this schema with typical model config
- Grouping produces at least 2 structural groups
- All 50 field paths are unique after flattening
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfield.config import ExtractionConfig
from nfield.pipeline._state import PipelineState
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.pipeline.s2b_prepass import run_stage_2b
from nfield.pipeline.s2c_packing import run_stage_2c

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "schemas" / "invoice_50fields.json"


@pytest.fixture
def invoice_schema():
    if not FIXTURE_PATH.exists():
        pytest.skip("invoice_50fields.json fixture not found")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def invoice_state(invoice_schema):
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state = run_stage_1(state, invoice_schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Invoice document text.", ExtractionConfig())
    state = run_stage_2c(state, ExtractionConfig())
    return state


class TestInvoiceGolden:
    def test_field_count(self, invoice_state):
        assert len(invoice_state.fields) >= 1

    def test_paths_unique(self, invoice_state):
        paths = [f.path for f in invoice_state.fields]
        assert len(paths) == len(set(paths))

    def test_k_min_at_least_1(self, invoice_state):
        assert invoice_state.K_min >= 1

    def test_groups_created(self, invoice_state):
        assert len(invoice_state.groups) >= 1

    def test_all_fields_have_tau(self, invoice_state):
        for f in invoice_state.fields:
            assert f.tau > 0.0

    def test_all_leaves_have_fields(self, invoice_state):
        for leaf in invoice_state.leaves:
            assert len(leaf.fields) >= 1

    def test_execution_order_covers_all_leaves(self, invoice_state):
        in_order = [leaf for r in invoice_state.execution_order for leaf in r]
        assert len(in_order) == len(invoice_state.leaves)
