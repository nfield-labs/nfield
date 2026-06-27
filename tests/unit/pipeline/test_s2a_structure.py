"""Tests for Stage 2A: Structural Grouping."""

from __future__ import annotations

from nfield.pipeline._state import PipelineState
from nfield.pipeline.s1_schema import run_stage_1
from nfield.pipeline.s2a_structure import run_stage_2a
from nfield.schema._types import FieldGroup


def _make_state() -> PipelineState:
    return PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)


FLAT_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
}

NESTED_SCHEMA = {
    "type": "object",
    "properties": {
        "address": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "zip": {"type": "string"},
            },
        },
        "name": {"type": "string"},
    },
}


class TestRunStage2a:
    def test_flat_schema_top_level_group(self):
        state = run_stage_2a(run_stage_1(_make_state(), FLAT_SCHEMA))
        assert len(state.groups) >= 1
        all_fields_covered = all(f.path in state.group_map for f in state.fields)
        assert all_fields_covered

    def test_nested_schema_address_group(self):
        state = run_stage_2a(run_stage_1(_make_state(), NESTED_SCHEMA))
        parent_paths = {g.parent_path for g in state.groups}
        # address.city and address.zip share parent "address"
        assert "address" in parent_paths

    def test_group_map_covers_all_fields(self):
        state = run_stage_2a(run_stage_1(_make_state(), NESTED_SCHEMA))
        for f in state.fields:
            assert f.path in state.group_map

    def test_groups_are_field_group_instances(self):
        state = run_stage_2a(run_stage_1(_make_state(), FLAT_SCHEMA))
        for g in state.groups:
            assert isinstance(g, FieldGroup)

    def test_each_group_has_fields(self):
        state = run_stage_2a(run_stage_1(_make_state(), FLAT_SCHEMA))
        for g in state.groups:
            assert len(g.fields) >= 1

    def test_address_group_has_two_fields(self):
        state = run_stage_2a(run_stage_1(_make_state(), NESTED_SCHEMA))
        address_groups = [g for g in state.groups if g.parent_path == "address"]
        assert len(address_groups) == 1
        assert len(address_groups[0].fields) == 2

    def test_returns_same_state(self):
        state = run_stage_1(_make_state(), FLAT_SCHEMA)
        returned = run_stage_2a(state)
        assert returned is state
