"""Tests for Stage 1: Schema Analysis."""

from __future__ import annotations

import pytest

from formatshield.assembly._blackboard import Blackboard
from formatshield.exceptions import SchemaError
from formatshield.pipeline._state import PipelineState
from formatshield.pipeline.s1_schema import run_stage_1


def _make_state(chars_per_token: float = 4.0) -> PipelineState:
    return PipelineState(chars_per_token=chars_per_token, C_eff=8192, M_O=1024, C_usable=4096.0)


SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
        "active": {"type": "boolean"},
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
    },
}


class TestRunStage1:
    def test_fields_populated(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        assert len(state.fields) == 3

    def test_field_paths_correct(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        paths = {f.path for f in state.fields}
        assert "name" in paths
        assert "age" in paths
        assert "active" in paths

    def test_field_by_path_index(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        assert "name" in state.field_by_path
        assert state.field_by_path["age"].type == "integer"

    def test_tau_computed(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        for f in state.fields:
            assert f.tau > 0.0

    def test_difficulty_in_range(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        for f in state.fields:
            assert 0.0 <= f.difficulty <= 1.0

    def test_blackboard_initialised(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        assert isinstance(state.blackboard, Blackboard)

    def test_blackboard_contains_all_paths(self):
        state = run_stage_1(_make_state(), SIMPLE_SCHEMA)
        bb_paths = set(state.blackboard.all_paths())
        field_paths = {f.path for f in state.fields}
        assert bb_paths == field_paths

    def test_nested_schema_paths(self):
        state = run_stage_1(_make_state(), NESTED_SCHEMA)
        paths = {f.path for f in state.fields}
        assert "address.city" in paths
        assert "address.zip" in paths

    def test_empty_schema_raises(self):
        with pytest.raises(SchemaError):
            run_stage_1(_make_state(), {"type": "object", "properties": {}})

    def test_dep_dag_populated(self):
        schema_with_deps = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
            "dependentRequired": {"b": ["a"]},
        }
        state = run_stage_1(_make_state(), schema_with_deps)
        assert isinstance(state.dep_dag, dict)

    def test_state_returned(self):
        state = _make_state()
        returned = run_stage_1(state, SIMPLE_SCHEMA)
        assert returned is state
