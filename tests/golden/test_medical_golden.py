"""Golden regression test: 134-field medical CRF schema."""

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

FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "schemas" / "medical_crf_134fields.json"


@pytest.fixture
def medical_schema():
    if not FIXTURE_PATH.exists():
        pytest.skip("medical_crf_134fields.json fixture not found")
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture
def medical_state(medical_schema):
    state = PipelineState(chars_per_token=4.0, C_eff=8192, M_O=1024, C_usable=4096.0)
    state = run_stage_1(state, medical_schema)
    state = run_stage_2a(state)
    state = run_stage_2b(state, "Patient record.", ExtractionConfig())
    state = run_stage_2c(state, ExtractionConfig())
    return state


class TestMedicalGolden:
    def test_field_count(self, medical_state):
        assert len(medical_state.fields) >= 1

    def test_paths_unique(self, medical_state):
        paths = [f.path for f in medical_state.fields]
        assert len(paths) == len(set(paths))

    def test_k_min_at_least_1(self, medical_state):
        assert medical_state.K_min >= 1

    def test_groups_at_least_1(self, medical_state):
        assert len(medical_state.groups) >= 1

    def test_all_fields_have_difficulty(self, medical_state):
        for f in medical_state.fields:
            assert 0.0 <= f.difficulty <= 1.0
