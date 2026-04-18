"""Tests for formatshield.oracle.schema_entropy — constraint tightness τ."""

from __future__ import annotations

import pytest

from formatshield.oracle.schema_entropy import constraint_tightness


class TestConstraintTightness:
    def test_empty_schema_returns_zero(self) -> None:
        assert constraint_tightness({}) == 0.0

    def test_non_dict_returns_zero(self) -> None:
        assert constraint_tightness(None) == 0.0  # type: ignore[arg-type]
        assert constraint_tightness("string") == 0.0  # type: ignore[arg-type]

    def test_boolean_field_high_tightness(self) -> None:
        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}},
            "required": ["flag"],
        }
        tau = constraint_tightness(schema)
        # boolean has entropy = 1 bit, H0 ≈ 17 bits → τ ≈ 1 - 1/17 ≈ 0.94
        assert tau > 0.9
        assert tau <= 1.0

    def test_unconstrained_string_low_tightness(self) -> None:
        schema = {
            "type": "object",
            "properties": {"text": {"type": "string"}},
        }
        tau = constraint_tightness(schema)
        # unconstrained string has entropy = H0 → τ ≈ 0
        assert tau == pytest.approx(0.0, abs=0.01)

    def test_enum_tightness(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]},
            },
            "required": ["color"],
        }
        tau = constraint_tightness(schema)
        # enum with 3 choices: h = log2(3) ≈ 1.58 bits → τ ≈ 1 - 1.58/17 ≈ 0.91
        assert tau > 0.85
        assert tau <= 1.0

    def test_mixed_schema_between_extremes(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},  # unconstrained
                "active": {"type": "boolean"},  # 1 bit
                "count": {"type": "integer", "minimum": 0, "maximum": 10},  # log2(11) bits
            },
        }
        tau = constraint_tightness(schema)
        assert 0.0 < tau < 1.0

    def test_integer_range(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "score": {"type": "integer", "minimum": 1, "maximum": 5},
            },
        }
        tau = constraint_tightness(schema)
        # h = log2(5) ≈ 2.32 bits → τ ≈ 1 - 2.32/17 ≈ 0.86
        assert tau > 0.8

    def test_string_with_format_half_tightness(self) -> None:
        schema_format = {
            "type": "object",
            "properties": {
                "date": {"type": "string", "format": "date"},
            },
        }
        tau_format = constraint_tightness(schema_format)
        schema_free = {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
            },
        }
        tau_free = constraint_tightness(schema_free)
        # format halves entropy → higher tightness than free string
        assert tau_format > tau_free

    def test_tightly_constrained_vs_unconstrained(self) -> None:
        tight = {
            "type": "object",
            "properties": {
                "a": {"type": "boolean"},
                "b": {"enum": ["x"]},
            },
        }
        loose = {
            "type": "object",
            "properties": {
                "a": {"type": "string"},
                "b": {"type": "string"},
            },
        }
        assert constraint_tightness(tight) > constraint_tightness(loose)

    def test_result_always_in_unit_interval(self) -> None:
        schemas = [
            {},
            {"type": "object", "properties": {}},
            {"type": "object", "properties": {"x": {"type": "boolean"}}},
            {
                "type": "object",
                "properties": {
                    "n": {"type": "integer", "minimum": 0, "maximum": 1000000},
                },
            },
        ]
        for schema in schemas:
            tau = constraint_tightness(schema)
            assert 0.0 <= tau <= 1.0, f"τ={tau} out of [0,1] for schema={schema}"
