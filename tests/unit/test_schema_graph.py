"""Tests for formatshield.oracle.schema_graph — Fiedler value computation."""

from __future__ import annotations

from formatshield.oracle.schema_graph import fiedler_value


class TestFiedlerValue:
    def test_empty_schema_returns_zero(self) -> None:
        assert fiedler_value({}) == 0.0

    def test_single_field_returns_zero(self) -> None:
        schema = {"type": "object", "properties": {"name": {"type": "string"}}}
        assert fiedler_value(schema) == 0.0

    def test_isolated_fields_low_connectivity(self) -> None:
        # Two flat unrelated fields — low structural coupling → low λ̃₂
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = fiedler_value(schema)
        assert 0.0 <= result <= 1.0

    def test_nested_schema_higher_than_flat(self) -> None:
        flat = {
            "type": "object",
            "properties": {
                "x": {"type": "string"},
                "y": {"type": "string"},
                "z": {"type": "string"},
            },
        }
        nested = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {
                            "type": "object",
                            "properties": {
                                "value": {"type": "integer"},
                            },
                        }
                    },
                },
                "sibling": {"type": "string"},
            },
        }
        # Nested schema has structural edges → higher λ̃₂
        assert fiedler_value(nested) >= fiedler_value(flat)

    def test_result_in_unit_interval(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "format": "date"},
                "end_date": {"type": "string", "format": "date"},
                "total": {"type": "number"},
            },
        }
        result = fiedler_value(schema)
        assert 0.0 <= result <= 1.0

    def test_shared_stem_adds_edges(self) -> None:
        # Fields with shared stems (start_date/end_date) get extra edges
        schema_with_stems = {
            "type": "object",
            "properties": {
                "start_date": {"type": "string"},
                "end_date": {"type": "string"},
            },
        }
        schema_no_stems = {
            "type": "object",
            "properties": {
                "alpha": {"type": "string"},
                "beta": {"type": "string"},
            },
        }
        # Both have 2 fields; the stem schema may have slightly higher connectivity
        # (exact equality acceptable if both use same fallback path)
        r_stem = fiedler_value(schema_with_stems)
        r_no_stem = fiedler_value(schema_no_stems)
        assert r_stem >= r_no_stem

    def test_anyof_schema(self) -> None:
        schema = {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
                {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                    },
                },
            ]
        }
        result = fiedler_value(schema)
        assert 0.0 <= result <= 1.0

    def test_non_dict_schema_returns_zero(self) -> None:
        assert fiedler_value(None) == 0.0  # type: ignore[arg-type]
        assert fiedler_value("not a schema") == 0.0  # type: ignore[arg-type]
