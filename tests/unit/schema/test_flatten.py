"""Tests for schema._flatten - Radix Trie Schema Flattener."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nfield.exceptions import SchemaError
from nfield.schema._flatten import flatten_schema

SCHEMAS_DIR = Path(__file__).parent.parent.parent / "fixtures" / "schemas"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_schema(name: str) -> dict:  # type: ignore[type-arg]
    """Load a JSON fixture by filename."""
    return json.loads((SCHEMAS_DIR / name).read_text())


# ---------------------------------------------------------------------------
# Basic tests
# ---------------------------------------------------------------------------


class TestFlattenBasic:
    def test_empty_schema_returns_empty_list(self) -> None:
        """An empty object schema with no properties yields no fields."""
        schema: dict = {"type": "object"}  # type: ignore[type-arg]
        result = flatten_schema(schema)
        assert result == []

    def test_flat_schema_5_fields(self) -> None:
        """simple_flat.json produces exactly 5 Field objects."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        assert len(fields) == 5

    def test_paths_are_strings(self) -> None:
        """All field paths must be non-empty strings."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        for f in fields:
            assert isinstance(f.path, str)
            assert len(f.path) > 0

    def test_all_paths_unique(self) -> None:
        """No two fields may share the same dot-notation path."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        paths = [f.path for f in fields]
        assert len(paths) == len(set(paths))

    def test_required_fields_marked(self) -> None:
        """Fields listed in 'required' have required=True."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        field_map = {f.path: f for f in fields}
        assert field_map["name"].required is True
        assert field_map["age"].required is True

    def test_non_required_fields_not_marked(self) -> None:
        """Fields not listed in 'required' have required=False."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        field_map = {f.path: f for f in fields}
        assert field_map["email"].required is False
        assert field_map["active"].required is False
        assert field_map["score"].required is False

    def test_parent_path_set_correctly(self) -> None:
        """All top-level fields have empty string as parent_path."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        for f in fields:
            assert f.parent_path == ""

    def test_top_level_fields_have_empty_parent_path(self) -> None:
        """Explicit check: top-level parent_path is always empty string."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "x": {"type": "integer"},
                "y": {"type": "string"},
            },
        }
        fields = flatten_schema(schema)
        for f in fields:
            assert f.parent_path == "", f"Expected '' but got {f.parent_path!r} for {f.path}"

    def test_field_types_preserved(self) -> None:
        """Field types from the schema are correctly assigned."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        field_map = {f.path: f for f in fields}
        assert field_map["name"].type == "string"
        assert field_map["age"].type == "integer"
        assert field_map["active"].type == "boolean"
        assert field_map["score"].type == "number"


# ---------------------------------------------------------------------------
# Nested tests
# ---------------------------------------------------------------------------


class TestFlattenNested:
    def test_nested_object_creates_dot_notation(self) -> None:
        """address.city appears as a dot-notation path."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "zip": {"type": "string"},
                    },
                }
            },
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "address.city" in paths
        assert "address.zip" in paths

    def test_nested_parent_path_set(self) -> None:
        """address.city has parent_path='address'."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "address": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                }
            },
        }
        fields = flatten_schema(schema)
        city_field = next(f for f in fields if f.path == "address.city")
        assert city_field.parent_path == "address"

    def test_deeply_nested_path(self) -> None:
        """a.b.c.d.e is created for a 5-level nesting."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "a": {
                    "type": "object",
                    "properties": {
                        "b": {
                            "type": "object",
                            "properties": {
                                "c": {
                                    "type": "object",
                                    "properties": {
                                        "d": {
                                            "type": "object",
                                            "properties": {"e": {"type": "string"}},
                                        }
                                    },
                                }
                            },
                        }
                    },
                }
            },
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "a.b.c.d.e" in paths

    def test_invoice_schema_fields_present(self) -> None:
        """invoice_50fields.json produces nested dot-notation fields."""
        schema = load_schema("invoice_50fields.json")
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        # Top-level fields
        assert "invoice_number" in paths
        assert "total_amount" in paths
        # Nested fields
        assert "vendor.name" in paths
        assert "vendor.address.city" in paths
        assert "customer.name" in paths
        assert "payment.method" in paths

    def test_invoice_all_paths_unique(self) -> None:
        """invoice_50fields.json produces no duplicate paths."""
        schema = load_schema("invoice_50fields.json")
        fields = flatten_schema(schema)
        paths = [f.path for f in fields]
        assert len(paths) == len(set(paths))

    def test_medical_schema_flattens(self) -> None:
        """medical_crf_134fields.json flattens without error."""
        schema = load_schema("medical_crf_134fields.json")
        fields = flatten_schema(schema)
        assert len(fields) > 0
        paths = [f.path for f in fields]
        assert len(paths) == len(set(paths)), "Duplicate paths found"

    def test_financial_schema_flattens(self) -> None:
        """financial_sec_369fields.json flattens without error."""
        schema = load_schema("financial_sec_369fields.json")
        fields = flatten_schema(schema)
        assert len(fields) > 0
        paths = [f.path for f in fields]
        assert len(paths) == len(set(paths)), "Duplicate paths found"


# ---------------------------------------------------------------------------
# Edge case tests
# ---------------------------------------------------------------------------


class TestFlattenEdgeCases:
    def test_anyof_takes_first_non_null(self) -> None:
        """anyOf with null and string picks string type."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "maybe_name": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "null"},
                    ]
                }
            },
        }
        fields = flatten_schema(schema)
        assert len(fields) == 1
        assert fields[0].path == "maybe_name"
        assert fields[0].type == "string"

    def test_anyof_null_first_skips_to_string(self) -> None:
        """anyOf with null first still picks string."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "val": {
                    "anyOf": [
                        {"type": "null"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        fields = flatten_schema(schema)
        assert len(fields) == 1
        assert fields[0].type == "integer"

    def test_ref_cycle_detected(self) -> None:
        """Circular $ref does not cause infinite loop - terminates."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "$defs": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "child": {"$ref": "#/$defs/Node"},
                    },
                }
            },
            "properties": {
                "root": {"$ref": "#/$defs/Node"},
            },
        }
        # Must terminate (not infinite loop)
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "root.value" in paths

    def test_array_items_creates_bracket_suffix(self) -> None:
        """A scalar array is one list-leaf carrying its item schema (whole list emitted)."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "tags": {
                    "type": "array",
                    "items": {"type": "string"},
                }
            },
        }
        by_path = {f.path: f for f in flatten_schema(schema)}
        assert set(by_path) == {"tags"}
        assert by_path["tags"].type == "array"
        assert by_path["tags"].constraints["items"]["type"] == "string"

    def test_object_array_expands_to_list_leaf(self) -> None:
        """An array of objects is one array list-leaf field carrying its item schema."""
        item_schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "value": {"type": "number"},
            },
        }
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {"rows": {"type": "array", "items": item_schema}},
        }
        by_path = {f.path: f for f in flatten_schema(schema)}
        assert set(by_path) == {"rows"}
        rows = by_path["rows"]
        assert rows.type == "array"
        assert rows.constraints["items"] == item_schema
        # The per-element template must NOT be emitted for an object array.
        assert "rows[].name" not in by_path

    def test_object_array_via_ref_expands_to_list_leaf(self) -> None:
        """A $ref item resolving to an object is also collapsed to one list-leaf."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "$defs": {
                "Entry": {
                    "type": "object",
                    "properties": {"period": {"type": "string"}, "amount": {"type": "number"}},
                }
            },
            "properties": {
                "entries": {"type": "array", "items": {"$ref": "#/$defs/Entry"}},
            },
        }
        by_path = {f.path: f for f in flatten_schema(schema)}
        assert set(by_path) == {"entries"}
        assert by_path["entries"].type == "array"

    def test_scalar_array_becomes_list_leaf(self) -> None:
        """An array of scalars is one list-leaf so the whole list is captured, not one item."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
        }
        paths = {f.path for f in flatten_schema(schema)}
        assert "tags" in paths
        assert "tags[]" not in paths

    def test_prefix_items_creates_indexed_paths(self) -> None:
        """prefixItems creates path[0], path[1], etc."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "coords": {
                    "type": "array",
                    "prefixItems": [
                        {"type": "number"},
                        {"type": "number"},
                    ],
                }
            },
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "coords[0]" in paths
        assert "coords[1]" in paths

    def test_allof_merges_properties(self) -> None:
        """allOf merges properties from all sub-schemas."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "allOf": [
                {"properties": {"name": {"type": "string"}}},
                {"properties": {"age": {"type": "integer"}}},
            ],
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "name" in paths
        assert "age" in paths

    def test_enum_type_inferred(self) -> None:
        """Node with 'enum' key gets type='enum' when no explicit type."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "color": {"enum": ["red", "green", "blue"]},
            },
        }
        fields = flatten_schema(schema)
        assert len(fields) == 1
        assert fields[0].type == "enum"

    def test_schema_not_dict_raises_schema_error(self) -> None:
        """Non-dict schema raises SchemaError."""
        with pytest.raises(SchemaError):
            flatten_schema("not a dict")  # type: ignore[arg-type]

    def test_schema_list_raises_schema_error(self) -> None:
        """List schema raises SchemaError."""
        with pytest.raises(SchemaError):
            flatten_schema(["not", "a", "dict"])  # type: ignore[arg-type]

    def test_oneof_takes_first_non_null(self) -> None:
        """oneOf with multiple options picks first non-null."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "amount": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "string"},
                    ]
                }
            },
        }
        fields = flatten_schema(schema)
        assert len(fields) == 1
        assert fields[0].type == "number"

    def test_additional_properties_dict_creates_open_map_leaf(self) -> None:
        """A pure additionalProperties object becomes one open-map list-leaf."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "metadata": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                }
            },
        }
        by_path = {f.path: f for f in flatten_schema(schema)}
        assert set(by_path) == {"metadata"}
        field = by_path["metadata"]
        assert field.type == "array"
        assert field.constraints["x-open-map"] is True
        item_props = field.constraints["items"]["properties"]
        assert set(item_props) == {"key", "value"}

    def test_additional_properties_bool_not_wildcard(self) -> None:
        """additionalProperties=True (bool) does NOT create wildcard."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "data": {
                    "type": "object",
                    "additionalProperties": True,
                }
            },
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "data.*" not in paths


# ---------------------------------------------------------------------------
# Constraint extraction tests
# ---------------------------------------------------------------------------


class TestFlattenConstraints:
    def test_constraints_extracted_correctly(self) -> None:
        """integer field with min/max has constraints populated."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        field_map = {f.path: f for f in fields}
        age = field_map["age"]
        assert age.constraints.get("minimum") == 0
        assert age.constraints.get("maximum") == 150

    def test_enum_constraints_preserved(self) -> None:
        """Enum values are preserved in constraints."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "status": {
                    "type": "string",
                    "enum": ["active", "inactive"],
                }
            },
        }
        fields = flatten_schema(schema)
        assert len(fields) == 1
        assert fields[0].constraints.get("enum") == ["active", "inactive"]

    def test_pattern_constraint_preserved(self) -> None:
        """Pattern constraint is extracted."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "zip": {
                    "type": "string",
                    "pattern": r"^\d{5}$",
                }
            },
        }
        fields = flatten_schema(schema)
        assert fields[0].constraints.get("pattern") == r"^\d{5}$"

    def test_minlength_maxlength_extracted(self) -> None:
        """minLength and maxLength are extracted into constraints."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "username": {
                    "type": "string",
                    "minLength": 3,
                    "maxLength": 20,
                }
            },
        }
        fields = flatten_schema(schema)
        assert fields[0].constraints.get("minLength") == 3
        assert fields[0].constraints.get("maxLength") == 20

    def test_format_constraint_extracted(self) -> None:
        """format is extracted into constraints."""
        schema = load_schema("simple_flat.json")
        fields = flatten_schema(schema)
        field_map = {f.path: f for f in fields}
        assert field_map["email"].constraints.get("format") == "email"

    def test_schema_node_preserved(self) -> None:
        """schema_node contains the original schema fragment."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "x": {"type": "integer", "minimum": 0},
            },
        }
        fields = flatten_schema(schema)
        assert "minimum" in fields[0].schema_node

    def test_ref_resolution_defs(self) -> None:
        """$ref with #/$defs/... resolves correctly."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "$defs": {
                "Address": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"},
                        "country": {"type": "string"},
                    },
                }
            },
            "properties": {
                "billing": {"$ref": "#/$defs/Address"},
            },
        }
        fields = flatten_schema(schema)
        paths = {f.path for f in fields}
        assert "billing.city" in paths
        assert "billing.country" in paths


# ---------------------------------------------------------------------------
# Resource bound - $ref fan-out / pathological expansion (DoS guard)
# ---------------------------------------------------------------------------
import nfield.schema._flatten as _flatten_mod  # noqa: E402


class TestNodeBudget:
    def _fanout_schema(self, levels: int) -> dict:
        """A $ref diamond: each level references the next twice → 2^levels nodes."""
        defs: dict = {}
        for i in range(levels):
            nxt = f"#/$defs/L{i + 1}"
            defs[f"L{i}"] = {
                "type": "object",
                "properties": {"a": {"$ref": nxt}, "b": {"$ref": nxt}},
            }
        defs[f"L{levels}"] = {"type": "object", "properties": {"v": {"type": "string"}}}
        return {"type": "object", "$defs": defs, "properties": {"root": {"$ref": "#/$defs/L0"}}}

    def test_fanout_exceeds_budget_raises(self, monkeypatch):
        # Tiny cap so the guard fires fast without building millions of nodes.
        monkeypatch.setattr(_flatten_mod, "MAX_TOTAL_NODES", 50)
        with pytest.raises(SchemaError, match="MAX_TOTAL_NODES"):
            flatten_schema(self._fanout_schema(levels=20))

    def test_normal_schema_under_budget(self, monkeypatch):
        # A small real schema stays well under even a modest cap.
        monkeypatch.setattr(_flatten_mod, "MAX_TOTAL_NODES", 1000)
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
        }
        assert {f.path for f in flatten_schema(schema)} == {"name", "age"}

    def test_default_budget_admits_large_schemas(self):
        # Constraint N_legit ≤ C: the cap must exceed the node count of any schema
        # we admit. A flat schema of F fields has N ≈ F, so C must clear a generous
        # field-count floor (1e6) while staying ≪ the exponential blow-up b^D.
        assert _flatten_mod.MAX_TOTAL_NODES >= 1_000_000


class TestStructuralUnion:
    """anyOf with both an array and an object branch is emitted as both."""

    def _skills_schema(self):
        return {
            "type": "object",
            "properties": {
                "skills": {
                    "anyOf": [
                        {"type": "array", "items": {"type": "string"}},
                        {
                            "type": "object",
                            "additionalProperties": {"type": "array", "items": {"type": "string"}},
                        },
                        {"type": "null"},
                    ]
                }
            },
        }

    def test_structural_union_emits_both_branches(self) -> None:
        fields = {f.path: f for f in flatten_schema(self._skills_schema())}
        assert "skills" in fields  # object (open-map) branch at base path
        assert "skills__uarr" in fields  # array branch at shadow path
        assert fields["skills"].constraints.get("x-union-kind") == "object"
        assert fields["skills__uarr"].constraints.get("x-union-kind") == "array"
        assert fields["skills"].constraints.get("x-union-base") == "skills"
        assert fields["skills__uarr"].constraints.get("x-union-base") == "skills"

    def test_scalar_union_unchanged(self) -> None:
        # string|int union is not structural: one field, no shadow, no union tags.
        schema = {
            "type": "object",
            "properties": {"d": {"anyOf": [{"type": "string"}, {"type": "integer"}]}},
        }
        fields = {f.path: f for f in flatten_schema(schema)}
        assert set(fields) == {"d"}
        assert "x-union-base" not in fields["d"].constraints


class TestArrayOfArray:
    """An array-of-array is one list-leaf carrying the nested item schema."""

    def test_array_of_array_is_single_list_leaf(self) -> None:
        schema = {
            "type": "object",
            "properties": {
                "m": {"type": "array", "items": {"type": "array", "items": {"type": "number"}}}
            },
        }
        fields = flatten_schema(schema)
        assert [f.path for f in fields] == ["m"]
        assert fields[0].type == "array"
        assert fields[0].constraints["items"]["type"] == "array"

    def test_array_of_array_assembles_without_extra_level(self) -> None:
        from nfield.assembly._trie import assemble_json

        assert assemble_json({"m": [[1, 2], [3, 4]]}) == {"m": [[1, 2], [3, 4]]}
