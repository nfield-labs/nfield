"""Unit tests for assembly._trie — radix trie assembler."""

from __future__ import annotations

import pytest

from formatshield.assembly._trie import (
    RadixTrie,
    assemble_json,
    parse_path_segments,
)
from formatshield.exceptions import AssemblyError

# ---------------------------------------------------------------------------
# parse_path_segments
# ---------------------------------------------------------------------------


class TestParsePathSegments:
    def test_simple_key(self):
        assert parse_path_segments("name") == ["name"]

    def test_nested_two_levels(self):
        assert parse_path_segments("address.city") == ["address", "city"]

    def test_nested_three_levels(self):
        assert parse_path_segments("a.b.c") == ["a", "b", "c"]

    def test_array_index(self):
        assert parse_path_segments("items[0]") == ["items", 0]

    def test_array_index_with_nested_key(self):
        assert parse_path_segments("items[0].name") == ["items", 0, "name"]

    def test_multi_level_array(self):
        assert parse_path_segments("matrix[0][1]") == ["matrix", 0, 1]

    def test_deep_mixed_path(self):
        result = parse_path_segments("a.b[0].c.d[2]")
        assert result == ["a", "b", 0, "c", "d", 2]

    def test_empty_path_raises(self):
        with pytest.raises(AssemblyError):
            parse_path_segments("")

    def test_whitespace_only_raises(self):
        with pytest.raises(AssemblyError):
            parse_path_segments("   ")

    def test_large_index(self):
        assert parse_path_segments("arr[99]") == ["arr", 99]


# ---------------------------------------------------------------------------
# assemble_json — flat dict to nested JSON
# ---------------------------------------------------------------------------


class TestAssembleJson:
    def test_empty_input(self):
        assert assemble_json({}) == {}

    def test_single_flat_field(self):
        assert assemble_json({"name": "Alice"}) == {"name": "Alice"}

    def test_two_flat_fields(self):
        result = assemble_json({"name": "Alice", "age": 30})
        assert result == {"name": "Alice", "age": 30}

    def test_nested_two_levels(self):
        result = assemble_json({"a.b": 1, "a.c": 2})
        assert result == {"a": {"b": 1, "c": 2}}

    def test_nested_three_levels(self):
        result = assemble_json({"x.y.z": 42})
        assert result == {"x": {"y": {"z": 42}}}

    def test_shared_parent_path(self):
        result = assemble_json(
            {
                "address.street": "123 Main St",
                "address.city": "Springfield",
                "address.zip": "12345",
            }
        )
        assert result == {
            "address": {
                "street": "123 Main St",
                "city": "Springfield",
                "zip": "12345",
            }
        }

    def test_array_single_element(self):
        result = assemble_json({"items[0].name": "Widget"})
        assert result == {"items": [{"name": "Widget"}]}

    def test_array_two_elements(self):
        result = assemble_json(
            {
                "items[0].name": "Alpha",
                "items[1].name": "Beta",
            }
        )
        assert result == {"items": [{"name": "Alpha"}, {"name": "Beta"}]}

    def test_array_with_scalar_value(self):
        result = assemble_json({"ids[0]": 1, "ids[1]": 2})
        assert result == {"ids": [1, 2]}

    def test_mixed_array_and_object(self):
        result = assemble_json(
            {
                "report.lines[0].amount": 100,
                "report.lines[1].amount": 200,
                "report.total": 300,
            }
        )
        assert result == {
            "report": {
                "lines": [{"amount": 100}, {"amount": 200}],
                "total": 300,
            }
        }

    def test_none_value_preserved(self):
        result = assemble_json({"notes": None})
        assert result == {"notes": None}

    def test_boolean_values(self):
        result = assemble_json({"active": True, "deleted": False})
        assert result == {"active": True, "deleted": False}

    def test_bijection_roundtrip(self):
        """SFEP bijection: flat dict re-assembles to expected nested structure."""
        flat = {
            "vendor.name": "Acme Corp",
            "vendor.address.city": "New York",
            "items[0].sku": "W001",
            "items[0].qty": 5,
            "items[1].sku": "W002",
            "items[1].qty": 3,
            "total": 450.0,
        }
        result = assemble_json(flat)
        assert result["vendor"]["name"] == "Acme Corp"
        assert result["vendor"]["address"]["city"] == "New York"
        assert result["items"][0]["sku"] == "W001"
        assert result["items"][1]["qty"] == 3
        assert result["total"] == 450.0

    def test_deep_nesting(self):
        result = assemble_json({"a.b.c.d.e": "deep"})
        assert result == {"a": {"b": {"c": {"d": {"e": "deep"}}}}}


class TestHomogeneousArrayBrackets:
    """The flattener emits ``[]`` for homogeneous arrays — assembler maps to 0."""

    def test_parse_empty_brackets_maps_to_index_zero(self):
        assert parse_path_segments("segments[].capex") == ["segments", 0, "capex"]

    def test_array_of_objects_single_element(self):
        result = assemble_json({"segments[].capex": 100, "segments[].revenue": 200})
        assert result == {"segments": [{"capex": 100, "revenue": 200}]}

    def test_scalar_homogeneous_array(self):
        assert assemble_json({"tags[]": "alpha"}) == {"tags": ["alpha"]}

    def test_empty_brackets_mixed_with_scalar_field(self):
        result = assemble_json({"segments[].capex": 1, "name": "Acme"})
        assert result == {"segments": [{"capex": 1}], "name": "Acme"}


# ---------------------------------------------------------------------------
# RadixTrie — direct usage
# ---------------------------------------------------------------------------


class TestRadixTrie:
    def test_empty_trie_builds_empty_dict(self):
        trie = RadixTrie()
        assert trie.build() == {}

    def test_single_insert(self):
        trie = RadixTrie()
        trie.insert("key", "value")
        assert trie.build() == {"key": "value"}

    def test_two_inserts_same_parent(self):
        trie = RadixTrie()
        trie.insert("x.a", 1)
        trie.insert("x.b", 2)
        assert trie.build() == {"x": {"a": 1, "b": 2}}

    def test_overwrite_same_path(self):
        trie = RadixTrie()
        trie.insert("key", "first")
        trie.insert("key", "second")
        result = trie.build()
        assert result["key"] == "second"

    def test_conflict_type_raises_assembly_error(self):
        trie = RadixTrie()
        trie.insert("a.b", 1)  # a -> dict
        with pytest.raises(AssemblyError):
            trie.insert("a[0]", 2)  # tries to make a -> list (conflict)

    def test_integer_values(self):
        trie = RadixTrie()
        trie.insert("count", 42)
        assert trie.build() == {"count": 42}
