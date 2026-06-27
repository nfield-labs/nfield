"""Tests for schema._deps - dependency extraction."""

from __future__ import annotations

from nfield.schema._deps import (
    _extract_allof_deps,
    _extract_dependent_required,
    _extract_if_then_else,
    extract_dependencies,
)

# ---------------------------------------------------------------------------
# Basic / empty tests
# ---------------------------------------------------------------------------


class TestExtractDepsBasic:
    def test_no_deps_returns_empty(self) -> None:
        """Schema with no dep keywords returns empty dict."""
        schema: dict = {"type": "object", "properties": {"name": {"type": "string"}}}  # type: ignore[type-arg]
        result = extract_dependencies(schema)
        assert result == {}

    def test_simple_schema_no_deps(self) -> None:
        """Flat schema with required array but no dep keywords returns empty."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name"],
        }
        result = extract_dependencies(schema)
        assert result == {}

    def test_non_dict_returns_empty(self) -> None:
        """Non-dict input returns empty dict (graceful fallback)."""
        result = extract_dependencies("not a dict")  # type: ignore[arg-type]
        assert result == {}

    def test_empty_dict_returns_empty(self) -> None:
        """Completely empty schema returns empty dict."""
        result = extract_dependencies({})
        assert result == {}


# ---------------------------------------------------------------------------
# dependentRequired
# ---------------------------------------------------------------------------


class TestDependentRequired:
    def test_dependent_required_creates_dep(self) -> None:
        """dependentRequired: city requires has_address → deps['city'] = {'has_address'}."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "properties": {
                "has_address": {"type": "boolean"},
                "city": {"type": "string"},
            },
            "dependentRequired": {"city": ["has_address"]},
        }
        deps = extract_dependencies(schema)
        assert "city" in deps
        assert "has_address" in deps["city"]

    def test_multiple_required_fields(self) -> None:
        """dependentRequired with multiple required fields captures all."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentRequired": {
                "billing_city": ["has_billing", "billing_enabled"],
            }
        }
        deps = extract_dependencies(schema)
        assert "has_billing" in deps["billing_city"]
        assert "billing_enabled" in deps["billing_city"]

    def test_multiple_trigger_fields(self) -> None:
        """Multiple triggers in dependentRequired are all captured."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentRequired": {
                "field_a": ["dep_1"],
                "field_b": ["dep_2", "dep_3"],
            }
        }
        deps = extract_dependencies(schema)
        assert deps["field_a"] == {"dep_1"}
        assert deps["field_b"] == {"dep_2", "dep_3"}

    def test_dependent_required_not_dict_returns_empty(self) -> None:
        """dependentRequired with non-dict value is ignored."""
        schema: dict = {"dependentRequired": "invalid"}  # type: ignore[type-arg]
        result = _extract_dependent_required(schema)
        assert result == {}

    def test_dependent_required_non_list_value_skipped(self) -> None:
        """dependentRequired entry with non-list value is skipped."""
        schema: dict = {"dependentRequired": {"field": "not_a_list"}}  # type: ignore[type-arg]
        result = _extract_dependent_required(schema)
        assert result == {}


# ---------------------------------------------------------------------------
# dependentSchemas
# ---------------------------------------------------------------------------


class TestDependentSchemas:
    def test_dependent_schemas_creates_dep(self) -> None:
        """dependentSchemas with required creates dep from required fields to trigger."""
        schema: dict = {  # type: ignore[type-arg]
            "type": "object",
            "dependentSchemas": {
                "credit_card": {
                    "type": "object",
                    "required": ["billing_address"],
                }
            },
        }
        deps = extract_dependencies(schema)
        assert "billing_address" in deps
        assert "credit_card" in deps["billing_address"]

    def test_dependent_schemas_no_required_no_dep(self) -> None:
        """dependentSchemas sub-schema without required produces no deps."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentSchemas": {
                "trigger_field": {
                    "type": "object",
                    "properties": {"optional": {"type": "string"}},
                }
            }
        }
        deps = extract_dependencies(schema)
        assert deps == {}

    def test_dependent_schemas_multiple_required(self) -> None:
        """dependentSchemas with multiple required fields captures all."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentSchemas": {
                "uses_shipping": {
                    "required": ["shipping_city", "shipping_zip"],
                }
            }
        }
        deps = extract_dependencies(schema)
        assert "uses_shipping" in deps["shipping_city"]
        assert "uses_shipping" in deps["shipping_zip"]


# ---------------------------------------------------------------------------
# if/then/else
# ---------------------------------------------------------------------------


class TestIfThenElse:
    def test_then_fields_depend_on_if_fields(self) -> None:
        """Fields required in 'then' depend on fields in 'if'."""
        schema: dict = {  # type: ignore[type-arg]
            "if": {"properties": {"country": {"const": "US"}}},
            "then": {"required": ["state", "zip_code"]},
        }
        deps = extract_dependencies(schema)
        assert "country" in deps.get("state", set())
        assert "country" in deps.get("zip_code", set())

    def test_else_fields_depend_on_if_fields(self) -> None:
        """Fields required in 'else' depend on fields in 'if'."""
        schema: dict = {  # type: ignore[type-arg]
            "if": {"properties": {"type": {"const": "individual"}}},
            "else": {"required": ["company_name"]},
        }
        deps = extract_dependencies(schema)
        assert "type" in deps.get("company_name", set())

    def test_no_if_no_deps(self) -> None:
        """Schema with 'then' but no 'if' produces no deps from if/then/else."""
        schema: dict = {"then": {"required": ["field_a"]}}  # type: ignore[type-arg]
        result = _extract_if_then_else(schema)
        assert result == {}

    def test_if_required_fields_also_captured(self) -> None:
        """Fields in 'if.required' are treated as condition fields."""
        schema: dict = {  # type: ignore[type-arg]
            "if": {"required": ["is_business"]},
            "then": {"required": ["tax_id"]},
        }
        deps = extract_dependencies(schema)
        assert "is_business" in deps.get("tax_id", set())

    def test_if_with_no_then_else_no_deps(self) -> None:
        """if without then or else produces no deps."""
        schema: dict = {  # type: ignore[type-arg]
            "if": {"properties": {"country": {"const": "US"}}},
        }
        deps = extract_dependencies(schema)
        assert deps == {}


# ---------------------------------------------------------------------------
# allOf
# ---------------------------------------------------------------------------


class TestAllOf:
    def test_allof_propagates_deps(self) -> None:
        """allOf sub-schemas with dependentRequired are propagated."""
        schema: dict = {  # type: ignore[type-arg]
            "allOf": [
                {
                    "dependentRequired": {
                        "billing_city": ["has_billing"],
                    }
                }
            ]
        }
        deps = extract_dependencies(schema)
        assert "has_billing" in deps.get("billing_city", set())

    def test_allof_merges_from_multiple_subschemas(self) -> None:
        """allOf merges deps from multiple sub-schemas."""
        schema: dict = {  # type: ignore[type-arg]
            "allOf": [
                {"dependentRequired": {"field_a": ["dep_1"]}},
                {"dependentRequired": {"field_b": ["dep_2"]}},
            ]
        }
        deps = extract_dependencies(schema)
        assert "dep_1" in deps.get("field_a", set())
        assert "dep_2" in deps.get("field_b", set())

    def test_allof_non_list_ignored(self) -> None:
        """allOf with non-list value returns no allof deps."""
        schema: dict = {"allOf": "not_a_list"}  # type: ignore[type-arg]
        result = _extract_allof_deps(schema)
        assert result == {}

    def test_allof_non_dict_items_skipped(self) -> None:
        """Non-dict items in allOf are skipped without error."""
        schema: dict = {"allOf": [None, "string", 42]}  # type: ignore[type-arg]
        result = _extract_allof_deps(schema)
        assert result == {}


# ---------------------------------------------------------------------------
# Return type contracts
# ---------------------------------------------------------------------------


class TestReturnType:
    def test_returns_dict(self) -> None:
        """extract_dependencies always returns a dict."""
        result = extract_dependencies({"type": "object"})
        assert isinstance(result, dict)

    def test_values_are_sets(self) -> None:
        """All values in the returned dict are sets."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentRequired": {"city": ["has_address"]},
        }
        deps = extract_dependencies(schema)
        for val in deps.values():
            assert isinstance(val, set)

    def test_all_paths_are_strings(self) -> None:
        """All keys and values in the returned dict are strings."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentRequired": {"city": ["has_address", "postal_code"]},
        }
        deps = extract_dependencies(schema)
        for key, dep_set in deps.items():
            assert isinstance(key, str)
            for dep in dep_set:
                assert isinstance(dep, str)

    def test_merging_multiple_sources(self) -> None:
        """Dependencies from multiple keywords are merged correctly."""
        schema: dict = {  # type: ignore[type-arg]
            "dependentRequired": {"city": ["has_address"]},
            "if": {"properties": {"country": {}}},
            "then": {"required": ["city"]},
        }
        deps = extract_dependencies(schema)
        # city should have deps from both dependentRequired and if/then
        assert "has_address" in deps.get("city", set())
        assert "country" in deps.get("city", set())
