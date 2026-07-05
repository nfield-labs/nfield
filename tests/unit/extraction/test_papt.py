"""Unit tests for extraction._papt - PAPT template selection."""

from __future__ import annotations

from nfield.extraction._papt import (
    TemplateType,
    describe_field,
    select_template,
)
from nfield.schema._types import Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_field(
    path: str,
    ftype: str,
    *,
    constraints: dict | None = None,
    difficulty: float = 0.0,
    schema_node: dict | None = None,
) -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints=constraints or {},
        parent_path="",
        schema_node=schema_node or {},
        difficulty=difficulty,
    )


# ---------------------------------------------------------------------------
# select_template - budget tier selection
# ---------------------------------------------------------------------------


class TestSelectTemplate:
    def test_very_tight_budget_returns_concise(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=50) == TemplateType.CONCISE

    def test_boundary_below_concise(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=299) == TemplateType.CONCISE

    def test_boundary_at_concise_max(self):
        f = make_field("x", "string")
        # _BUDGET_CONCISE_MAX = 300 → 300 is STANDARD
        assert select_template([f], budget_tokens=300) == TemplateType.STANDARD

    def test_medium_budget_returns_standard(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=500) == TemplateType.STANDARD

    def test_boundary_below_verbose(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=799) == TemplateType.STANDARD

    def test_boundary_at_verbose_min(self):
        f = make_field("x", "string")
        # _BUDGET_VERBOSE_MIN = 800 → 800 is VERBOSE
        assert select_template([f], budget_tokens=800) == TemplateType.VERBOSE

    def test_large_budget_returns_verbose(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=2000) == TemplateType.VERBOSE

    def test_zero_budget_returns_concise(self):
        f = make_field("x", "string")
        assert select_template([f], budget_tokens=0) == TemplateType.CONCISE


# ---------------------------------------------------------------------------
# describe_field - field description at each tier
# ---------------------------------------------------------------------------


class TestDescribeField:
    def test_no_description_or_constraints_is_path_and_type(self):
        f = make_field("age", "integer")
        # No description and no constraints in the schema → just path (type).
        assert describe_field(f, TemplateType.CONCISE) == "age (integer)"

    def test_description_always_sent_even_in_concise(self):
        # Description is never dropped - the model needs it to understand the field.
        f = Field(
            path="age",
            type="integer",
            constraints={},
            parent_path="",
            schema_node={"description": "Patient age in years"},
        )
        for tier in (TemplateType.CONCISE, TemplateType.STANDARD, TemplateType.VERBOSE):
            assert describe_field(f, tier) == "age (integer): Patient age in years"

    def test_constraints_always_sent_with_description(self):
        # type + constraints + description all travel with the field at every tier.
        f = Field(
            path="year_first_elected",
            type="integer",
            constraints={"minimum": 1950},
            parent_path="",
            schema_node={"description": "Year joined board"},
        )
        for tier in (TemplateType.CONCISE, TemplateType.STANDARD, TemplateType.VERBOSE):
            result = describe_field(f, tier)
            assert result == "year_first_elected (integer): Year joined board - >= 1950"

    def test_standard_no_description_falls_back(self):
        f = make_field("age", "integer")
        assert describe_field(f, TemplateType.STANDARD) == "age (integer)"

    def test_verbose_with_enum_constraint(self):
        f = Field(
            path="status",
            type="enum",
            constraints={"enum": ["active", "inactive"]},
            parent_path="",
            schema_node={"description": "Account status"},
        )
        result = describe_field(f, TemplateType.VERBOSE)
        assert "status (enum)" in result
        assert "active" in result or "one of" in result

    def test_verbose_with_range_constraint(self):
        f = Field(
            path="score",
            type="number",
            constraints={"minimum": 0, "maximum": 100},
            parent_path="",
            schema_node={},
        )
        result = describe_field(f, TemplateType.VERBOSE)
        assert "range" in result or "0" in result

    def test_verbose_no_constraints_no_extra(self):
        f = make_field("name", "string")
        result = describe_field(f, TemplateType.VERBOSE)
        assert result == "name (string)"

    def test_template_type_enum_values(self):
        assert TemplateType.CONCISE.value == "concise"
        assert TemplateType.STANDARD.value == "standard"
        assert TemplateType.VERBOSE.value == "verbose"


class TestDimensionDirective:
    """Array-of-objects with enum item props get an enumeration directive."""

    def _metric_field(self) -> Field:
        items = {
            "type": "object",
            "properties": {
                "segment_type": {
                    "enum": ["company", "business_segment", "geographic_segment"],
                    "type": "string",
                },
                "data_period": {"type": "string"},
                "value": {"anyOf": [{"type": "null"}, {"type": "number"}]},
            },
        }
        return make_field("revenue", "array", constraints={"items": items})

    def test_enum_item_prop_emits_enumerate_directive(self):
        line = describe_field(self._metric_field(), TemplateType.STANDARD)
        assert "enumerate" in line
        assert "segment_type" in line
        assert "do not emit only the total" in line

    def test_anyof_item_prop_resolves_to_real_type(self):
        # value: anyOf[null, number] must render as number, not the string default.
        line = describe_field(self._metric_field(), TemplateType.STANDARD)
        assert "value: number" in line

    def test_scalar_array_gets_no_directive(self):
        f = make_field("tags", "array", constraints={"items": {"type": "string"}})
        assert "enumerate" not in describe_field(f, TemplateType.STANDARD)

    def test_object_array_without_enum_gets_no_directive(self):
        items = {"type": "object", "properties": {"name": {"type": "string"}}}
        f = make_field("people", "array", constraints={"items": items})
        assert "enumerate" not in describe_field(f, TemplateType.STANDARD)

    def test_single_value_enum_is_not_a_dimension(self):
        # A one-value enum is a constant attribute, not an axis the list spans.
        items = {
            "type": "object",
            "properties": {"kind": {"enum": ["only"], "type": "string"}},
        }
        f = make_field("rows", "array", constraints={"items": items})
        assert "enumerate" not in describe_field(f, TemplateType.STANDARD)


class TestNestedArrayShape:
    """A nested-array field shows its inner shape, not a bare 'array'."""

    def test_array_of_scalar_array_shape(self) -> None:
        f = Field(
            "m",
            "array",
            {"items": {"type": "array", "items": {"type": "number"}}},
            "",
            {"items": {"type": "array", "items": {"type": "number"}}},
        )
        assert "items: array of number" in describe_field(f, TemplateType.STANDARD)

    def test_array_of_object_array_shape(self) -> None:
        inner = {
            "type": "array",
            "items": {"type": "object", "properties": {"v": {"type": "string"}}},
        }
        f = Field("g", "array", {"items": inner}, "", {"items": inner})
        assert "array of object {v: string}" in describe_field(f, TemplateType.STANDARD)
