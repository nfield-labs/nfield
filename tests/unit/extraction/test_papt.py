"""Unit tests for extraction._papt — PAPT template selection."""

from __future__ import annotations

from nfield.extraction._papt import (
    ClusterType,
    TemplateType,
    classify_cluster,
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
# select_template — budget tier selection
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
# classify_cluster — cluster type detection
# ---------------------------------------------------------------------------


class TestClassifyCluster:
    def test_empty_fields_returns_standard(self):
        assert classify_cluster([]) == ClusterType.STANDARD

    def test_all_boolean_returns_simple(self):
        fields = [make_field(f"f{i}", "boolean") for i in range(3)]
        assert classify_cluster(fields) == ClusterType.SIMPLE

    def test_all_integer_returns_simple(self):
        fields = [make_field(f"f{i}", "integer") for i in range(3)]
        assert classify_cluster(fields) == ClusterType.SIMPLE

    def test_majority_array_returns_list(self):
        fields = [
            make_field("a", "array"),
            make_field("b", "array"),
            make_field("c", "string"),
        ]
        assert classify_cluster(fields) == ClusterType.LIST

    def test_mixed_types_returns_standard(self):
        fields = [
            make_field("name", "string"),
            make_field("age", "integer"),
            make_field("score", "number"),
        ]
        assert classify_cluster(fields) == ClusterType.STANDARD

    def test_high_difficulty_returns_complex(self):
        fields = [make_field(f"f{i}", "string", difficulty=0.8) for i in range(5)]
        assert classify_cluster(fields) == ClusterType.COMPLEX

    def test_ref_in_schema_node_returns_reference(self):
        f = Field(
            path="ref_field",
            type="object",
            constraints={},
            parent_path="",
            schema_node={"$ref": "#/definitions/Address"},
        )
        assert classify_cluster([f]) == ClusterType.REFERENCE


# ---------------------------------------------------------------------------
# describe_field — field description at each tier
# ---------------------------------------------------------------------------


class TestDescribeField:
    def test_no_description_or_constraints_is_path_and_type(self):
        f = make_field("age", "integer")
        # No description and no constraints in the schema → just path (type).
        assert describe_field(f, TemplateType.CONCISE) == "age (integer)"

    def test_description_always_sent_even_in_concise(self):
        # Description is never dropped — the model needs it to understand the field.
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
            assert result == "year_first_elected (integer): Year joined board — >= 1950"

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

    def test_cluster_type_enum_values(self):
        assert ClusterType.SIMPLE.value == "simple"
        assert ClusterType.LIST.value == "list"
        assert ClusterType.COMPLEX.value == "complex"
        assert ClusterType.REFERENCE.value == "reference"
