"""Tests for schema._difficulty — D(f) difficulty scoring."""
from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from formatshield.schema._difficulty import (
    _D_TYPE,
    _D_TYPE_STRING_UNCONSTRAINED,
    _D_WEIGHT_CONSTRAINT,
    _D_WEIGHT_DEP,
    _D_WEIGHT_TYPE,
    _MAX_DEP_DEGREE,
    compute_difficulty,
)
from formatshield.schema._types import Field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_field(
    type_: str,
    constraints: dict | None = None,  # type: ignore[type-arg]
    path: str = "x",
) -> Field:
    """Build a minimal Field for testing."""
    return Field(
        path=path,
        type=type_,
        constraints=constraints or {},
        parent_path="",
        schema_node={},
    )


# ---------------------------------------------------------------------------
# Range tests
# ---------------------------------------------------------------------------


class TestDifficultyRange:
    @pytest.mark.parametrize(
        "type_",
        ["boolean", "null", "enum", "integer", "number", "string", "array", "object"],
    )
    def test_difficulty_always_in_0_1(self, type_: str) -> None:
        """D(f) is always in [0.0, 1.0] for all standard types."""
        f = make_field(type_)
        d = compute_difficulty(f, dep_dag={})
        assert 0.0 <= d <= 1.0, f"D(f)={d} out of range for type={type_}"

    def test_boolean_is_lowest_difficulty(self) -> None:
        """Boolean (no constraints, no deps) is the easiest field type."""
        bool_field = make_field("boolean")
        other_types = ["integer", "number", "string", "array", "object"]
        d_bool = compute_difficulty(bool_field, dep_dag={})
        for type_ in other_types:
            d_other = compute_difficulty(make_field(type_), dep_dag={})
            assert d_bool <= d_other, f"boolean D={d_bool} > {type_} D={d_other}"

    def test_object_is_highest_difficulty(self) -> None:
        """Object (unconstrained) is the hardest field type with no deps."""
        d_obj = compute_difficulty(make_field("object"), dep_dag={})
        for type_ in ("boolean", "null", "enum", "integer", "number"):
            d_other = compute_difficulty(make_field(type_), dep_dag={})
            assert d_obj >= d_other, f"object D={d_obj} < {type_} D={d_other}"


# ---------------------------------------------------------------------------
# Type component tests
# ---------------------------------------------------------------------------


class TestDifficultyTypeComponent:
    def test_boolean_d_type_formula(self) -> None:
        """D(boolean) = 0.5 * 0.05 = 0.025 (no constraints, no deps)."""
        f = make_field("boolean")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["boolean"]
        assert abs(d - expected) < 1e-9

    def test_integer_d_type_formula(self) -> None:
        """D(integer) = 0.5 * 0.15 = 0.075 (no constraints, no deps)."""
        f = make_field("integer")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["integer"]
        assert abs(d - expected) < 1e-9

    def test_number_d_type_formula(self) -> None:
        """D(number) = 0.5 * 0.20 = 0.10 (no constraints, no deps)."""
        f = make_field("number")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["number"]
        assert abs(d - expected) < 1e-9

    def test_enum_d_type_formula(self) -> None:
        """D(enum) = 0.5 * 0.10 = 0.05 (no constraints, no deps)."""
        f = make_field("enum")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["enum"]
        assert abs(d - expected) < 1e-9

    def test_null_d_type_formula(self) -> None:
        """D(null) = 0.5 * 0.05 = 0.025 (no constraints, no deps)."""
        f = make_field("null")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["null"]
        assert abs(d - expected) < 1e-9

    def test_string_constrained_d_type(self) -> None:
        """Constrained string uses D_type=0.40."""
        f = make_field("string", {"maxLength": 100})
        d = compute_difficulty(f, dep_dag={})
        # D_type=0.40, D_constraint=0.1 (maxLength), D_dep=0.0
        d_type_component = _D_WEIGHT_TYPE * _D_TYPE["string"]
        assert d >= d_type_component

    def test_string_unconstrained_d_type(self) -> None:
        """Unconstrained string uses D_type=0.70."""
        f = make_field("string")
        d = compute_difficulty(f, dep_dag={})
        expected_type_component = _D_WEIGHT_TYPE * _D_TYPE_STRING_UNCONSTRAINED
        assert abs(d - expected_type_component) < 1e-9

    def test_array_d_type_formula(self) -> None:
        """D(array) = 0.5 * 0.60 = 0.30 (no constraints, no deps)."""
        f = make_field("array")
        d = compute_difficulty(f, dep_dag={})
        expected = _D_WEIGHT_TYPE * _D_TYPE["array"]
        assert abs(d - expected) < 1e-9


# ---------------------------------------------------------------------------
# Constraint component tests
# ---------------------------------------------------------------------------


class TestDifficultyConstraintComponent:
    def test_no_constraints_zero_d_constraint(self) -> None:
        """Fields with no constraints have D_constraint=0."""
        f = make_field("integer")
        d_no_constraint = compute_difficulty(f, dep_dag={})
        d_type_only = _D_WEIGHT_TYPE * _D_TYPE["integer"]
        assert abs(d_no_constraint - d_type_only) < 1e-9

    def test_pattern_increases_difficulty(self) -> None:
        """Adding 'pattern' constraint does not reduce difficulty vs unconstrained.

        Note: unconstrained string D_type=0.70; constrained string D_type=0.40 but
        gains D_constraint=0.50 for pattern. Net effect: equal or greater than plain.
        """
        f_plain = make_field("string")
        f_pattern = make_field("string", {"pattern": r"^\d{5}$"})
        d_plain = compute_difficulty(f_plain, dep_dag={})
        d_pattern = compute_difficulty(f_pattern, dep_dag={})
        assert d_pattern >= d_plain

    def test_format_increases_difficulty(self) -> None:
        """Adding 'format' constraint raises difficulty above a no-constraint string of same type.

        Comparing a constrained string (format only) against an otherwise
        identical constrained string with no constraints at all.
        """
        # Both strings are "constrained" type (D_type=0.40).
        # One has no constraints; the other has format (weight 0.30).
        f_constrained_no_extra = make_field("string", {"maxLength": 200})
        f_format = make_field("string", {"maxLength": 200, "format": "email"})
        d_base = compute_difficulty(f_constrained_no_extra, dep_dag={})
        d_format = compute_difficulty(f_format, dep_dag={})
        assert d_format > d_base

    def test_multiple_constraints_accumulate(self) -> None:
        """Multiple constraints accumulate up to 1.0."""
        f_few = make_field("string", {"maxLength": 100})
        f_many = make_field(
            "string",
            {"maxLength": 100, "minLength": 1, "pattern": r"^\w+$", "format": "email"},
        )
        d_few = compute_difficulty(f_few, dep_dag={})
        d_many = compute_difficulty(f_many, dep_dag={})
        assert d_many >= d_few

    def test_d_constraint_capped_at_1(self) -> None:
        """D_constraint component is capped at 1.0."""
        # Many heavy constraints
        constraints = {
            "pattern": ".*",
            "format": "email",
            "minimum": 0,
            "maximum": 100,
            "minLength": 1,
            "maxLength": 50,
            "uniqueItems": True,
            "multipleOf": 2,
            "minItems": 1,
            "maxItems": 10,
        }
        f = make_field("string", constraints)
        d = compute_difficulty(f, dep_dag={})
        assert d <= 1.0


# ---------------------------------------------------------------------------
# Dependency component tests
# ---------------------------------------------------------------------------


class TestDifficultyDepComponent:
    def test_no_deps_zero_d_dep(self) -> None:
        """No deps → D_dep component = 0."""
        f = make_field("boolean", path="active")
        d = compute_difficulty(f, dep_dag={})
        # Should equal type-only difficulty
        expected = _D_WEIGHT_TYPE * _D_TYPE["boolean"]
        assert abs(d - expected) < 1e-9

    def test_in_degree_increases_difficulty(self) -> None:
        """Fields with in-degree > 0 have higher difficulty."""
        f = make_field("string", path="city")
        d_no_dep = compute_difficulty(f, dep_dag={})
        # city depends on has_address
        dep_dag: dict[str, set[str]] = {"city": {"has_address"}}
        d_with_dep = compute_difficulty(f, dep_dag=dep_dag)
        assert d_with_dep > d_no_dep

    def test_out_degree_increases_difficulty(self) -> None:
        """Fields that are depended on (out-degree > 0) have higher difficulty."""
        f = make_field("boolean", path="has_address")
        d_no_dep = compute_difficulty(f, dep_dag={})
        # has_address is depended on by city
        dep_dag: dict[str, set[str]] = {"city": {"has_address"}}
        d_with_out = compute_difficulty(f, dep_dag=dep_dag)
        assert d_with_out > d_no_dep

    def test_d_dep_capped_at_1(self) -> None:
        """D_dep is capped at 1.0 regardless of degree."""
        f = make_field("string", path="hub")
        # Create many deps to exceed _MAX_DEP_DEGREE
        dep_dag: dict[str, set[str]] = {
            f"field_{i}": {"hub"} for i in range(_MAX_DEP_DEGREE + 5)
        }
        d = compute_difficulty(f, dep_dag=dep_dag)
        assert d <= 1.0

    def test_dep_dag_empty_dict_no_crash(self) -> None:
        """Empty dep_dag dict works without error."""
        f = make_field("integer")
        d = compute_difficulty(f, dep_dag={})
        assert d >= 0.0


# ---------------------------------------------------------------------------
# Weight sum test
# ---------------------------------------------------------------------------


class TestDifficultyWeights:
    def test_weights_sum_to_1(self) -> None:
        """The three weights must sum to 1.0."""
        total = _D_WEIGHT_TYPE + _D_WEIGHT_CONSTRAINT + _D_WEIGHT_DEP
        assert abs(total - 1.0) < 1e-9

    def test_max_possible_difficulty_is_1(self) -> None:
        """Maximum possible D(f) is 1.0 (all components at maximum)."""
        # object type (0.80), many constraints, many deps
        constraints = {
            "pattern": ".*",
            "format": "email",
            "minimum": 0,
            "maximum": 100,
            "minLength": 1,
            "maxLength": 50,
            "uniqueItems": True,
            "multipleOf": 2,
            "minItems": 1,
            "maxItems": 10,
        }
        f = make_field("object", constraints, path="x")
        dep_dag: dict[str, set[str]] = {
            f"field_{i}": {"x"} for i in range(_MAX_DEP_DEGREE + 5)
        }
        d = compute_difficulty(f, dep_dag=dep_dag)
        assert 0.0 <= d <= 1.0


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestDifficultyProperties:
    """Hypothesis property tests to catch rare edge cases (principles §10.4)."""

    @given(
        type_=st.sampled_from(
            ["boolean", "null", "enum", "integer", "number", "string", "array", "object"]
        ),
        max_length=st.integers(min_value=0, max_value=10000) | st.none(),
    )
    def test_difficulty_always_in_valid_range(self, type_: str, max_length: int | None) -> None:
        """Property: D(f) ∈ [0.0, 1.0] for any field type and constraints."""
        constraints = {}
        if type_ == "string" and max_length is not None:
            constraints["maxLength"] = max_length
        elif type_ == "enum":
            constraints["enum"] = ["a", "b"]

        f = make_field(type_, constraints)
        d = compute_difficulty(f, dep_dag={})
        assert 0.0 <= d <= 1.0, f"D(f)={d} out of range for type={type_}"

    @given(
        constraint_count_range=st.tuples(
            st.integers(min_value=0, max_value=3),
            st.integers(min_value=3, max_value=8),
        ),
    )
    def test_more_constraints_increase_difficulty(self, constraint_count_range):
        """Property: given same base field, adding more constraints increases difficulty."""
        fewer, more = constraint_count_range
        constraint_options = ["minLength", "maxLength", "minimum", "maximum", "pattern", "enum"]

        constraints_fewer = dict.fromkeys(constraint_options[:fewer], 100)
        constraints_more = dict.fromkeys(constraint_options[:more], 100)

        # Use same type (integer) to keep D_type constant
        f_fewer = make_field("integer", constraints_fewer)
        f_more = make_field("integer", constraints_more)

        d_fewer = compute_difficulty(f_fewer, dep_dag={})
        d_more = compute_difficulty(f_more, dep_dag={})

        # With same type and more constraints, difficulty should increase or stay same
        assert d_more >= d_fewer - 1e-9, (
            f"Adding constraints reduced difficulty: "
            f"{fewer} constraints → D={d_fewer}, {more} constraints → D={d_more}"
        )

    @given(
        dep_degree=st.integers(min_value=0, max_value=100),
    )
    def test_more_dependencies_increase_difficulty(self, dep_degree: int) -> None:
        """Property: higher dependency degree → higher difficulty."""
        f1 = make_field("string", path="x")
        f2 = make_field("string", path="x")

        # No deps
        d_no_dep = compute_difficulty(f1, dep_dag={})

        # Create dep_dag where "x" depends on many fields
        if dep_degree > 0:
            dep_dag = {"x": {f"dep_{i}" for i in range(dep_degree)}}
            d_with_dep = compute_difficulty(f2, dep_dag=dep_dag)
            assert d_with_dep >= d_no_dep - 1e-9

    @given(
        type_=st.sampled_from(
            ["boolean", "null", "enum", "integer", "number", "string", "array", "object"]
        ),
        reverse_dep_count=st.integers(min_value=0, max_value=20),
    )
    def test_compute_difficulty_with_precomputed_reverse_index(
        self, type_: str, reverse_dep_count: int
    ) -> None:
        """Property: passing reverse_dep_dag produces same result as computing it."""
        f = make_field(type_, path="field_x")

        # Create a dep_dag where many fields depend on "field_x"
        dep_dag: dict[str, set[str]] = {
            f"field_{i}": {"field_x"} for i in range(reverse_dep_count)
        }

        # Compute with auto-computed reverse index
        d_auto = compute_difficulty(f, dep_dag=dep_dag)

        # Compute with pre-computed reverse index
        reverse_dep_dag = {"field_x": {f"field_{i}" for i in range(reverse_dep_count)}}
        d_precomputed = compute_difficulty(f, dep_dag=dep_dag, reverse_dep_dag=reverse_dep_dag)

        # Both should give the same result
        assert abs(d_auto - d_precomputed) < 1e-9, (
            f"Results differ: auto={d_auto}, precomputed={d_precomputed}"
        )
