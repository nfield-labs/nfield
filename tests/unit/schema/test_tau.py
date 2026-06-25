"""Tests for schema._tau — SOTP token predictor."""

from __future__ import annotations

import math

from hypothesis import given
from hypothesis import strategies as st

from nfield.schema._tau import _compute_enum_tau, compute_tau
from nfield.schema._types import Field

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def make_field(
    type_: str,
    constraints: dict | None = None,  # type: ignore[type-arg]
) -> Field:
    """Build a minimal Field for testing."""
    # flatten_schema stores items in schema_node, not constraints — mirror that here
    schema_node: dict = {}  # type: ignore[type-arg]
    final_constraints: dict = constraints or {}  # type: ignore[type-arg]

    if type_ == "array" and final_constraints.get("items"):
        schema_node["items"] = final_constraints.pop("items")

    return Field(
        path="x",
        type=type_,
        constraints=final_constraints,
        parent_path="",
        schema_node=schema_node,
    )


# ---------------------------------------------------------------------------
# Basic types
# ---------------------------------------------------------------------------


class TestComputeTauBasicTypes:
    def test_boolean_tau_is_1(self) -> None:
        """Boolean field always produces tau=1.0."""
        f = make_field("boolean")
        tau, _ = compute_tau(f, 4.0)
        assert tau == 1.0

    def test_boolean_var_is_0(self) -> None:
        """Boolean field has zero variance."""
        f = make_field("boolean")
        _, var_tau = compute_tau(f, 4.0)
        assert var_tau == 0.0

    def test_null_tau_is_1(self) -> None:
        """Null field always produces tau=1.0."""
        f = make_field("null")
        tau, _ = compute_tau(f, 4.0)
        assert tau == 1.0

    def test_null_var_is_0(self) -> None:
        """Null field has zero variance."""
        f = make_field("null")
        _, var_tau = compute_tau(f, 4.0)
        assert var_tau == 0.0

    def test_integer_tau_positive(self) -> None:
        """Integer field tau is positive."""
        f = make_field("integer")
        tau, _ = compute_tau(f, 4.0)
        assert tau >= 1.0

    def test_integer_tau_formula(self) -> None:
        """Integer tau = ceil(5 / chars_per_token)."""
        f = make_field("integer")
        tau, var_tau = compute_tau(f, 4.0)
        expected = float(math.ceil(5 / 4.0))
        assert tau == expected
        assert var_tau == 0.5

    def test_number_tau_greater_than_or_equal_integer(self) -> None:
        """Number tau >= integer tau (more chars needed for decimals)."""
        fi = make_field("integer")
        fn = make_field("number")
        tau_i, _ = compute_tau(fi, 4.0)
        tau_n, _ = compute_tau(fn, 4.0)
        assert tau_n >= tau_i

    def test_number_tau_formula(self) -> None:
        """Number tau = ceil((6 + 3) / chars_per_token)."""
        f = make_field("number")
        tau, var_tau = compute_tau(f, 4.0)
        expected = float(math.ceil(9 / 4.0))
        assert tau == expected
        assert var_tau == 1.0

    def test_enum_tau_based_on_longest_option(self) -> None:
        """Enum tau is based on longest value divided by chars_per_token."""
        f = make_field("enum", {"enum": ["A", "BB", "CCC"]})
        tau, var_tau = compute_tau(f, 4.0)
        # longest = "CCC" → 3 chars → ceil(3/4) = 1
        assert tau == max(1.0, float(math.ceil(3 / 4.0)))
        assert var_tau == 0.0

    def test_enum_var_is_0(self) -> None:
        """Enum field always has zero variance."""
        f = make_field("enum", {"enum": ["draft", "sent", "paid"]})
        _, var_tau = compute_tau(f, 4.0)
        assert var_tau == 0.0

    def test_enum_with_long_values(self) -> None:
        """Enum tau uses ceil of longest value over chars_per_token."""
        f = make_field("enum", {"enum": ["bank_transfer", "credit_card", "cash"]})
        tau, _ = compute_tau(f, 4.0)
        # "bank_transfer" = 13 chars → ceil(13/4) = 4
        assert tau == float(math.ceil(13 / 4.0))

    def test_enum_no_values_returns_1(self) -> None:
        """Enum with no values still returns tau >= 1.0."""
        f = make_field("enum", {"enum": []})
        tau, _ = compute_tau(f, 4.0)
        assert tau >= 1.0


# ---------------------------------------------------------------------------
# String type
# ---------------------------------------------------------------------------


class TestComputeTauString:
    def test_constrained_string_uses_maxlength(self) -> None:
        """String with maxLength uses that as char bound."""
        f = make_field("string", {"maxLength": 80})
        tau, _ = compute_tau(f, 4.0)
        assert tau == float(math.ceil(80 / 4.0))

    def test_constrained_string_variance(self) -> None:
        """Constrained string variance = (tau * 0.3)^2."""
        f = make_field("string", {"maxLength": 80})
        tau, var_tau = compute_tau(f, 4.0)
        assert abs(var_tau - (tau * 0.3) ** 2) < 1e-9

    def test_unconstrained_string_uses_p90(self) -> None:
        """String without maxLength uses p90_string_tokens."""
        f = make_field("string")
        tau, _ = compute_tau(f, 4.0, p90_string_tokens=50)
        assert tau == 50.0

    def test_unconstrained_string_variance_positive(self) -> None:
        """Unconstrained string variance = (tau * 0.6)^2 > 0."""
        f = make_field("string")
        _tau, var_tau = compute_tau(f, 4.0, p90_string_tokens=35)
        assert var_tau > 0.0
        assert abs(var_tau - (35.0 * 0.6) ** 2) < 1e-9

    def test_string_maxlength_zero_still_returns_1(self) -> None:
        """maxLength=0 is edge case; tau is still at least 1.0."""
        f = make_field("string", {"maxLength": 0})
        tau, _ = compute_tau(f, 4.0)
        assert tau >= 1.0


# ---------------------------------------------------------------------------
# Array type
# ---------------------------------------------------------------------------


class TestComputeTauArray:
    def test_array_tau_scales_with_expected_size(self) -> None:
        """Array tau = element_tau * expected_array_size."""
        f = make_field("array", {"items": {"type": "boolean"}})
        tau, _ = compute_tau(f, 4.0, expected_array_size=5)
        # boolean element_tau = 1.0 → 1.0 * 5 = 5.0
        assert tau == 5.0

    def test_array_tau_minimum_1(self) -> None:
        """Array tau is always at least 1.0."""
        f = make_field("array")
        tau, _ = compute_tau(f, 4.0, expected_array_size=0)
        assert tau >= 1.0

    def test_array_no_items_uses_string_defaults(self) -> None:
        """Array without items info uses p90 string defaults."""
        f = make_field("array")
        tau, _ = compute_tau(f, 4.0, p90_string_tokens=35, expected_array_size=3)
        assert tau == 35.0 * 3

    def test_array_string_items_variance(self) -> None:
        """Array variance scales with element variance."""
        f = make_field("array", {"items": {"type": "string", "maxLength": 20}})
        _tau, var_tau = compute_tau(f, 4.0, expected_array_size=3)
        element_tau = float(math.ceil(20 / 4.0))
        element_var = (element_tau * 0.3) ** 2
        assert abs(var_tau - element_var * 3) < 1e-9


# ---------------------------------------------------------------------------
# chars_per_token sensitivity
# ---------------------------------------------------------------------------


class TestComputeTauCharPerToken:
    def test_cjk_chars_per_token_lower(self) -> None:
        """Lower chars_per_token (CJK ~1.5) increases tau for strings."""
        f = make_field("string", {"maxLength": 30})
        tau_en, _ = compute_tau(f, 4.0)
        tau_cjk, _ = compute_tau(f, 1.5)
        assert tau_cjk >= tau_en

    def test_english_chars_per_token_standard(self) -> None:
        """English chars_per_token ~4.0 is the baseline."""
        f = make_field("string", {"maxLength": 40})
        tau, _ = compute_tau(f, 4.0)
        assert tau == float(math.ceil(40 / 4.0))

    def test_tau_always_at_least_1(self) -> None:
        """tau is always >= 1.0 regardless of input."""
        for type_ in ("boolean", "null", "integer", "number", "string", "array", "object"):
            f = make_field(type_)
            tau, _ = compute_tau(f, 4.0)
            assert tau >= 1.0, f"Expected tau >= 1.0 for type={type_}, got {tau}"

    def test_zero_chars_per_token_uses_default(self) -> None:
        """chars_per_token=0 falls back to default without crashing."""
        f = make_field("string", {"maxLength": 20})
        tau, _ = compute_tau(f, 0.0)
        assert tau >= 1.0


# ---------------------------------------------------------------------------
# Return type tests
# ---------------------------------------------------------------------------


class TestComputeTauReturnType:
    def test_returns_tuple_of_two_floats(self) -> None:
        """compute_tau returns a 2-tuple."""
        f = make_field("string")
        result = compute_tau(f, 4.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_tau_is_float(self) -> None:
        """First element of return tuple is a float."""
        f = make_field("integer")
        tau, _ = compute_tau(f, 4.0)
        assert isinstance(tau, float)

    def test_var_tau_non_negative(self) -> None:
        """Second element (var_tau) is always >= 0.0."""
        for type_ in ("boolean", "null", "integer", "number", "string", "array", "object"):
            f = make_field(type_)
            _, var_tau = compute_tau(f, 4.0)
            assert var_tau >= 0.0, f"var_tau negative for type={type_}"


# ---------------------------------------------------------------------------
# _compute_enum_tau helper
# ---------------------------------------------------------------------------


class TestComputeEnumTau:
    def test_single_char_returns_1(self) -> None:
        """Single character enum value → tau=1.0."""
        result = _compute_enum_tau(["A"], 4.0)
        assert result == 1.0

    def test_empty_list_returns_1(self) -> None:
        """Empty enum values list → tau=1.0."""
        result = _compute_enum_tau([], 4.0)
        assert result == 1.0

    def test_longer_value_increases_tau(self) -> None:
        """Longer enum values produce higher tau."""
        short = _compute_enum_tau(["AB"], 4.0)
        long_ = _compute_enum_tau(["ABCDEFGHIJK"], 4.0)
        assert long_ >= short


# ---------------------------------------------------------------------------
# Property-based tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestComputeTauProperties:
    """Hypothesis property tests to catch rare edge cases."""

    @given(
        type_=st.sampled_from(
            ["boolean", "null", "integer", "number", "string", "enum", "array", "object"]
        ),
        chars_per_token=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_tau_always_returns_positive(self, type_: str, chars_per_token: float) -> None:
        """Property: tau(f) >= 1.0 for any field type and chars_per_token."""
        constraints = {}
        if type_ == "enum":
            constraints["enum"] = ["a", "b", "c"]
        elif type_ == "string":
            constraints["maxLength"] = 100

        f = make_field(type_, constraints)
        tau, _ = compute_tau(f, chars_per_token)
        assert tau >= 1.0, f"tau={tau} < 1.0 for type={type_}"

    @given(
        type_=st.sampled_from(
            ["boolean", "null", "integer", "number", "string", "enum", "array", "object"]
        ),
        chars_per_token=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_var_tau_always_non_negative(self, type_: str, chars_per_token: float) -> None:
        """Property: var_tau(f) >= 0 for any field."""
        constraints = {}
        if type_ == "enum":
            constraints["enum"] = ["x", "y", "z"]
        elif type_ == "string":
            constraints["maxLength"] = 50

        f = make_field(type_, constraints)
        _, var_tau = compute_tau(f, chars_per_token)
        assert var_tau >= 0.0, f"var_tau={var_tau} < 0.0 for type={type_}"

    @given(
        max_length=st.integers(min_value=1, max_value=10000),
        chars_per_token=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_string_tau_monotonic_in_maxlength(
        self, max_length: int, chars_per_token: float
    ) -> None:
        """Property: tau increases (or stays same) as maxLength increases."""
        if max_length < 2:
            return

        f1 = make_field("string", {"maxLength": max_length})
        f2 = make_field("string", {"maxLength": max_length * 2})

        tau1, _ = compute_tau(f1, chars_per_token)
        tau2, _ = compute_tau(f2, chars_per_token)

        assert tau2 >= tau1, (
            f"tau not monotonic: τ({max_length})={tau1} > τ({max_length * 2})={tau2}"
        )

    @given(
        expected_size=st.integers(min_value=1, max_value=100),
        chars_per_token=st.floats(min_value=0.1, max_value=10.0),
    )
    def test_array_tau_scales_with_size(self, expected_size: int, chars_per_token: float) -> None:
        """Property: array tau ~ element_tau * expected_size (for boolean elements)."""
        f = make_field("array", {"items": {"type": "boolean"}})
        tau, _ = compute_tau(f, chars_per_token, expected_array_size=expected_size)

        # For boolean elements, element_tau = 1.0
        # So array tau should be exactly expected_size (but >= 1.0 at minimum)
        expected_tau = max(float(expected_size), 1.0)
        assert tau == expected_tau, f"array tau={tau}, expected {expected_tau}"

    @given(st.text(min_size=1, max_size=1000))
    def test_enum_never_crashes_on_any_strings(self, text: str) -> None:
        """Property: _compute_enum_tau never crashes on any string input."""
        try:
            result = _compute_enum_tau([text, "other", "values"], 4.0)
            assert result >= 1.0
        except Exception as e:
            raise AssertionError(f"_compute_enum_tau crashed on input {text!r}: {e}") from e
