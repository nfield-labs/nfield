"""Tests for formatshield.oracle.routing_score — closed-form Φ routing score."""

from __future__ import annotations

from formatshield.oracle.routing_score import RoutingScore, compute_routing_score

SIMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer"},
    },
}

COMPLEX_SCHEMA = {
    "type": "object",
    "properties": {
        "itinerary": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "day": {"type": "integer"},
                    "hotel": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "stars": {"type": "integer", "minimum": 1, "maximum": 5},
                        },
                    },
                    "total_cost": {"type": "number"},
                    "budget_remaining": {"type": "number"},
                },
            },
        }
    },
}

SIMPLE_PROMPT = "Extract name and age from: John Smith is 34 years old."
COMPLEX_PROMPT = "Plan a detailed 10-day European trip with a budget of $8000 including hotels."


class TestComputeRoutingScore:
    def test_returns_routing_score_dataclass(self) -> None:
        result = compute_routing_score(SIMPLE_PROMPT, SIMPLE_SCHEMA)
        assert isinstance(result, RoutingScore)

    def test_phi_in_unit_interval(self) -> None:
        for prompt, schema in [
            (SIMPLE_PROMPT, SIMPLE_SCHEMA),
            (COMPLEX_PROMPT, COMPLEX_SCHEMA),
        ]:
            rs = compute_routing_score(prompt, schema)
            assert 0.0 <= rs.phi <= 1.0, f"Φ={rs.phi} out of [0,1]"

    def test_components_in_unit_interval(self) -> None:
        rs = compute_routing_score(SIMPLE_PROMPT, SIMPLE_SCHEMA)
        assert 0.0 <= rs.lambda2 <= 1.0
        assert 0.0 <= rs.tau <= 1.0
        assert 0.0 <= rs.delta_k <= 1.0

    def test_simple_schema_aligned_prompt_below_threshold(self) -> None:
        # Simple 2-field schema with aligned prompt → Φ < 0.5 → direct
        rs = compute_routing_score(SIMPLE_PROMPT, SIMPLE_SCHEMA)
        assert rs.phi < 0.5, f"Expected Φ < 0.5 for simple aligned schema, got Φ={rs.phi:.3f}"

    def test_complex_schema_distant_prompt_above_threshold(self) -> None:
        # Deeply nested schema with semantically distant prompt → Φ > 0.5 → TTF
        rs = compute_routing_score(COMPLEX_PROMPT, COMPLEX_SCHEMA)
        assert rs.phi > 0.5, f"Expected Φ > 0.5 for complex nested schema, got Φ={rs.phi:.3f}"

    def test_non_dict_schema_returns_neutral(self) -> None:
        rs = compute_routing_score("some prompt", None)  # type: ignore[arg-type]
        assert rs.phi == 0.5
        assert rs.explanation == "Φ=0.500 (no schema — neutral routing)"

    def test_monotonicity_complexity(self) -> None:
        # Adding more nested fields should push Φ higher
        simple_rs = compute_routing_score(COMPLEX_PROMPT, SIMPLE_SCHEMA)
        complex_rs = compute_routing_score(COMPLEX_PROMPT, COMPLEX_SCHEMA)
        assert complex_rs.phi >= simple_rs.phi, (
            f"Complex schema should have Φ ≥ simple schema: "
            f"complex={complex_rs.phi:.3f} vs simple={simple_rs.phi:.3f}"
        )

    def test_explanation_contains_components(self) -> None:
        rs = compute_routing_score(SIMPLE_PROMPT, SIMPLE_SCHEMA)
        assert "Φ=" in rs.explanation
        assert "λ̃₂=" in rs.explanation
        assert "τ=" in rs.explanation
        assert "ΔK=" in rs.explanation

    def test_empty_schema_neutral_to_low(self) -> None:
        rs = compute_routing_score(SIMPLE_PROMPT, {})
        # Empty schema: no fields → λ̃₂=0, τ=0, ΔK=0.5 → Φ driven by ΔK alone
        assert 0.0 <= rs.phi <= 1.0
