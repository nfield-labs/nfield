"""Tests for formatshield.oracle.ncd — Normalized Compression Distance."""

from __future__ import annotations

from formatshield.oracle.ncd import prompt_schema_ncd


class TestPromptSchemaNcd:
    def test_non_dict_schema_returns_neutral(self) -> None:
        assert prompt_schema_ncd("hello", None) == 0.5  # type: ignore[arg-type]
        assert prompt_schema_ncd("hello", "not a dict") == 0.5  # type: ignore[arg-type]
        assert prompt_schema_ncd("hello", []) == 0.5  # type: ignore[arg-type]

    def test_empty_schema_returns_neutral(self) -> None:
        # Empty schema has no fields → _flatten_schema returns "" → neutral
        assert prompt_schema_ncd("hello world", {}) == 0.5

    def test_short_inputs_return_neutral(self) -> None:
        # Inputs shorter than 32 bytes → guard activates → 0.5
        short_schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert prompt_schema_ncd("hi", short_schema) == 0.5

    def test_result_in_unit_interval(self) -> None:
        prompt = "Extract the person's name and age: John Smith is 34 years old."
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        result = prompt_schema_ncd(prompt, schema)
        assert 0.0 <= result <= 1.0

    def test_aligned_prompt_lower_ncd(self) -> None:
        # Prompt that closely mirrors the schema field names → lower NCD (more alignment)
        aligned = (
            "Extract the following fields: name (string), age (integer), "
            "name name name name name name name name name name name"
        )
        unrelated = (
            "Describe the history of the Byzantine Empire and its cultural influence "
            "on medieval European art and architecture during the 9th century."
        )
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
        }
        # Both must pass the 32-byte guard
        ncd_aligned = prompt_schema_ncd(aligned, schema)
        ncd_unrelated = prompt_schema_ncd(unrelated, schema)
        # Aligned prompt shares more tokens with schema field names
        assert ncd_aligned <= ncd_unrelated

    def test_nested_schema_produces_valid_result(self) -> None:
        prompt = "Plan a 10-day European trip with a budget of 800 dollars including hotels."
        schema = {
            "type": "object",
            "properties": {
                "itinerary": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "day": {"type": "integer"},
                            "hotel": {"type": "string"},
                            "cost": {"type": "number"},
                        },
                    },
                }
            },
        }
        result = prompt_schema_ncd(prompt, schema)
        assert 0.0 <= result <= 1.0

    def test_identical_prompt_and_schema_low_ncd(self) -> None:
        # Prompt that exactly repeats the schema flat representation → very low NCD
        schema = {
            "type": "object",
            "properties": {
                "field_alpha": {"type": "string"},
                "field_beta": {"type": "integer"},
            },
        }
        # Build a prompt from the field names (simulates exact alignment)
        prompt = "field_alpha: string\nfield_beta: integer\n" * 5
        result = prompt_schema_ncd(prompt, schema)
        # Should be low (high alignment)
        assert result < 0.5
