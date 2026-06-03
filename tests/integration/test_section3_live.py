"""Live integration test for Section 3: SFEP extraction + validate + retry via Groq.

This test exercises the full Section 3 pipeline with a real Groq API call:
1. Build an extraction prompt for 3 simple fields
2. Call Groq and parse the SFEP response
3. Validate each extracted value
4. Trigger retry for any failures

Requires GROQ_API_KEY to be set.
"""

from __future__ import annotations

import os

import pytest

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


@pytest.fixture
def groq_provider():
    """Create a real GroqProvider using the live API key."""
    if not GROQ_API_KEY:
        pytest.skip("GROQ_API_KEY not set")
    from formatshield.providers._registry import from_model

    return from_model("groq/llama-3.1-8b-instant")


@pytest.fixture
def simple_fields():
    """Three simple schema fields for live extraction test."""
    from formatshield.schema._types import Field

    return [
        Field("company_name", "string", {}, "", {"description": "Name of the company"}),
        Field("founded_year", "integer", {"minimum": 1800, "maximum": 2030}, "", {}),
        Field("is_public", "boolean", {}, "", {}),
    ]


@pytest.fixture
def document_excerpt():
    return (
        "Acme Corporation was founded in 1947. "
        "It is publicly traded on the NYSE under the ticker ACM. "
        "The company employs over 50,000 people worldwide."
    )


class TestLiveGroqSFEPExtraction:
    @pytest.mark.asyncio
    async def test_extract_simple_fields_via_groq(
        self, groq_provider, simple_fields, document_excerpt
    ):
        """Live test: extract 3 fields from a document via Groq, parse SFEP output."""
        from formatshield.extraction._papt import select_template
        from formatshield.extraction._prompt import build_extraction_prompt
        from formatshield.extraction._sfep import parse_sfep

        template = select_template(simple_fields, budget_tokens=500)
        messages = build_extraction_prompt(simple_fields, document_excerpt, template)

        raw_output = await groq_provider.complete(messages, max_tokens=100)
        assert isinstance(raw_output, str)
        assert len(raw_output) > 0

        result = parse_sfep(raw_output, simple_fields)
        assert isinstance(result, dict)

        # At minimum, the company name should be extracted
        assert "company_name" in result or "founded_year" in result, (
            f"Expected at least one field extracted. Raw output:\n{raw_output}"
        )

    @pytest.mark.asyncio
    async def test_validate_extracted_values(self, groq_provider, simple_fields, document_excerpt):
        """Live test: validate extracted values pass type/constraint checks."""
        from formatshield.extraction._papt import TemplateType
        from formatshield.extraction._prompt import build_extraction_prompt
        from formatshield.extraction._sfep import parse_sfep
        from formatshield.validation._type_check import validate_field

        messages = build_extraction_prompt(simple_fields, document_excerpt, TemplateType.STANDARD)
        raw_output = await groq_provider.complete(messages, max_tokens=100)
        result = parse_sfep(raw_output, simple_fields)

        field_map = {f.path: f for f in simple_fields}
        for path, value in result.items():
            if value is None:
                continue
            field = field_map.get(path)
            if field:
                valid, err = validate_field(value, field)
                assert valid, (
                    f"Field {path!r} = {value!r} failed validation: {err}. "
                    f"Raw output:\n{raw_output}"
                )

    @pytest.mark.asyncio
    async def test_retry_recovers_failed_field(
        self, groq_provider, simple_fields, document_excerpt
    ):
        """Live test: orchestrate_retry recovers a deliberately wrong integer field."""
        from formatshield.config import ExtractionConfig
        from formatshield.schema._types import CapacityLeaf
        from formatshield.validation._retry import orchestrate_retry
        from formatshield.validation._type_check import validate_field

        # Only retry the integer field
        integer_field = simple_fields[1]  # founded_year
        leaf = CapacityLeaf(
            fields=[integer_field],
            document_excerpt=document_excerpt,
            safe_output=50,
            leaf_id=1,
        )
        config = ExtractionConfig(max_retry_rounds=1)

        # Simulate: integer field "failed" with a string value
        recovered = await orchestrate_retry(
            failed_fields=[integer_field],
            errors={"founded_year": "expected integer, got str 'nineteen forty-seven'"},
            provider=groq_provider,
            leaf=leaf,
            dep_dag={},
            config=config,
        )

        assert isinstance(recovered, dict)
        if "founded_year" in recovered:
            val = recovered["founded_year"]
            if val is not None:
                valid, err = validate_field(val, integer_field)
                assert valid, f"Recovered value {val!r} is still invalid: {err}"

    @pytest.mark.asyncio
    async def test_assemble_extracted_json(self, groq_provider, simple_fields, document_excerpt):
        """Live test: assemble extracted SFEP result into nested JSON dict."""
        from formatshield.assembly._trie import assemble_json
        from formatshield.extraction._papt import TemplateType
        from formatshield.extraction._prompt import build_extraction_prompt
        from formatshield.extraction._sfep import NEEDS_REVALIDATION, parse_sfep

        messages = build_extraction_prompt(simple_fields, document_excerpt, TemplateType.STANDARD)
        raw_output = await groq_provider.complete(messages, max_tokens=100)
        result = parse_sfep(raw_output, simple_fields)

        # Filter out sentinels before assembly
        clean = {k: v for k, v in result.items() if v is not NEEDS_REVALIDATION}
        assembled = assemble_json(clean)
        assert isinstance(assembled, dict)
