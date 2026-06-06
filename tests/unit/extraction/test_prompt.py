"""Unit tests for extraction._prompt — prompt construction."""

from __future__ import annotations

import pytest

from formatshield.extraction._papt import TemplateType
from formatshield.extraction._prompt import (
    build_extraction_prompt,
    build_retry_system_message,
    build_schema_description_block,
    estimate_prompt_tokens,
)
from formatshield.schema._types import Field

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_field(path: str, ftype: str, description: str = "") -> Field:
    return Field(
        path=path,
        type=ftype,
        constraints={},
        parent_path="",
        schema_node={"description": description} if description else {},
    )


class TestPromptContextPrepend:
    """Caller system/user prompts are prepended, SFEP contract preserved."""

    def test_extraction_prompt_prepends_system_and_user(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt(
            [f],
            "doc",
            TemplateType.STANDARD,
            system_prompt="DOMAIN: clinical notes.",
            user_prompt="Be precise about dosages.",
        )
        system, user = msgs[0]["content"], msgs[1]["content"]
        # Caller context appears...
        assert system.startswith("DOMAIN: clinical notes.")
        assert "Be precise about dosages." in user
        # ...and the built-in SFEP contract is still there (parsing stays valid).
        assert "OUTPUT FORMAT" in system
        assert "field.path = value" in system

    def test_empty_context_leaves_prompt_unchanged(self):
        f = make_field("name", "string")
        base = build_extraction_prompt([f], "doc", TemplateType.STANDARD)
        with_empty = build_extraction_prompt(
            [f], "doc", TemplateType.STANDARD, system_prompt="", user_prompt="  "
        )
        assert base[0]["content"] == with_empty[0]["content"]
        assert base[1]["content"] == with_empty[1]["content"]

    def test_retry_prompt_prepends_system(self):
        f = make_field("age", "integer")
        msgs = build_retry_system_message(
            [f], {"age": "bad"}, "doc", system_prompt="DOMAIN: finance."
        )
        assert msgs[0]["content"].startswith("DOMAIN: finance.")
        assert "RE-EXTRACTION" in msgs[0]["content"]


# ---------------------------------------------------------------------------
# build_extraction_prompt
# ---------------------------------------------------------------------------


class TestBuildExtractionPrompt:
    def test_returns_two_messages(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        assert msgs[0]["role"] == "system"

    def test_second_message_is_user(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        assert msgs[1]["role"] == "user"

    def test_system_contains_sfep_instructions(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        system = msgs[0]["content"]
        assert "BEGIN EXTRACTION" in system
        assert "field.path = value" in system

    def test_system_contains_null_rule(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        assert "NULL" in msgs[0]["content"]

    def test_user_contains_field_name(self):
        f = make_field("invoice_number", "string")
        msgs = build_extraction_prompt([f], "doc text", TemplateType.STANDARD)
        assert "invoice_number" in msgs[1]["content"]

    def test_user_contains_document(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "Patient: Alice Smith", TemplateType.STANDARD)
        assert "Alice Smith" in msgs[1]["content"]

    def test_user_contains_field_descriptions_in_standard(self):
        f = make_field("age", "integer", description="Patient age in years")
        msgs = build_extraction_prompt([f], "doc", TemplateType.STANDARD)
        assert "Patient age in years" in msgs[1]["content"]

    def test_descriptions_always_sent_even_in_concise(self):
        f = make_field("age", "integer", description="Patient age")
        msgs = build_extraction_prompt([f], "doc", TemplateType.CONCISE)
        # Description is never dropped — the model needs it to understand the field.
        assert "Patient age" in msgs[1]["content"]
        assert "age (integer)" in msgs[1]["content"]

    def test_empty_fields_raises_value_error(self):
        with pytest.raises(ValueError, match="fields must be non-empty"):
            build_extraction_prompt([], "doc", TemplateType.STANDARD)

    def test_empty_document_excerpt(self):
        f = make_field("x", "string")
        msgs = build_extraction_prompt([f], "", TemplateType.STANDARD)
        assert "no document provided" in msgs[1]["content"]

    def test_multiple_fields_all_present(self):
        fields = [
            make_field("name", "string"),
            make_field("age", "integer"),
            make_field("active", "boolean"),
        ]
        msgs = build_extraction_prompt(fields, "doc", TemplateType.STANDARD)
        user = msgs[1]["content"]
        assert "name" in user
        assert "age" in user
        assert "active" in user

    def test_messages_have_string_content(self):
        f = make_field("x", "string")
        msgs = build_extraction_prompt([f], "doc", TemplateType.STANDARD)
        for msg in msgs:
            assert isinstance(msg["content"], str)


# ---------------------------------------------------------------------------
# build_retry_system_message
# ---------------------------------------------------------------------------


class TestBuildRetrySystemMessage:
    def test_returns_two_messages(self):
        f = make_field("age", "integer")
        msgs = build_retry_system_message([f], {"age": "expected integer"}, "doc")
        assert len(msgs) == 2

    def test_system_message_has_retry_instructions(self):
        f = make_field("age", "integer")
        msgs = build_retry_system_message([f], {"age": "expected integer"}, "doc")
        assert "RE-EXTRACTION" in msgs[0]["content"] or "re-extract" in msgs[0]["content"].lower()

    def test_user_contains_error_message(self):
        f = make_field("age", "integer")
        msgs = build_retry_system_message([f], {"age": "Cannot cast 'thirty' to integer"}, "doc")
        assert "thirty" in msgs[1]["content"]

    def test_user_contains_field_path(self):
        f = make_field("invoice.total", "number")
        msgs = build_retry_system_message([f], {"invoice.total": "parse error"}, "doc")
        assert "invoice.total" in msgs[1]["content"]

    def test_user_contains_document(self):
        f = make_field("x", "string")
        msgs = build_retry_system_message([f], {"x": "error"}, "Patient: Bob Jones")
        assert "Bob Jones" in msgs[1]["content"]


class TestKnowledgeFallback:
    """The knowledge_fallback flag swaps the sourcing rule in the system message."""

    def test_strict_grounding_is_default(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc", TemplateType.STANDARD)
        system = msgs[0]["content"]
        assert "Use NULL if a field is not found in the document" in system
        assert "well-established knowledge" not in system

    def test_knowledge_fallback_changes_sourcing_rule(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt([f], "doc", TemplateType.STANDARD, knowledge_fallback=True)
        system = msgs[0]["content"]
        assert "well-established knowledge" in system
        # The SFEP format contract is otherwise intact.
        assert "field.path = value" in system

    def test_retry_message_honours_knowledge_fallback(self):
        f = make_field("notable_trait", "string")
        strict = build_retry_system_message([f], {"notable_trait": "err"}, "doc")
        loose = build_retry_system_message(
            [f], {"notable_trait": "err"}, "doc", knowledge_fallback=True
        )
        assert "well-established knowledge" not in strict[0]["content"]
        assert "well-established knowledge" in loose[0]["content"]


# ---------------------------------------------------------------------------
# estimate_prompt_tokens + build_schema_description_block
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_estimate_tokens_positive(self):
        f = make_field("name", "string")
        tokens = estimate_prompt_tokens([f], "short document", TemplateType.STANDARD)
        assert tokens > 0

    def test_estimate_tokens_increases_with_more_fields(self):
        fields_few = [make_field("x", "string")]
        fields_many = [make_field(f"f{i}", "string") for i in range(20)]
        few_tokens = estimate_prompt_tokens(fields_few, "doc", TemplateType.STANDARD)
        many_tokens = estimate_prompt_tokens(fields_many, "doc", TemplateType.STANDARD)
        assert many_tokens > few_tokens

    def test_schema_description_block_contains_field(self):
        f = make_field("name", "string")
        block = build_schema_description_block([f], TemplateType.CONCISE)
        assert "name" in block

    def test_schema_description_block_multiple_fields(self):
        fields = [make_field("a", "string"), make_field("b", "integer")]
        block = build_schema_description_block(fields, TemplateType.STANDARD)
        assert "a" in block
        assert "b" in block
