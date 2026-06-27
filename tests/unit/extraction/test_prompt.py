"""Unit tests for extraction._prompt - prompt construction."""

from __future__ import annotations

import pytest

from nfield.extraction._papt import TemplateType
from nfield.extraction._prompt import (
    build_extraction_prompt,
    build_retry_system_message,
)
from nfield.schema._types import Field

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
    """Caller instructions lead the USER message; SFEP contract stays in system."""

    def test_extraction_prompt_prepends_instructions(self):
        f = make_field("name", "string")
        msgs = build_extraction_prompt(
            [f],
            "doc",
            TemplateType.STANDARD,
            instructions="DOMAIN: clinical notes. Be precise about dosages.",
        )
        system, user = msgs[0]["content"], msgs[1]["content"]
        # Caller instructions lead the USER message (Llama follows user-turn
        # instructions more reliably than system-prompt ones).
        assert user.startswith("DOMAIN: clinical notes. Be precise about dosages.")
        # The system message stays the pure SFEP contract (parsing stays valid).
        assert "OUTPUT FORMAT" in system
        assert "field.path = value" in system
        assert "DOMAIN: clinical notes" not in system

    def test_empty_instructions_leaves_prompt_unchanged(self):
        f = make_field("name", "string")
        base = build_extraction_prompt([f], "doc", TemplateType.STANDARD)
        with_empty = build_extraction_prompt([f], "doc", TemplateType.STANDARD, instructions="  ")
        assert base[0]["content"] == with_empty[0]["content"]
        assert base[1]["content"] == with_empty[1]["content"]

    def test_retry_prompt_prepends_instructions(self):
        f = make_field("age", "integer")
        msgs = build_retry_system_message(
            [f], {"age": "bad"}, "doc", instructions="DOMAIN: finance."
        )
        # Instructions lead the user message; the system message is the retry contract.
        assert msgs[1]["content"].startswith("DOMAIN: finance.")
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
        # Description is never dropped - the model needs it to understand the field.
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
        assert "Use NULL only when" in system
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
