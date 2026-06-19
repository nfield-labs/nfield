"""Tests for the document-first extraction prompt.

Locks the prompt contract: the document precedes the field list, the task is framed
as the value being present (anti-null), and a worked example is shown.
"""

from __future__ import annotations

from formatshield.extraction._papt import TemplateType
from formatshield.extraction._prompt import build_extraction_prompt
from formatshield.schema._types import Field


def _fields() -> list[Field]:
    return [
        Field("a.x", "integer", {}, "a", {"description": "the x"}),
        Field("a.y", "string", {}, "a", {"description": "the y"}),
    ]


class TestDocumentFirstPrompt:
    def test_document_comes_before_fields(self):
        msgs = build_extraction_prompt(_fields(), "X is 42.", TemplateType.STANDARD)
        user = msgs[1]["content"]
        assert user.index("Document:") < user.index("Fields to extract")

    def test_anti_null_framing_and_example_in_system(self):
        msgs = build_extraction_prompt(_fields(), "X is 42.", TemplateType.STANDARD)
        system = msgs[0]["content"]
        assert "value appears in the document" in system
        assert "a.y = NULL" in system  # worked example shows the absent case

    def test_instructions_first_then_document_then_fields(self):
        msgs = build_extraction_prompt(
            _fields(), "X is 42.", TemplateType.STANDARD, instructions="Be exact."
        )
        user = msgs[1]["content"]
        assert user.index("Be exact.") < user.index("Document:") < user.index("Fields to extract")
