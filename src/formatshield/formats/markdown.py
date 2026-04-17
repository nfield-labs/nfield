"""Markdown output format for FormatShield.

Provides prompt wrapping and answer extraction for Markdown-structured outputs.
Markdown is optimal for reports, lists, and general prose with structure.
"""

from __future__ import annotations

import re
from typing import Any

# Regex to extract content from ```answer ... ``` code fences
_FENCE_RE: re.Pattern[str] = re.compile(r"```answer\s*\n(.*?)\n```", re.DOTALL)
# Regex for ## Answer sections
_SECTION_RE: re.Pattern[str] = re.compile(r"##\s+Answer\s*\n(.*?)(?:\n##|\Z)", re.DOTALL)


class MarkdownFormat:
    """Markdown output format handler.

    Wraps prompts with Markdown formatting instructions and extracts
    structured answers from Markdown output.

    Example::

        fmt = MarkdownFormat()
        prompt = fmt.wrap_prompt("Summarize the following...", schema=None)
        response = model.generate(prompt)
        answer = fmt.extract_answer(response)
    """

    name: str = "markdown"

    def wrap_prompt(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        """Wrap a prompt with Markdown output instructions.

        Args:
            prompt: The original user prompt.
            schema: Optional schema hint for Markdown section structure.

        Returns:
            Prompt with Markdown formatting instructions.
        """
        schema_hint = ""
        if schema and isinstance(schema, dict):
            keys = list(schema.get("properties", {}).keys())
            if keys:
                section_hints = "\n".join(f"### {k.replace('_', ' ').title()}" for k in keys[:5])
                schema_hint = f"\n\nUse these sections:\n{section_hints}"

        return (
            f"{prompt}{schema_hint}\n\nProvide your answer under a `## Answer` section in Markdown."
        )

    def extract_answer(self, text: str) -> str:
        """Extract the answer from Markdown-formatted model output.

        Tries ```answer fences first, then ## Answer sections.

        Args:
            text: Raw model output potentially containing Markdown answer markers.

        Returns:
            Extracted answer text, or the full text if no markers found.
        """
        match = _FENCE_RE.search(text)
        if match:
            return match.group(1).strip()
        match = _SECTION_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def load_grammar(self) -> str:
        """Return an EBNF grammar string for Markdown-constrained decoding.

        Returns:
            EBNF grammar string.
        """
        return 'root ::= content "## Answer\\n" answer\ncontent ::= [^#]*\nanswer ::= .+'


def wrap_markdown_prompt(prompt: str, schema: dict[str, Any] | None = None) -> str:
    """Module-level convenience for Markdown prompt wrapping.

    Args:
        prompt: The original user prompt.
        schema: Optional schema hint.

    Returns:
        Prompt with Markdown formatting instructions.
    """
    return MarkdownFormat().wrap_prompt(prompt, schema)


def extract_markdown_answer(text: str) -> str:
    """Module-level convenience for Markdown answer extraction.

    Args:
        text: Raw model output text.

    Returns:
        Extracted answer from Markdown structure.
    """
    return MarkdownFormat().extract_answer(text)
