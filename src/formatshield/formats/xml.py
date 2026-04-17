"""XML output format for FormatShield.

Provides prompt wrapping and answer extraction for XML-structured outputs.
XML is optimal for document extraction, NER, and hierarchical data tasks.
"""

from __future__ import annotations

import re
from typing import Any

# Tag used to wrap the final answer in XML format
_ANSWER_TAG: str = "answer"
# Regex to extract content from answer tags (non-greedy)
_ANSWER_RE: re.Pattern[str] = re.compile(rf"<{_ANSWER_TAG}>(.*?)</{_ANSWER_TAG}>", re.DOTALL)


class XMLFormat:
    """XML output format handler.

    Wraps prompts with XML schema hints and extracts XML answers
    from model responses.

    Example::

        fmt = XMLFormat()
        prompt = fmt.wrap_prompt("Extract entities", {"entities": "list"})
        response = model.generate(prompt)
        answer = fmt.extract_answer(response)
    """

    name: str = "xml"

    def wrap_prompt(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        """Wrap a prompt with XML output instructions.

        Args:
            prompt: The original user prompt.
            schema: Optional schema hint for XML structure.

        Returns:
            Prompt augmented with XML formatting instructions.
        """
        schema_hint = ""
        if schema:
            # Build XML structure hint from schema keys
            keys = list(schema.get("properties", {}).keys()) if isinstance(schema, dict) else []
            if keys:
                example_tags = "\n".join(f"  <{k}>...</{k}>" for k in keys[:5])
                schema_hint = (
                    f"\n\nExpected XML structure:\n"
                    f"<{_ANSWER_TAG}>\n{example_tags}\n</{_ANSWER_TAG}>"
                )

        return (
            f"{prompt}"
            f"{schema_hint}"
            f"\n\nRespond with your answer wrapped in <{_ANSWER_TAG}>...</{_ANSWER_TAG}> tags."
        )

    def extract_answer(self, text: str) -> str:
        """Extract the answer from XML-tagged model output.

        Args:
            text: Raw model output potentially containing XML answer tags.

        Returns:
            Extracted answer text, or the full text if no tags found.
        """
        match = _ANSWER_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def load_grammar(self) -> str:
        """Return an EBNF grammar string for XML-constrained decoding.

        Returns:
            EBNF grammar string compatible with SGLang/outlines grammar mode.
        """
        # Simplified EBNF for XML answer blocks
        return f'root ::= "<{_ANSWER_TAG}>" content "</{_ANSWER_TAG}>"\ncontent ::= [^<]*'


def wrap_xml_prompt(prompt: str, schema: dict[str, Any] | None = None) -> str:
    """Module-level convenience function for XML prompt wrapping.

    Args:
        prompt: The original user prompt.
        schema: Optional schema dict for structure hints.

    Returns:
        Prompt with XML output instructions.
    """
    return XMLFormat().wrap_prompt(prompt, schema)


def extract_xml_answer(text: str) -> str:
    """Module-level convenience function for XML answer extraction.

    Args:
        text: Raw model output text.

    Returns:
        Extracted answer from XML tags, or full text if not found.
    """
    return XMLFormat().extract_answer(text)
