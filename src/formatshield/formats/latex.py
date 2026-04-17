"""LaTeX output format for FormatShield.

Provides prompt wrapping and answer extraction for LaTeX-structured outputs.
LaTeX is optimal for math reasoning tasks where answers appear in \\boxed{}.
"""

from __future__ import annotations

import re
from typing import Any

# Regex to extract content from \boxed{} — handles nested braces
_BOXED_RE: re.Pattern[str] = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
# Regex for \begin{answer}...\end{answer} environment
_ENV_RE: re.Pattern[str] = re.compile(r"\\begin\{answer\}(.*?)\\end\{answer\}", re.DOTALL)


class LaTeXFormat:
    """LaTeX output format handler.

    Wraps prompts with LaTeX formatting instructions and extracts
    answers from \\boxed{} or answer environments.

    Example::

        fmt = LaTeXFormat()
        prompt = fmt.wrap_prompt("Solve: x^2 + 3x + 2 = 0")
        response = model.generate(prompt)
        answer = fmt.extract_answer(response)  # extracts from \\boxed{}
    """

    name: str = "latex"

    def wrap_prompt(self, prompt: str, schema: dict[str, Any] | None = None) -> str:
        """Wrap a prompt with LaTeX output instructions.

        Args:
            prompt: The original user prompt.
            schema: Ignored for LaTeX format (LaTeX is structurally self-describing).

        Returns:
            Prompt with LaTeX formatting instructions.
        """
        return (
            f"{prompt}"
            "\n\nShow your reasoning step by step. "
            "Place your final answer inside \\boxed{} — for example: \\boxed{42}"
        )

    def extract_answer(self, text: str) -> str:
        """Extract the answer from LaTeX-formatted model output.

        Tries \\boxed{} first, then \\begin{answer}...\\end{answer}.

        Args:
            text: Raw model output potentially containing LaTeX answer markers.

        Returns:
            Extracted answer text, or the full text if no markers found.
        """
        # Try \boxed{} first (most common in math)
        match = _BOXED_RE.search(text)
        if match:
            return match.group(1).strip()
        # Try \begin{answer}...\end{answer}
        match = _ENV_RE.search(text)
        if match:
            return match.group(1).strip()
        return text.strip()

    def load_grammar(self) -> str:
        """Return an EBNF grammar string for LaTeX-constrained decoding.

        Returns:
            EBNF grammar string.
        """
        return 'root ::= text "\\\\boxed{" answer "}"\ntext ::= [^\\\\]*\nanswer ::= [^}]*'


def wrap_latex_prompt(prompt: str, schema: dict[str, Any] | None = None) -> str:
    """Module-level convenience for LaTeX prompt wrapping.

    Args:
        prompt: The original user prompt.
        schema: Ignored for LaTeX format.

    Returns:
        Prompt with LaTeX formatting instructions.
    """
    return LaTeXFormat().wrap_prompt(prompt, schema)


def extract_latex_answer(text: str) -> str:
    """Module-level convenience for LaTeX answer extraction.

    Args:
        text: Raw model output text.

    Returns:
        Extracted answer from \\boxed{} or answer environment.
    """
    return LaTeXFormat().extract_answer(text)
